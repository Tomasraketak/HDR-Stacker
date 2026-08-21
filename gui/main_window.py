"""
Main Application Window for Astro HDR Stacker.
Features fast proxy-resolution interactive workflow, interactive ROI (e.g. 300x300 px crop around Sun for real-time sub-50ms editing),
manual & assisted frame-by-frame subpixel alignment, and asynchronous full-resolution background rendering.
"""

import os
from typing import List, Optional, Dict, Any, Tuple
import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QProgressBar, QLabel, QFileDialog, QMessageBox, QApplication
)

try:
    from core.exif_and_analysis import ExposureItem
    from core.aligner import ImageAligner, calculate_moon_shifts, apply_shifts_to_images, detect_black_circle_in_light
    from core.merger import HDRMerger
    from core.postprocess import apply_postprocessing, save_image
    from gui.exposure_list_widget import ExposureListWidget
    from gui.image_viewer import ImageViewerContainer
    from gui.controls_panel import ControlsPanel
    from gui.manual_align_dialog import ManualAlignDialog
    from gui.styles import DARK_THEME
except ImportError:
    from ..core.exif_and_analysis import ExposureItem
    from ..core.aligner import ImageAligner, calculate_moon_shifts, apply_shifts_to_images, detect_black_circle_in_light
    from ..core.merger import HDRMerger
    from ..core.postprocess import apply_postprocessing, save_image
    from .exposure_list_widget import ExposureListWidget
    from .image_viewer import ImageViewerContainer
    from .controls_panel import ControlsPanel
    from .manual_align_dialog import ManualAlignDialog
    from .styles import DARK_THEME


class StackingWorker(QThread):
    """Background worker for fast proxy-resolution or ROI crop stacking."""
    progress = pyqtSignal(int, str)
    finished_success = pyqtSignal(object, object)  # (base_f32_bgr, hdr_radiance_map_or_None)
    failed = pyqtSignal(str)

    def __init__(
        self,
        items: List[ExposureItem],
        settings: Dict[str, Any],
        scale: float = 0.25,
        roi_rect: Optional[Tuple[int, int, int, int]] = None
    ):
        super().__init__()
        self.items = items
        self.settings = settings
        self.scale = max(0.05, min(1.0, float(scale)))
        self.roi_rect = roi_rect  # (x, y, w, h) in base coordinates

    def run(self):
        try:
            if len(self.items) < 2:
                self.failed.emit("K HDR složení jsou potřeba alespoň 2 snímky.")
                return

            images = []
            times = []
            total_items = len(self.items)

            for idx, item in enumerate(self.items):
                pct = int(10 + (idx / total_items) * 30)
                self.progress.emit(pct, f"Načítání snímku {idx+1}/{total_items}: {item.filename}")
                
                img = cv2.imdecode(np.fromfile(item.filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    self.failed.emit(f"Nelze načíst soubor: {item.filepath}")
                    return

                # If ROI Crop Mode is active, crop directly from original
                if self.roi_rect is not None:
                    rx, ry, rw, rh = self.roi_rect
                    h_orig, w_orig = img.shape[:2]
                    # Clamp ROI to image boundaries
                    rx_cl = max(0, min(w_orig - 10, rx))
                    ry_cl = max(0, min(h_orig - 10, ry))
                    rw_cl = min(rw, w_orig - rx_cl)
                    rh_cl = min(rh, h_orig - ry_cl)
                    img = img[ry_cl:ry_cl+rh_cl, rx_cl:rx_cl+rw_cl]

                # Downscale for ultra-fast proxy editing if not in ROI mode
                elif self.scale < 0.99:
                    h, w = img.shape[:2]
                    nw, nh = max(16, int(w * self.scale)), max(16, int(h * self.scale))
                    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

                images.append(img)
                times.append(max(1e-6, item.exposure_time))

            h0, w0 = images[0].shape[:2]
            for i, img in enumerate(images):
                if img.shape[:2] != (h0, w0):
                    images[i] = cv2.resize(img, (w0, h0), interpolation=cv2.INTER_AREA)

            # Alignment Check
            has_manual_shifts = any(abs(it.shift_x) > 0.01 or abs(it.shift_y) > 0.01 for it in self.items)
            align_method = self.settings.get('align_method', 'none')

            effective_scale = 1.0 if self.roi_rect is not None else self.scale

            if has_manual_shifts:
                self.progress.emit(45, "Aplikuji manuálně nastavené posuny...")
                shifts = [(it.shift_x, it.shift_y) for it in self.items]
                images = apply_shifts_to_images(images, shifts, scale_factor=effective_scale)
            elif align_method == 'eclipse_disc':
                self.progress.emit(45, "Hledám černý disk Měsíce v záři korony...")
                shifts = calculate_moon_shifts(images)
                images = apply_shifts_to_images(images, shifts, scale_factor=1.0)
                for idx, (dx, dy) in enumerate(shifts):
                    if effective_scale > 0:
                        self.items[idx].shift_x = round(dx / effective_scale, 1)
                        self.items[idx].shift_y = round(dy / effective_scale, 1)

            # Merge
            algo = self.settings.get('algo', 'mertens')
            hdr_radiance = None

            if algo == 'mertens':
                self.progress.emit(75, "Probíhá rychlá Laplaceova fúze...")
                base_merged = HDRMerger.merge_mertens(
                    images,
                    contrast_weight=self.settings.get('mertens_contrast', 1.0),
                    saturation_weight=self.settings.get('mertens_saturation', 1.0),
                    exposure_weight=self.settings.get('mertens_exposure', 1.0),
                    progress_callback=lambda p, msg: self.progress.emit(int(75 + p * 0.2), msg)
                )
            elif algo in ('debevec', 'robertson'):
                self.progress.emit(75, f"Generování {algo.capitalize()} HDR mapy...")
                if algo == 'debevec':
                    hdr_radiance, _ = HDRMerger.merge_debevec(images, times)
                else:
                    hdr_radiance, _ = HDRMerger.merge_robertson(images, times)
                self.progress.emit(90, "Tonemapping...")
                base_merged = HDRMerger.tonemap(
                    hdr_radiance,
                    method=self.settings.get('tonemap_method', 'reinhard')
                )
            else:
                self.failed.emit(f"Neznámý algoritmus: {algo}")
                return

            self.progress.emit(100, "Složení dokončeno.")
            self.finished_success.emit(base_merged, hdr_radiance)

        except Exception as e:
            self.failed.emit(f"Chyba při skládání: {str(e)}")


class FullResExportWorker(QThread):
    """Background worker for rendering and saving final 100% full-resolution file."""
    progress = pyqtSignal(int, str)
    finished_success = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, items: List[ExposureItem], settings: Dict[str, Any], export_filepath: str):
        super().__init__()
        self.items = items
        self.settings = settings
        self.export_filepath = export_filepath

    def run(self):
        try:
            total_items = len(self.items)
            images = []
            times = []

            for idx, item in enumerate(self.items):
                pct = int(5 + (idx / total_items) * 35)
                self.progress.emit(pct, f"Načítání plného rozlišení {idx+1}/{total_items}: {item.filename}")
                img = cv2.imdecode(np.fromfile(item.filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    self.failed.emit(f"Nelze načíst soubor: {item.filepath}")
                    return
                images.append(img)
                times.append(max(1e-6, item.exposure_time))

            h0, w0 = images[0].shape[:2]
            for i, img in enumerate(images):
                if img.shape[:2] != (h0, w0):
                    images[i] = cv2.resize(img, (w0, h0), interpolation=cv2.INTER_AREA)

            # Alignment in full resolution
            has_manual_shifts = any(abs(it.shift_x) > 0.01 or abs(it.shift_y) > 0.01 for it in self.items)
            align_method = self.settings.get('align_method', 'none')

            if has_manual_shifts:
                self.progress.emit(42, "Aplikuji posuny v plném rozlišení...")
                shifts = [(it.shift_x, it.shift_y) for it in self.items]
                images = apply_shifts_to_images(images, shifts, scale_factor=1.0)
            elif align_method == 'eclipse_disc':
                self.progress.emit(42, "Zarovnávám disk Měsíce v plném rozlišení...")
                shifts = calculate_moon_shifts(images)
                images = apply_shifts_to_images(images, shifts, scale_factor=1.0)

            algo = self.settings.get('algo', 'mertens')
            hdr_radiance = None

            if algo == 'mertens':
                self.progress.emit(70, "Skládání plné kvality (Mertens Exposure Fusion)...")
                base_merged = HDRMerger.merge_mertens(
                    images,
                    contrast_weight=self.settings.get('mertens_contrast', 1.0),
                    saturation_weight=self.settings.get('mertens_saturation', 1.0),
                    exposure_weight=self.settings.get('mertens_exposure', 1.0),
                    progress_callback=lambda p, msg: self.progress.emit(int(70 + p * 0.15), msg)
                )
            else:
                self.progress.emit(70, f"Generování {algo.capitalize()} 32-bit HDR...")
                if algo == 'debevec':
                    hdr_radiance, _ = HDRMerger.merge_debevec(images, times)
                else:
                    hdr_radiance, _ = HDRMerger.merge_robertson(images, times)
                self.progress.emit(82, "Tonemapping...")
                base_merged = HDRMerger.tonemap(hdr_radiance, method=self.settings.get('tonemap_method', 'reinhard'))

            self.progress.emit(88, "Aplikace postprocessingu a barev v plné kvalitě...")
            final_proc = apply_postprocessing(
                base_merged,
                brightness=self.settings['brightness'],
                contrast=self.settings['contrast'],
                gamma=self.settings['gamma'],
                saturation=self.settings['saturation'],
                coronal_boost=self.settings['coronal_boost'],
                coronal_radius=self.settings['coronal_radius'],
                shadow_lift=self.settings['shadows'],
                highlight_drop=self.settings['highlights'],
                denoise_strength=self.settings['denoise']
            )

            self.progress.emit(95, f"Ukládání souboru {os.path.basename(self.export_filepath)}...")
            success = save_image(self.export_filepath, final_proc, hdr_radiance_map=hdr_radiance, jpeg_quality=100)

            if success:
                self.progress.emit(100, "Export úspěšně dokončen.")
                self.finished_success.emit(self.export_filepath)
            else:
                self.failed.emit("Chyba při zápisu souboru na disk.")

        except Exception as e:
            self.failed.emit(f"Chyba při exportu: {str(e)}")


class MainWindow(QMainWindow):
    """
    Main Application Window.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Astro HDR Stacker — Skládání expozic & Zatmění Slunce")
        self.resize(1380, 880)
        self.setAcceptDrops(True)
        self.setStyleSheet(DARK_THEME)

        self._base_merged_bgr: Optional[np.ndarray] = None
        self._hdr_radiance_map: Optional[np.ndarray] = None
        self._worker: Optional[StackingWorker] = None
        self._export_worker: Optional[FullResExportWorker] = None

        # ROI Crop Mode state
        self._roi_active: bool = False
        self._roi_rect: Optional[Tuple[int, int, int, int]] = None

        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 1. Left panel: Exposure list
        self.exposure_list = ExposureListWidget()
        self.exposure_list.setMinimumWidth(280)
        self.exposure_list.item_selected.connect(self._on_preview_single_exposure)
        splitter.addWidget(self.exposure_list)

        # 2. Center panel: Interactive Image Viewer
        self.viewer_container = ImageViewerContainer()
        self.viewer_container.roi_mode_toggled.connect(self._on_roi_mode_toggled)
        self.viewer_container.center_sun_requested.connect(self._center_roi_on_sun)
        splitter.addWidget(self.viewer_container)

        # 3. Right panel: Controls
        self.controls = ControlsPanel()
        self.controls.setMinimumWidth(320)
        self.controls.stack_requested.connect(self.start_stacking)
        self.controls.manual_align_requested.connect(self.open_manual_alignment)
        self.controls.live_adjust_requested.connect(self._apply_postprocessing_live)
        self.controls.export_requested.connect(self.export_result)
        splitter.addWidget(self.controls)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 6)
        splitter.setStretchFactor(2, 3)
        main_layout.addWidget(splitter)

        # Bottom Bar
        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(4, 2, 4, 2)
        
        self.lbl_status = QLabel("Přetáhněte sem sérii fotografií nebo klikněte na '+ Přidat fotky'.")
        self.lbl_status.setStyleSheet("color: #b0bac9; font-weight: 500;")
        bottom_bar.addWidget(self.lbl_status)
        bottom_bar.addStretch()

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(260)
        self.progress_bar.setVisible(False)
        bottom_bar.addWidget(self.progress_bar)

        main_layout.addLayout(bottom_bar)

    # ------------------ Drag and Drop ------------------
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        filepaths = []
        for u in urls:
            path = u.toLocalFile()
            if os.path.isfile(path):
                ext = os.path.splitext(path)[1].lower()
                if ext in ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng', '.bmp'):
                    filepaths.append(path)
            elif os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for f in files:
                        ext = os.path.splitext(f)[1].lower()
                        if ext in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
                            filepaths.append(os.path.join(root, f))
        
        if filepaths:
            self.exposure_list.load_files(filepaths)
            self.lbl_status.setText(f"Načteno {len(filepaths)} snímků. Připraveno ke složení.")

    # ------------------ Preview single exposure ------------------
    def _on_preview_single_exposure(self, filepath: str):
        try:
            img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.viewer_container.viewer.set_image_rgb_uint8(rgb, keep_view=True)
                self.viewer_container.viewer.set_compare_image_bgr_float(img.astype(np.float32) / 255.0)
                self.lbl_status.setText(f"Náhled expozice: {os.path.basename(filepath)}")
        except Exception as e:
            print(f"Error previewing file {filepath}: {e}")

    # ------------------ ROI Mode & Sun Centering ------------------
    def _on_roi_mode_toggled(self, enabled: bool, x: int, y: int, w: int, h: int):
        self._roi_active = enabled
        self._roi_rect = (x, y, w, h) if enabled else None
        
        if enabled:
            self.lbl_status.setText(f"🎯 Aktivován bleskový výřez ROI {w}x{h} px. Provádím okamžité složení...")
            self.start_stacking()
        else:
            self.lbl_status.setText("Zobrazen celý snímek. Provádím složení plné scény...")
            self.start_stacking()

    def _center_roi_on_sun(self):
        """Finds Sun / Moon disc and centers the ROI box around it."""
        active_items = self.exposure_list.get_active_items()
        if not active_items:
            return
        
        ref_item = active_items[len(active_items) // 2]
        img = cv2.imdecode(np.fromfile(ref_item.filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return

        disc = detect_black_circle_in_light(img)
        if disc is not None:
            cx, cy, _ = disc
            self.viewer_container.viewer.set_roi_center(int(cx), int(cy))
            self.lbl_status.setText(f"☀️ Slunce nalezeno na souřadnicích [{int(cx)}, {int(cy)}]. Výřez vycentrován.")
        else:
            h, w = img.shape[:2]
            self.viewer_container.viewer.set_roi_center(w // 2, h // 2)
            self.lbl_status.setText("Slunce nebylo automaticky nalezeno, výřez vycentrován na střed snímku.")

    # ------------------ Manual Alignment Dialog ------------------
    def open_manual_alignment(self):
        items = self.exposure_list.get_active_items()
        if len(items) < 2:
            QMessageBox.warning(self, "Nedostatek snímků", "Pro zarovnání načtěte alespoň 2 aktivní snímky.")
            return

        dialog = ManualAlignDialog(items, parent=self)
        dialog.shifts_applied.connect(self.start_stacking)
        dialog.exec()

    # ------------------ Stacking Execution ------------------
    def start_stacking(self):
        items = self.exposure_list.get_active_items()
        if len(items) < 2:
            QMessageBox.warning(
                self,
                "Nedostatek snímků",
                "Pro HDR složení vyberte v seznamu alespoň 2 aktivní expozice."
            )
            return

        self.controls.btn_stack.setEnabled(False)
        self.controls.btn_export.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        settings = self.controls.get_settings()
        scale = settings.get('proxy_scale', 0.25)
        
        roi = self._roi_rect if self._roi_active else None

        if roi:
            self.lbl_status.setText(f"Zahajuji bleskové složení výřezu {roi[2]}x{roi[3]} px...")
        else:
            self.lbl_status.setText("Zahajuji rychlé skládání scény...")

        self._worker = StackingWorker(items, settings, scale=scale, roi_rect=roi)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished_success.connect(self._on_stacking_success)
        self._worker.failed.connect(self._on_stacking_failed)
        self._worker.start()

    def _on_worker_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.lbl_status.setText(msg)

    def _on_stacking_success(self, base_bgr: np.ndarray, hdr_radiance: Optional[np.ndarray]):
        self.controls.btn_stack.setEnabled(True)
        self.controls.btn_export.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self._base_merged_bgr = base_bgr
        self._hdr_radiance_map = hdr_radiance

        active_items = self.exposure_list.get_active_items()
        if active_items and not self._roi_active:
            mid_idx = len(active_items) // 2
            mid_path = active_items[mid_idx].filepath
            mid_img = cv2.imdecode(np.fromfile(mid_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if mid_img is not None:
                h, w = base_bgr.shape[:2]
                mid_small = cv2.resize(mid_img, (w, h), interpolation=cv2.INTER_AREA)
                self.viewer_container.viewer.set_compare_image_bgr_float(mid_small.astype(np.float32) / 255.0)

        self._apply_postprocessing_live()
        
        if self._roi_active:
            self.lbl_status.setText("⚡ Bleskový výřez ROI složen (odezva <50ms)! Upravujte posuvníky v reálném čase.")
        else:
            scale_pct = int(self.controls.get_settings().get('proxy_scale', 0.25) * 100)
            self.lbl_status.setText(f"✅ Složeno ({scale_pct}% rychlý náhled). Posuvníky reagují živě!")

    def _on_stacking_failed(self, err_msg: str):
        self.controls.btn_stack.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_status.setText(f"❌ Chyba: {err_msg}")
        QMessageBox.critical(self, "Chyba skládání", err_msg)

    # ------------------ Real-time Postprocessing ------------------
    def _apply_postprocessing_live(self):
        if self._base_merged_bgr is None:
            return

        settings = self.controls.get_settings()
        proc = apply_postprocessing(
            self._base_merged_bgr,
            brightness=settings['brightness'],
            contrast=settings['contrast'],
            gamma=settings['gamma'],
            saturation=settings['saturation'],
            coronal_boost=settings['coronal_boost'],
            coronal_radius=settings['coronal_radius'],
            shadow_lift=settings['shadows'],
            highlight_drop=settings['highlights'],
            denoise_strength=settings['denoise']
        )
        self.viewer_container.viewer.set_image_bgr_float(proc)

    # ------------------ Full Resolution Background Export ------------------
    def export_result(self):
        items = self.exposure_list.get_active_items()
        if not items:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Uložit výsledný HDR snímek v plné kvalitě",
            "eclipse_hdr_fullres.tif",
            "16-bit TIFF (*.tif *.tiff);;JPEG vysoká kvalita (*.jpg);;PNG (*.png);;32-bit Radiance HDR (*.hdr)"
        )

        if not filepath:
            return

        self.controls.btn_stack.setEnabled(False)
        self.controls.btn_export.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText(f"Probíhá výpočet v plné kvalitě a export do {os.path.basename(filepath)}...")

        settings = self.controls.get_settings()
        self._export_worker = FullResExportWorker(items, settings, filepath)
        self._export_worker.progress.connect(self._on_worker_progress)
        self._export_worker.finished_success.connect(self._on_export_success)
        self._export_worker.failed.connect(self._on_export_failed)
        self._export_worker.start()

    def _on_export_success(self, filepath: str):
        self.controls.btn_stack.setEnabled(True)
        self.controls.btn_export.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_status.setText(f"✅ Snímek úspěšně vyexportován v plné kvalitě: {os.path.basename(filepath)}")
        QMessageBox.information(
            self,
            "Export dokončen",
            f"HDR snímek byl v plné 100% kvalitě úspěšně složen a uložen do:\n{filepath}"
        )

    def _on_export_failed(self, err_msg: str):
        self.controls.btn_stack.setEnabled(True)
        self.controls.btn_export.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_status.setText(f"❌ Chyba při exportu: {err_msg}")
        QMessageBox.critical(self, "Chyba exportu", err_msg)
