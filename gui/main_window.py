"""
Main Application Window for Astro HDR Stacker.
Orchestrates UI components, threading, drag & drop, and processing pipelines.
"""

import os
from typing import List, Optional, Dict, Any
import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QIcon, QAction
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QProgressBar, QLabel, QFileDialog, QMessageBox, QStatusBar,
    QApplication
)

try:
    from core.exif_and_analysis import ExposureItem
    from core.aligner import ImageAligner
    from core.merger import HDRMerger
    from core.postprocess import apply_postprocessing, save_image
    from gui.exposure_list_widget import ExposureListWidget
    from gui.image_viewer import ImageViewerContainer
    from gui.controls_panel import ControlsPanel
    from gui.styles import DARK_THEME
except ImportError:
    from ..core.exif_and_analysis import ExposureItem
    from ..core.aligner import ImageAligner
    from ..core.merger import HDRMerger
    from ..core.postprocess import apply_postprocessing, save_image
    from .exposure_list_widget import ExposureListWidget
    from .image_viewer import ImageViewerContainer
    from .controls_panel import ControlsPanel
    from .styles import DARK_THEME


class StackingWorker(QThread):
    """Background worker for loading, aligning, and merging exposures."""
    progress = pyqtSignal(int, str)
    finished_success = pyqtSignal(object, object)  # (base_f32_bgr, hdr_radiance_map_or_None)
    failed = pyqtSignal(str)

    def __init__(self, items: List[ExposureItem], settings: Dict[str, Any]):
        super().__init__()
        self.items = items
        self.settings = settings

    def run(self):
        try:
            if len(self.items) < 2:
                self.failed.emit("K HDR složení jsou potřeba alespoň 2 snímky.")
                return

            # 1. Load full resolution images
            images = []
            times = []
            total_items = len(self.items)

            for idx, item in enumerate(self.items):
                pct = int(10 + (idx / total_items) * 30)
                self.progress.emit(pct, f"Načítání snímku {idx+1}/{total_items}: {item.filename}")
                
                # Load with OpenCV supporting UTF-8 filepaths on Windows
                img = cv2.imdecode(np.fromfile(item.filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    self.failed.emit(f"Nelze načíst soubor: {item.filepath}")
                    return
                images.append(img)
                times.append(max(1e-6, item.exposure_time))

            # 2. Check dimensions consistency
            h0, w0 = images[0].shape[:2]
            for i, img in enumerate(images):
                if img.shape[:2] != (h0, w0):
                    self.progress.emit(40, f"Přizpůsobení rozměrů snímku {i+1}...")
                    images[i] = cv2.resize(img, (w0, h0), interpolation=cv2.INTER_AREA)

            # 3. Alignment (MTB)
            if self.settings.get('align', True):
                self.progress.emit(45, "Zarovnávání snímků (Median Threshold Bitmap)...")
                aligner = ImageAligner(max_bits=5, exclude_range=4, cut=True)
                images = aligner.align(images, progress_callback=lambda p, msg: self.progress.emit(int(45 + p * 0.25), msg))

            # 4. Fusion / HDR Stacking
            algo = self.settings.get('algo', 'mertens')
            hdr_radiance = None

            if algo == 'mertens':
                self.progress.emit(75, "Probíhá víceúrovňová Laplaceova fúze (Mertens)...")
                base_merged = HDRMerger.merge_mertens(
                    images,
                    contrast_weight=self.settings.get('mertens_contrast', 1.0),
                    saturation_weight=self.settings.get('mertens_saturation', 1.0),
                    exposure_weight=self.settings.get('mertens_exposure', 0.0),
                    progress_callback=lambda p, msg: self.progress.emit(int(75 + p * 0.2), msg)
                )
            elif algo == 'debevec':
                self.progress.emit(75, "Generování Debevec HDR Radiance mapy...")
                hdr_radiance, crf = HDRMerger.merge_debevec(
                    images, times,
                    progress_callback=lambda p, msg: self.progress.emit(int(75 + p * 0.15), msg)
                )
                self.progress.emit(90, "Tonemapping...")
                base_merged = HDRMerger.tonemap(
                    hdr_radiance,
                    method=self.settings.get('tonemap_method', 'reinhard')
                )
            elif algo == 'robertson':
                self.progress.emit(75, "Generování Robertson HDR mapy...")
                hdr_radiance, crf = HDRMerger.merge_robertson(
                    images, times,
                    progress_callback=lambda p, msg: self.progress.emit(int(75 + p * 0.15), msg)
                )
                self.progress.emit(90, "Tonemapping...")
                base_merged = HDRMerger.tonemap(
                    hdr_radiance,
                    method=self.settings.get('tonemap_method', 'reinhard')
                )
            else:
                self.failed.emit(f"Neznámý algoritmus: {algo}")
                return

            self.progress.emit(100, "Skládání úspěšně dokončeno.")
            self.finished_success.emit(base_merged, hdr_radiance)

        except Exception as e:
            self.failed.emit(f"Chyba při zpracování: {str(e)}")


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

        # Internal state
        self._raw_merged_bgr: Optional[np.ndarray] = None    # Raw float32 merged image [0, 1]
        self._processed_bgr: Optional[np.ndarray] = None     # Current postprocessed float32 image
        self._hdr_radiance_map: Optional[np.ndarray] = None  # 32-bit linear radiance map if available
        self._preview_cached_bgr: Optional[np.ndarray] = None # Downscaled for ultrafast slider updates
        self._worker: Optional[StackingWorker] = None
        
        # Debounce timer for slider adjustments
        self._slider_timer = QTimer()
        self._slider_timer.setSingleShot(True)
        self._slider_timer.setInterval(40)  # 40ms debounce
        self._slider_timer.timeout.connect(self._apply_postprocessing_live)

        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # Splitter: Left (Exposures), Center (Viewer), Right (Controls)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 1. Left panel: Exposure list
        self.exposure_list = ExposureListWidget()
        self.exposure_list.setMinimumWidth(280)
        self.exposure_list.item_selected.connect(self._on_preview_single_exposure)
        splitter.addWidget(self.exposure_list)

        # 2. Center panel: Interactive Image Viewer
        self.viewer_container = ImageViewerContainer()
        splitter.addWidget(self.viewer_container)

        # 3. Right panel: Controls
        self.controls = ControlsPanel()
        self.controls.setMinimumWidth(320)
        self.controls.stack_requested.connect(self.start_stacking)
        self.controls.parameters_changed.connect(self._on_parameters_changed)
        self.controls.export_requested.connect(self.export_result)
        splitter.addWidget(self.controls)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 6)
        splitter.setStretchFactor(2, 3)
        main_layout.addWidget(splitter)

        # Bottom Bar: Status label + Progress Bar
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

        # Prepare UI
        self.controls.btn_stack.setEnabled(False)
        self.controls.btn_export.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Zahajuji skládání expozic...")

        settings = self.controls.get_settings()
        self._worker = StackingWorker(items, settings)
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
        
        self._raw_merged_bgr = base_bgr
        self._hdr_radiance_map = hdr_radiance
        
        # Set reference for comparison slider (median exposure from list)
        active_items = self.exposure_list.get_active_items()
        if active_items:
            mid_idx = len(active_items) // 2
            mid_path = active_items[mid_idx].filepath
            mid_img = cv2.imdecode(np.fromfile(mid_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if mid_img is not None:
                self.viewer_container.viewer.set_compare_image_bgr_float(mid_img.astype(np.float32) / 255.0)

        # Apply postprocessing
        self._apply_postprocessing_full()
        self.lbl_status.setText("✅ HDR složení úspěšně dokončeno! Upravte posuvníky nebo exportujte.")

    def _on_stacking_failed(self, err_msg: str):
        self.controls.btn_stack.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_status.setText(f"❌ Chyba: {err_msg}")
        QMessageBox.critical(self, "Chyba skládání", err_msg)

    # ------------------ Real-time Postprocessing ------------------
    def _on_parameters_changed(self):
        if self._raw_merged_bgr is not None:
            self._slider_timer.start()

    def _apply_postprocessing_live(self):
        """Fast live preview update."""
        if self._raw_merged_bgr is None:
            return
        
        settings = self.controls.get_settings()
        
        # For fast responsive preview, if image is very large, compute on downscaled copy
        h, w = self._raw_merged_bgr.shape[:2]
        if max(h, w) > 2000:
            scale = 1600.0 / max(h, w)
            small = cv2.resize(self._raw_merged_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            proc_small = apply_postprocessing(
                small,
                brightness=settings['brightness'],
                contrast=settings['contrast'],
                gamma=settings['gamma'],
                saturation=settings['saturation'],
                coronal_boost=settings['coronal_boost'],
                coronal_radius=settings['coronal_radius'] * scale,
                shadow_lift=settings['shadows'],
                highlight_drop=settings['highlights']
            )
            self.viewer_container.viewer.set_image_bgr_float(proc_small)
        else:
            self._apply_postprocessing_full()

    def _apply_postprocessing_full(self):
        """Full resolution render."""
        if self._raw_merged_bgr is None:
            return

        settings = self.controls.get_settings()
        self._processed_bgr = apply_postprocessing(
            self._raw_merged_bgr,
            brightness=settings['brightness'],
            contrast=settings['contrast'],
            gamma=settings['gamma'],
            saturation=settings['saturation'],
            coronal_boost=settings['coronal_boost'],
            coronal_radius=settings['coronal_radius'],
            shadow_lift=settings['shadows'],
            highlight_drop=settings['highlights']
        )
        self.viewer_container.viewer.set_image_bgr_float(self._processed_bgr)

    # ------------------ Export ------------------
    def export_result(self):
        if self._processed_bgr is None and self._raw_merged_bgr is not None:
            self._apply_postprocessing_full()
            
        if self._processed_bgr is None:
            return

        filepath, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Uložit složený HDR snímek",
            "eclipse_hdr_result.tif",
            "16-bit TIFF (*.tif *.tiff);;JPEG vysoká kvalita (*.jpg);;PNG (*.png);;32-bit Radiance HDR (*.hdr)"
        )

        if not filepath:
            return

        self.lbl_status.setText(f"Exportuji snímek do: {filepath}...")
        QApplication.processEvents()

        # Ensure full resolution postprocessing is up to date
        self._apply_postprocessing_full()

        success = save_image(
            filepath,
            self._processed_bgr,
            hdr_radiance_map=self._hdr_radiance_map,
            jpeg_quality=100
        )

        if success:
            self.lbl_status.setText(f"✅ Snímek úspěšně uložen: {os.path.basename(filepath)}")
            QMessageBox.information(
                self,
                "Export dokončen",
                f"Složený snímek byl úspěšně uložen do:\n{filepath}"
            )
        else:
            self.lbl_status.setText("❌ Chyba při ukládání souboru.")
            QMessageBox.critical(self, "Chyba", f"Nepodařilo se uložit soubor do:\n{filepath}")
