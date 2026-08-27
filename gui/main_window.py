"""
Main application window for Astro HDR Stacker.

Interactive workflow:
  * proxy-resolution or ROI-crop stacking on a background thread,
  * debounced re-stacking so dragging the ROI cannot spawn a thread per mouse move,
  * a shared decoded-image cache so a frame is read from disk at most once,
  * memory-aware full-resolution export that degrades gracefully instead of
    being killed by the OS.

Thread lifecycle rule: a QThread is never terminated and never dropped while it
is still running. Cancelled workers are disconnected, asked to stop, and parked
until they finish on their own.
"""

import os
from typing import List, Optional, Dict, Any, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QCloseEvent, QAction, QKeySequence
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QProgressBar, QLabel, QFileDialog, QMessageBox, QFrame, QSizePolicy
)

try:
    from core.exif_and_analysis import ExposureItem, estimate_stack_megapixels
    from core.aligner import calculate_moon_shifts, apply_shifts_to_images, find_sun_or_moon_center
    from core.merger import HDRMerger, HDRMergeError
    from core.postprocess import apply_postprocessing, save_image
    from core.image_cache import GLOBAL_IMAGE_CACHE, available_memory_bytes
    from gui.exposure_list_widget import ExposureListWidget
    from gui.image_viewer import ImageViewerContainer
    from gui.controls_panel import ControlsPanel
    from gui.manual_align_dialog import ManualAlignDialog
    from gui.styles import DARK_THEME
except ImportError:  # pragma: no cover
    from ..core.exif_and_analysis import ExposureItem, estimate_stack_megapixels
    from ..core.aligner import calculate_moon_shifts, apply_shifts_to_images, find_sun_or_moon_center
    from ..core.merger import HDRMerger, HDRMergeError
    from ..core.postprocess import apply_postprocessing, save_image
    from ..core.image_cache import GLOBAL_IMAGE_CACHE, available_memory_bytes
    from .exposure_list_widget import ExposureListWidget
    from .image_viewer import ImageViewerContainer
    from .controls_panel import ControlsPanel
    from .manual_align_dialog import ManualAlignDialog
    from .styles import DARK_THEME

SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp')

# Live slider adjustments run on the GUI thread, so the preview buffer is capped.
# The export always recomputes from the originals, so this costs no final quality.
PREVIEW_MAX_PIXELS = 2_500_000

# How long the ROI must sit still before a re-stack is launched.
RESTACK_DEBOUNCE_MS = 220


def _crop_to_rect(img: np.ndarray, rect: Tuple[int, int, int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crops to `rect`, clamped to the image. Returns the crop and the rect that was
    actually used, so the viewer can draw the patch exactly where it came from.
    """
    rx, ry, rw, rh = rect
    h, w = img.shape[:2]
    rx = int(np.clip(rx, 0, max(0, w - 1)))
    ry = int(np.clip(ry, 0, max(0, h - 1)))
    rw = int(np.clip(rw, 1, w - rx))
    rh = int(np.clip(rh, 1, h - ry))
    return img[ry:ry + rh, rx:rx + rw], (rx, ry, rw, rh)


class _CancellableWorker(QThread):
    """Common cancellation and error plumbing for the background workers."""

    progress = pyqtSignal(int, str)
    failed = pyqtSignal(str)

    def __init__(self, generation: int = 0):
        super().__init__()
        self.generation = generation

    def cancelled(self) -> bool:
        return self.isInterruptionRequested()

    def emit_progress(self, pct: int, msg: str):
        if not self.cancelled():
            self.progress.emit(int(np.clip(pct, 0, 100)), msg)

    def report_failure(self, msg: str):
        if not self.cancelled():
            self.failed.emit(msg)


class StackingWorker(_CancellableWorker):
    """Fast proxy-resolution or ROI-crop stacking for the interactive preview."""

    # (base_f32_bgr, hdr_radiance_or_None, orig_w, orig_h, used_roi_or_None, generation)
    finished_success = pyqtSignal(object, object, int, int, object, int)

    def __init__(
        self,
        items: List[ExposureItem],
        settings: Dict[str, Any],
        scale: float = 0.25,
        roi_rect: Optional[Tuple[int, int, int, int]] = None,
        generation: int = 0,
    ):
        super().__init__(generation=generation)
        # Snapshot the per-frame state: the list widget may mutate its items
        # while this thread runs, and reading them from here would be a race.
        self.frames = [
            (it.filepath, it.filename, float(it.exposure_time), float(it.shift_x), float(it.shift_y))
            for it in items
        ]
        self.settings = dict(settings)
        self.scale = float(np.clip(scale, 0.05, 1.0))
        self.roi_rect = roi_rect
        self.detected_shifts: Optional[List[Tuple[float, float]]] = None

    def run(self):
        try:
            self._run_inner()
        except HDRMergeError as e:
            self.report_failure(str(e))
        except MemoryError:
            self.report_failure(
                "Došla operační paměť. Zvolte nižší pracovní rozlišení "
                "(⚡ 1/4 nebo 🚀 1/8), nebo zapněte režim výřezu 🎯 ROI."
            )
        except Exception as e:  # a background crash must never take the app down
            self.report_failure(f"Chyba při skládání: {type(e).__name__}: {e}")

    def _run_inner(self):
        if len(self.frames) < 2:
            self.report_failure("K HDR složení jsou potřeba alespoň 2 snímky.")
            return

        # In ROI mode the crop comes out of the full-resolution frame, so the
        # crop pixels *are* full-resolution pixels; otherwise we work on a proxy.
        load_scale = 1.0 if self.roi_rect is not None else self.scale

        images: List[np.ndarray] = []
        times: List[float] = []
        orig_w = orig_h = 0
        used_roi = self.roi_rect

        total = len(self.frames)
        for idx, (path, filename, exp_time, _sx, _sy) in enumerate(self.frames):
            if self.cancelled():
                return

            label = "Bleskový výřez" if self.roi_rect is not None else "Načítání snímku"
            self.emit_progress(10 + int(30 * idx / total), f"{label} {idx + 1}/{total}: {filename}")

            img = GLOBAL_IMAGE_CACHE.get(path, load_scale)
            if img is None:
                self.report_failure(f"Nelze načíst soubor: {path}")
                return

            if self.roi_rect is not None:
                orig_h, orig_w = img.shape[:2]
                img, used_roi = _crop_to_rect(img, self.roi_rect)
            else:
                # Recover the full size from the proxy and the scale we asked for.
                orig_h = int(round(img.shape[0] / load_scale))
                orig_w = int(round(img.shape[1] / load_scale))

            images.append(img)
            times.append(exp_time)

        if self.cancelled():
            return

        # Frames from a different body or orientation are resized rather than
        # rejected, so a mixed folder still produces something usable.
        h0, w0 = images[0].shape[:2]
        for i, img in enumerate(images):
            if img.shape[:2] != (h0, w0):
                images[i] = cv2.resize(img, (w0, h0), interpolation=cv2.INTER_AREA)

        images = self._align(images, load_scale)
        if self.cancelled():
            return

        base_merged, hdr_radiance = self._merge(images, times)
        if self.cancelled():
            return

        self.emit_progress(100, "Složení dokončeno.")
        self.finished_success.emit(
            base_merged, hdr_radiance, int(orig_w), int(orig_h), used_roi, self.generation
        )

    def _align(self, images: List[np.ndarray], load_scale: float) -> List[np.ndarray]:
        manual = [(sx, sy) for (_p, _f, _t, sx, sy) in self.frames]
        has_manual = any(abs(sx) > 0.01 or abs(sy) > 0.01 for sx, sy in manual)
        method = self.settings.get('align_method', 'none')

        if has_manual:
            self.emit_progress(45, "Aplikuji manuálně nastavené posuny...")
            # Manual shifts are stored in full-resolution pixels. In ROI mode the
            # crop is already at full resolution, so the factor is 1.
            factor = 1.0 if self.roi_rect is not None else load_scale
            return apply_shifts_to_images(images, manual, scale_factor=factor)

        if method == 'eclipse_disc':
            self.emit_progress(45, "Hledám černý disk Měsíce v záři korony...")
            shifts = calculate_moon_shifts(
                images,
                progress_callback=lambda p, m: self.emit_progress(45 + int(p * 0.2), m),
                should_cancel=self.cancelled,
            )
            if self.cancelled():
                return images
            # Shifts were measured in the pixel scale of `images`; report them
            # back in full-resolution pixels so the GUI and export agree.
            inv = 1.0 if self.roi_rect is not None else (1.0 / load_scale if load_scale > 0 else 1.0)
            self.detected_shifts = [(dx * inv, dy * inv) for dx, dy in shifts]
            return apply_shifts_to_images(images, shifts, scale_factor=1.0)

        return images

    def _merge(self, images, times) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        algo = self.settings.get('algo', 'mertens')

        if algo == 'mertens':
            self.emit_progress(75, "Probíhá rychlá Laplaceova fúze...")
            merged = HDRMerger.merge_mertens(
                images,
                contrast_weight=self.settings.get('mertens_contrast', 1.0),
                saturation_weight=self.settings.get('mertens_saturation', 1.0),
                exposure_weight=self.settings.get('mertens_exposure', 1.0),
                progress_callback=lambda p, m: self.emit_progress(75 + int(p * 0.2), m),
            )
            return merged, None

        if algo in ('debevec', 'robertson'):
            self.emit_progress(75, f"Generování {algo.capitalize()} HDR mapy...")
            if algo == 'debevec':
                radiance, _ = HDRMerger.merge_debevec(images, times)
            else:
                radiance, _ = HDRMerger.merge_robertson(images, times)
            self.emit_progress(90, "Tonemapping...")
            merged = HDRMerger.tonemap(radiance, method=self.settings.get('tonemap_method', 'reinhard'))
            return merged, radiance

        raise HDRMergeError(f"Neznámý algoritmus: {algo}")


class FullResExportWorker(_CancellableWorker):
    """Renders and saves the final full-resolution file."""

    finished_success = pyqtSignal(str)

    def __init__(
        self,
        items: List[ExposureItem],
        settings: Dict[str, Any],
        export_filepath: str,
        export_scale: float = 1.0,
    ):
        super().__init__()
        self.frames = [
            (it.filepath, it.filename, float(it.exposure_time), float(it.shift_x), float(it.shift_y))
            for it in items
        ]
        self.settings = dict(settings)
        self.export_filepath = export_filepath
        self.export_scale = float(np.clip(export_scale, 0.1, 1.0))

    def run(self):
        try:
            self._run_inner()
        except HDRMergeError as e:
            self.report_failure(str(e))
        except MemoryError:
            self.report_failure(
                "Došla paměť při exportu v plném rozlišení. Zavřete ostatní programy, "
                "nebo export potvrďte ve zmenšeném rozlišení."
            )
        except Exception as e:
            self.report_failure(f"Chyba při exportu: {type(e).__name__}: {e}")

    def _run_inner(self):
        if len(self.frames) < 2:
            self.report_failure("K exportu jsou potřeba alespoň 2 snímky.")
            return

        images: List[np.ndarray] = []
        times: List[float] = []
        total = len(self.frames)

        for idx, (path, filename, exp_time, _sx, _sy) in enumerate(self.frames):
            if self.cancelled():
                return
            self.emit_progress(5 + int(35 * idx / total),
                               f"Načítání plného rozlišení {idx + 1}/{total}: {filename}")
            img = GLOBAL_IMAGE_CACHE.get(path, self.export_scale)
            if img is None:
                self.report_failure(f"Nelze načíst soubor: {path}")
                return
            images.append(img)
            times.append(exp_time)

        if self.cancelled():
            return

        h0, w0 = images[0].shape[:2]
        for i, img in enumerate(images):
            if img.shape[:2] != (h0, w0):
                images[i] = cv2.resize(img, (w0, h0), interpolation=cv2.INTER_AREA)

        manual = [(sx, sy) for (_p, _f, _t, sx, sy) in self.frames]
        has_manual = any(abs(sx) > 0.01 or abs(sy) > 0.01 for sx, sy in manual)

        if has_manual:
            self.emit_progress(42, "Aplikuji posuny v plném rozlišení...")
            images = apply_shifts_to_images(images, manual, scale_factor=self.export_scale)
        elif self.settings.get('align_method') == 'eclipse_disc':
            self.emit_progress(42, "Zarovnávám disk Měsíce v plném rozlišení...")
            shifts = calculate_moon_shifts(
                images,
                progress_callback=lambda p, m: self.emit_progress(42 + int(p * 0.15), m),
                should_cancel=self.cancelled,
            )
            if self.cancelled():
                return
            images = apply_shifts_to_images(images, shifts, scale_factor=1.0)

        if self.cancelled():
            return

        algo = self.settings.get('algo', 'mertens')
        hdr_radiance = None

        if algo == 'mertens':
            self.emit_progress(60, "Skládání plné kvality (Mertens Exposure Fusion)...")
            base_merged = HDRMerger.merge_mertens(
                images,
                contrast_weight=self.settings.get('mertens_contrast', 1.0),
                saturation_weight=self.settings.get('mertens_saturation', 1.0),
                exposure_weight=self.settings.get('mertens_exposure', 1.0),
                progress_callback=lambda p, m: self.emit_progress(60 + int(p * 0.22), m),
            )
        else:
            self.emit_progress(60, f"Generování {algo.capitalize()} 32-bit HDR...")
            if algo == 'debevec':
                hdr_radiance, _ = HDRMerger.merge_debevec(images, times)
            else:
                hdr_radiance, _ = HDRMerger.merge_robertson(images, times)
            self.emit_progress(80, "Tonemapping...")
            base_merged = HDRMerger.tonemap(hdr_radiance, method=self.settings.get('tonemap_method', 'reinhard'))

        # The uint8 stack is no longer needed; release it before post-processing
        # allocates its own full-resolution float buffers.
        images = None

        if self.cancelled():
            return

        self.emit_progress(88, "Aplikace postprocessingu a barev v plné kvalitě...")
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
            denoise_strength=self.settings['denoise'],
        )

        if self.cancelled():
            return

        self.emit_progress(95, f"Ukládání souboru {os.path.basename(self.export_filepath)}...")
        if save_image(self.export_filepath, final_proc, hdr_radiance_map=hdr_radiance, jpeg_quality=100):
            self.emit_progress(100, "Export úspěšně dokončen.")
            self.finished_success.emit(self.export_filepath)
        else:
            self.report_failure(
                "Soubor se nepodařilo zapsat na disk. Zkontrolujte, že cílová složka "
                "existuje, je zapisovatelná a soubor není otevřený v jiném programu."
            )


class SunDetectWorker(_CancellableWorker):
    """Locates the Sun / lunar disc off the GUI thread."""

    found = pyqtSignal(int, int)

    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath

    def run(self):
        try:
            img = GLOBAL_IMAGE_CACHE.get(self.filepath, 1.0)
            if img is None or self.cancelled():
                return
            cx, cy = find_sun_or_moon_center(img)
            if not self.cancelled():
                self.found.emit(int(cx), int(cy))
        except Exception as e:
            self.report_failure(f"Detekce Slunce selhala: {e}")


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Astro HDR Stacker — Skládání expozic & Zatmění Slunce")
        self.resize(1440, 900)
        self.setMinimumSize(1024, 680)
        self.setAcceptDrops(True)
        self.setStyleSheet(DARK_THEME)

        self._base_merged_bgr: Optional[np.ndarray] = None
        self._preview_base_bgr: Optional[np.ndarray] = None
        self._hdr_radiance_map: Optional[np.ndarray] = None

        self._worker: Optional[StackingWorker] = None
        self._export_worker: Optional[FullResExportWorker] = None
        self._sun_worker: Optional[SunDetectWorker] = None
        # Workers that were cancelled but have not finished unwinding yet.
        # Holding a reference is what prevents "QThread destroyed while running".
        self._retired_workers: List[QThread] = []

        self._stack_generation = 0
        self._roi_active = False
        self._roi_rect: Optional[Tuple[int, int, int, int]] = None

        # Coalesces bursts of ROI drags / setting changes into one stacking run.
        self._restack_timer = QTimer(self)
        self._restack_timer.setSingleShot(True)
        self._restack_timer.setInterval(RESTACK_DEBOUNCE_MS)
        self._restack_timer.timeout.connect(self._run_stacking)

        self._init_ui()
        self._init_shortcuts()

    # --------------------------------------------------------------------- UI

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 8)
        main_layout.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)
        splitter.setChildrenCollapsible(False)

        self.exposure_list = ExposureListWidget()
        self.exposure_list.item_selected.connect(self._on_preview_single_exposure)
        self.exposure_list.items_changed.connect(self._on_items_changed)
        splitter.addWidget(self.exposure_list)

        self.viewer_container = ImageViewerContainer()
        self.viewer_container.roi_mode_toggled.connect(self._on_roi_mode_toggled)
        self.viewer_container.center_sun_requested.connect(self._center_roi_on_sun)
        splitter.addWidget(self.viewer_container)

        self.controls = ControlsPanel()
        self.controls.stack_requested.connect(self.start_stacking)
        self.controls.manual_align_requested.connect(self.open_manual_alignment)
        self.controls.live_adjust_requested.connect(self._apply_postprocessing_live)
        self.controls.merge_param_changed.connect(self.request_restack)
        self.controls.export_requested.connect(self.export_result)
        splitter.addWidget(self.controls)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 6)
        splitter.setStretchFactor(2, 3)
        splitter.setSizes([330, 720, 390])
        # Stretch 1 on the splitter: without it the status strip below competes
        # for vertical space and grows to fill half the window.
        main_layout.addWidget(splitter, 1)

        # Status bar strip
        bar = QFrame()
        bar.setObjectName("StatusStrip")
        bottom_bar = QHBoxLayout(bar)
        bottom_bar.setContentsMargins(12, 6, 12, 6)
        bottom_bar.setSpacing(12)

        self.lbl_status = QLabel("Přetáhněte sem sérii fotografií nebo klikněte na '+ Přidat fotky'.")
        self.lbl_status.setObjectName("StatusLabel")
        bottom_bar.addWidget(self.lbl_status, 1)

        self.lbl_memory = QLabel("")
        self.lbl_memory.setObjectName("StatusHint")
        bottom_bar.addWidget(self.lbl_memory)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(260)
        self.progress_bar.setVisible(False)
        bottom_bar.addWidget(self.progress_bar)

        bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        bar.setFixedHeight(38)
        main_layout.addWidget(bar, 0)
        self._update_memory_readout()

    def _init_shortcuts(self):
        def add(seq: str, slot):
            act = QAction(self)
            act.setShortcut(QKeySequence(seq))
            act.triggered.connect(slot)
            self.addAction(act)

        add("Ctrl+O", self.exposure_list._on_add_files)
        add("Ctrl+R", self.start_stacking)
        add("Ctrl+S", self.export_result)
        add("Ctrl+M", self.open_manual_alignment)
        add("Ctrl+0", lambda: self.viewer_container.viewer.fit_to_window())
        add("Ctrl+1", lambda: self.viewer_container.viewer.actual_size_100())

    def _update_memory_readout(self):
        avail = available_memory_bytes()
        if avail is None:
            self.lbl_memory.setText("")
        else:
            self.lbl_memory.setText(f"Volná RAM: {avail / (1024 ** 3):.1f} GB")

    # ---------------------------------------------------------- Thread safety

    def _retire_worker(self, worker: Optional[QThread]):
        """
        Stops a worker without ever calling terminate().

        QThread.terminate() kills the thread at an arbitrary instruction — often
        in the middle of an OpenCV allocation — which corrupts the process heap
        and crashes the app moments later. Instead the worker is disconnected so
        its stale results are ignored, asked to stop, and kept referenced until
        it finishes on its own.
        """
        if worker is None:
            return
        try:
            worker.progress.disconnect()
        except (TypeError, RuntimeError):
            pass
        try:
            worker.finished_success.disconnect()
        except (TypeError, RuntimeError, AttributeError):
            pass
        try:
            worker.failed.disconnect()
        except (TypeError, RuntimeError):
            pass

        if worker.isRunning():
            worker.requestInterruption()
            self._retired_workers.append(worker)
            worker.finished.connect(lambda w=worker: self._forget_retired(w))
            worker.wait(50)  # most cancellations land within one loop iteration

    def _forget_retired(self, worker: QThread):
        if worker in self._retired_workers:
            self._retired_workers.remove(worker)

    def _wait_for_all_workers(self, msec: int = 4000):
        for worker in [self._worker, self._export_worker, self._sun_worker] + list(self._retired_workers):
            if worker is not None and worker.isRunning():
                worker.requestInterruption()
        for worker in [self._worker, self._export_worker, self._sun_worker] + list(self._retired_workers):
            if worker is not None and worker.isRunning():
                worker.wait(msec)

    def closeEvent(self, event: QCloseEvent):
        self._restack_timer.stop()
        if self._export_worker is not None and self._export_worker.isRunning():
            reply = QMessageBox.question(
                self, "Probíhá export",
                "Export v plné kvalitě ještě běží. Opravdu chcete aplikaci zavřít "
                "a export zrušit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self._wait_for_all_workers()
        GLOBAL_IMAGE_CACHE.invalidate()
        event.accept()

    # ----------------------------------------------------------- Drag and drop

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        filepaths: List[str] = []
        for u in event.mimeData().urls():
            path = u.toLocalFile()
            if os.path.isfile(path):
                if os.path.splitext(path)[1].lower() in SUPPORTED_EXTENSIONS:
                    filepaths.append(path)
            elif os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for f in sorted(files):
                        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS:
                            filepaths.append(os.path.join(root, f))

        if not filepaths:
            self.lbl_status.setText("Přetažené soubory nejsou podporované obrázky.")
            return

        self.exposure_list.load_files(filepaths)
        self.lbl_status.setText(f"Načteno {len(filepaths)} snímků. Připraveno ke složení.")
        self._setup_initial_preview()

    def _on_items_changed(self):
        self._update_memory_readout()

    def _setup_initial_preview(self):
        """Shows the middle exposure and starts an async Sun search."""
        active_items = self.exposure_list.get_active_items()
        if not active_items:
            return

        mid_item = active_items[len(active_items) // 2]
        img = GLOBAL_IMAGE_CACHE.get(mid_item.filepath, 1.0)
        if img is None:
            self.lbl_status.setText(f"Nelze načíst náhled: {mid_item.filename}")
            return

        h, w = img.shape[:2]
        self.viewer_container.viewer.set_original_size(w, h)
        self.viewer_container.viewer.set_base_image_bgr_uint8(img, keep_view=False)
        self.viewer_container.viewer.set_compare_image_bgr_uint8(img)
        self._start_sun_detection(mid_item.filepath, announce=False)

    def _start_sun_detection(self, filepath: str, announce: bool = True):
        self._retire_worker(self._sun_worker)
        self._sun_worker = SunDetectWorker(filepath)
        self._sun_worker.found.connect(lambda x, y: self._on_sun_found(x, y, announce))
        self._sun_worker.failed.connect(lambda m: self.lbl_status.setText(m))
        if announce:
            self.lbl_status.setText("☀️ Hledám Slunce / disk Měsíce...")
        self._sun_worker.start()

    def _on_sun_found(self, cx: int, cy: int, announce: bool):
        self.viewer_container.viewer.set_roi_center(cx, cy, emit_signal=announce)
        if announce:
            self.lbl_status.setText(f"☀️ Slunce zaměřeno na [{cx}, {cy}].")

    # -------------------------------------------------------- Single exposure

    def _on_preview_single_exposure(self, filepath: str):
        img = GLOBAL_IMAGE_CACHE.get(filepath, 1.0)
        if img is None:
            self.lbl_status.setText(f"Nelze načíst {os.path.basename(filepath)}")
            return
        self.viewer_container.viewer.set_base_image_bgr_uint8(img, keep_view=True)
        self.viewer_container.viewer.set_compare_image_bgr_uint8(img)
        self.lbl_status.setText(f"Náhled expozice: {os.path.basename(filepath)}")

    # ------------------------------------------------------ ROI mode & Sun

    def _on_roi_mode_toggled(self, enabled: bool, x: int, y: int, w: int, h: int):
        self._roi_active = enabled
        self._roi_rect = (x, y, w, h) if enabled else None
        if enabled:
            self.lbl_status.setText("🎯 Klikněte do fotky pro vycentrování výřezu na Slunce.")
        else:
            self.lbl_status.setText("Zobrazen celý snímek. Provádím složení plné scény...")
        self.request_restack()

    def _on_manual_shifts_applied(self):
        self.lbl_status.setText("Manuální posuny uloženy. Spouštím složení...")
        self.exposure_list.refresh_table()
        self.request_restack()

    def _center_roi_on_sun(self):
        active_items = self.exposure_list.get_active_items()
        if not active_items:
            self.lbl_status.setText("Nejsou načteny žádné snímky.")
            return
        self._start_sun_detection(active_items[len(active_items) // 2].filepath, announce=True)

    # ---------------------------------------------------- Manual alignment

    def open_manual_alignment(self):
        items = self.exposure_list.get_active_items()
        if len(items) < 2:
            QMessageBox.warning(self, "Nedostatek snímků",
                                "Pro zarovnání načtěte alespoň 2 aktivní snímky.")
            return

        # A modal dialog that loads its own copies must not race the preview worker.
        self._restack_timer.stop()
        self._retire_worker(self._worker)
        self._worker = None

        dialog = ManualAlignDialog(items, parent=self)
        dialog.shifts_applied.connect(self._on_manual_shifts_applied)
        dialog.exec()

    # ------------------------------------------------------------- Stacking

    def request_restack(self):
        """
        Schedules a re-stack. Repeated calls (ROI dragging, slider sweeps) only
        ever produce one run once the input settles.
        """
        if len(self.exposure_list.get_active_items()) >= 2:
            self._restack_timer.start()

    def start_stacking(self):
        """Explicit user request: validate loudly, then run immediately."""
        if len(self.exposure_list.get_active_items()) < 2:
            QMessageBox.warning(self, "Nedostatek snímků",
                                "Pro HDR složení vyberte v seznamu alespoň 2 aktivní expozice.")
            return
        self._restack_timer.stop()
        self._run_stacking()

    def _run_stacking(self):
        items = self.exposure_list.get_active_items()
        if len(items) < 2:
            return

        self._retire_worker(self._worker)
        self._stack_generation += 1

        self.controls.btn_stack.setEnabled(False)
        self.controls.btn_export.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        settings = self.controls.get_settings()
        roi = self._roi_rect if self._roi_active else None

        if roi:
            self.lbl_status.setText(f"Zahajuji bleskové složení výřezu {roi[2]}x{roi[3]} px...")
        else:
            self.lbl_status.setText("Zahajuji rychlé skládání scény...")

        worker = StackingWorker(items, settings,
                                scale=settings.get('proxy_scale', 0.25),
                                roi_rect=roi,
                                generation=self._stack_generation)
        worker.progress.connect(self._on_worker_progress)
        worker.finished_success.connect(self._on_stacking_success)
        worker.failed.connect(self._on_stacking_failed)
        self._worker = worker
        worker.start()

    def _on_worker_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.lbl_status.setText(msg)

    def _on_stacking_success(self, base_bgr, hdr_radiance, orig_w: int, orig_h: int,
                             used_roi, generation: int):
        if generation != self._stack_generation:
            return  # a newer run has already superseded this result

        self.controls.btn_stack.setEnabled(True)
        self.controls.btn_export.setEnabled(True)
        self.progress_bar.setVisible(False)

        # Carry any auto-detected shifts back onto the items so the export and
        # the manual dialog start from the same alignment the preview used.
        worker = self.sender()
        if isinstance(worker, StackingWorker) and worker.detected_shifts:
            # One decimal everywhere: it matches the manual dialog's spin boxes,
            # so opening that dialog never silently re-rounds a detected shift.
            for item, (dx, dy) in zip(self.exposure_list.get_active_items(), worker.detected_shifts):
                item.shift_x = round(dx, 1)
                item.shift_y = round(dy, 1)

        if orig_w > 0 and orig_h > 0:
            self.viewer_container.viewer.set_original_size(orig_w, orig_h)

        self._base_merged_bgr = base_bgr
        self._hdr_radiance_map = hdr_radiance
        self._preview_base_bgr = self._downscale_for_preview(base_bgr)
        if used_roi is not None:
            self._roi_rect = tuple(used_roi)

        # Refresh the split-view "before" image with the middle exposure.
        active_items = self.exposure_list.get_active_items()
        if active_items and not self._roi_active:
            mid = GLOBAL_IMAGE_CACHE.get(active_items[len(active_items) // 2].filepath, 1.0)
            if mid is not None:
                h, w = self._preview_base_bgr.shape[:2]
                self.viewer_container.viewer.set_compare_image_bgr_uint8(
                    cv2.resize(mid, (w, h), interpolation=cv2.INTER_AREA))

        self._apply_postprocessing_live()

        if self._roi_active and self._roi_rect is not None:
            self.lbl_status.setText(
                f"⚡ Výřez {self._roi_rect[2]}x{self._roi_rect[3]} px složen. "
                "Kliknutím do fotky ho přesunete."
            )
        else:
            pct = int(self.controls.get_settings().get('proxy_scale', 0.25) * 100)
            self.lbl_status.setText(f"✅ Složeno ({pct}% náhled). Posuvníky reagují živě.")
        self._update_memory_readout()

    @staticmethod
    def _downscale_for_preview(img: np.ndarray) -> np.ndarray:
        """Caps the live-preview buffer so slider drags stay responsive."""
        h, w = img.shape[:2]
        if h * w <= PREVIEW_MAX_PIXELS:
            return img
        scale = (PREVIEW_MAX_PIXELS / float(h * w)) ** 0.5
        return cv2.resize(img, (max(16, int(w * scale)), max(16, int(h * scale))),
                          interpolation=cv2.INTER_AREA)

    def _on_stacking_failed(self, err_msg: str):
        self.controls.btn_stack.setEnabled(True)
        # A failed re-stack does not invalidate a result already on screen.
        self.controls.btn_export.setEnabled(self._base_merged_bgr is not None)
        self.progress_bar.setVisible(False)
        self.lbl_status.setText(f"❌ {err_msg}")
        QMessageBox.critical(self, "Chyba skládání", err_msg)

    # ------------------------------------------------- Live post-processing

    def _apply_postprocessing_live(self):
        source = self._preview_base_bgr
        if source is None:
            return

        settings = self.controls.get_settings()
        try:
            proc = apply_postprocessing(
                source,
                brightness=settings['brightness'],
                contrast=settings['contrast'],
                gamma=settings['gamma'],
                saturation=settings['saturation'],
                coronal_boost=settings['coronal_boost'],
                coronal_radius=settings['coronal_radius'],
                shadow_lift=settings['shadows'],
                highlight_drop=settings['highlights'],
                denoise_strength=settings['denoise'],
            )
        except Exception as e:
            self.lbl_status.setText(f"❌ Úpravy selhaly: {e}")
            return

        if self._roi_active and self._roi_rect is not None:
            rx, ry, rw, rh = self._roi_rect
            self.viewer_container.viewer.set_roi_crop_bgr_float(proc, rx, ry, rw, rh)
        else:
            self.viewer_container.viewer.set_base_image_bgr_float(proc, keep_view=True)

    # -------------------------------------------------------------- Export

    def _choose_export_scale(self, items: List[ExposureItem]) -> Optional[float]:
        """
        Decides whether a full-resolution export fits in RAM, asking the user
        when it does not. Returns the scale to export at, or None to abort.
        """
        megapixels = estimate_stack_megapixels(items)
        avail = available_memory_bytes()
        if megapixels <= 0 or avail is None:
            return 1.0

        n = len(items)
        # uint8 source stack + the float32 result + post-processing scratch.
        needed = (megapixels * 1e6) * (n * 3 + 3 * 4 * 3)
        if needed < avail * 0.65:
            return 1.0

        safe_scale = float(np.clip(((avail * 0.55) / needed) ** 0.5, 0.25, 1.0))
        reply = QMessageBox.question(
            self,
            "Málo volné paměti",
            f"Export {n} snímků o {megapixels:.0f} Mpx by potřeboval přibližně "
            f"{needed / (1024 ** 3):.1f} GB RAM, ale volných je jen "
            f"{avail / (1024 ** 3):.1f} GB.\n\n"
            f"Chcete exportovat ve zmenšeném rozlišení ({int(safe_scale * 100)} %)?\n\n"
            "Ano = bezpečný zmenšený export\n"
            "Ne = zkusit přesto plné rozlišení (aplikace může spadnout)\n"
            "Zrušit = export zrušit",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Cancel:
            return None
        return safe_scale if reply == QMessageBox.StandardButton.Yes else 1.0

    def export_result(self):
        items = self.exposure_list.get_active_items()
        if len(items) < 2:
            QMessageBox.warning(self, "Nedostatek snímků",
                                "Pro export vyberte alespoň 2 aktivní expozice.")
            return

        if self._export_worker is not None and self._export_worker.isRunning():
            QMessageBox.information(self, "Export již probíhá",
                                    "Počkejte prosím na dokončení probíhajícího exportu.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Uložit výsledný HDR snímek v plné kvalitě",
            "eclipse_hdr_fullres.tif",
            "16-bit TIFF (*.tif *.tiff);;JPEG vysoká kvalita (*.jpg);;"
            "16-bit PNG (*.png);;32-bit Radiance HDR (*.hdr)",
        )
        if not filepath:
            return
        if not os.path.splitext(filepath)[1]:
            filepath += '.tif'

        export_scale = self._choose_export_scale(items)
        if export_scale is None:
            return

        self.controls.btn_stack.setEnabled(False)
        self.controls.btn_export.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText(f"Probíhá výpočet a export do {os.path.basename(filepath)}...")

        worker = FullResExportWorker(items, self.controls.get_settings(), filepath, export_scale)
        worker.progress.connect(self._on_worker_progress)
        worker.finished_success.connect(self._on_export_success)
        worker.failed.connect(self._on_export_failed)
        self._export_worker = worker
        worker.start()

    def _on_export_success(self, filepath: str):
        self.controls.btn_stack.setEnabled(True)
        self.controls.btn_export.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_status.setText(f"✅ Vyexportováno: {os.path.basename(filepath)}")
        QMessageBox.information(self, "Export dokončen",
                                f"HDR snímek byl uložen do:\n{filepath}")
        self._update_memory_readout()

    def _on_export_failed(self, err_msg: str):
        self.controls.btn_stack.setEnabled(True)
        self.controls.btn_export.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_status.setText(f"❌ Chyba při exportu: {err_msg}")
        QMessageBox.critical(self, "Chyba exportu", err_msg)
