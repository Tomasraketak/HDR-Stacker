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
from PyQt6.QtCore import Qt, QThread, QTimer, QStandardPaths, QSettings, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QCloseEvent, QAction, QKeySequence
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QProgressBar, QLabel, QFileDialog, QMessageBox, QFrame, QSizePolicy
)

try:
    from core.exif_and_analysis import ExposureItem, estimate_stack_megapixels
    from core.aligner import (calculate_moon_shifts, apply_shifts_to_images,
                              find_sun_or_moon_center, calculate_light_pattern_shifts)
    from core.merger import HDRMerger, HDRMergeError
    from core.postprocess import apply_postprocessing, save_image
    from core.image_cache import GLOBAL_IMAGE_CACHE, available_memory_bytes
    from core.project import (Project, ProjectError, PROJECT_EXTENSION, PROJECT_FILTER,
                              build_project, save_project, load_project,
                              resolved_paths, apply_frame_records)
    from gui.exposure_list_widget import ExposureListWidget
    from gui.image_viewer import ImageViewerContainer
    from gui.controls_panel import ControlsPanel
    from gui.manual_align_dialog import ManualAlignDialog
    from gui.styles import DARK_THEME
    from gui.ui_utils import fit_window_to_screen
except ImportError:  # pragma: no cover
    from ..core.exif_and_analysis import ExposureItem, estimate_stack_megapixels
    from ..core.aligner import (calculate_moon_shifts, apply_shifts_to_images,
                                find_sun_or_moon_center, calculate_light_pattern_shifts)
    from ..core.merger import HDRMerger, HDRMergeError
    from ..core.postprocess import apply_postprocessing, save_image
    from ..core.image_cache import GLOBAL_IMAGE_CACHE, available_memory_bytes
    from ..core.project import (Project, ProjectError, PROJECT_EXTENSION, PROJECT_FILTER,
                                build_project, save_project, load_project,
                                resolved_paths, apply_frame_records)
    from .exposure_list_widget import ExposureListWidget
    from .image_viewer import ImageViewerContainer
    from .controls_panel import ControlsPanel
    from .manual_align_dialog import ManualAlignDialog
    from .styles import DARK_THEME
    from .ui_utils import fit_window_to_screen

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


def _as_saved_rect(value) -> Optional[Tuple[int, int, int, int]]:
    """Coerces a rectangle stored inside the settings blob back to a tuple."""
    if not value or len(value) != 4:
        return None
    try:
        return tuple(int(v) for v in value)
    except (TypeError, ValueError):
        return None


def _apply_crop(img: np.ndarray, crop_rect: Optional[Tuple[int, int, int, int]],
                scale: float = 1.0) -> np.ndarray:
    """
    Crops to `crop_rect`, which is expressed in ORIGINAL full-resolution pixels
    and scaled here to match whatever proxy `img` actually is.

    Cropping always happens AFTER alignment: warping a frame that has already
    been cropped would pull border-replicated pixels in from the crop edge
    instead of the real neighbouring image data.
    """
    if crop_rect is None:
        return img
    x, y, w, h = crop_rect
    return _crop_to_rect(img, (int(round(x * scale)), int(round(y * scale)),
                               max(1, int(round(w * scale))),
                               max(1, int(round(h * scale)))))[0]


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
        crop_rect: Optional[Tuple[int, int, int, int]] = None,
        generation: int = 0,
    ):
        super().__init__(generation=generation)
        self.crop_rect = crop_rect
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
        self.align_report: str = ""

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

        # Align on the uncropped frames, then crop: the other order would warp
        # in replicated border pixels along the crop edge.
        images = self._align(images, load_scale)
        if self.cancelled():
            return

        if self.crop_rect is not None:
            self.emit_progress(68, "Aplikuji ořez na všechny expozice...")
            images = [_apply_crop(im, self.crop_rect, load_scale) for im in images]

        # The scene the viewer works in is whatever survived the crop.
        scene_h, scene_w = images[0].shape[:2]
        orig_h = int(round(scene_h / load_scale))
        orig_w = int(round(scene_w / load_scale))

        if self.roi_rect is not None:
            cropped = []
            for im in images:
                patch, used_roi = _crop_to_rect(im, self.roi_rect)
                cropped.append(patch)
            images = cropped

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

        if method in ('eclipse_disc', 'static_lights'):
            if method == 'static_lights':
                self.emit_progress(45, "Hledám statická pouliční světla...")
                shifts, match_counts = calculate_light_pattern_shifts(
                    images,
                    progress_callback=lambda p, m: self.emit_progress(45 + int(p * 0.2), m),
                    should_cancel=self.cancelled,
                )
                self.align_report = self._describe_light_alignment(match_counts)
            else:
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

    @staticmethod
    def _describe_light_alignment(match_counts: List[int]) -> str:
        """
        Summarises how well the light pattern was matched, per frame.

        A count of 0 means the frame could not be aligned at all and was left
        untouched — the user needs to know which ones, because those are exactly
        the frames worth nudging by hand.
        """
        total = len(match_counts)
        confident = sum(1 for c in match_counts if c > 0)
        weak = [i + 1 for i, c in enumerate(match_counts) if c == -1]
        failed = [i + 1 for i, c in enumerate(match_counts) if c == 0]

        parts = [f"💡 Zarovnáno podle světel: {confident}/{total} snímků spolehlivě"]
        if weak:
            parts.append("přibližně u č. " + ", ".join(map(str, weak)))
        if failed:
            parts.append("NEPODAŘILO SE u č. " + ", ".join(map(str, failed))
                         + " — dolaďte je ručně (Ctrl+M)")
        return "  ·  ".join(parts)

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
        crop_rect: Optional[Tuple[int, int, int, int]] = None,
    ):
        super().__init__()
        self.crop_rect = crop_rect
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
        elif self.settings.get('align_method') == 'static_lights':
            self.emit_progress(42, "Zarovnávám podle statických světel v plném rozlišení...")
            shifts, _counts = calculate_light_pattern_shifts(
                images,
                progress_callback=lambda p, m: self.emit_progress(42 + int(p * 0.15), m),
                should_cancel=self.cancelled,
            )
            if self.cancelled():
                return
            images = apply_shifts_to_images(images, shifts, scale_factor=1.0)
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

        # Crop every exposure identically, after alignment.
        if self.crop_rect is not None:
            self.emit_progress(56, "Aplikuji ořez na všechny expozice...")
            images = [_apply_crop(im, self.crop_rect, self.export_scale) for im in images]
            if images[0].size == 0:
                self.report_failure("Oříznutá oblast je prázdná. Zkontrolujte nastavení ořezu.")
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

    def __init__(self, session_persistence: bool = True):
        """
        `session_persistence` controls the automatic last-session snapshot.

        Tests construct windows freely and must not pick up — or overwrite —
        the real user's saved session, so they turn it off.
        """
        super().__init__()
        self._session_persistence = bool(session_persistence)
        self.setWindowTitle("Astro HDR Stacker — Skládání expozic & Zatmění Slunce")
        # A hard minimum this large cannot be honoured on a 1280x688 desktop
        # (a 15" 1080p laptop at 150 % scaling), so keep it genuinely small and
        # let the panels scroll instead.
        self.setMinimumSize(900, 560)
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
        self._project_path: Optional[str] = None
        self._settings_store = QSettings("AstroHDRStacker", "AstroHDRStacker")
        self._roi_active = False
        self._roi_rect: Optional[Tuple[int, int, int, int]] = None
        # User-defined output crop, in original full-resolution coordinates.
        self._crop_rect: Optional[Tuple[int, int, int, int]] = None

        # Coalesces bursts of ROI drags / setting changes into one stacking run.
        self._restack_timer = QTimer(self)
        self._restack_timer.setSingleShot(True)
        self._restack_timer.setInterval(RESTACK_DEBOUNCE_MS)
        self._restack_timer.timeout.connect(self._run_stacking)

        self._init_ui()
        self._init_menu()
        self._init_shortcuts()
        fit_window_to_screen(self, 1440, 900)
        if self._session_persistence:
            QTimer.singleShot(150, self._maybe_restore_last_session)

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
        self.viewer_container.viewer.crop_selected.connect(self._on_crop_drawn)
        splitter.addWidget(self.viewer_container)

        self.controls = ControlsPanel()
        self.controls.stack_requested.connect(self.start_stacking)
        self.controls.manual_align_requested.connect(self.open_manual_alignment)
        self.controls.live_adjust_requested.connect(self._apply_postprocessing_live)
        self.controls.merge_param_changed.connect(self.request_restack)
        self.controls.crop_changed.connect(self._on_crop_changed)
        self.controls.crop_select_requested.connect(self._on_crop_select_requested)
        self.controls.crop_defaults_requested.connect(self._seed_default_crop)
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

    # ------------------------------------------------------------------ Menu

    def _init_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&Projekt")

        def add(menu, text, slot, shortcut=None, tip=""):
            action = QAction(text, self)
            if shortcut:
                action.setShortcut(QKeySequence(shortcut))
            if tip:
                action.setStatusTip(tip)
            action.triggered.connect(slot)
            menu.addAction(action)
            return action

        add(file_menu, "&Nový projekt", self.new_project, "Ctrl+N",
            "Zavře fotky a vrátí nastavení na výchozí.")
        add(file_menu, "&Otevřít projekt…", self.open_project, "Ctrl+Shift+O",
            "Načte fotky, zarovnání i nastavení z dříve uloženého projektu.")
        self.act_recent = add(file_menu, "Obnovit &poslední relaci",
                              self.restore_last_session, None,
                              "Načte stav, ve kterém jste program naposledy zavřeli.")
        file_menu.addSeparator()
        add(file_menu, "&Uložit projekt", self.save_project_action, "Ctrl+S",
            "Uloží fotky, zarovnání, ořez i všechna nastavení.")
        add(file_menu, "Uložit projekt &jako…", self.save_project_as, "Ctrl+Shift+S")
        file_menu.addSeparator()

        self.act_autorestore = QAction("Obnovovat poslední relaci při startu", self)
        self.act_autorestore.setCheckable(True)
        self.act_autorestore.setChecked(self._autorestore_enabled())
        self.act_autorestore.toggled.connect(
            lambda on: self._settings_store.setValue("autorestore", bool(on)))
        file_menu.addAction(self.act_autorestore)
        file_menu.addSeparator()

        add(file_menu, "Přidat &fotky…", self.exposure_list._on_add_files, "Ctrl+O")
        add(file_menu, "&Exportovat výsledek…", self.export_result, "Ctrl+E")
        file_menu.addSeparator()
        add(file_menu, "&Konec", self.close, "Ctrl+Q")

        self._refresh_recent_action()

    def _autorestore_enabled(self) -> bool:
        stored = self._settings_store.value("autorestore", True)
        return stored if isinstance(stored, bool) else str(stored).lower() != "false"

    def _session_file(self) -> str:
        """Path of the automatic 'last session' snapshot in the user's app data."""
        base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
        if not base:
            base = os.path.join(os.path.expanduser("~"), ".astro_hdr_stacker")
        return os.path.join(base, "last_session" + PROJECT_EXTENSION)

    def _refresh_recent_action(self):
        self.act_recent.setEnabled(os.path.isfile(self._session_file()))

    def _init_shortcuts(self):
        def add(seq: str, slot):
            act = QAction(self)
            act.setShortcut(QKeySequence(seq))
            act.triggered.connect(slot)
            self.addAction(act)

        add("Ctrl+R", self.start_stacking)
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
        self._autosave_session()
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

    # --------------------------------------------------------------- Projects

    def _snapshot_project(self, project_path: Optional[str]) -> Project:
        """Captures the whole session — frames, per-frame state and settings."""
        return build_project(
            self.exposure_list.items,           # every frame, not just the active ones
            self.controls.get_settings(),
            project_path=project_path,
            crop_rect=self._crop_rect,
            roi_active=self._roi_active,
            roi_rect=self._roi_rect,
            roi_size=self.viewer_container.combo_roi_size.currentData() or 300,
            ev_step=self.exposure_list.spin_ev_step.value(),
            preset_name=self.controls.combo_preset.currentText(),
            compare_mode=self.viewer_container.btn_split.isChecked(),
            histogram_visible=self.viewer_container.btn_hist.isChecked(),
        )

    def new_project(self):
        self._restack_timer.stop()
        self._retire_worker(self._worker)
        self._worker = None
        self._stack_generation += 1

        self.exposure_list.clear_all()
        self.controls.chk_crop.setChecked(False)
        self.viewer_container.btn_roi_toggle.setChecked(False)
        self.controls.reset_adjustments()
        self._crop_rect = None
        self._base_merged_bgr = None
        self._preview_base_bgr = None
        self._hdr_radiance_map = None
        self._project_path = None
        self.viewer_container.viewer.set_crop_rect(None)
        self.controls.btn_export.setEnabled(False)
        self._update_title()
        self.lbl_status.setText("Nový projekt. Přetáhněte sem fotky nebo je přidejte tlačítkem.")

    def save_project_action(self):
        if self._project_path:
            self._write_project(self._project_path)
        else:
            self.save_project_as()

    def save_project_as(self):
        if not self.exposure_list.items:
            QMessageBox.information(self, "Prázdný projekt",
                                    "Nejdřív načtěte fotky, které chcete uložit.")
            return
        suggested = self._project_path or os.path.join(
            os.path.dirname(self.exposure_list.items[0].filepath),
            "zatmeni_projekt" + PROJECT_EXTENSION)
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Uložit projekt", suggested, PROJECT_FILTER)
        if filepath:
            self._write_project(filepath)

    def _write_project(self, filepath: str) -> bool:
        try:
            save_project(self._snapshot_project(filepath), filepath)
        except ProjectError as e:
            QMessageBox.critical(self, "Chyba uložení projektu", str(e))
            return False
        self._project_path = filepath
        self._update_title()
        self.lbl_status.setText(
            f"💾 Projekt uložen: {os.path.basename(filepath)} "
            f"({len(self.exposure_list.items)} snímků včetně zarovnání a nastavení)")
        return True

    def open_project(self):
        start_dir = os.path.dirname(self._project_path) if self._project_path else ""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Otevřít projekt", start_dir, PROJECT_FILTER)
        if filepath:
            self._load_project_file(filepath, remember_path=True)

    def restore_last_session(self):
        session = self._session_file()
        if not os.path.isfile(session):
            self.lbl_status.setText("Žádná uložená relace nebyla nalezena.")
            return
        # The auto-snapshot is not a project the user named, so it does not
        # become the current project path — Ctrl+S must not overwrite it.
        self._load_project_file(session, remember_path=False)

    def _maybe_restore_last_session(self):
        if not self._autorestore_enabled():
            return
        session = self._session_file()
        if os.path.isfile(session) and not self.exposure_list.items:
            self._load_project_file(session, remember_path=False, quiet=True)

    def _load_project_file(self, filepath: str, remember_path: bool,
                           quiet: bool = False) -> bool:
        """Loads a project and applies it to every part of the UI."""
        try:
            project, missing = load_project(filepath)
        except ProjectError as e:
            if not quiet:
                QMessageBox.critical(self, "Chyba otevření projektu", str(e))
            return False

        found = resolved_paths(project, filepath)
        if not found:
            message = ("Žádnou z fotek uložených v projektu se nepodařilo najít.\n\n"
                       "Fotky byly zřejmě přesunuty nebo smazány. Projekt ukládá jen "
                       "cesty k souborům, ne samotné fotografie.")
            if quiet:
                self.lbl_status.setText("Poslední relaci nelze obnovit — fotky nebyly nalezeny.")
            else:
                QMessageBox.warning(self, "Fotky nenalezeny", message)
            return False

        self._restack_timer.stop()
        self._retire_worker(self._worker)
        self._worker = None
        self._stack_generation += 1

        # Order matters: load the frames, restore their per-frame state, then the
        # settings, and only then trigger a single stack.
        self.exposure_list.clear_all()
        self.exposure_list.spin_ev_step.setValue(project.ev_step)
        self.exposure_list.load_files(found)
        matched = apply_frame_records(project, self.exposure_list.items, filepath)
        self.exposure_list.refresh_table()

        self.controls.apply_settings(project.settings)
        self.controls.set_preset_name(project.preset_name)
        self._crop_rect = project.crop_rect or _as_saved_rect(project.settings.get('crop_rect'))
        self.viewer_container.viewer.set_crop_rect(None)

        self.viewer_container.btn_hist.setChecked(project.histogram_visible)
        self.viewer_container.btn_split.setChecked(project.compare_mode)

        roi_index = self.viewer_container.combo_roi_size.findData(project.roi_size)
        if roi_index >= 0:
            self.viewer_container.combo_roi_size.setCurrentIndex(roi_index)

        self._setup_initial_preview()

        self._roi_rect = project.roi_rect
        self.viewer_container.btn_roi_toggle.setChecked(bool(project.roi_active))
        if project.roi_active and project.roi_rect:
            self.viewer_container.viewer.set_roi_center(
                project.roi_rect[0] + project.roi_rect[2] // 2,
                project.roi_rect[1] + project.roi_rect[3] // 2,
                emit_signal=False)
            self._roi_active = True
            self._roi_rect = project.roi_rect

        self._project_path = filepath if remember_path else None
        self._update_title()

        shifted = sum(1 for it in self.exposure_list.items
                      if abs(it.shift_x) > 0.01 or abs(it.shift_y) > 0.01)
        summary = (f"📂 Načteno {len(found)} snímků, obnoveno zarovnání u {matched} z nich "
                   f"({shifted} s posunem) a všechna nastavení.")
        if missing:
            summary += f"  ⚠ Chybí {len(missing)} souborů."
            if not quiet:
                QMessageBox.warning(
                    self, "Některé fotky chybí",
                    f"{len(missing)} fotek z projektu se nepodařilo najít a byly vynechány:\n\n"
                    + "\n".join(os.path.basename(m) for m in missing[:12])
                    + ("\n…" if len(missing) > 12 else ""))
        self.lbl_status.setText(summary)

        self.request_restack()
        return True

    def _update_title(self):
        base = "Astro HDR Stacker — Skládání expozic & Zatmění Slunce"
        if self._project_path:
            self.setWindowTitle(f"{os.path.basename(self._project_path)} — {base}")
        else:
            self.setWindowTitle(base)

    def _autosave_session(self):
        """Snapshots the session on exit so it can be picked up next time."""
        if not self._session_persistence or not self.exposure_list.items:
            return
        session = self._session_file()
        try:
            os.makedirs(os.path.dirname(session), exist_ok=True)
            save_project(self._snapshot_project(session), session)
        except (ProjectError, OSError) as e:
            # Never let a failed convenience snapshot block closing the app.
            print(f"Could not autosave session: {e}")

    # ------------------------------------------------------------------ Crop

    def _on_crop_changed(self, rect: Optional[Tuple[int, int, int, int]]):
        """The crop was enabled, disabled or edited numerically."""
        self._crop_rect = tuple(rect) if rect else None
        selecting = self.controls.btn_crop_select.isChecked()
        if selecting or self._crop_rect is None:
            self.viewer_container.viewer.set_crop_rect(self._crop_rect)
        if self._crop_rect:
            x, y, w, h = self._crop_rect
            self.lbl_status.setText(
                f"✂️ Ořez {w}×{h} px od [{x}, {y}] — použije se na všechny expozice i na export.")
        else:
            self.lbl_status.setText("Ořez vypnut — pracuje se s celým snímkem.")
        if not selecting:
            self.request_restack()

    def _seed_default_crop(self):
        """
        Seeds the crop with the full frame when it is switched on with nothing set.
        Starting from 0x0 would be an invalid state the user has to dig out of.
        """
        width, height = self._current_scene_size()
        if width <= 0 or height <= 0:
            self.lbl_status.setText(
                "Ořez lze nastavit až po načtení fotek — nejdřív přidejte expozice.")
            return
        self.controls.set_crop_rect((0, 0, width, height))
        self._crop_rect = (0, 0, width, height)
        self._show_uncropped_reference()
        self.viewer_container.viewer.set_crop_rect(self._crop_rect)
        self.lbl_status.setText(
            f"✂️ Ořez zapnut na celý snímek ({width}×{height} px). "
            "Táhněte myší přes snímek nebo zadejte rozměry.")

    def _on_crop_select_requested(self, active: bool):
        """
        Toggles drag-a-rectangle crop selection.

        While selecting, the full uncropped frame is shown. Drawing a crop on an
        already-cropped preview would define a crop relative to a crop, and the
        overlay rectangle would no longer line up with what is on screen.
        """
        viewer = self.viewer_container.viewer
        viewer.set_crop_select_mode(active)

        if active:
            # A restack queued or already running from the previous crop edit
            # would land moments later and replace the full-frame reference —
            # leaving the user composing against the wrong image.
            self._restack_timer.stop()
            self._retire_worker(self._worker)
            self._worker = None
            self._stack_generation += 1
            self.progress_bar.setVisible(False)
            self.controls.btn_stack.setEnabled(True)

            self._show_uncropped_reference()
            viewer.set_crop_rect(self._crop_rect)
            self.lbl_status.setText(
                "✂️ Táhněte myší přes snímek a vyberte oblast ořezu. "
                "Zobrazen je celý needitovaný snímek.")
        else:
            # Back to the processed, cropped preview.
            self.request_restack()

    def _show_uncropped_reference(self):
        """Displays the middle exposure at full frame, for composing the crop."""
        active_items = self.exposure_list.get_active_items()
        if not active_items:
            return
        img = GLOBAL_IMAGE_CACHE.get(active_items[len(active_items) // 2].filepath, 1.0)
        if img is None:
            return
        h, w = img.shape[:2]
        viewer = self.viewer_container.viewer
        viewer.set_original_size(w, h)
        viewer.set_base_image_bgr_uint8(img, keep_view=False)

    def _on_crop_drawn(self, x: int, y: int, w: int, h: int):
        """
        The crop rectangle was set on the image (dragged, or set programmatically).

        The viewer is refreshed explicitly rather than relying on it having drawn
        the rectangle itself: that only holds for the drag path, and letting the
        window and the viewer hold different rectangles is exactly the kind of
        divergence that shows the user one crop and exports another.
        """
        self._crop_rect = (x, y, w, h)
        self.controls.set_crop_rect(self._crop_rect, from_drag=True)
        self.viewer_container.viewer.set_crop_rect(self._crop_rect)
        if self.controls.btn_crop_select.isChecked():
            self.lbl_status.setText(
                f"✂️ Vybrán ořez {w}×{h} px. Vypněte „Vybrat oblast myší“ "
                "pro náhled oříznutého výsledku.")
        else:
            self.lbl_status.setText(f"✂️ Ořez {w}×{h} px. Skládám náhled…")
            self.request_restack()

    def _current_scene_size(self) -> Tuple[int, int]:
        """Full pixel size of the loaded frames, before any crop."""
        for item in self.exposure_list.items:
            if item.width > 0 and item.height > 0:
                return item.width, item.height
        return 0, 0

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
                                crop_rect=self._crop_rect,
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

        if self.controls.btn_crop_select.isChecked():
            # Composing a crop: the viewer must keep showing the full frame.
            return

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

        # The preview now *is* the cropped region, so its own coordinate space
        # starts at the crop corner. Drawing the crop rectangle on top of it
        # would place the marker at the wrong offset. While the user is still
        # composing the crop the marker is what they are aiming with, so it stays.
        if not self.controls.btn_crop_select.isChecked():
            self.viewer_container.viewer.set_crop_rect(None)

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

        report = worker.align_report if isinstance(worker, StackingWorker) else ""
        if report:
            self.lbl_status.setText(report)
        elif self._roi_active and self._roi_rect is not None:
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

        worker = FullResExportWorker(items, self.controls.get_settings(), filepath,
                                     export_scale, crop_rect=self._crop_rect)
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
