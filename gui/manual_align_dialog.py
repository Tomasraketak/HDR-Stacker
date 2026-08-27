"""
Interactive manual and assisted alignment dialog.

Frame-by-frame subpixel nudging with live difference, alpha-blend and flicker
previews, plus one-click automatic detection of the black lunar disc.

Frames are loaded on a background thread and cached at a bounded proxy size, so
opening the dialog on a 9-frame 24 MP bracket neither freezes the UI nor
allocates gigabytes.
"""

from typing import List, Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QPointF
from PyQt6.QtGui import QKeyEvent, QShowEvent, QCloseEvent
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QGroupBox, QRadioButton, QButtonGroup,
    QMessageBox, QSplitter, QWidget, QCheckBox, QProgressBar,
    QScrollArea, QFrame, QSizePolicy
)

try:
    from core.exif_and_analysis import ExposureItem
    from core.aligner import detect_black_circle_in_light, calculate_moon_shifts
    from core.image_cache import GLOBAL_IMAGE_CACHE
    from gui.image_viewer import InteractiveImageViewer
    from gui.ui_utils import fit_window_to_screen, center_on_screen
except ImportError:  # pragma: no cover
    from ..core.exif_and_analysis import ExposureItem
    from ..core.aligner import detect_black_circle_in_light, calculate_moon_shifts
    from ..core.image_cache import GLOBAL_IMAGE_CACHE
    from .image_viewer import InteractiveImageViewer
    from .ui_utils import fit_window_to_screen, center_on_screen

# Frames are compared at this size: large enough for confident subpixel work,
# small enough that a whole bracket fits comfortably in memory.
ALIGN_PROXY_MAX_DIM = 1600


def normalize_for_comparison(img_f32_bgr: np.ndarray) -> np.ndarray:
    """
    Equalises exposure so a -4 EV and a +4 EV frame show comparable edge contrast,
    which is what makes the difference view usable across a whole bracket.
    """
    u8 = (np.clip(np.nan_to_num(img_f32_bgr), 0.0, 1.0) * 255.0).astype(np.uint8)
    gray = cv2.cvtColor(u8, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray).astype(np.float32) / 255.0


class _FrameLoader(QThread):
    """Loads and normalises the alignment proxies off the GUI thread."""

    progress = pyqtSignal(int, str)
    # (index, bgr_float32_proxy, normalised_gray, proxy_scale)
    frame_ready = pyqtSignal(int, object, object, float)
    finished_all = pyqtSignal()

    def __init__(self, filepaths: List[str], parent=None):
        super().__init__(parent)
        self.filepaths = list(filepaths)

    def run(self):
        total = len(self.filepaths)
        for i, path in enumerate(self.filepaths):
            if self.isInterruptionRequested():
                return
            self.progress.emit(int(100 * i / max(1, total)),
                               f"Načítám snímky do paměti ({i + 1}/{total})…")

            img = GLOBAL_IMAGE_CACHE.get(path, 1.0)
            if img is None:
                # Emit a placeholder so indices stay aligned with self.items.
                self.frame_ready.emit(i, np.zeros((600, 600, 3), np.float32),
                                      np.zeros((600, 600), np.float32), 1.0)
                continue

            h, w = img.shape[:2]
            scale = min(1.0, ALIGN_PROXY_MAX_DIM / float(max(w, h)))
            if scale < 0.999:
                img = cv2.resize(img, (max(16, int(w * scale)), max(16, int(h * scale))),
                                 interpolation=cv2.INTER_AREA)

            f32 = img.astype(np.float32) / 255.0
            self.frame_ready.emit(i, f32, normalize_for_comparison(f32), scale)

        if not self.isInterruptionRequested():
            self.finished_all.emit()


class _AutoDetectWorker(QThread):
    """Runs the lunar-disc search across the whole bracket off the GUI thread."""

    progress = pyqtSignal(int, str)
    done = pyqtSignal(object)   # list of (dx, dy) in proxy pixels
    failed = pyqtSignal(str)

    def __init__(self, images_u8: List[np.ndarray], ref_idx: int, parent=None):
        super().__init__(parent)
        self.images = images_u8
        self.ref_idx = ref_idx

    def run(self):
        try:
            shifts = calculate_moon_shifts(
                self.images,
                ref_idx=self.ref_idx,
                progress_callback=lambda p, m: self.progress.emit(p, m),
                should_cancel=self.isInterruptionRequested,
            )
            if not self.isInterruptionRequested():
                self.done.emit(shifts)
        except Exception as e:
            self.failed.emit(f"Automatická detekce selhala: {e}")


class ManualAlignDialog(QDialog):
    """Interactive dialog for precise manual / assisted frame-by-frame alignment."""

    shifts_applied = pyqtSignal()

    def __init__(self, items: List[ExposureItem], parent=None):
        super().__init__(parent)
        self.setWindowTitle("🛠️ Manuální a asistované zarovnání snímků zatmění")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizeGripEnabled(True)

        self.items = items
        self.ref_idx = len(items) // 2
        self.current_idx = 0 if self.ref_idx != 0 else min(1, len(items) - 1)

        n = len(items)
        self._cached_images_f32: List[Optional[np.ndarray]] = [None] * n
        self._norm_grays_f32: List[Optional[np.ndarray]] = [None] * n
        self._proxy_scales: List[float] = [1.0] * n

        # Snapshot for Cancel: the dialog mutates the caller's items live.
        self._original_shifts = [(it.shift_x, it.shift_y, it.is_valid) for it in items]

        self._loader: Optional[_FrameLoader] = None
        self._auto_worker: Optional[_AutoDetectWorker] = None

        self._flicker_state = False
        self._flicker_timer = QTimer(self)
        self._flicker_timer.setInterval(250)
        self._flicker_timer.timeout.connect(self._on_flicker_tick)

        self._init_ui()
        # Fit to the usable desktop rather than a fixed 1320x860: on a 15" laptop
        # at 125 % scaling that height puts the action buttons off-screen.
        fit_window_to_screen(self, 1320, 860)
        center_on_screen(self)

    # ------------------------------------------------------------------ Build

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Scrolling the controls means the dialog can be made short without the
        # bottom button row ever being pushed out of reach.
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        left_scroll.setWidget(self._build_left_panel())
        left_scroll.setMinimumWidth(330)
        splitter.addWidget(left_scroll)
        splitter.addWidget(self._build_viewer_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        splitter.setSizes([360, 900])
        main_layout.addWidget(splitter, 1)

        action_bar = QFrame()
        action_bar.setObjectName("StatusStrip")
        action_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        bottom = QHBoxLayout(action_bar)
        bottom.setContentsMargins(10, 6, 10, 6)
        self.lbl_status = QLabel("Zarovnejte snímky tak, aby hrany disku na sebe seděly.")
        self.lbl_status.setWordWrap(False)
        self.lbl_status.setObjectName("StatusLabel")
        bottom.addWidget(self.lbl_status, 1)

        self.progress = QProgressBar()
        self.progress.setFixedWidth(200)
        self.progress.setVisible(False)
        bottom.addWidget(self.progress)

        self.btn_cancel = QPushButton("Zrušit")
        self.btn_cancel.clicked.connect(self.reject)
        bottom.addWidget(self.btn_cancel)

        self.btn_apply = QPushButton("✅  Použít zarovnání a složit")
        self.btn_apply.setObjectName("PrimaryButton")
        self.btn_apply.setMinimumHeight(38)
        self.btn_apply.clicked.connect(self._on_apply)
        bottom.addWidget(self.btn_apply)

        main_layout.addWidget(action_bar, 0)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(2, 2, 6, 2)
        layout.setSpacing(10)

        # --- Frame selection
        nav_group = QGroupBox("Výběr zarovnávaného snímku")
        nav_layout = QVBoxLayout(nav_group)
        row = QHBoxLayout()
        row.addWidget(QLabel("Snímek:"))

        self.combo_frame = QComboBox()
        for i, it in enumerate(self.items):
            marker = "★ ref" if i == self.ref_idx else f"{i + 1}"
            self.combo_frame.addItem(f"[{marker}] {it.filename} ({it.shutter_str})", i)
        self.combo_frame.setCurrentIndex(self.current_idx)
        self.combo_frame.currentIndexChanged.connect(self._on_frame_selected)
        row.addWidget(self.combo_frame, 1)

        self.btn_prev = QPushButton("◄")
        self.btn_prev.setFixedWidth(32)
        self.btn_prev.setToolTip("Předchozí snímek (Page Up)")
        self.btn_prev.clicked.connect(self._prev_frame)
        row.addWidget(self.btn_prev)

        self.btn_next = QPushButton("►")
        self.btn_next.setFixedWidth(32)
        self.btn_next.setToolTip("Další snímek (Page Down)")
        self.btn_next.clicked.connect(self._next_frame)
        row.addWidget(self.btn_next)
        nav_layout.addLayout(row)

        self.chk_active = QCheckBox("Zahrnout tento snímek do skládání")
        self.chk_active.setChecked(self.items[self.current_idx].is_valid)
        self.chk_active.toggled.connect(self._on_chk_active_changed)
        nav_layout.addWidget(self.chk_active)
        layout.addWidget(nav_group)

        # --- Display mode
        mode_group = QGroupBox("Režim zobrazení zarovnání")
        mode_layout = QVBoxLayout(mode_group)
        self.bg_modes = QButtonGroup(self)

        self.rb_diff = QRadioButton("Rozdíl hran (Difference) — doporučeno")
        self.rb_diff.setChecked(True)
        self.rb_blend = QRadioButton("50% průhledné překrytí (Blend)")
        self.rb_flicker = QRadioButton("Blikání (Flicker)")
        for idx, rb in enumerate((self.rb_diff, self.rb_blend, self.rb_flicker)):
            self.bg_modes.addButton(rb, idx)
            mode_layout.addWidget(rb)
        self.bg_modes.idToggled.connect(self._on_mode_changed)
        layout.addWidget(mode_group)

        # --- Nudge controls
        nudge_group = QGroupBox("Posun vybraného snímku")
        nudge_layout = QVBoxLayout(nudge_group)

        spin_row = QHBoxLayout()
        spin_row.setSpacing(4)
        spin_row.addWidget(QLabel("Δ X:"))
        self.spin_dx = self._make_shift_spin(self.items[self.current_idx].shift_x)
        spin_row.addWidget(self.spin_dx, 1)
        spin_row.addWidget(QLabel("Δ Y:"))
        self.spin_dy = self._make_shift_spin(self.items[self.current_idx].shift_y)
        spin_row.addWidget(self.spin_dy, 1)
        nudge_layout.addLayout(spin_row)

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Krok šipek:"))
        self.combo_step = QComboBox()
        for text, val in (("Jemný: 0.2 px", 0.2), ("Standardní: 1.0 px", 1.0),
                          ("Hrubý: 5.0 px", 5.0)):
            self.combo_step.addItem(text, val)
        self.combo_step.setCurrentIndex(1)
        step_row.addWidget(self.combo_step, 1)
        nudge_layout.addLayout(step_row)

        btn_up = QPushButton("▲")
        btn_up.setFixedWidth(80)
        btn_up.clicked.connect(lambda: self._nudge(0, -self._step()))
        nudge_layout.addWidget(btn_up, alignment=Qt.AlignmentFlag.AlignCenter)

        mid = QHBoxLayout()
        btn_left = QPushButton("◀")
        btn_left.clicked.connect(lambda: self._nudge(-self._step(), 0))
        mid.addWidget(btn_left)
        btn_center = QPushButton("0,0")
        btn_center.setToolTip("Vynulovat posun tohoto snímku")
        btn_center.clicked.connect(lambda: self._set_current_shift(0.0, 0.0))
        mid.addWidget(btn_center)
        btn_right = QPushButton("▶")
        btn_right.clicked.connect(lambda: self._nudge(self._step(), 0))
        mid.addWidget(btn_right)
        nudge_layout.addLayout(mid)

        btn_down = QPushButton("▼")
        btn_down.setFixedWidth(80)
        btn_down.clicked.connect(lambda: self._nudge(0, self._step()))
        nudge_layout.addWidget(btn_down, alignment=Qt.AlignmentFlag.AlignCenter)

        hint = QLabel("💡 Šipky na klávesnici posouvají v reálném čase.\n"
                      "Shift = hrubý krok (5 px), Ctrl = jemný krok (0.2 px).")
        hint.setObjectName("StatusHint")
        hint.setWordWrap(True)
        nudge_layout.addWidget(hint)
        layout.addWidget(nudge_group)

        # --- Automation
        auto_group = QGroupBox("Automatická asistence")
        auto_layout = QVBoxLayout(auto_group)
        self.btn_auto_moon = QPushButton("🌑  Najít černý disk Měsíce")
        self.btn_auto_moon.setObjectName("PrimaryButton")
        self.btn_auto_moon.setMinimumHeight(36)
        self.btn_auto_moon.setToolTip(
            "Vyhledá kruhový černý disk Měsíce v záři korony\n"
            "a předvyplní posuny pro všechny snímky.")
        self.btn_auto_moon.clicked.connect(self._auto_detect_moon)
        auto_layout.addWidget(self.btn_auto_moon)

        self.btn_reset_all = QPushButton("Vynulovat posuny všech snímků")
        self.btn_reset_all.clicked.connect(self._reset_all_shifts)
        auto_layout.addWidget(self.btn_reset_all)
        layout.addWidget(auto_group)

        layout.addStretch()
        return panel

    def _make_shift_spin(self, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-2000.0, 2000.0)
        spin.setSingleStep(0.5)
        spin.setDecimals(1)
        spin.setSuffix(" px")
        spin.setMinimumWidth(84)
        spin.setValue(value)
        spin.valueChanged.connect(self._on_spin_changed)
        return spin

    def _build_viewer_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        top_bar = QHBoxLayout()
        for text, slot, tip in (
            ("Přizpůsobit oknu", lambda: self.viewer.fit_to_window(), "Zobrazit celý snímek"),
            ("100 % (1:1)", lambda: self.viewer.actual_size_100(), "Skutečná velikost pixelů"),
            ("🔍 Přiblížit na Slunce", self._zoom_on_moon, "Najde disk a přiblíží se na něj"),
        ):
            btn = QPushButton(text)
            btn.setFixedHeight(30)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            top_bar.addWidget(btn)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        self.viewer = InteractiveImageViewer(self)
        layout.addWidget(self.viewer, 1)
        return container

    # ------------------------------------------------------------- Life cycle

    def showEvent(self, event: QShowEvent):
        super().showEvent(event)
        if self._loader is None:
            QTimer.singleShot(30, self._start_loading)

    @property
    def _frames_loaded(self) -> int:
        """Derived from the cache itself, so a re-entrant load cannot inflate it."""
        return sum(1 for img in self._cached_images_f32 if img is not None)

    def _start_loading(self):
        # showEvent can fire more than once (restore, re-show); starting a second
        # loader would orphan the first and leave a QThread running unowned.
        if self._loader is not None:
            return
        self.progress.setVisible(True)
        self.btn_auto_moon.setEnabled(False)
        self._loader = _FrameLoader([it.filepath for it in self.items], self)
        self._loader.progress.connect(self._on_load_progress)
        self._loader.frame_ready.connect(self._on_frame_ready)
        self._loader.finished_all.connect(self._on_load_finished)
        self._loader.start()

    def _on_load_progress(self, pct: int, msg: str):
        self.progress.setValue(pct)
        self.lbl_status.setText(msg)

    def _on_frame_ready(self, idx: int, f32: np.ndarray, norm: np.ndarray, scale: float):
        if not (0 <= idx < len(self.items)):
            return
        self._cached_images_f32[idx] = f32
        self._norm_grays_f32[idx] = norm
        self._proxy_scales[idx] = scale
        # Show something as soon as the reference frame is available.
        if idx in (self.ref_idx, self.current_idx):
            self._update_view()

    def _on_load_finished(self):
        self.progress.setVisible(False)
        self.btn_auto_moon.setEnabled(True)
        self._update_view()
        self.viewer.fit_to_window()
        self.lbl_status.setText("Snímky načteny. Zarovnejte je tak, aby hrany disku na sebe seděly.")

    def _stop_workers(self):
        self._flicker_timer.stop()
        for worker in (self._loader, self._auto_worker):
            if worker is not None and worker.isRunning():
                worker.requestInterruption()
                worker.wait(3000)

    def done(self, result: int):
        """
        Every way of closing a QDialog funnels through done() — accept, reject,
        Escape and the window-manager close button alike. Stopping the workers
        here is what guarantees no QThread outlives the dialog that owns it.
        """
        self._stop_workers()
        super().done(result)

    def closeEvent(self, event: QCloseEvent):
        self._stop_workers()
        super().closeEvent(event)

    def reject(self):
        """Cancel restores every shift the dialog changed."""
        for it, (sx, sy, valid) in zip(self.items, self._original_shifts):
            it.shift_x, it.shift_y, it.is_valid = sx, sy, valid
        super().reject()

    def _on_apply(self):
        self.shifts_applied.emit()
        self.accept()

    # --------------------------------------------------------------- Controls

    def _ready(self, idx: int) -> bool:
        return 0 <= idx < len(self.items) and self._cached_images_f32[idx] is not None

    def _step(self) -> float:
        return float(self.combo_step.currentData() or 1.0)

    def _zoom_on_moon(self):
        if not self._ready(self.ref_idx):
            return
        raw = (self._cached_images_f32[self.ref_idx] * 255.0).astype(np.uint8)
        disc = detect_black_circle_in_light(raw)
        if disc is not None:
            cx, cy = disc[0], disc[1]
        else:
            h, w = raw.shape[:2]
            cx, cy = w / 2.0, h / 2.0

        zoom = 3.0
        self.viewer._zoom = zoom
        self.viewer._pan_pos = QPointF(self.viewer.width() / 2.0 - cx * zoom,
                                       self.viewer.height() / 2.0 - cy * zoom)
        self.viewer._needs_fit = False
        self.viewer.update()

    def _on_frame_selected(self, idx: int):
        if not (0 <= idx < len(self.items)):
            return
        self.current_idx = idx
        it = self.items[idx]
        for widget, value in ((self.spin_dx, it.shift_x), (self.spin_dy, it.shift_y)):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self.chk_active.blockSignals(True)
        self.chk_active.setChecked(it.is_valid)
        self.chk_active.blockSignals(False)
        self._update_view()

    def _on_chk_active_changed(self, checked: bool):
        self.items[self.current_idx].is_valid = bool(checked)

    def _prev_frame(self):
        self.combo_frame.setCurrentIndex(max(0, self.current_idx - 1))

    def _next_frame(self):
        self.combo_frame.setCurrentIndex(min(len(self.items) - 1, self.current_idx + 1))

    def _nudge(self, dx: float, dy: float):
        it = self.items[self.current_idx]
        self._set_current_shift(round(it.shift_x + dx, 1), round(it.shift_y + dy, 1))

    def _set_current_shift(self, dx: float, dy: float):
        it = self.items[self.current_idx]
        it.shift_x, it.shift_y = dx, dy
        for widget, value in ((self.spin_dx, dx), (self.spin_dy, dy)):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self._update_view()

    def _on_spin_changed(self):
        it = self.items[self.current_idx]
        it.shift_x = self.spin_dx.value()
        it.shift_y = self.spin_dy.value()
        self._update_view()

    def _on_mode_changed(self):
        if self.bg_modes.checkedId() == 2:
            self._flicker_timer.start()
        else:
            self._flicker_timer.stop()
        self._update_view()

    def _on_flicker_tick(self):
        self._flicker_state = not self._flicker_state
        self._update_view()

    # ------------------------------------------------------------------ View

    def _update_view(self):
        """Renders the alignment preview with exposure-normalised frames."""
        if not self._ready(self.ref_idx) or not self._ready(self.current_idx):
            return

        ref_img = self._cached_images_f32[self.ref_idx]
        cur_img = self._cached_images_f32[self.current_idx]
        ref_norm = self._norm_grays_f32[self.ref_idx]
        cur_norm = self._norm_grays_f32[self.current_idx]

        if self.current_idx == self.ref_idx:
            self.viewer.set_image_bgr_float(ref_img)
            self.lbl_status.setText("Zobrazen referenční snímek — podle něj se zarovnávají ostatní.")
            return

        it = self.items[self.current_idx]
        scale = self._proxy_scales[self.current_idx]
        # Shifts are stored in full-resolution pixels; the preview is a proxy.
        proxy_dx, proxy_dy = it.shift_x * scale, it.shift_y * scale

        h, w = cur_img.shape[:2]
        if abs(proxy_dx) > 0.01 or abs(proxy_dy) > 0.01:
            M = np.float32([[1.0, 0.0, proxy_dx], [0.0, 1.0, proxy_dy]])
            cur_shifted = cv2.warpAffine(cur_img, M, (w, h), flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)
            cur_norm_shifted = cv2.warpAffine(cur_norm, M, (w, h), flags=cv2.INTER_CUBIC,
                                              borderMode=cv2.BORDER_REPLICATE)
        else:
            cur_shifted, cur_norm_shifted = cur_img, cur_norm

        # The reference proxy may differ in size if the frames do; match them.
        if ref_norm.shape != cur_norm_shifted.shape:
            ref_norm = cv2.resize(ref_norm, (w, h), interpolation=cv2.INTER_AREA)
        if ref_img.shape != cur_shifted.shape:
            ref_img = cv2.resize(ref_img, (w, h), interpolation=cv2.INTER_AREA)

        mode = self.bg_modes.checkedId()
        label = f"Snímek {self.current_idx + 1}: ΔX={it.shift_x:+.1f} px, ΔY={it.shift_y:+.1f} px"

        if mode == 1:
            self.viewer.set_image_bgr_float(0.5 * ref_img + 0.5 * cur_shifted)
            self.lbl_status.setText(label)
        elif mode == 2:
            self.viewer.set_image_bgr_float(ref_img if self._flicker_state else cur_shifted)
            tag = "REFERENČNÍ" if self._flicker_state else f"SNÍMEK {self.current_idx + 1}"
            self.lbl_status.setText(f"Blikání [{tag}] — pokud disk poskakuje, dolaďte ΔX a ΔY.")
        else:
            # False-colour difference: misaligned edges glow, aligned ones vanish.
            diff = np.abs(ref_norm - cur_norm_shifted)
            vis = np.empty_like(cur_shifted)
            vis[:, :, 0] = np.clip(ref_norm * 0.4, 0.0, 1.0)
            vis[:, :, 1] = np.clip(diff * 2.5 + cur_norm_shifted * 0.3, 0.0, 1.0)
            vis[:, :, 2] = np.clip(diff * 2.0, 0.0, 1.0)
            self.viewer.set_image_bgr_float(vis)
            self.lbl_status.setText(f"{label} — minimalizujte barevné hrany disku.")

    # ------------------------------------------------------------ Auto detect

    def _auto_detect_moon(self):
        if self._frames_loaded < len(self.items):
            QMessageBox.information(self, "Snímky se ještě načítají",
                                    "Počkejte prosím na dokončení načítání snímků.")
            return
        if self._auto_worker is not None and self._auto_worker.isRunning():
            return

        self.btn_auto_moon.setEnabled(False)
        self.progress.setVisible(True)
        self.lbl_status.setText("Probíhá automatické vyhledávání černého disku Měsíce…")

        raw = [(img * 255.0).astype(np.uint8) for img in self._cached_images_f32]
        self._auto_worker = _AutoDetectWorker(raw, self.ref_idx, self)
        self._auto_worker.progress.connect(self._on_load_progress)
        self._auto_worker.done.connect(self._on_auto_done)
        self._auto_worker.failed.connect(self._on_auto_failed)
        self._auto_worker.start()

    def _on_auto_done(self, shifts):
        self.progress.setVisible(False)
        self.btn_auto_moon.setEnabled(True)

        detected = 0
        for idx, (proxy_dx, proxy_dy) in enumerate(shifts):
            if idx >= len(self.items):
                break
            scale = self._proxy_scales[idx] or 1.0
            # Convert proxy-pixel shifts back to full-resolution pixels.
            self.items[idx].shift_x = round(proxy_dx / scale, 1)
            self.items[idx].shift_y = round(proxy_dy / scale, 1)
            if abs(proxy_dx) > 0.01 or abs(proxy_dy) > 0.01:
                detected += 1

        self._set_current_shift(self.items[self.current_idx].shift_x,
                                self.items[self.current_idx].shift_y)

        QMessageBox.information(
            self, "Automatická detekce dokončena",
            f"Vypočteny posuny pro {len(self.items)} snímků "
            f"({detected} z nich bylo potřeba posunout).\n\n"
            "Projděte si jednotlivé snímky šipkami ◄ ► a případně dolaďte.")

    def _on_auto_failed(self, msg: str):
        self.progress.setVisible(False)
        self.btn_auto_moon.setEnabled(True)
        self.lbl_status.setText(f"❌ {msg}")

    def _reset_all_shifts(self):
        for it in self.items:
            it.shift_x = 0.0
            it.shift_y = 0.0
        self._set_current_shift(0.0, 0.0)

    # ------------------------------------------------------------- Keyboard

    def keyPressEvent(self, event: QKeyEvent):
        step = self._step()
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            step = 5.0
        elif event.modifiers() & (Qt.KeyboardModifier.ControlModifier
                                  | Qt.KeyboardModifier.AltModifier):
            step = 0.2

        key = event.key()
        if key == Qt.Key.Key_Left:
            self._nudge(-step, 0)
        elif key == Qt.Key.Key_Right:
            self._nudge(step, 0)
        elif key == Qt.Key.Key_Up:
            self._nudge(0, -step)
        elif key == Qt.Key.Key_Down:
            self._nudge(0, step)
        elif key == Qt.Key.Key_PageUp:
            self._prev_frame()
        elif key == Qt.Key.Key_PageDown:
            self._next_frame()
        elif key == Qt.Key.Key_Escape:
            self.reject()
        else:
            super().keyPressEvent(event)
