"""
Interactive Manual & Assisted Alignment Dialog.
Allows frame-by-frame subpixel manual nudging with live Difference, Alpha-blend, and Flicker modes,
as well as one-click automatic detection of the black lunar circle.
"""

from typing import List, Optional
import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont, QKeyEvent
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QGroupBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QRadioButton, QButtonGroup,
    QMessageBox, QSplitter, QWidget
)

try:
    from core.exif_and_analysis import ExposureItem
    from core.aligner import detect_black_circle_in_light, calculate_moon_shifts
    from gui.image_viewer import InteractiveImageViewer
except ImportError:
    from ..core.exif_and_analysis import ExposureItem
    from ..core.aligner import detect_black_circle_in_light, calculate_moon_shifts
    from .image_viewer import InteractiveImageViewer


class ManualAlignDialog(QDialog):
    """
    Interactive dialog for precise manual / assisted frame-by-frame alignment.
    """
    shifts_applied = pyqtSignal()

    def __init__(self, items: List[ExposureItem], parent=None):
        super().__init__(parent)
        self.setWindowTitle("🛠️ Manuální a asistované zarovnání snímků zatmění")
        self.resize(1200, 780)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.items = items
        self.ref_idx = len(items) // 2
        self.current_idx = 0
        
        # Cache loaded proxy images for fast display
        self._cached_images_f32: List[np.ndarray] = []
        self._load_cached_images()

        # Flicker timer
        self._flicker_state = False
        self._flicker_timer = QTimer(self)
        self._flicker_timer.setInterval(250)
        self._flicker_timer.timeout.connect(self._on_flicker_tick)

        self._init_ui()
        self._update_view()

    def _load_cached_images(self):
        """Loads downscaled proxy images for instant responsive manipulation."""
        self._cached_images_f32 = []
        for it in self.items:
            img = cv2.imdecode(np.fromfile(it.filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((400, 400, 3), dtype=np.uint8)
            h, w = img.shape[:2]
            scale = min(1.0, 1600.0 / max(w, h))
            if scale < 0.99:
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            self._cached_images_f32.append(img.astype(np.float32) / 255.0)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 1. Left Control Panel
        left_panel = QWidget()
        l_layout = QVBoxLayout(left_panel)
        l_layout.setContentsMargins(4, 4, 4, 4)
        l_layout.setSpacing(8)

        # Frame navigation
        nav_group = QGroupBox("Výběr zarovnávaného snímku")
        nav_layout = QVBoxLayout(nav_group)

        self.combo_frame = QComboBox()
        for idx, it in enumerate(self.items):
            is_ref = " (REFERENČNÍ BÁZE)" if idx == self.ref_idx else ""
            self.combo_frame.addItem(f"{idx+1}. {it.filename} [{it.shutter_str}]{is_ref}", idx)
        self.combo_frame.currentIndexChanged.connect(self._on_frame_selected)
        nav_layout.addWidget(self.combo_frame)

        nav_btns = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Předchozí")
        self.btn_prev.clicked.connect(self._prev_frame)
        nav_btns.addWidget(self.btn_prev)
        self.btn_next = QPushButton("Další ▶")
        self.btn_next.clicked.connect(self._next_frame)
        nav_btns.addWidget(self.btn_next)
        nav_layout.addLayout(nav_btns)
        l_layout.addWidget(nav_group)

        # Display Mode (Difference, Blend, Flicker)
        mode_group = QGroupBox("Režim zobrazení zarovnání")
        mode_layout = QVBoxLayout(mode_group)
        self.bg_modes = QButtonGroup(self)

        self.rb_diff = QRadioButton("Rozdíl hran (Difference) — ideální")
        self.rb_diff.setChecked(True)
        self.bg_modes.addButton(self.rb_diff, 0)
        mode_layout.addWidget(self.rb_diff)

        self.rb_blend = QRadioButton("50% Průhledné překrytí (Blend)")
        self.bg_modes.addButton(self.rb_blend, 1)
        mode_layout.addWidget(self.rb_blend)

        self.rb_flicker = QRadioButton("Blikání (Flicker toggle)")
        self.bg_modes.addButton(self.rb_flicker, 2)
        mode_layout.addWidget(self.rb_flicker)

        self.bg_modes.idToggled.connect(self._on_mode_changed)
        l_layout.addWidget(mode_group)

        # Nudge Pad & SpinBoxes
        nudge_group = QGroupBox("Posun vybraného snímku (Nudge)")
        nudge_layout = QVBoxLayout(nudge_group)

        # Spinboxes
        spin_row = QHBoxLayout()
        spin_row.addWidget(QLabel("Δ X:"))
        self.spin_dx = QDoubleSpinBox()
        self.spin_dx.setRange(-200.0, 200.0)
        self.spin_dx.setSingleStep(0.5)
        self.spin_dx.setDecimals(1)
        self.spin_dx.setSuffix(" px")
        self.spin_dx.valueChanged.connect(self._on_spin_changed)
        spin_row.addWidget(self.spin_dx)

        spin_row.addWidget(QLabel("Δ Y:"))
        self.spin_dy = QDoubleSpinBox()
        self.spin_dy.setRange(-200.0, 200.0)
        self.spin_dy.setSingleStep(0.5)
        self.spin_dy.setDecimals(1)
        self.spin_dy.setSuffix(" px")
        self.spin_dy.valueChanged.connect(self._on_spin_changed)
        spin_row.addWidget(self.spin_dy)
        nudge_layout.addLayout(spin_row)

        # Step size
        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Krok šipek:"))
        self.combo_step = QComboBox()
        self.combo_step.addItem("Jemný: 0.5 px", 0.5)
        self.combo_step.addItem("Standardní: 1.0 px", 1.0)
        self.combo_step.addItem("Hrubý: 5.0 px", 5.0)
        self.combo_step.setCurrentIndex(1)
        step_row.addWidget(self.combo_step)
        nudge_layout.addLayout(step_row)

        # Arrow Button Pad
        btn_up = QPushButton("▲ Nahoru")
        btn_up.clicked.connect(lambda: self._nudge(0, -self.combo_step.currentData()))
        nudge_layout.addWidget(btn_up, alignment=Qt.AlignmentFlag.AlignCenter)

        mid_pad = QHBoxLayout()
        btn_left = QPushButton("◀ Vlevo")
        btn_left.clicked.connect(lambda: self._nudge(-self.combo_step.currentData(), 0))
        mid_pad.addWidget(btn_left)

        btn_center = QPushButton("0,0")
        btn_center.setToolTip("Vynulovat posun")
        btn_center.clicked.connect(lambda: self._set_current_shift(0.0, 0.0))
        mid_pad.addWidget(btn_center)

        btn_right = QPushButton("Vpravo ▶")
        btn_right.clicked.connect(lambda: self._nudge(self.combo_step.currentData(), 0))
        mid_pad.addWidget(btn_right)
        nudge_layout.addLayout(mid_pad)

        btn_down = QPushButton("▼ Dolů")
        btn_down.clicked.connect(lambda: self._nudge(0, self.combo_step.currentData()))
        nudge_layout.addWidget(btn_down, alignment=Qt.AlignmentFlag.AlignCenter)

        lbl_hint = QLabel("💡 Tip: Můžete posouvat i přímo klávesovými šipkami (←, →, ↑, ↓)!")
        lbl_hint.setStyleSheet("color: #8c9ba5; font-size: 11px;")
        lbl_hint.setWordWrap(True)
        nudge_layout.addWidget(lbl_hint)

        l_layout.addWidget(nudge_group)

        # Automation button: Auto-detect black circle
        auto_group = QGroupBox("Automatická detekce")
        auto_layout = QVBoxLayout(auto_group)

        self.btn_auto_moon = QPushButton("🌑 Automaticky najít černý disk Měsíce")
        self.btn_auto_moon.setToolTip("Vyhledá kruhový černý disk Měsíce v záři korony na všech snímcích a automaticky vypočte posuny.")
        self.btn_auto_moon.setStyleSheet("background-color: #1a73e8; color: white; font-weight: bold;")
        self.btn_auto_moon.clicked.connect(self._auto_detect_moon)
        auto_layout.addWidget(self.btn_auto_moon)

        self.btn_reset_all = QPushButton("Vynulovat všechny posuny")
        self.btn_reset_all.clicked.connect(self._reset_all_shifts)
        auto_layout.addWidget(self.btn_reset_all)

        l_layout.addWidget(auto_group)
        l_layout.addStretch()

        splitter.addWidget(left_panel)

        # 2. Right: Interactive Image Viewer
        self.viewer = InteractiveImageViewer(self)
        splitter.addWidget(self.viewer)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        main_layout.addWidget(splitter)

        # Bottom Buttons
        bottom_row = QHBoxLayout()
        self.lbl_status = QLabel("Zarovnejte snímky tak, aby hrany na sebe seděly.")
        self.lbl_status.setStyleSheet("color: #4da6ff; font-weight: 500;")
        bottom_row.addWidget(self.lbl_status)
        bottom_row.addStretch()

        self.btn_cancel = QPushButton("Zrušit")
        self.btn_cancel.clicked.connect(self.reject)
        bottom_row.addWidget(self.btn_cancel)

        self.btn_apply = QPushButton("✅ Použít zarovnání a složit")
        self.btn_apply.setObjectName("PrimaryButton")
        self.btn_apply.clicked.connect(self._on_apply)
        bottom_row.addWidget(self.btn_apply)

        main_layout.addLayout(bottom_row)

    def _on_frame_selected(self, idx: int):
        self.current_idx = idx
        it = self.items[idx]
        self.spin_dx.blockSignals(True)
        self.spin_dy.blockSignals(True)
        self.spin_dx.setValue(it.shift_x)
        self.spin_dy.setValue(it.shift_y)
        self.spin_dx.blockSignals(False)
        self.spin_dy.blockSignals(False)
        self._update_view()

    def _prev_frame(self):
        new_idx = max(0, self.current_idx - 1)
        self.combo_frame.setCurrentIndex(new_idx)

    def _next_frame(self):
        new_idx = min(len(self.items) - 1, self.current_idx + 1)
        self.combo_frame.setCurrentIndex(new_idx)

    def _nudge(self, dx: float, dy: float):
        it = self.items[self.current_idx]
        it.shift_x = round(it.shift_x + dx, 1)
        it.shift_y = round(it.shift_y + dy, 1)
        self.spin_dx.setValue(it.shift_x)
        self.spin_dy.setValue(it.shift_y)
        self._update_view()

    def _set_current_shift(self, dx: float, dy: float):
        it = self.items[self.current_idx]
        it.shift_x = dx
        it.shift_y = dy
        self.spin_dx.setValue(dx)
        self.spin_dy.setValue(dy)
        self._update_view()

    def _on_spin_changed(self):
        it = self.items[self.current_idx]
        it.shift_x = self.spin_dx.value()
        it.shift_y = self.spin_dy.value()
        self._update_view()

    def _on_mode_changed(self):
        mode = self.bg_modes.checkedId()
        if mode == 2:  # Flicker
            self._flicker_timer.start()
        else:
            self._flicker_timer.stop()
        self._update_view()

    def _on_flicker_tick(self):
        self._flicker_state = not self._flicker_state
        self._update_view()

    def _update_view(self):
        """Renders the alignment preview according to selected mode."""
        if not self._cached_images_f32:
            return

        ref_img = self._cached_images_f32[self.ref_idx]
        cur_img = self._cached_images_f32[self.current_idx]
        it = self.items[self.current_idx]

        h, w = cur_img.shape[:2]
        dx, dy = it.shift_x, it.shift_y

        # Shift current image
        if abs(dx) > 0.01 or abs(dy) > 0.01:
            M = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
            cur_shifted = cv2.warpAffine(cur_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
        else:
            cur_shifted = cur_img.copy()

        mode = self.bg_modes.checkedId()

        if self.current_idx == self.ref_idx:
            # Viewing reference itself
            self.viewer.set_image_bgr_float(ref_img)
            self.lbl_status.setText("Zobrazen referenční snímek báze.")
            return

        if mode == 0:  # Difference
            # Absolute difference in luminance
            gray_ref = cv2.cvtColor((ref_img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray_cur = cv2.cvtColor((cur_shifted * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            diff = np.abs(gray_ref - gray_cur) / 255.0
            # Highlight edges in high contrast green
            diff_bgr = np.zeros_like(ref_img)
            diff_bgr[:, :, 1] = np.clip(diff * 3.0, 0.0, 1.0)
            diff_bgr[:, :, 2] = np.clip(diff * 1.5, 0.0, 1.0)
            self.viewer.set_image_bgr_float(diff_bgr)
            self.lbl_status.setText(f"Snímek {self.current_idx+1}: ΔX={dx:+.1f}px, ΔY={dy:+.1f}px. Minimalizujte zelené hrany.")

        elif mode == 1:  # 50% Blend
            blend = 0.5 * ref_img + 0.5 * cur_shifted
            self.viewer.set_image_bgr_float(blend)
            self.lbl_status.setText(f"Snímek {self.current_idx+1}: ΔX={dx:+.1f}px, ΔY={dy:+.1f}px.")

        elif mode == 2:  # Flicker
            img_to_show = ref_img if self._flicker_state else cur_shifted
            self.viewer.set_image_bgr_float(img_to_show)
            tag = "REFERENČNÍ" if self._flicker_state else f"SNÍMEK {self.current_idx+1}"
            self.lbl_status.setText(f"Blikání [{tag}]: Pokud obraz poskakuje, posuňte ΔX a ΔY.")

    def _auto_detect_moon(self):
        """Runs automatic black circle detection on all frames and sets shifts."""
        self.lbl_status.setText("Probíhá automatické vyhledávání černého disku Měsíce...")
        raw_bgrs = [(img * 255.0).astype(np.uint8) for img in self._cached_images_f32]
        shifts = calculate_moon_shifts(raw_bgrs, ref_idx=self.ref_idx)

        detected_count = 0
        for idx, (dx, dy) in enumerate(shifts):
            self.items[idx].shift_x = round(dx, 1)
            self.items[idx].shift_y = round(dy, 1)
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                detected_count += 1

        self.spin_dx.setValue(self.items[self.current_idx].shift_x)
        self.spin_dy.setValue(self.items[self.current_idx].shift_y)
        self._update_view()

        QMessageBox.information(
            self,
            "Automatická detekce dokončena",
            f"Úspěšně vypočteny posuny pro {len(self.items)} snímků.\nNyní můžete jednotlivé snímky zkontrolovat a doladit."
        )

    def _reset_all_shifts(self):
        for it in self.items:
            it.shift_x = 0.0
            it.shift_y = 0.0
        self.spin_dx.setValue(0.0)
        self.spin_dy.setValue(0.0)
        self._update_view()

    def keyPressEvent(self, event: QKeyEvent):
        """Arrow key shortcuts for direct nudge."""
        step = self.combo_step.currentData()
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            step = 5.0
        elif event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier):
            step = 0.2

        if event.key() == Qt.Key.Key_Left:
            self._nudge(-step, 0)
        elif event.key() == Qt.Key.Key_Right:
            self._nudge(step, 0)
        elif event.key() == Qt.Key.Key_Up:
            self._nudge(0, -step)
        elif event.key() == Qt.Key.Key_Down:
            self._nudge(0, step)
        elif event.key() == Qt.Key.Key_PageUp:
            self._prev_frame()
        elif event.key() == Qt.Key.Key_PageDown:
            self._next_frame()
        else:
            super().keyPressEvent(event)

    def _on_apply(self):
        self._flicker_timer.stop()
        self.shifts_applied.emit()
        self.accept()
