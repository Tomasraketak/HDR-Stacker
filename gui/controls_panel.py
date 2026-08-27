"""
Controls panel.

Houses the fusion engine, alignment strategy, working resolution, noise
reduction, coronal enhancement and live post-processing adjustments.

Two distinct signals are exposed so the main window knows how much work a change
implies: `live_adjust_requested` only needs the post-processing pipeline re-run
on the existing merge, while `merge_param_changed` requires a full re-stack.
"""

from typing import Dict, Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QSlider, QComboBox, QPushButton, QScrollArea, QFrame, QSizePolicy,
    QSpinBox, QCheckBox, QGridLayout
)
from typing import Optional, Tuple


class SliderRow(QWidget):
    """A labelled slider with a value readout badge and a double-click reset."""

    valueChanged = pyqtSignal(float)

    def __init__(self, label: str, min_val: float, max_val: float, default_val: float,
                 step: float = 0.05, suffix: str = "", tip: str = "", parent=None):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.suffix = suffix
        self.default_val = default_val
        # Integer slider positions map onto float values through this multiplier.
        self.multiplier = max(1, int(round(1.0 / step)))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(8)

        self.lbl_title = QLabel(label)
        self.lbl_title.setObjectName("SliderLabel")
        self.lbl_title.setMinimumWidth(112)
        layout.addWidget(self.lbl_title)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(round(min_val * self.multiplier)), int(round(max_val * self.multiplier)))
        self.slider.setValue(int(round(default_val * self.multiplier)))
        self.slider.setCursor(Qt.CursorShape.PointingHandCursor)
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider, 1)

        self.lbl_val = QLabel(self._format(default_val))
        self.lbl_val.setObjectName("SliderValue")
        self.lbl_val.setMinimumWidth(58)
        self.lbl_val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.lbl_val)

        if tip:
            self.setToolTip(tip)
            self.lbl_title.setToolTip(tip)

    def _format(self, val: float) -> str:
        return f"{val:.2f}{self.suffix}"

    def _on_slider_changed(self, val: int):
        float_val = val / float(self.multiplier)
        self.lbl_val.setText(self._format(float_val))
        self.valueChanged.emit(float_val)

    def mouseDoubleClickEvent(self, event):
        """Double-clicking a row restores its default — a standard editor gesture."""
        self.setValue(self.default_val)

    def value(self) -> float:
        return self.slider.value() / float(self.multiplier)

    def setValue(self, val: float):
        self.slider.setValue(int(round(val * self.multiplier)))


class ControlsPanel(QWidget):
    """Sidebar with stacking, alignment, working resolution and live adjustments."""

    stack_requested = pyqtSignal()
    live_adjust_requested = pyqtSignal()
    merge_param_changed = pyqtSignal()
    crop_changed = pyqtSignal(object)          # (x, y, w, h) or None
    crop_select_requested = pyqtSignal(bool)
    crop_defaults_requested = pyqtSignal()     # enabled with no usable rectangle yet
    manual_align_requested = pyqtSignal()
    export_requested = pyqtSignal()

    # name -> (brightness, contrast, gamma, saturation, shadows, highlights,
    #          denoise, coronal_boost, coronal_radius)
    PRESETS = {
        "Vlastní nastavení": None,
        "🌑 Vnitřní korona (detail)": (0.00, 1.20, 0.95, 1.10, 0.05, 0.35, 0.15, 0.80, 4.0),
        "🌟 Vnější korona (jemné paprsky)": (0.06, 1.05, 1.35, 1.05, 0.45, 0.10, 0.35, 1.20, 12.0),
        "💎 Diamantový prsten": (-0.04, 1.30, 0.85, 1.20, 0.00, 0.50, 0.10, 0.40, 3.0),
        "🏔️ Krajina se zatměním": (0.05, 1.10, 1.15, 1.15, 0.35, 0.25, 0.20, 0.30, 8.0),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(300)
        self._loading_preset = False
        self._init_ui()

    # ------------------------------------------------------------------ Build

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 0, 4, 4)
        main_layout.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 8, 4)
        layout.setSpacing(14)

        layout.addWidget(self._build_engine_group())
        layout.addWidget(self._build_crop_group())
        layout.addWidget(self._build_corona_group())
        layout.addWidget(self._build_adjust_group())
        layout.addStretch()

        scroll.setWidget(content)
        main_layout.addWidget(scroll, 1)

        self.btn_stack = QPushButton("⚡  Složit snímky  (Ctrl+R)")
        self.btn_stack.setObjectName("PrimaryButton")
        self.btn_stack.setMinimumHeight(46)
        self.btn_stack.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stack.clicked.connect(self.stack_requested.emit)
        main_layout.addWidget(self.btn_stack)

        self.btn_export = QPushButton("💾  Exportovat v plné kvalitě  (Ctrl+S)")
        self.btn_export.setObjectName("ExportButton")
        self.btn_export.setMinimumHeight(42)
        self.btn_export.setEnabled(False)
        self.btn_export.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_export.clicked.connect(self.export_requested.emit)
        main_layout.addWidget(self.btn_export)

    def _build_engine_group(self) -> QGroupBox:
        group = QGroupBox("Výpočet a zarovnání expozic")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(9)

        self.combo_proxy = self._labelled_combo(vbox, "Pracovní rychlost:", [
            ("⚡  1/4 rozlišení — bleskové", 0.25),
            ("🚀  1/8 rozlišení — ultra rychlé", 0.125),
            ("⚖️  1/2 rozlišení — vyvážené", 0.5),
            ("🎯  1/1 plné rozlišení — pomalé", 1.0),
        ], tip="Náhled se počítá ze zmenšených kopií, aby byl okamžitý.\n"
               "Export vždy proběhne v plné kvalitě z originálů.")
        self.combo_proxy.currentIndexChanged.connect(self.merge_param_changed.emit)

        self.combo_algo = self._labelled_combo(vbox, "Metoda HDR:", [
            ("Mertens Exposure Fusion", "mertens"),
            ("Debevec 32-bit HDR", "debevec"),
            ("Robertson 32-bit HDR", "robertson"),
        ], tip="Mertens je pro zatmění nejlepší a nepotřebuje expoziční časy.\n"
               "Debevec a Robertson vyžadují známé časy závěrky z EXIF.")
        self.combo_algo.currentIndexChanged.connect(self._on_algo_changed)

        self.combo_align = self._labelled_combo(vbox, "Zarovnání:", [
            ("💡  Podle statických světel (lampy)", "static_lights"),
            ("🌑  Detekce černého disku Měsíce", "eclipse_disc"),
            ("🚫  Bez zarovnání (stativ)", "none"),
        ], tip="💡 Podle světel: najde vzor pouličních lamp a vzdálených světel\n"
               "v dolní části snímku. Ta se nehýbou, takže měří přímo otřesy\n"
               "fotoaparátu — nejpřesnější volba, když je v záběru krajina.\n\n"
               "🌑 Podle disku Měsíce: zarovná na korónu. Použijte, když je\n"
               "v záběru jen obloha, nebo když se Slunce mezi snímky posunulo.")
        self.combo_align.currentIndexChanged.connect(self.merge_param_changed.emit)

        self.btn_manual_align = QPushButton("🛠️  Ruční dozarovnání snímek po snímku…")
        self.btn_manual_align.setToolTip(
            "Interaktivní okno pro přesné posouvání každé fotky pomocí šipek\n"
            "s rozdílovým, prolínacím a blikacím náhledem. (Ctrl+M)")
        self.btn_manual_align.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_manual_align.clicked.connect(self.manual_align_requested.emit)
        vbox.addWidget(self.btn_manual_align)

        # Mertens weights (only meaningful for exposure fusion)
        self.mertens_box = QWidget()
        m_layout = QVBoxLayout(self.mertens_box)
        m_layout.setContentsMargins(0, 4, 0, 0)
        m_layout.setSpacing(2)
        self.slider_m_contrast = SliderRow(
            "Váha kontrastu:", 0.0, 3.0, 1.0, step=0.1,
            tip="Vyšší hodnota upřednostní snímky s ostrými detaily korony.")
        self.slider_m_sat = SliderRow(
            "Váha saturace:", 0.0, 3.0, 1.0, step=0.1,
            tip="Vyšší hodnota zvýrazní barevné protuberance.")
        self.slider_m_exp = SliderRow(
            "Váha expozice (šum):", 0.0, 3.0, 1.0, step=0.1,
            tip="Vyšší hodnota upřednostní dobře exponované pixely a potlačí šum.")
        for s in (self.slider_m_contrast, self.slider_m_sat, self.slider_m_exp):
            s.valueChanged.connect(lambda _v: self.merge_param_changed.emit())
            m_layout.addWidget(s)
        vbox.addWidget(self.mertens_box)

        # Tonemapping (only for Debevec / Robertson)
        self.tonemap_box = QWidget()
        t_layout = QVBoxLayout(self.tonemap_box)
        t_layout.setContentsMargins(0, 4, 0, 0)
        self.combo_tonemap = self._labelled_combo(t_layout, "Tonemapping:", [
            ("Reinhard — přirozený", "reinhard"),
            ("Drago — světlé stíny", "drago"),
            ("Mantiuk — vysoký kontrast", "mantiuk"),
        ])
        self.combo_tonemap.currentIndexChanged.connect(self.merge_param_changed.emit)
        self.tonemap_box.setVisible(False)
        vbox.addWidget(self.tonemap_box)

        return group

    def _build_crop_group(self) -> QGroupBox:
        """Output crop, applied identically to every exposure and to the export."""
        group = QGroupBox("Ořez snímku")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(7)

        self.chk_crop = QCheckBox("Oříznout výsledek")
        self.chk_crop.setToolTip(
            "Ořez se použije na všechny expozice stejně — v náhledu i při exportu.\n"
            "Zarovnání proběhne ještě před ořezem, takže se u okrajů neztrácí data.")
        self.chk_crop.toggled.connect(self._on_crop_toggled)
        vbox.addWidget(self.chk_crop)

        self.btn_crop_select = QPushButton("🖱️  Vybrat oblast myší")
        self.btn_crop_select.setCheckable(True)
        self.btn_crop_select.setToolTip("Táhněte myší přes snímek a vyberte oblast ořezu.")
        self.btn_crop_select.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_crop_select.toggled.connect(self._on_crop_select_toggled)
        vbox.addWidget(self.btn_crop_select)

        grid = QGridLayout()
        grid.setSpacing(5)
        self.spin_crop = {}
        for col, (key, label) in enumerate((("x", "X:"), ("y", "Y:"))):
            lbl = QLabel(label)
            lbl.setObjectName("SliderLabel")
            grid.addWidget(lbl, 0, col * 2)
            grid.addWidget(self._make_crop_spin(key), 0, col * 2 + 1)
        for col, (key, label) in enumerate((("w", "Šířka:"), ("h", "Výška:"))):
            lbl = QLabel(label)
            lbl.setObjectName("SliderLabel")
            grid.addWidget(lbl, 1, col * 2)
            grid.addWidget(self._make_crop_spin(key), 1, col * 2 + 1)
        vbox.addLayout(grid)

        self.btn_crop_reset = QPushButton("Celý snímek (zrušit ořez)")
        self.btn_crop_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_crop_reset.clicked.connect(lambda: self.chk_crop.setChecked(False))
        vbox.addWidget(self.btn_crop_reset)

        self.lbl_crop_info = QLabel("Ořez vypnut — pracuje se s celým snímkem.")
        self.lbl_crop_info.setObjectName("StatusHint")
        self.lbl_crop_info.setWordWrap(True)
        vbox.addWidget(self.lbl_crop_info)

        self._set_crop_controls_enabled(False)
        return group

    def _make_crop_spin(self, key: str) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(0, 100000)
        spin.setSingleStep(10)
        spin.setSuffix(" px")
        spin.setMinimumWidth(78)
        spin.valueChanged.connect(self._on_crop_spin_changed)
        self.spin_crop[key] = spin
        return spin

    def _set_crop_controls_enabled(self, enabled: bool):
        for spin in self.spin_crop.values():
            spin.setEnabled(enabled)
        self.btn_crop_select.setEnabled(enabled)
        self.btn_crop_reset.setEnabled(enabled)

    def _on_crop_toggled(self, checked: bool):
        self._set_crop_controls_enabled(checked)
        if not checked:
            self.btn_crop_select.setChecked(False)
            self.lbl_crop_info.setText("Ořez vypnut — pracuje se s celým snímkem.")
            self.crop_changed.emit(None)
        else:
            self._emit_crop()

    def _on_crop_select_toggled(self, checked: bool):
        if checked and not self.chk_crop.isChecked():
            self.chk_crop.setChecked(True)
        self.crop_select_requested.emit(checked)

    def _on_crop_spin_changed(self, _value: int = 0):
        if self._loading_preset or not self.chk_crop.isChecked():
            return
        self._emit_crop()

    def _emit_crop(self):
        rect = self.get_crop_rect()
        if rect is None:
            # Enabled but empty: ask the window to seed it from the real frame
            # size, so the user starts from the whole image rather than 0x0.
            self.lbl_crop_info.setText("Nastavuji výchozí ořez…")
            self.crop_defaults_requested.emit()
            return
        self.lbl_crop_info.setText(
            f"Ořez {rect[2]} × {rect[3]} px  ·  {rect[2] * rect[3] / 1e6:.1f} Mpx")
        self.crop_changed.emit(rect)

    def set_crop_rect(self, rect: Optional[Tuple[int, int, int, int]],
                      from_drag: bool = False):
        """
        Pushes a crop rectangle into the numeric fields.

        `from_drag` marks the one case that should also leave selection mode:
        the user finished dragging on the image. Seeding defaults must not turn
        the mode off, or enabling it would immediately cancel itself.
        """
        if rect is None:
            self.chk_crop.setChecked(False)
            return
        self._loading_preset = True
        try:
            if not self.chk_crop.isChecked():
                self.chk_crop.blockSignals(True)
                self.chk_crop.setChecked(True)
                self.chk_crop.blockSignals(False)
                self._set_crop_controls_enabled(True)
            for key, value in zip(("x", "y", "w", "h"), rect):
                self.spin_crop[key].setValue(int(value))
        finally:
            self._loading_preset = False
        self.lbl_crop_info.setText(
            f"Ořez {rect[2]} × {rect[3]} px  ·  {rect[2] * rect[3] / 1e6:.1f} Mpx")
        # Drawing is a one-shot gesture: drop back out of drag mode so the
        # cropped preview is rendered straight away.
        if from_drag and self.btn_crop_select.isChecked():
            self.btn_crop_select.setChecked(False)

    def get_crop_rect(self) -> Optional[Tuple[int, int, int, int]]:
        if not self.chk_crop.isChecked():
            return None
        w = self.spin_crop["w"].value()
        h = self.spin_crop["h"].value()
        if w < 16 or h < 16:
            return None
        return (self.spin_crop["x"].value(), self.spin_crop["y"].value(), w, h)

    def _build_corona_group(self) -> QGroupBox:
        group = QGroupBox("Potlačení šumu a korona")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(2)

        self.slider_denoise = SliderRow(
            "Redukce šumu:", 0.0, 1.0, 0.0, step=0.05,
            tip="Bilaterální filtr — vyhladí zrno oblohy, ale zachová paprsky korony.")
        self.slider_coronal_boost = SliderRow(
            "Detaily korony:", 0.0, 2.0, 0.0, step=0.05,
            tip="Zvýrazní jemné struktury magnetického pole.\n"
                "Tmavá obloha je chráněna, takže se nezvýrazňuje šum.")
        self.slider_coronal_radius = SliderRow(
            "Poloměr detailů:", 1.0, 25.0, 6.0, step=0.5, suffix=" px",
            tip="Malý poloměr = jemné struktury u disku.\n"
                "Velký poloměr = dlouhé paprsky vnější korony.")
        for s in (self.slider_denoise, self.slider_coronal_boost, self.slider_coronal_radius):
            s.valueChanged.connect(self._on_live_param_changed)
            vbox.addWidget(s)
        return group

    def _build_adjust_group(self) -> QGroupBox:
        group = QGroupBox("Okamžité živé úpravy")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(2)

        self.combo_preset = QComboBox()
        for name in self.PRESETS:
            self.combo_preset.addItem(name)
        self.combo_preset.setToolTip("Výchozí nastavení pro typické situace při zatmění.")
        self.combo_preset.currentTextChanged.connect(self._on_preset_selected)
        vbox.addWidget(self.combo_preset)

        self.slider_brightness = SliderRow("Jas:", -0.5, 0.5, 0.0, step=0.02)
        self.slider_contrast = SliderRow("Kontrast:", 0.5, 2.5, 1.0, step=0.05)
        self.slider_gamma = SliderRow(
            "Gamma:", 0.4, 2.5, 1.0, step=0.05,
            tip="Nad 1.0 vytáhne slabou vnější korónu z temné oblohy.")
        self.slider_saturation = SliderRow("Sytost barev:", 0.0, 2.5, 1.0, step=0.05)
        self.slider_shadows = SliderRow("Projasnění stínů:", 0.0, 1.0, 0.0, step=0.05)
        self.slider_highlights = SliderRow("Tlumení světel:", 0.0, 1.0, 0.0, step=0.05)
        for s in (self.slider_brightness, self.slider_contrast, self.slider_gamma,
                  self.slider_saturation, self.slider_shadows, self.slider_highlights):
            s.valueChanged.connect(self._on_live_param_changed)
            vbox.addWidget(s)

        self.btn_reset_adj = QPushButton("↺  Obnovit posuvníky")
        self.btn_reset_adj.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reset_adj.clicked.connect(self.reset_adjustments)
        vbox.addWidget(self.btn_reset_adj)
        return group

    def _labelled_combo(self, parent_layout, label: str, entries, tip: str = "") -> QComboBox:
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setObjectName("SliderLabel")
        lbl.setMinimumWidth(118)
        row.addWidget(lbl)
        combo = QComboBox()
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        for text, data in entries:
            combo.addItem(text, data)
        if tip:
            combo.setToolTip(tip)
            lbl.setToolTip(tip)
        row.addWidget(combo, 1)
        parent_layout.addLayout(row)
        return combo

    # ----------------------------------------------------------- Interaction

    def _on_algo_changed(self, _idx: int):
        is_mertens = (self.combo_algo.currentData() == "mertens")
        self.mertens_box.setVisible(is_mertens)
        self.tonemap_box.setVisible(not is_mertens)
        self.merge_param_changed.emit()

    def _on_live_param_changed(self, _=None):
        # A manual slider move means the result no longer matches the preset.
        if not self._loading_preset and self.combo_preset.currentIndex() != 0:
            self.combo_preset.blockSignals(True)
            self.combo_preset.setCurrentIndex(0)
            self.combo_preset.blockSignals(False)
        self.live_adjust_requested.emit()

    def _on_preset_selected(self, name: str):
        values = self.PRESETS.get(name)
        if values is None:
            return
        (brightness, contrast, gamma, saturation, shadows,
         highlights, denoise, boost, radius) = values

        self._loading_preset = True
        try:
            self.slider_brightness.setValue(brightness)
            self.slider_contrast.setValue(contrast)
            self.slider_gamma.setValue(gamma)
            self.slider_saturation.setValue(saturation)
            self.slider_shadows.setValue(shadows)
            self.slider_highlights.setValue(highlights)
            self.slider_denoise.setValue(denoise)
            self.slider_coronal_boost.setValue(boost)
            self.slider_coronal_radius.setValue(radius)
        finally:
            self._loading_preset = False
        self.live_adjust_requested.emit()

    def reset_adjustments(self):
        self._loading_preset = True
        try:
            for slider in (self.slider_brightness, self.slider_contrast, self.slider_gamma,
                           self.slider_saturation, self.slider_shadows, self.slider_highlights,
                           self.slider_denoise, self.slider_coronal_boost, self.slider_coronal_radius):
                slider.setValue(slider.default_val)
            self.combo_preset.blockSignals(True)
            self.combo_preset.setCurrentIndex(0)
            self.combo_preset.blockSignals(False)
        finally:
            self._loading_preset = False
        self.live_adjust_requested.emit()

    def get_settings(self) -> Dict[str, Any]:
        return {
            'proxy_scale': self.combo_proxy.currentData(),
            'algo': self.combo_algo.currentData(),
            'align_method': self.combo_align.currentData(),
            'mertens_contrast': self.slider_m_contrast.value(),
            'mertens_saturation': self.slider_m_sat.value(),
            'mertens_exposure': self.slider_m_exp.value(),
            'tonemap_method': self.combo_tonemap.currentData(),
            'brightness': self.slider_brightness.value(),
            'contrast': self.slider_contrast.value(),
            'gamma': self.slider_gamma.value(),
            'saturation': self.slider_saturation.value(),
            'shadows': self.slider_shadows.value(),
            'highlights': self.slider_highlights.value(),
            'denoise': self.slider_denoise.value(),
            'coronal_boost': self.slider_coronal_boost.value(),
            'coronal_radius': self.slider_coronal_radius.value(),
            'crop_rect': self.get_crop_rect(),
        }
