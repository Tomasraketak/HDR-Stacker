"""
Controls Panel Widget.
Houses parameters for HDR fusion, alignment strategy, proxy working resolution,
noise reduction, coronal enhancement, and live post-processing adjustments.
"""

from typing import Dict, Any
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QSlider, QComboBox, QPushButton, QScrollArea, QFrame
)


class SliderRow(QWidget):
    """A clean slider row with label and value readout badge."""
    valueChanged = pyqtSignal(float)

    def __init__(self, label: str, min_val: float, max_val: float, default_val: float, step: float = 0.05, suffix: str = "", parent=None):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.suffix = suffix
        self.multiplier = int(1.0 / step)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        self.lbl_title = QLabel(label)
        self.lbl_title.setStyleSheet("color: #b0bac9; font-size: 12px;")
        layout.addWidget(self.lbl_title)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(min_val * self.multiplier), int(max_val * self.multiplier))
        self.slider.setValue(int(default_val * self.multiplier))
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider)

        self.lbl_val = QLabel(f"{default_val:.2f}{suffix}")
        self.lbl_val.setStyleSheet("color: #4da6ff; font-weight: bold; min-width: 45px; text-align: right;")
        self.lbl_val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.lbl_val)

    def _on_slider_changed(self, val: int):
        float_val = val / float(self.multiplier)
        self.lbl_val.setText(f"{float_val:.2f}{self.suffix}")
        self.valueChanged.emit(float_val)

    def value(self) -> float:
        return self.slider.value() / float(self.multiplier)

    def setValue(self, val: float):
        self.slider.setValue(int(val * self.multiplier))


class ControlsPanel(QWidget):
    """
    Control sidebar with Stacking, Alignment, Proxy Mode, and Real-time adjustments.
    """
    stack_requested = pyqtSignal()
    live_adjust_requested = pyqtSignal()
    manual_align_requested = pyqtSignal()
    export_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(12)

        # 1. Stacking Engine & Proxy Speed Group
        algo_group = QGroupBox("Výpočet a Zarovnání expozic")
        algo_layout = QVBoxLayout(algo_group)
        algo_layout.setSpacing(8)

        # Proxy Quality Selector
        proxy_row = QHBoxLayout()
        proxy_row.addWidget(QLabel("Pracovní rychlost:"))
        self.combo_proxy = QComboBox()
        self.combo_proxy.addItem("⚡ 1/4 rozlišení (Bleskově rychlé)", 0.25)
        self.combo_proxy.addItem("🚀 1/8 rozlišení (Ultra rychlé)", 0.125)
        self.combo_proxy.addItem("⚖️ 1/2 rozlišení (Vyvážené)", 0.5)
        self.combo_proxy.addItem("🎯 1/1 Plné rozlišení", 1.0)
        self.combo_proxy.setToolTip("Při práci se fotky zmenší pro okamžité složení a editaci. Plná kvalita se spočítá automaticky až při exportu!")
        proxy_row.addWidget(self.combo_proxy)
        algo_layout.addLayout(proxy_row)

        # HDR Method Selector
        algo_row = QHBoxLayout()
        algo_row.addWidget(QLabel("Metoda HDR:"))
        self.combo_algo = QComboBox()
        self.combo_algo.addItem("Mertens Exposure Fusion", "mertens")
        self.combo_algo.addItem("Debevec 32-bit HDR", "debevec")
        self.combo_algo.addItem("Robertson 32-bit HDR", "robertson")
        self.combo_algo.currentIndexChanged.connect(self._on_algo_changed)
        algo_row.addWidget(self.combo_algo)
        algo_layout.addLayout(algo_row)

        # Alignment Mode Selector
        align_row = QHBoxLayout()
        align_row.addWidget(QLabel("Zarovnání:"))
        self.combo_align = QComboBox()
        self.combo_align.addItem("🌑 Detekce černého disku Měsíce", "eclipse_disc")
        self.combo_align.addItem("🚫 Bez zarovnání (Stativ / Krajina + Slunce)", "none")
        self.combo_align.setToolTip("Detekuje černý kruhový disk Měsíce obklopený světlem korony a subpixelově zarovná snímky.")
        align_row.addWidget(self.combo_align)
        algo_layout.addLayout(align_row)

        # Manual Alignment Button
        self.btn_manual_align = QPushButton("🛠️ Ruční dozarovnání fotku po fotce...")
        self.btn_manual_align.setToolTip("Otevře interaktivní okno pro přesné manuální posouvání každé fotky pomocí šipek a rozdílového náhledu.")
        self.btn_manual_align.clicked.connect(self.manual_align_requested.emit)
        algo_layout.addWidget(self.btn_manual_align)

        # Mertens Weights Container
        self.mertens_box = QWidget()
        m_layout = QVBoxLayout(self.mertens_box)
        m_layout.setContentsMargins(0, 4, 0, 0)
        self.slider_m_contrast = SliderRow("Váha kontrastu:", 0.0, 3.0, 1.0, step=0.1)
        m_layout.addWidget(self.slider_m_contrast)

        self.slider_m_sat = SliderRow("Váha saturace:", 0.0, 3.0, 1.0, step=0.1)
        m_layout.addWidget(self.slider_m_sat)

        self.slider_m_exp = SliderRow("Potlačení šumu (Váha expozice):", 0.0, 3.0, 1.0, step=0.1)
        m_layout.addWidget(self.slider_m_exp)
        algo_layout.addWidget(self.mertens_box)

        # Tonemapping container (for Debevec/Robertson)
        self.tonemap_box = QWidget()
        t_layout = QVBoxLayout(self.tonemap_box)
        t_layout.setContentsMargins(0, 4, 0, 0)
        t_row = QHBoxLayout()
        t_row.addWidget(QLabel("Tonemapping:"))
        self.combo_tonemap = QComboBox()
        self.combo_tonemap.addItem("Reinhard", "reinhard")
        self.combo_tonemap.addItem("Drago", "drago")
        self.combo_tonemap.addItem("Mantiuk", "mantiuk")
        t_row.addWidget(self.combo_tonemap)
        t_layout.addLayout(t_row)
        self.tonemap_box.setVisible(False)
        algo_layout.addWidget(self.tonemap_box)

        layout.addWidget(algo_group)

        # 2. Noise reduction & Coronal detail enhancer
        corona_group = QGroupBox("Potlačení šumu & Korona")
        corona_layout = QVBoxLayout(corona_group)
        corona_layout.setSpacing(6)

        self.slider_denoise = SliderRow("Redukce šumu (Denoise):", 0.0, 1.0, 0.0, step=0.05)
        self.slider_denoise.valueChanged.connect(self._on_live_param_changed)
        corona_layout.addWidget(self.slider_denoise)

        self.slider_coronal_boost = SliderRow("Zvýraznění detailů korony:", 0.0, 2.0, 0.0, step=0.05)
        self.slider_coronal_boost.valueChanged.connect(self._on_live_param_changed)
        corona_layout.addWidget(self.slider_coronal_boost)

        self.slider_coronal_radius = SliderRow("Poloměr detailů:", 1.0, 25.0, 6.0, step=0.5, suffix=" px")
        self.slider_coronal_radius.valueChanged.connect(self._on_live_param_changed)
        corona_layout.addWidget(self.slider_coronal_radius)

        layout.addWidget(corona_group)

        # 3. Live Fine Adjustments
        adj_group = QGroupBox("Okamžité živé úpravy")
        adj_layout = QVBoxLayout(adj_group)
        adj_layout.setSpacing(6)

        self.slider_brightness = SliderRow("Jas:", -0.5, 0.5, 0.0, step=0.02)
        self.slider_brightness.valueChanged.connect(self._on_live_param_changed)
        adj_layout.addWidget(self.slider_brightness)

        self.slider_contrast = SliderRow("Kontrast:", 0.5, 2.5, 1.0, step=0.05)
        self.slider_contrast.valueChanged.connect(self._on_live_param_changed)
        adj_layout.addWidget(self.slider_contrast)

        self.slider_gamma = SliderRow("Gamma:", 0.4, 2.5, 1.0, step=0.05)
        self.slider_gamma.valueChanged.connect(self._on_live_param_changed)
        adj_layout.addWidget(self.slider_gamma)

        self.slider_saturation = SliderRow("Sytost barev:", 0.0, 2.5, 1.0, step=0.05)
        self.slider_saturation.valueChanged.connect(self._on_live_param_changed)
        adj_layout.addWidget(self.slider_saturation)

        self.slider_shadows = SliderRow("Projasnění stínů:", 0.0, 1.0, 0.0, step=0.05)
        self.slider_shadows.valueChanged.connect(self._on_live_param_changed)
        adj_layout.addWidget(self.slider_shadows)

        self.slider_highlights = SliderRow("Tlumení světel:", 0.0, 1.0, 0.0, step=0.05)
        self.slider_highlights.valueChanged.connect(self._on_live_param_changed)
        adj_layout.addWidget(self.slider_highlights)

        self.btn_reset_adj = QPushButton("Obnovit posuvníky")
        self.btn_reset_adj.clicked.connect(self.reset_adjustments)
        adj_layout.addWidget(self.btn_reset_adj)

        layout.addWidget(adj_group)
        layout.addStretch()

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        # 4. Action Buttons
        self.btn_stack = QPushButton("⚡ Složit snímky (HDR Merge)")
        self.btn_stack.setObjectName("PrimaryButton")
        self.btn_stack.setMinimumHeight(44)
        self.btn_stack.clicked.connect(self.stack_requested.emit)
        main_layout.addWidget(self.btn_stack)

        self.btn_export = QPushButton("💾 Exportovat v plné kvalitě (TIFF/JPG)")
        self.btn_export.setObjectName("ExportButton")
        self.btn_export.setMinimumHeight(40)
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_requested.emit)
        main_layout.addWidget(self.btn_export)

    def _on_algo_changed(self, idx: int):
        algo = self.combo_algo.currentData()
        is_mertens = (algo == "mertens")
        self.mertens_box.setVisible(is_mertens)
        self.tonemap_box.setVisible(not is_mertens)

    def _on_live_param_changed(self, _=None):
        self.live_adjust_requested.emit()

    def reset_adjustments(self):
        self.slider_brightness.setValue(0.0)
        self.slider_contrast.setValue(1.0)
        self.slider_gamma.setValue(1.0)
        self.slider_saturation.setValue(1.0)
        self.slider_shadows.setValue(0.0)
        self.slider_highlights.setValue(0.0)
        self.slider_denoise.setValue(0.0)
        self.slider_coronal_boost.setValue(0.0)
        self.slider_coronal_radius.setValue(6.0)
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
        }
