"""
Controls Panel Widget.
Houses parameters for HDR fusion, alignment, astronomical coronal enhancement,
live post-processing sliders, and export actions.
"""

from typing import Dict, Any
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QSlider, QComboBox, QCheckBox, QPushButton,
    QScrollArea, QFrame
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
    Control sidebar with tabs for Stacking & Alignment, Post-processing, and Coronal Filters.
    """
    stack_requested = pyqtSignal()
    parameters_changed = pyqtSignal()
    export_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(10)

        # Scroll area for compact height adaptability
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(12)

        # 1. Stacking Engine & Algorithm Group
        algo_group = QGroupBox("Algoritmus skládání expozic")
        algo_layout = QVBoxLayout(algo_group)
        algo_layout.setSpacing(8)

        self.combo_algo = QComboBox()
        self.combo_algo.addItem("Mertens Exposure Fusion (Doporučeno pro zatmění)", "mertens")
        self.combo_algo.addItem("Debevec 32-bit HDR + Tonemapping", "debevec")
        self.combo_algo.addItem("Robertson 32-bit HDR + Tonemapping", "robertson")
        self.combo_algo.currentIndexChanged.connect(self._on_algo_changed)
        algo_layout.addWidget(self.combo_algo)

        # Alignment Checkbox
        self.chk_align = QCheckBox("Automatické zarovnání snímků (MTB)")
        self.chk_align.setChecked(True)
        self.chk_align.setToolTip("Median Threshold Bitmap zarovnání kompenzuje drobné posuny fotoaparátu mezi jednotlivými expozicemi.")
        algo_layout.addWidget(self.chk_align)

        # Mertens Weights Container
        self.mertens_box = QWidget()
        m_layout = QVBoxLayout(self.mertens_box)
        m_layout.setContentsMargins(0, 4, 0, 0)
        self.slider_m_contrast = SliderRow("Váha kontrastu:", 0.0, 3.0, 1.0, step=0.1)
        self.slider_m_contrast.valueChanged.connect(self._on_param_changed)
        m_layout.addWidget(self.slider_m_contrast)

        self.slider_m_sat = SliderRow("Váha saturace:", 0.0, 3.0, 1.0, step=0.1)
        self.slider_m_sat.valueChanged.connect(self._on_param_changed)
        m_layout.addWidget(self.slider_m_sat)

        self.slider_m_exp = SliderRow("Váha expozice:", 0.0, 2.0, 0.0, step=0.1)
        self.slider_m_exp.valueChanged.connect(self._on_param_changed)
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
        self.combo_tonemap.currentIndexChanged.connect(self._on_param_changed)
        t_row.addWidget(self.combo_tonemap)
        t_layout.addLayout(t_row)
        
        self.tonemap_box.setVisible(False)
        algo_layout.addWidget(self.tonemap_box)

        layout.addWidget(algo_group)

        # 2. Solar Eclipse Coronal Detail Enhancer
        corona_group = QGroupBox("Zvýraznění sluneční korony (Eclipse Filter)")
        corona_layout = QVBoxLayout(corona_group)
        corona_layout.setSpacing(6)

        self.slider_coronal_boost = SliderRow("Zvýraznění detailů:", 0.0, 2.0, 0.35, step=0.05)
        self.slider_coronal_boost.valueChanged.connect(self._on_param_changed)
        corona_layout.addWidget(self.slider_coronal_boost)

        self.slider_coronal_radius = SliderRow("Poloměr filtru:", 1.0, 25.0, 6.0, step=0.5, suffix=" px")
        self.slider_coronal_radius.valueChanged.connect(self._on_param_changed)
        corona_layout.addWidget(self.slider_coronal_radius)

        layout.addWidget(corona_group)

        # 3. Fine Adjustments (Exposure / Gamma / Colors)
        adj_group = QGroupBox("Úprava expozice a barev")
        adj_layout = QVBoxLayout(adj_group)
        adj_layout.setSpacing(6)

        self.slider_brightness = SliderRow("Jas:", -0.5, 0.5, 0.0, step=0.02)
        self.slider_brightness.valueChanged.connect(self._on_param_changed)
        adj_layout.addWidget(self.slider_brightness)

        self.slider_contrast = SliderRow("Kontrast:", 0.5, 2.5, 1.1, step=0.05)
        self.slider_contrast.valueChanged.connect(self._on_param_changed)
        adj_layout.addWidget(self.slider_contrast)

        self.slider_gamma = SliderRow("Gamma:", 0.4, 2.5, 1.0, step=0.05)
        self.slider_gamma.valueChanged.connect(self._on_param_changed)
        adj_layout.addWidget(self.slider_gamma)

        self.slider_saturation = SliderRow("Sytost barev:", 0.0, 2.5, 1.15, step=0.05)
        self.slider_saturation.valueChanged.connect(self._on_param_changed)
        adj_layout.addWidget(self.slider_saturation)

        self.slider_shadows = SliderRow("Projasnění stínů:", 0.0, 1.0, 0.0, step=0.05)
        self.slider_shadows.valueChanged.connect(self._on_param_changed)
        adj_layout.addWidget(self.slider_shadows)

        self.slider_highlights = SliderRow("Tlumení světel:", 0.0, 1.0, 0.0, step=0.05)
        self.slider_highlights.valueChanged.connect(self._on_param_changed)
        adj_layout.addWidget(self.slider_highlights)

        # Reset button for adjustments
        self.btn_reset_adj = QPushButton("Obnovit posuvníky")
        self.btn_reset_adj.clicked.connect(self.reset_adjustments)
        adj_layout.addWidget(self.btn_reset_adj)

        layout.addWidget(adj_group)
        layout.addStretch()

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        # 4. Primary Action Buttons
        self.btn_stack = QPushButton("⚡ Složit snímky (HDR Merge)")
        self.btn_stack.setObjectName("PrimaryButton")
        self.btn_stack.setMinimumHeight(44)
        self.btn_stack.setToolTip("Zarovná vybrané expozice a sloučí je do jednoho výsledného HDR snímku")
        self.btn_stack.clicked.connect(self.stack_requested.emit)
        main_layout.addWidget(self.btn_stack)

        self.btn_export = QPushButton("💾 Exportovat výsledek (TIFF / PNG / JPG)")
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
        self.parameters_changed.emit()

    def _on_param_changed(self, _=None):
        self.parameters_changed.emit()

    def reset_adjustments(self):
        self.slider_brightness.setValue(0.0)
        self.slider_contrast.setValue(1.1)
        self.slider_gamma.setValue(1.0)
        self.slider_saturation.setValue(1.15)
        self.slider_shadows.setValue(0.0)
        self.slider_highlights.setValue(0.0)
        self.slider_coronal_boost.setValue(0.35)
        self.slider_coronal_radius.setValue(6.0)
        self.parameters_changed.emit()

    def get_settings(self) -> Dict[str, Any]:
        return {
            'algo': self.combo_algo.currentData(),
            'align': self.chk_align.isChecked(),
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
            'coronal_boost': self.slider_coronal_boost.value(),
            'coronal_radius': self.slider_coronal_radius.value(),
        }
