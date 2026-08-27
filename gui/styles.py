"""
Dark astronomy theme for PyQt6.

Palette: deep space-blue neutrals with a single cyan accent, chosen so the
photograph in the middle of the window is always the brightest thing on screen.
Only Qt-supported QSS properties are used — unsupported ones (box-shadow,
transform, transition) are silently ignored by Qt and only add noise.
"""

# --- Palette ---------------------------------------------------------------
BG_DEEP = "#0d0f14"     # window / canvas
BG_BASE = "#151922"     # panels
BG_RAISED = "#1c212c"   # group boxes, inputs
BG_HOVER = "#252b39"
BORDER = "#2a3140"
BORDER_STRONG = "#3a4356"

TEXT = "#e2e8f0"
TEXT_DIM = "#94a3b8"
TEXT_FAINT = "#64748b"

ACCENT = "#38bdf8"
ACCENT_DEEP = "#0284c7"
SUCCESS = "#10b981"
SUCCESS_DEEP = "#059669"

DARK_THEME = f"""
QWidget {{
    background-color: {BG_BASE};
    color: {TEXT};
    font-family: 'Segoe UI', 'SF Pro Text', 'Inter', 'Helvetica Neue', Arial, sans-serif;
    font-size: 13px;
}}

QMainWindow, QDialog {{
    background-color: {BG_DEEP};
}}

/* ---------------------------------------------------------------- Group box */
QGroupBox {{
    background-color: {BG_RAISED};
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-top: 22px;
    padding: 16px 12px 12px 12px;
    font-weight: 600;
    color: {ACCENT};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    top: 2px;
    padding: 2px 10px;
    background-color: {BG_DEEP};
    border: 1px solid {BORDER};
    border-radius: 9px;
    font-size: 12px;
    letter-spacing: 0.3px;
}}

/* ----------------------------------------------------------------- Buttons */
QPushButton {{
    background-color: {BG_HOVER};
    color: {TEXT};
    border: 1px solid {BORDER_STRONG};
    border-radius: 7px;
    padding: 7px 14px;
    font-weight: 600;
}}

QPushButton:hover {{
    background-color: #2f374a;
    border-color: {ACCENT};
    color: #ffffff;
}}

QPushButton:pressed {{
    background-color: #1a1f2b;
}}

QPushButton:disabled {{
    background-color: #171b24;
    color: {TEXT_FAINT};
    border-color: #232936;
}}

QPushButton:checked {{
    background-color: {ACCENT_DEEP};
    border-color: {ACCENT};
    color: #ffffff;
}}

QPushButton#PrimaryButton {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 {ACCENT_DEEP}, stop:1 {ACCENT});
    color: #04121c;
    border: none;
    padding: 10px 20px;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.3px;
    border-radius: 9px;
}}

QPushButton#PrimaryButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 #0ea5e9, stop:1 #7dd3fc);
}}

QPushButton#PrimaryButton:pressed {{
    background: #026aa2;
    color: #ffffff;
}}

QPushButton#PrimaryButton:disabled {{
    background: #1d2430;
    color: {TEXT_FAINT};
}}

QPushButton#ExportButton {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 {SUCCESS_DEEP}, stop:1 {SUCCESS});
    color: #03201a;
    border: none;
    padding: 10px 20px;
    font-size: 14px;
    font-weight: 700;
    border-radius: 9px;
}}

QPushButton#ExportButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 #10b981, stop:1 #6ee7b7);
}}

QPushButton#ExportButton:disabled {{
    background: #1d2430;
    color: {TEXT_FAINT};
}}

QPushButton#AddButton {{
    background-color: {ACCENT_DEEP};
    border: 1px solid {ACCENT};
    color: #ffffff;
    font-weight: 700;
}}

QPushButton#AddButton:hover {{
    background-color: {ACCENT};
    color: #04121c;
}}

QPushButton#RoiToggle:checked {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 {ACCENT_DEEP}, stop:1 {ACCENT});
    color: #04121c;
    border: 1px solid #7dd3fc;
}}

/* -------------------------------------------------------- Viewer chrome */
QFrame#ViewerToolbar {{
    background-color: {BG_RAISED};
    border: 1px solid {BORDER};
    border-radius: 9px;
}}

QFrame#ViewerToolbar QPushButton {{
    padding: 5px 12px;
}}

QFrame#ToolbarSeparator {{
    color: {BORDER_STRONG};
    background-color: {BORDER_STRONG};
    max-width: 1px;
}}

QLabel#PixelReadout {{
    color: {ACCENT};
    font-family: 'Consolas', 'SF Mono', 'DejaVu Sans Mono', monospace;
    font-size: 11px;
    background: transparent;
}}

QFrame#StatusStrip {{
    background-color: {BG_RAISED};
    border: 1px solid {BORDER};
    border-radius: 9px;
}}

QFrame#StatusStrip QLabel {{
    background: transparent;
}}

QLabel#StatusLabel {{
    color: {TEXT};
    font-weight: 500;
}}

QLabel#StatusHint, QLabel#SliderLabel {{
    color: {TEXT_DIM};
    font-size: 11px;
    background: transparent;
}}

QLabel#SliderLabel {{
    font-size: 12px;
}}

QLabel#SliderValue {{
    color: {ACCENT};
    font-family: 'Consolas', 'SF Mono', 'DejaVu Sans Mono', monospace;
    font-weight: 700;
    font-size: 11px;
    background: transparent;
}}

/* ----------------------------------------------------------------- Sliders */
QSlider::groove:horizontal {{
    height: 5px;
    background: #0f131b;
    border-radius: 3px;
    border: 1px solid {BORDER};
}}

QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 {ACCENT_DEEP}, stop:1 {ACCENT});
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: #ffffff;
    border: 2px solid {ACCENT};
    width: 13px;
    height: 13px;
    margin: -6px 0;
    border-radius: 8px;
}}

QSlider::handle:horizontal:hover {{
    background: {ACCENT};
    border-color: #ffffff;
}}

QSlider::handle:horizontal:disabled {{
    background: #3a4356;
    border-color: #2a3140;
}}

/* ------------------------------------------------------------------ Tables */
QTableWidget {{
    background-color: {BG_DEEP};
    alternate-background-color: #12161e;
    gridline-color: {BORDER};
    border: 1px solid {BORDER};
    border-radius: 8px;
    selection-background-color: {ACCENT_DEEP};
    selection-color: #ffffff;
}}

QTableWidget::item {{
    padding: 4px;
    border: none;
}}

QTableWidget::item:selected {{
    background-color: {ACCENT_DEEP};
}}

QHeaderView::section {{
    background-color: {BG_RAISED};
    color: {TEXT_DIM};
    padding: 7px 6px;
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    font-weight: 700;
    font-size: 11px;
}}

/* ------------------------------------------------------------------ Inputs */
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background-color: {BG_DEEP};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 6px 10px;
    color: {TEXT};
    min-height: 18px;
}}

QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{
    border-color: {ACCENT};
}}

QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
    border-color: {ACCENT};
    background-color: #11151d;
}}

QComboBox::drop-down {{
    border: none;
    width: 22px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_DIM};
    width: 0;
    height: 0;
    margin-right: 8px;
}}

QComboBox QAbstractItemView {{
    background-color: {BG_RAISED};
    border: 1px solid {BORDER_STRONG};
    border-radius: 6px;
    selection-background-color: {ACCENT_DEEP};
    selection-color: #ffffff;
    padding: 4px;
    outline: none;
}}

/* -------------------------------------------------------------- Checkboxes */
QCheckBox, QRadioButton {{
    spacing: 8px;
    background: transparent;
    color: {TEXT};
}}

QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {BORDER_STRONG};
    background-color: {BG_DEEP};
}}

QCheckBox::indicator {{
    border-radius: 4px;
}}

QRadioButton::indicator {{
    border-radius: 9px;
}}

QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color: {ACCENT};
}}

QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {ACCENT};
    border-color: #7dd3fc;
}}

/* ------------------------------------------------------------ Progress bar */
QProgressBar {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    text-align: center;
    background-color: {BG_DEEP};
    color: {TEXT};
    font-weight: 700;
    font-size: 11px;
    max-height: 18px;
}}

QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 {ACCENT_DEEP}, stop:1 {ACCENT});
    border-radius: 7px;
}}

/* --------------------------------------------------------------- Scrollbars */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}

QScrollBar::handle:vertical {{
    background: {BORDER_STRONG};
    min-height: 28px;
    border-radius: 5px;
}}

QScrollBar::handle:vertical:hover {{
    background: {ACCENT_DEEP};
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}

QScrollBar::handle:horizontal {{
    background: {BORDER_STRONG};
    min-width: 28px;
    border-radius: 5px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {ACCENT_DEEP};
}}

QScrollBar::add-line, QScrollBar::sub-line {{
    width: 0px;
    height: 0px;
}}

QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
}}

QScrollArea {{
    background: transparent;
    border: none;
}}

/* ------------------------------------------------------------------ Splitter */
QSplitter::handle {{
    background-color: transparent;
}}

QSplitter::handle:horizontal {{
    width: 6px;
}}

QSplitter::handle:hover {{
    background-color: {ACCENT_DEEP};
}}

/* ------------------------------------------------------------------ Tooltip */
QToolTip {{
    background-color: #0a0d13;
    color: {TEXT};
    border: 1px solid {ACCENT};
    padding: 7px 9px;
    border-radius: 6px;
    font-size: 12px;
}}

/* -------------------------------------------------------------- Message box */
QMessageBox {{
    background-color: {BG_RAISED};
}}

QMessageBox QLabel {{
    background: transparent;
    color: {TEXT};
}}

QMessageBox QPushButton {{
    min-width: 84px;
}}
"""
