"""
Dark Professional Astronomy & Photography Theme for PyQt6.
"""

DARK_THEME = """
QWidget {
    background-color: #1a1d24;
    color: #e0e6ed;
    font-family: 'Segoe UI', 'SF Pro Display', 'Helvetica Neue', Arial, sans-serif;
    font-size: 13px;
}

QMainWindow {
    background-color: #14161c;
}

QGroupBox {
    background-color: #21252f;
    border: 1px solid #313746;
    border-radius: 8px;
    margin-top: 24px;
    padding: 14px 10px 10px 10px;
    font-weight: bold;
    color: #4da6ff;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    top: 4px;
    padding: 0 6px;
    background-color: #21252f;
    border-radius: 4px;
}

QPushButton {
    background-color: #2b3342;
    color: #ffffff;
    border: 1px solid #3e485d;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #384257;
    border-color: #4da6ff;
}

QPushButton:pressed {
    background-color: #1e2430;
}

QPushButton:disabled {
    background-color: #1a1d24;
    color: #555e70;
    border-color: #252a35;
}

QPushButton#PrimaryButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1a73e8, stop:1 #0099ff);
    color: #ffffff;
    border: none;
    padding: 10px 20px;
    font-size: 14px;
    font-weight: bold;
}

QPushButton#PrimaryButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2b82f6, stop:1 #1ab0ff);
}

QPushButton#PrimaryButton:pressed {
    background: #145cb8;
}

QPushButton#ExportButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0f8a5f, stop:1 #10b981);
    color: #ffffff;
    border: none;
    padding: 10px 20px;
    font-size: 14px;
    font-weight: bold;
}

QPushButton#ExportButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #13a371, stop:1 #22c55e);
}

QTabWidget::pane {
    border: 1px solid #313746;
    background-color: #21252f;
    border-radius: 8px;
    top: -1px;
}

QTabBar::tab {
    background-color: #1a1d24;
    color: #8c9ba5;
    padding: 8px 16px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 4px;
    border: 1px solid #2b3342;
    border-bottom: none;
}

QTabBar::tab:selected {
    background-color: #21252f;
    color: #4da6ff;
    font-weight: bold;
    border: 1px solid #313746;
    border-bottom: 2px solid #4da6ff;
}

QTabBar::tab:hover:!selected {
    background-color: #252a35;
    color: #e0e6ed;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #151820;
    border-radius: 3px;
    border: 1px solid #2b3342;
}

QSlider::sub-page:horizontal {
    background: #4da6ff;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #ffffff;
    border: 2px solid #4da6ff;
    width: 16px;
    margin-top: -6px;
    margin-bottom: -6px;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #e6f2ff;
}

QTableWidget {
    background-color: #1a1d24;
    gridline-color: #2b3342;
    border: 1px solid #313746;
    border-radius: 6px;
    selection-background-color: #2b4566;
    selection-color: #ffffff;
}

QHeaderView::section {
    background-color: #21252f;
    color: #8c9ba5;
    padding: 6px;
    border: none;
    border-right: 1px solid #2b3342;
    border-bottom: 1px solid #313746;
    font-weight: bold;
}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #1a1d24;
    border: 1px solid #313746;
    border-radius: 6px;
    padding: 6px 10px;
    color: #e0e6ed;
}

QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
    border-color: #4da6ff;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    background-color: #21252f;
    border: 1px solid #313746;
    selection-background-color: #2b4566;
}

QCheckBox {
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid #3e485d;
    background-color: #1a1d24;
}

QCheckBox::indicator:checked {
    background-color: #1a73e8;
    border-color: #4da6ff;
}

QProgressBar {
    border: 1px solid #313746;
    border-radius: 6px;
    text-align: center;
    background-color: #1a1d24;
    color: #ffffff;
    font-weight: bold;
    height: 18px;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1a73e8, stop:1 #00b4d8);
    border-radius: 5px;
}

QScrollBar:vertical {
    background: #14161c;
    width: 10px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #2b3342;
    min-height: 20px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #3e485d;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background: #14161c;
    height: 10px;
    margin: 0;
}

QScrollBar::handle:horizontal {
    background: #2b3342;
    min-width: 20px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal:hover {
    background: #3e485d;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

QToolTip {
    background-color: #2b3342;
    color: #ffffff;
    border: 1px solid #4da6ff;
    padding: 6px;
    border-radius: 4px;
}
"""
