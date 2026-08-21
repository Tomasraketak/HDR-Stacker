"""
Exposure List Widget.
Displays loaded image thumbnails, EXIF data, calculated EV steps,
and allows sorting, enabling/disabling, and auto-recognizing EV sequences.
"""

from typing import List, Optional
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap, QImage
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QFileDialog, QLabel, QDoubleSpinBox,
    QGroupBox, QCheckBox, QMessageBox
)

try:
    from core.exif_and_analysis import ExposureItem, inspect_exposure_files, format_shutter_speed
except ImportError:
    from ..core.exif_and_analysis import ExposureItem, inspect_exposure_files, format_shutter_speed


class ExposureListWidget(QWidget):
    """
    Manages loaded exposure frames.
    """
    items_changed = pyqtSignal()
    item_selected = pyqtSignal(str)  # filepath of selected image to preview

    def __init__(self, parent=None):
        super().__init__(parent)
        self.items: List[ExposureItem] = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # 1. Top action buttons
        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("+ Přidat fotky")
        self.btn_add.clicked.connect(self._on_add_files)
        btn_layout.addWidget(self.btn_add)

        self.btn_clear = QPushButton("Vyčistit")
        self.btn_clear.clicked.connect(self.clear_all)
        btn_layout.addWidget(self.btn_clear)
        layout.addLayout(btn_layout)

        # 2. EV Sequence Auto-Detector / Wizard Group
        wizard_group = QGroupBox("Automatické rozpoznání EV řady")
        wiz_layout = QVBoxLayout(wizard_group)
        wiz_layout.setSpacing(6)

        wiz_row = QHBoxLayout()
        wiz_row.addWidget(QLabel("Krok expozice:"))
        self.spin_ev_step = QDoubleSpinBox()
        self.spin_ev_step.setRange(0.1, 5.0)
        self.spin_ev_step.setSingleStep(0.33)
        self.spin_ev_step.setValue(1.0)
        self.spin_ev_step.setSuffix(" EV")
        self.spin_ev_step.setFixedWidth(85)
        wiz_row.addWidget(self.spin_ev_step)
        wiz_layout.addLayout(wiz_row)

        self.btn_auto_sort = QPushButton("🔄 Seřadit a spočítat EV")
        self.btn_auto_sort.setToolTip("Automaticky seřadí načtené snímky od nejtmavšího (-EV) po nejsvětlejší (+EV) a namapuje expozice")
        self.btn_auto_sort.clicked.connect(self.auto_sort_and_map_ev)
        wiz_layout.addWidget(self.btn_auto_sort)

        layout.addWidget(wizard_group)

        # 3. Table of exposures
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Aktivní", "Náhled", "Soubor", "Čas", "EV"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.cellClicked.connect(self._on_row_clicked)
        self.table.cellChanged.connect(self._on_cell_changed)
        layout.addWidget(self.table)

        # Bottom info label
        self.lbl_count = QLabel("Načteno snímků: 0")
        self.lbl_count.setStyleSheet("color: #8c9ba5; font-size: 11px;")
        layout.addWidget(self.lbl_count)

    def _on_add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Vyberte expoziční sérii fotografií",
            "",
            "Obrázky (*.jpg *.jpeg *.png *.tif *.tiff *.dng);;Všechny soubory (*.*)"
        )
        if files:
            self.load_files(files)

    def load_files(self, filepaths: List[str]):
        """Inspects and loads list of files."""
        ev_step = self.spin_ev_step.value()
        new_items = inspect_exposure_files(filepaths, user_ev_step=ev_step)
        
        # Merge or replace
        existing_paths = {it.filepath for it in self.items}
        for item in new_items:
            if item.filepath not in existing_paths:
                self.items.append(item)

        self.auto_sort_and_map_ev()

    def auto_sort_and_map_ev(self):
        """Automatically sorts items and assigns EV sequence."""
        if not self.items:
            self.refresh_table()
            return

        ev_step = self.spin_ev_step.value()
        paths = [it.filepath for it in self.items]
        self.items = inspect_exposure_files(paths, user_ev_step=ev_step)
        self.refresh_table()
        self.items_changed.emit()

    def clear_all(self):
        self.items.clear()
        self.refresh_table()
        self.items_changed.emit()

    def refresh_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.items))
        
        for row, item in enumerate(self.items):
            # Column 0: Checkbox
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            chk_item.setCheckState(Qt.CheckState.Checked if item.is_valid else Qt.CheckState.Unchecked)
            self.table.setItem(row, 0, chk_item)

            # Column 1: Thumbnail
            thumb_item = QTableWidgetItem()
            if item.thumbnail is not None:
                h, w, ch = item.thumbnail.shape
                qimg = QImage(item.thumbnail.data, w, h, ch * w, QImage.Format.Format_RGB888)
                pix = QPixmap.fromImage(qimg).scaled(48, 36, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                thumb_item.setIcon(QIcon(pix))
            thumb_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(row, 1, thumb_item)

            # Column 2: Filename
            name_item = QTableWidgetItem(item.filename)
            name_item.setToolTip(item.filepath)
            name_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(row, 2, name_item)

            # Column 3: Shutter time
            shutter_item = QTableWidgetItem(item.shutter_str)
            shutter_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            shutter_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(row, 3, shutter_item)

            # Column 4: EV
            ev_str = f"{item.calculated_ev:+.1f} EV" if item.calculated_ev is not None else "0.0 EV"
            ev_item = QTableWidgetItem(ev_str)
            ev_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            ev_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(row, 4, ev_item)

        self.table.blockSignals(False)
        self.table.setIconSize(self.table.iconSize())
        active_count = sum(1 for it in self.items if it.is_valid)
        self.lbl_count.setText(f"Celkem snímků: {len(self.items)} (Aktivních k fúzi: {active_count})")

    def _on_cell_changed(self, row: int, col: int):
        if col == 0 and 0 <= row < len(self.items):
            item_widget = self.table.item(row, 0)
            if item_widget:
                self.items[row].is_valid = (item_widget.checkState() == Qt.CheckState.Checked)
                active_count = sum(1 for it in self.items if it.is_valid)
                self.lbl_count.setText(f"Celkem snímků: {len(self.items)} (Aktivních k fúzi: {active_count})")
                self.items_changed.emit()

    def _on_row_clicked(self, row: int, col: int):
        if 0 <= row < len(self.items):
            self.item_selected.emit(self.items[row].filepath)

    def get_active_items(self) -> List[ExposureItem]:
        return [it for it in self.items if it.is_valid]
