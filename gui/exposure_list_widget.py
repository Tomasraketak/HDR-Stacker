"""
Exposure list widget.

Shows loaded thumbnails, EXIF data and calculated EV steps, and lets the user
enable/disable frames and re-detect the EV sequence. Re-scanning preserves
per-frame user state (manual shifts, include/exclude), which is what makes the
manual alignment dialog and the auto-sort button safe to use together.
"""

from typing import List

from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QPixmap, QImage
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QFileDialog, QLabel, QDoubleSpinBox,
    QGroupBox, QAbstractItemView
)

try:
    from core.exif_and_analysis import ExposureItem, inspect_exposure_files
except ImportError:  # pragma: no cover
    from ..core.exif_and_analysis import ExposureItem, inspect_exposure_files

THUMB_SIZE = QSize(64, 44)


class ExposureListWidget(QWidget):
    """Manages the loaded exposure frames."""

    items_changed = pyqtSignal()
    item_selected = pyqtSignal(str)  # filepath of the frame to preview

    def __init__(self, parent=None):
        super().__init__(parent)
        self.items: List[ExposureItem] = []
        # Below this the filename column collapses to an ellipsis and the list
        # stops being useful for picking frames.
        self.setMinimumWidth(250)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 4, 4)
        layout.setSpacing(10)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)
        self.btn_add = QPushButton("+  Přidat fotky")
        self.btn_add.setObjectName("AddButton")
        self.btn_add.setMinimumHeight(38)
        self.btn_add.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_add.setToolTip("Vybrat expoziční sérii (Ctrl+O). Fotky lze také přetáhnout myší.")
        self.btn_add.clicked.connect(self._on_add_files)
        btn_layout.addWidget(self.btn_add, 2)

        self.btn_clear = QPushButton("Vyčistit")
        self.btn_clear.setMinimumHeight(38)
        self.btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear.clicked.connect(self.clear_all)
        btn_layout.addWidget(self.btn_clear, 1)
        layout.addLayout(btn_layout)

        wizard_group = QGroupBox("Automatické rozpoznání EV řady")
        wiz_layout = QVBoxLayout(wizard_group)
        wiz_layout.setSpacing(7)

        wiz_row = QHBoxLayout()
        lbl_step = QLabel("Krok expozice:")
        lbl_step.setObjectName("SliderLabel")
        wiz_row.addWidget(lbl_step)
        self.spin_ev_step = QDoubleSpinBox()
        self.spin_ev_step.setRange(0.1, 5.0)
        self.spin_ev_step.setSingleStep(0.33)
        self.spin_ev_step.setValue(1.0)
        self.spin_ev_step.setSuffix(" EV")
        self.spin_ev_step.setFixedWidth(92)
        self.spin_ev_step.setToolTip(
            "Rozestup mezi expozicemi. Použije se jen tehdy, když ve fotkách chybí EXIF.")
        wiz_row.addWidget(self.spin_ev_step)
        wiz_row.addStretch()
        wiz_layout.addLayout(wiz_row)

        self.btn_auto_sort = QPushButton("🔄  Seřadit a spočítat EV")
        self.btn_auto_sort.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_auto_sort.setToolTip(
            "Seřadí snímky od nejtmavšího (-EV) po nejsvětlejší (+EV)\n"
            "a namapuje expoziční časy. Ruční posuny zůstanou zachovány.")
        self.btn_auto_sort.clicked.connect(self.auto_sort_and_map_ev)
        wiz_layout.addWidget(self.btn_auto_sort)
        layout.addWidget(wizard_group)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["", "Náhled", "Soubor", "Čas", "EV"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setIconSize(THUMB_SIZE)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.cellClicked.connect(self._on_row_clicked)
        self.table.cellChanged.connect(self._on_cell_changed)
        layout.addWidget(self.table, 1)

        self.lbl_count = QLabel("Načteno snímků: 0")
        self.lbl_count.setObjectName("StatusHint")
        layout.addWidget(self.lbl_count)

    # ------------------------------------------------------------- Load/clear

    def _on_add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Vyberte expoziční sérii fotografií",
            "",
            "Obrázky (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp);;Všechny soubory (*.*)",
        )
        if files:
            self.load_files(files)

    def load_files(self, filepaths: List[str]):
        """Adds files, skipping any already in the list, then re-sorts."""
        existing = {it.filepath for it in self.items}
        new_paths = [p for p in filepaths if p not in existing]
        if not new_paths:
            return
        all_paths = [it.filepath for it in self.items] + new_paths
        self._rescan(all_paths)

    def auto_sort_and_map_ev(self):
        if not self.items:
            self.refresh_table()
            return
        self._rescan([it.filepath for it in self.items])

    def _rescan(self, paths: List[str]):
        """
        Re-inspects the given files, carrying user-set state across.

        Without the `preserve` map, re-sorting would silently reset every manual
        alignment shift and every include/exclude tick — losing real work.
        """
        preserve = {it.filepath: it for it in self.items}
        self.items = inspect_exposure_files(
            paths, user_ev_step=self.spin_ev_step.value(), preserve=preserve)
        self.refresh_table()
        self.items_changed.emit()

    def clear_all(self):
        self.items.clear()
        self.refresh_table()
        self.items_changed.emit()

    # ----------------------------------------------------------------- Table

    def refresh_table(self):
        # Signals stay blocked while rebuilding: setItem fires cellChanged, which
        # would otherwise write half-built rows back onto the model.
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(len(self.items))
            for row, item in enumerate(self.items):
                self.table.setItem(row, 0, self._checkbox_cell(item))
                self.table.setItem(row, 1, self._thumbnail_cell(item))
                self.table.setItem(row, 2, self._name_cell(item))
                self.table.setItem(row, 3, self._readonly_cell(item.shutter_str))
                self.table.setItem(row, 4, self._readonly_cell(self._ev_text(item)))
                self.table.setRowHeight(row, THUMB_SIZE.height() + 8)
        finally:
            self.table.blockSignals(False)
        self._update_count_label()

    @staticmethod
    def _ev_text(item: ExposureItem) -> str:
        return f"{item.calculated_ev:+.1f} EV" if item.calculated_ev is not None else "0.0 EV"

    @staticmethod
    def _checkbox_cell(item: ExposureItem) -> QTableWidgetItem:
        cell = QTableWidgetItem()
        cell.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        cell.setCheckState(Qt.CheckState.Checked if item.is_valid else Qt.CheckState.Unchecked)
        cell.setToolTip("Zahrnout tento snímek do skládání")
        return cell

    @staticmethod
    def _thumbnail_cell(item: ExposureItem) -> QTableWidgetItem:
        cell = QTableWidgetItem()
        if item.thumbnail is not None and item.thumbnail.size > 0:
            h, w, ch = item.thumbnail.shape
            # The QImage must be copied: it does not own the numpy buffer.
            qimg = QImage(item.thumbnail.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
            cell.setIcon(QIcon(QPixmap.fromImage(qimg).scaled(
                THUMB_SIZE, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)))
        cell.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        return cell

    @staticmethod
    def _name_cell(item: ExposureItem) -> QTableWidgetItem:
        cell = QTableWidgetItem(item.filename)
        details = [item.filepath]
        if item.width and item.height:
            details.append(f"{item.width} × {item.height} px "
                           f"({item.width * item.height / 1e6:.1f} Mpx)")
        if item.iso:
            details.append(f"ISO {item.iso}")
        if item.aperture:
            details.append(f"f/{item.aperture:.1f}")
        details.append("Expoziční čas z EXIF" if item.has_exif_time
                       else "Čas odhadnut z jasu scény")
        if abs(item.shift_x) > 0.01 or abs(item.shift_y) > 0.01:
            details.append(f"Ruční posun: ΔX {item.shift_x:+.1f} px, ΔY {item.shift_y:+.1f} px")
        cell.setToolTip("\n".join(details))
        cell.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        return cell

    @staticmethod
    def _readonly_cell(text: str) -> QTableWidgetItem:
        cell = QTableWidgetItem(text)
        cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        cell.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        return cell

    def _update_count_label(self):
        active = sum(1 for it in self.items if it.is_valid)
        if not self.items:
            self.lbl_count.setText("Načteno snímků: 0")
            return
        megapixels = max((it.width * it.height / 1e6) for it in self.items)
        suffix = f"   ·   {megapixels:.1f} Mpx" if megapixels > 0 else ""
        self.lbl_count.setText(f"Snímků: {len(self.items)}   ·   aktivních: {active}{suffix}")

    def _on_cell_changed(self, row: int, col: int):
        if col != 0 or not (0 <= row < len(self.items)):
            return
        cell = self.table.item(row, 0)
        if cell is None:
            return
        self.items[row].is_valid = (cell.checkState() == Qt.CheckState.Checked)
        self._update_count_label()
        self.items_changed.emit()

    def _on_row_clicked(self, row: int, col: int):
        if 0 <= row < len(self.items):
            self.item_selected.emit(self.items[row].filepath)

    def get_active_items(self) -> List[ExposureItem]:
        return [it for it in self.items if it.is_valid]
