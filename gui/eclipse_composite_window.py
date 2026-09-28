"""
Eclipse sequence composite editor.

A stacked totality frame is the background; partial-phase frames shot through a
solar filter are dropped in, and each Sun is placed on the Sun's real path
through the sky at its own timestamp. The user calibrates the background once
— horizon, Sun, diameter — and the program draws the Sun's daily path (and the
ecliptic, if wanted) over the photograph so the result can be checked by eye.

Grass blades or birds in the background's sky can be painted away with a
retouch brush; the Suns are drawn on top of the retouched background.

Every Sun is also equalised to the same surface brightness automatically.
Brightness, contrast, mid-tones, saturation, temperature and tint can then be
set for all Suns at once (the master grade), for each frame on top of that,
and for the totality background. Each Sun can be nudged by hand: drag it on
the canvas or use the arrow keys.

Heavy work (decoding a 24 Mpx background, finding the Sun in every partial
frame, the full-resolution export) runs on background threads. The interactive
preview renders on a bounded proxy of the background, so moving a slider only
ever re-blends a handful of small cut-outs.
"""

import copy
import math
import os
from dataclasses import astuple
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, QPointF, QRectF, QDate, QTime, pyqtSignal
from PyQt6.QtGui import (QColor, QFont, QPainter, QPainterPath, QPen, QKeyEvent, QMouseEvent,
                         QCloseEvent, QKeySequence)
from PyQt6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QDoubleSpinBox, QCheckBox, QScrollArea, QFrame,
    QSplitter, QProgressBar, QFileDialog, QMessageBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QDateEdit, QTimeEdit,
    QSizePolicy, QTabWidget, QDialogButtonBox
)

try:
    from core.eclipse_composite import (
        CompositeSettings, CompositeError, PartialFrame, SunCutout, CalibrationReport,
        ColorGrade, Placement, apply_grade, load_image_float, trace_sun_limb,
        harmonise_sun_radii, recentre_cutout, detect_totality_disc, cut_out_sun,
        calibrate, compute_placements, path_polyline, ecliptic_polyline,
        horizon_polyline, render_composite, frame_gains, format_time, parse_time,
        describe_clock_offset,
    )
    from core.exif_and_analysis import (extract_capture_time, extract_gps_position,
                                        extract_focal_length)
    from core.solar_position import sun_position
    from core.postprocess import save_image
    from core.retouch import RetouchStroke, RetouchCache, count_strokes
    from gui.image_viewer import InteractiveImageViewer, ACCENT, TEXT_DIM, RETOUCH_HINT
    from gui.controls_panel import SliderRow
    from gui.ui_utils import fit_window_to_screen, center_on_screen
except ImportError:  # pragma: no cover
    from ..core.eclipse_composite import (
        CompositeSettings, CompositeError, PartialFrame, SunCutout, CalibrationReport,
        ColorGrade, Placement, apply_grade, load_image_float, trace_sun_limb,
        harmonise_sun_radii, recentre_cutout, detect_totality_disc, cut_out_sun,
        calibrate, compute_placements, path_polyline, ecliptic_polyline,
        horizon_polyline, render_composite, frame_gains, format_time, parse_time,
        describe_clock_offset,
    )
    from ..core.exif_and_analysis import (extract_capture_time, extract_gps_position,
                                          extract_focal_length)
    from ..core.solar_position import sun_position
    from ..core.postprocess import save_image
    from ..core.retouch import RetouchStroke, RetouchCache, count_strokes
    from .image_viewer import InteractiveImageViewer, ACCENT, TEXT_DIM, RETOUCH_HINT
    from .controls_panel import SliderRow
    from .ui_utils import fit_window_to_screen, center_on_screen

IMAGE_FILTER = "Obrázky (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp);;Všechny soubory (*.*)"

# The interactive preview renders on a background no larger than this.
PREVIEW_MAX_DIM = 2400

# How long input must settle before the preview is re-rendered.
RENDER_DEBOUNCE_MS = 40

# A few places on the 12 August 2026 totality path, plus home.
LOCATION_PRESETS = [
    ("Vlastní poloha", None),
    ("León (ES)", (42.5987, -5.5671)),
    ("Burgos (ES)", (42.3439, -3.6969)),
    ("Valladolid (ES)", (41.6523, -4.7245)),
    ("Oviedo (ES)", (43.3614, -5.8494)),
    ("Zaragoza (ES)", (41.6488, -0.8891)),
    ("Teruel (ES)", (40.3456, -1.1065)),
    ("Valencia (ES)", (39.4699, -0.3763)),
    ("Palma de Mallorca (ES)", (39.5696, 2.6502)),
    ("Reykjavík (IS)", (64.1466, -21.9426)),
    ("Praha (CZ)", (50.0755, 14.4378)),
    ("Brno (CZ)", (49.1951, 16.6068)),
]

# Overlay palette.
PATH_COLOUR = QColor(251, 191, 36, 215)
TICK_COLOUR = QColor(253, 230, 138, 230)
ECLIPTIC_COLOUR = QColor(236, 72, 153, 200)
HORIZON_COLOUR = QColor(56, 189, 248, 230)
MODEL_HORIZON_COLOUR = QColor(74, 222, 128, 200)
SUN_MARK_COLOUR = QColor(250, 204, 21, 240)
MARKER_COLOUR = QColor(226, 232, 240, 210)


# ------------------------------------------------------------------ Workers

class _Task(QThread):
    """Runs one callable off the GUI thread; the callable receives this task."""

    progress = pyqtSignal(int, str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn: Callable[["_Task"], Any], parent=None):
        super().__init__(parent)
        self._fn = fn

    def cancelled(self) -> bool:
        return self.isInterruptionRequested()

    def report(self, pct: int, msg: str):
        if not self.cancelled():
            self.progress.emit(int(np.clip(pct, 0, 100)), msg)

    def run(self):
        try:
            result = self._fn(self)
        except MemoryError:
            if not self.cancelled():
                self.failed.emit("Došla operační paměť. Zavřete ostatní programy a zkuste to znovu.")
        except CompositeError as e:
            if not self.cancelled():
                self.failed.emit(str(e))
        except Exception as e:  # a background error must never take the app down
            if not self.cancelled():
                self.failed.emit(f"{type(e).__name__}: {e}")
        else:
            if not self.cancelled():
                self.done.emit(result)


def _load_background(path: str, task: _Task) -> Dict[str, Any]:
    task.report(10, f"Načítám pozadí {os.path.basename(path)}…")
    full = load_image_float(path)
    if full is None:
        raise CompositeError(f"Soubor nelze načíst: {path}")
    h, w = full.shape[:2]
    scale = min(1.0, PREVIEW_MAX_DIM / float(max(w, h)))
    proxy = (cv2.resize(full, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                        interpolation=cv2.INTER_AREA) if scale < 1.0 else full)
    task.report(60, "Hledám disk Měsíce v koróně…")
    disc = detect_totality_disc(full)
    moment, offset = extract_capture_time(path)
    return dict(path=path, proxy=np.ascontiguousarray(proxy), scale=proxy.shape[1] / float(w),
                width=w, height=h, disc=disc, moment=moment, offset=offset,
                gps=extract_gps_position(path))


def _load_partials(paths: List[str], known_discs: Dict[str, Tuple[float, float, float]],
                   task: _Task) -> List[Dict[str, Any]]:
    """
    Decodes each filtered frame once, finds its Sun and keeps only the cut-out.

    The newly traced Suns are then given one common radius per lens (see
    harmonise_sun_radii), which keeps thin crescents the right size.
    """
    results = []
    traced: List[Tuple[int, Any, Optional[float]]] = []    # (result index, LimbFit, focal)
    for i, path in enumerate(paths):
        if task.cancelled():
            break
        task.report(int(100 * i / max(1, len(paths))),
                    f"Hledám Slunce ve snímku {i + 1}/{len(paths)}: {os.path.basename(path)}")
        entry: Dict[str, Any] = dict(path=path, cutout=None, disc=None, error="")
        moment, offset = extract_capture_time(path)
        entry.update(moment=moment, offset=offset, gps=extract_gps_position(path))
        if not os.path.isfile(path):
            entry["error"] = "soubor chybí"
            results.append(entry)
            continue
        img = load_image_float(path)
        if img is None:
            entry["error"] = "nelze načíst"
            results.append(entry)
            continue
        disc = known_discs.get(path)
        if disc is None:
            fit = trace_sun_limb(img)
            if fit is not None:
                disc = fit.circle
                traced.append((len(results), fit, extract_focal_length(path)))
        if disc is None:
            entry["error"] = "Slunce nenalezeno"
        else:
            entry["disc"] = tuple(float(v) for v in disc)
            entry["cutout"] = cut_out_sun(img, disc)
        results.append(entry)

    if len(traced) > 1 and not task.cancelled():
        circles = harmonise_sun_radii([fit for _i, fit, _f in traced],
                                      groups=[focal for _i, _fit, focal in traced])
        for (i, fit, _focal), circle in zip(traced, circles):
            if circle is not None and circle != fit.circle:
                entry = results[i]
                entry["cutout"] = recentre_cutout(entry["cutout"], fit.circle, circle)
                entry["disc"] = tuple(float(v) for v in circle)
    return results


# ------------------------------------------------------------------- Canvas

class CompositeCanvas(InteractiveImageViewer):
    """The image viewer plus calibration tools and sky overlays."""

    horizon_drawn = pyqtSignal(float, float, float, float)
    sun_marked = pyqtSignal(float, float, float)      # x, y, diameter (0 = keep)
    frame_clicked = pyqtSignal(int)
    frame_moved = pyqtSignal(int, float, float)       # index, new centre x, y

    TOOL_NONE, TOOL_HORIZON, TOOL_SUN = range(3)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tool = self.TOOL_NONE
        self._drag_start: Optional[Tuple[float, float]] = None
        self._drag_now: Optional[Tuple[float, float]] = None
        self._drag_frame: Optional[int] = None
        self._drag_grab = (0.0, 0.0)
        self._overlay: Dict[str, Any] = {}

    # ------------------------------------------------------------ Public API

    def set_tool(self, tool: int):
        self._tool = tool
        self._drag_start = self._drag_now = None
        self.setCursor(Qt.CursorShape.CrossCursor if tool != self.TOOL_NONE
                       else Qt.CursorShape.ArrowCursor)
        self.update()

    def tool(self) -> int:
        return self._tool

    def set_overlay(self, overlay: Dict[str, Any]):
        self._overlay = overlay or {}
        self.update()

    def zoom_to(self, x: float, y: float, zoom: float):
        self._zoom = float(np.clip(zoom, self.MIN_ZOOM, self.MAX_ZOOM))
        self._pan_pos = QPointF(self.width() / 2.0 - (x + 0.5) * self._zoom,
                                self.height() / 2.0 - (y + 0.5) * self._zoom)
        self._needs_fit = False
        self.update()

    # ---------------------------------------------------------- Coordinates

    def _to_screen(self, x: float, y: float) -> QPointF:
        """Scene pixel centres (OpenCV convention) to widget coordinates."""
        return QPointF(self._pan_pos.x() + (x + 0.5) * self._zoom,
                       self._pan_pos.y() + (y + 0.5) * self._zoom)

    def _to_scene(self, pos: QPointF) -> Tuple[float, float]:
        z = max(self._zoom, 1e-6)
        return ((pos.x() - self._pan_pos.x()) / z - 0.5,
                (pos.y() - self._pan_pos.y()) / z - 0.5)

    def _hit_marker(self, pos: QPointF) -> Optional[int]:
        best, best_d = None, float("inf")
        for marker in self._overlay.get("markers", []):
            idx, x, y, r = marker["index"], marker["x"], marker["y"], marker["r"]
            centre = self._to_screen(x, y)
            d = math.hypot(pos.x() - centre.x(), pos.y() - centre.y())
            if d <= max(r * self._zoom, 9.0) + 3.0 and d < best_d:
                best, best_d = idx, d
        return best

    # ---------------------------------------------------------------- Mouse

    def mousePressEvent(self, event: QMouseEvent):
        if self._retouch_enabled:
            # The brush owns the left button; markers stay put while retouching.
            super().mousePressEvent(event)
            return
        if event.button() == Qt.MouseButton.LeftButton and self._base_pixmap is not None:
            scene = self._to_scene(event.position())
            if self._tool in (self.TOOL_HORIZON, self.TOOL_SUN):
                self._drag_start = self._drag_now = scene
                self.update()
                return
            hit = self._hit_marker(event.position()) if self._overlay.get("show_markers") else None
            if hit is not None:
                marker = next(m for m in self._overlay["markers"] if m["index"] == hit)
                self._drag_frame = hit
                self._drag_grab = (scene[0] - marker["x"], scene[1] - marker["y"])
                self.frame_clicked.emit(hit)
                self.setCursor(Qt.CursorShape.SizeAllCursor)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_start is not None:
            self._drag_now = self._to_scene(event.position())
            self.update()
            self._emit_hover(event.position())
            return
        if self._drag_frame is not None:
            x, y = self._to_scene(event.position())
            self.frame_moved.emit(self._drag_frame, x - self._drag_grab[0], y - self._drag_grab[1])
            self._emit_hover(event.position())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._drag_start is not None:
            (x0, y0), (x1, y1) = self._drag_start, self._drag_now or self._drag_start
            self._drag_start = self._drag_now = None
            if self._tool == self.TOOL_HORIZON:
                if math.hypot(x1 - x0, y1 - y0) >= 10.0:
                    self.horizon_drawn.emit(x0, y0, x1, y1)
            elif self._tool == self.TOOL_SUN:
                radius = math.hypot(x1 - x0, y1 - y0)
                # A plain click only moves the centre and keeps the diameter.
                self.sun_marked.emit(x0, y0, 2.0 * radius if radius >= 1.0 else 0.0)
            self.update()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._drag_frame is not None:
            self._drag_frame = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        super().mouseReleaseEvent(event)

    # -------------------------------------------------------------- Painting

    def _paint_status_pill(self, painter: QPainter):
        """Overlays are painted here: last in the base paint pass, on the same painter."""
        self._paint_overlays(painter)
        hints = {
            self.TOOL_HORIZON: "Horizont: táhněte myší podél obzoru zleva doprava",
            self.TOOL_SUN: "Slunce: klikněte do středu disku a táhněte k jeho okraji",
        }
        text = hints.get(self._tool,
                         "Tažením Slunce ho posunete · šipky = jemný posun · kolečko = zoom")
        if self._retouch_enabled:
            text = RETOUCH_HINT + " · Ctrl+Z zpět"
        text = f"Zoom {int(self._zoom * 100)} %   ·   {text}"
        painter.setFont(QFont("Segoe UI", 9))
        metrics = painter.fontMetrics()
        rect = QRectF(12, self.height() - metrics.height() - 18,
                      metrics.horizontalAdvance(text) + 22, metrics.height() + 10)
        path = QPainterPath()
        path.addRoundedRect(rect, rect.height() / 2.0, rect.height() / 2.0)
        painter.fillPath(path, QColor(8, 10, 16, 200))
        painter.setPen(ACCENT if self._tool != self.TOOL_NONE or self._retouch_enabled else TEXT_DIM)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _polyline(self, painter: QPainter, points, pen: QPen):
        path = QPainterPath()
        pen_down = False
        for x, y in points:
            if not (np.isfinite(x) and np.isfinite(y)):
                pen_down = False
                continue
            p = self._to_screen(x, y)
            if pen_down:
                path.lineTo(p)
            else:
                path.moveTo(p)
                pen_down = True
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

    def _label(self, painter: QPainter, anchor: QPointF, text: str, colour: QColor,
               size: int = 8, dx: float = 8.0, dy: float = -8.0):
        painter.setFont(QFont("Segoe UI", size, QFont.Weight.DemiBold))
        metrics = painter.fontMetrics()
        rect = QRectF(anchor.x() + dx, anchor.y() + dy - metrics.height(),
                      metrics.horizontalAdvance(text) + 10, metrics.height() + 4)
        path = QPainterPath()
        path.addRoundedRect(rect, 4, 4)
        painter.fillPath(path, QColor(8, 10, 16, 190))
        painter.setPen(colour)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_overlays(self, painter: QPainter):
        ov = self._overlay
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if ov.get("show_ecliptic") and ov.get("ecliptic"):
            pen = QPen(ECLIPTIC_COLOUR, 1.4, Qt.PenStyle.DashLine)
            self._polyline(painter, ov["ecliptic"], pen)
            pts = [p for p in ov["ecliptic"] if np.isfinite(p[0])]
            if pts:
                visible = [p for p in pts if 0 <= p[0] < self._orig_w and 0 <= p[1] < self._orig_h]
                if visible:
                    self._label(painter, self._to_screen(*visible[len(visible) // 5]),
                                "ekliptika", ECLIPTIC_COLOUR)

        if ov.get("show_path") and ov.get("path"):
            self._polyline(painter, [(x, y) for _t, x, y in ov["path"]], QPen(PATH_COLOUR, 1.6))
            if ov.get("show_ticks"):
                for label, x, y in ov.get("ticks", []):
                    p = self._to_screen(x, y)
                    painter.setPen(QPen(TICK_COLOUR, 1.5))
                    painter.setBrush(TICK_COLOUR)
                    painter.drawEllipse(p, 2.6, 2.6)
                    self._label(painter, p, label, TICK_COLOUR, size=8)

        if ov.get("show_calibration"):
            if ov.get("model_horizon"):
                self._polyline(painter, ov["model_horizon"], QPen(MODEL_HORIZON_COLOUR, 1.0))
            horizon = ov.get("horizon")
            if horizon:
                x1, y1, x2, y2 = horizon
                dx, dy = x2 - x1, y2 - y1
                n = math.hypot(dx, dy) or 1.0
                ext = 4.0 * max(self._orig_w, self._orig_h)
                a = (x1 - dx / n * ext, y1 - dy / n * ext)
                b = (x2 + dx / n * ext, y2 + dy / n * ext)
                self._polyline(painter, [a, b], QPen(HORIZON_COLOUR, 1.2, Qt.PenStyle.DashLine))
                for hx, hy in ((x1, y1), (x2, y2)):
                    p = self._to_screen(hx, hy)
                    painter.setPen(QPen(HORIZON_COLOUR, 2))
                    painter.setBrush(QColor(8, 10, 16, 200))
                    painter.drawRect(QRectF(p.x() - 4, p.y() - 4, 8, 8))
                self._label(painter, self._to_screen(x1, y1), "horizont", HORIZON_COLOUR, dy=-10)
            sun = ov.get("sun")
            if sun:
                sx, sy, d = sun
                p = self._to_screen(sx, sy)
                r = max(4.0, d / 2.0 * self._zoom)
                painter.setPen(QPen(SUN_MARK_COLOUR, 1.4))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(p, r, r)
                arm = r + 8
                painter.drawLine(QPointF(p.x() - arm, p.y()), QPointF(p.x() - r - 2, p.y()))
                painter.drawLine(QPointF(p.x() + r + 2, p.y()), QPointF(p.x() + arm, p.y()))
                painter.drawLine(QPointF(p.x(), p.y() - arm), QPointF(p.x(), p.y() - r - 2))
                painter.drawLine(QPointF(p.x(), p.y() + r + 2), QPointF(p.x(), p.y() + arm))
                if ov.get("sun_label"):
                    self._label(painter, p, ov["sun_label"], SUN_MARK_COLOUR, dx=arm, dy=-arm)

        if ov.get("show_markers"):
            for marker in ov.get("markers", []):
                p = self._to_screen(marker["x"], marker["y"])
                r = max(6.0, marker["r"] * self._zoom + 3.0)
                selected = marker.get("selected", False)
                painter.setPen(QPen(ACCENT if selected else MARKER_COLOUR, 2.0 if selected else 1.0,
                                    Qt.PenStyle.SolidLine if selected else Qt.PenStyle.DotLine))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(p, r, r)
                self._label(painter, p, marker["label"], ACCENT if selected else MARKER_COLOUR,
                            dx=r * 0.7 + 4, dy=-r * 0.7)

        # Tool feedback while dragging.
        if self._drag_start is not None and self._drag_now is not None:
            a, b = self._to_screen(*self._drag_start), self._to_screen(*self._drag_now)
            if self._tool == self.TOOL_HORIZON:
                painter.setPen(QPen(HORIZON_COLOUR, 2))
                painter.drawLine(a, b)
            elif self._tool == self.TOOL_SUN:
                r = math.hypot(b.x() - a.x(), b.y() - a.y())
                painter.setPen(QPen(SUN_MARK_COLOUR, 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(a, r, r)
                d = 2.0 * r / max(self._zoom, 1e-6)
                self._label(painter, b, f"Ø {d:.1f} px", SUN_MARK_COLOUR)
        painter.restore()


# -------------------------------------------------------------- Small widgets

class DateTimeRow(QWidget):
    """Date + time editors without any time-zone conversion behind the scenes."""

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.date_edit = QDateEdit()
        self.date_edit.setDisplayFormat("dd.MM.yyyy")
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate(2026, 8, 12))
        self.time_edit = QTimeEdit()
        self.time_edit.setDisplayFormat("HH:mm:ss")
        layout.addWidget(self.date_edit, 3)
        layout.addWidget(self.time_edit, 2)
        self.date_edit.dateChanged.connect(lambda *_: self.changed.emit())
        self.time_edit.timeChanged.connect(lambda *_: self.changed.emit())

    def set_value(self, moment: Optional[datetime]):
        if moment is None:
            return
        for w in (self.date_edit, self.time_edit):
            w.blockSignals(True)
        self.date_edit.setDate(QDate(moment.year, moment.month, moment.day))
        self.time_edit.setTime(QTime(moment.hour, moment.minute, moment.second))
        for w in (self.date_edit, self.time_edit):
            w.blockSignals(False)

    def value(self) -> datetime:
        d = self.date_edit.date()
        t = self.time_edit.time()
        return datetime(d.year(), d.month(), d.day(), t.hour(), t.minute(), t.second())


def _zone_text(hours: float) -> str:
    return f"UTC{'+' if hours >= 0 else '−'}{abs(hours):g}"


class ClockSyncDialog(QDialog):
    """Works out the camera clock correction from one photo whose true time is known."""

    def __init__(self, settings: CompositeSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Seřídit hodiny fotoaparátu")
        self.settings = settings
        layout = QVBoxLayout(self)
        intro = QLabel("Vyberte snímek, u kterého znáte přesný čas — třeba začátek úplné fáze "
                       "(2. kontakt) z tabulky místních okolností zatmění — a zadejte ho. "
                       "Rozdíl proti hodinám fotoaparátu se pak započítá u všech snímků.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        grid = QGridLayout()
        grid.addWidget(QLabel("Snímek:"), 0, 0)
        self.combo = QComboBox()
        background = parse_time(settings.background_time)
        if background is not None:
            self.combo.addItem(f"Pozadí (úplná fáze) — {background:%H:%M:%S}", -1)
        for i, frame in enumerate(settings.frames):
            if frame.moment is not None:
                self.combo.addItem(f"#{i + 1} {frame.filename} — {frame.moment:%H:%M:%S}", i)
        grid.addWidget(self.combo, 0, 1)
        grid.addWidget(QLabel("Podle fotoaparátu:"), 1, 0)
        self.lbl_camera = QLabel("")
        grid.addWidget(self.lbl_camera, 1, 1)
        grid.addWidget(QLabel("Skutečný čas:"), 2, 0)
        self.true_time = DateTimeRow()
        grid.addWidget(self.true_time, 2, 1)
        zone = QLabel(f"Zadejte ho v pásmu {_zone_text(settings.utc_offset_hours)}, stejném jako "
                      f"čas fotoaparátu (nastavení Časové pásmo).")
        zone.setObjectName("StatusHint")
        zone.setWordWrap(True)
        grid.addWidget(zone, 3, 0, 1, 2)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        self.lbl_result = QLabel("")
        self.lbl_result.setWordWrap(True)
        layout.addWidget(self.lbl_result)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                        | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.combo.currentIndexChanged.connect(self._on_reference)
        self.true_time.changed.connect(self._update_result)
        self._on_reference()
        self.setMinimumWidth(460)

    def _reference(self) -> Tuple[Optional[PartialFrame], Optional[datetime]]:
        """(frame or None for the background, its camera time)."""
        index = self.combo.currentData()
        if index is None:
            return None, None
        if index < 0:
            return None, parse_time(self.settings.background_time)
        frame = self.settings.frames[index]
        return frame, frame.moment

    def _on_reference(self, *_):
        frame, camera_time = self._reference()
        self.lbl_camera.setText(camera_time.strftime("%d.%m.%Y %H:%M:%S") if camera_time else "—")
        # Start from the time the program currently believes.
        believed = (self.settings.background_moment if frame is None
                    else self.settings.frame_moment(frame))
        self.true_time.set_value(believed)
        self._update_result()

    def offset(self) -> Optional[float]:
        frame, camera_time = self._reference()
        if camera_time is None:
            return None
        return self.settings.clock_offset_for(self.true_time.value(), frame)

    def _update_result(self, *_):
        value = self.offset()
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok.setEnabled(value is not None)
        if value is None:
            self.lbl_result.setText("Žádný snímek nemá čas pořízení.")
        else:
            self.lbl_result.setText(f"Korekce hodin: <b>{value:+.1f} s</b> — "
                                    f"{describe_clock_offset(value)}.")


class GradeEditor(QWidget):
    """Brightness, contrast, mid-tones, saturation and colour balance sliders."""

    changed = pyqtSignal()

    # attribute, label, lowest, highest, step, decimals, suffix, tooltip
    SLIDERS = (
        ("exposure", "Jas", -3.0, 3.0, 0.05, 2, " EV", "Jas v expozičních stupních (+1 EV = 2× jasnější)."),
        ("contrast", "Kontrast", -100.0, 100.0, 1.0, 0, "",
         "U Slunce se mění kolem jasu jeho povrchu: + prohloubí okrajové\n"
         "ztemnění a skvrny, − obraz zploští."),
        ("midtones", "Střední tóny", -100.0, 100.0, 1.0, 0, "",
         "Zesvětlí nebo ztmaví střední tóny; černá i bílá zůstanou."),
        ("saturation", "Saturace", -100.0, 100.0, 1.0, 0, "", "−100 = černobíle."),
        ("temperature", "Teplota", -100.0, 100.0, 1.0, 0, "", "− chladnější (modřejší), + teplejší (žlutější)."),
        ("tint", "Odstín", -100.0, 100.0, 1.0, 0, "", "− do zelena, + do purpurova."),
    )

    def __init__(self, hint: str = "", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(2)
        self.lbl_hint = QLabel(hint)
        self.lbl_hint.setObjectName("StatusHint")
        self.lbl_hint.setWordWrap(True)
        self.lbl_hint.setVisible(bool(hint))
        layout.addWidget(self.lbl_hint)
        self.rows: Dict[str, SliderRow] = {}
        for name, label, lo, hi, step, decimals, suffix, tip in self.SLIDERS:
            row = SliderRow(label, lo, hi, 0.0, step=step, suffix=suffix,
                            tip=tip + "\nDvojklik = 0.", decimals=decimals)
            row.valueChanged.connect(lambda *_: self.changed.emit())
            layout.addWidget(row)
            self.rows[name] = row
        self.btn_reset = QPushButton("↺  Vynulovat úpravy")
        self.btn_reset.clicked.connect(self.reset)
        layout.addWidget(self.btn_reset)

    def set_grade(self, grade: ColorGrade):
        """Shows `grade` without emitting `changed`."""
        for name, row in self.rows.items():
            _set_quiet(row, getattr(grade, name))

    def write_into(self, grade: ColorGrade):
        for name, row in self.rows.items():
            setattr(grade, name, row.value())

    def reset(self):
        self.set_grade(ColorGrade())
        self.changed.emit()


def _spin(lo: float, hi: float, step: float, decimals: int, suffix: str = "",
          tip: str = "") -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setSingleStep(step)
    spin.setDecimals(decimals)
    if suffix:
        spin.setSuffix(suffix)
    if tip:
        spin.setToolTip(tip)
    spin.setKeyboardTracking(False)
    return spin


def _set_quiet(widget, value):
    """Sets a spin box / check box / slider row without firing its signals."""
    widget.blockSignals(True)
    try:
        if isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        else:
            widget.setValue(value)
    finally:
        widget.blockSignals(False)


# ------------------------------------------------------------------- Window

class EclipseCompositeWindow(QDialog):
    """Editor for the partial-phase sequence composite."""

    def __init__(self, settings: CompositeSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🌗 Časosběrný kompozit zatmění — dráha Slunce nad krajinou")
        self.setSizeGripEnabled(True)
        self.setModal(False)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.settings = settings
        self._bg_proxy: Optional[np.ndarray] = None
        self._bg_graded: Optional[np.ndarray] = None    # _bg_proxy retouched and graded
        self._bg_graded_key: Optional[tuple] = None
        self._retouch_cache = RetouchCache()
        self._bg_scale = 1.0
        self._bg_size = (0, 0)
        self._cutouts: Dict[str, Optional[SunCutout]] = {}
        self._frame_errors: Dict[str, str] = {}
        self._report: Optional[CalibrationReport] = None
        self._calib_error = ""
        self._placements: List[Optional[Placement]] = []
        self._selected = 0 if settings.frames else -1
        self._tasks: List[_Task] = []
        self._export_task: Optional[_Task] = None
        self._first_image = True
        self._syncing = False

        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(RENDER_DEBOUNCE_MS)
        self._render_timer.timeout.connect(self._refresh)

        self._init_ui()
        self._load_settings_into_ui()
        fit_window_to_screen(self, 1500, 940)
        center_on_screen(self)

    # ================================================================ Build

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        # Squeeze rather than scroll sideways: a hidden right edge would hide
        # buttons such as "Odebrat" with no hint that they exist.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(2, 2, 8, 2)
        layout.setSpacing(12)
        layout.addWidget(self._build_background_group())
        layout.addWidget(self._build_calibration_group())
        layout.addWidget(self._build_frames_group())
        layout.addWidget(self._build_selected_group())
        layout.addWidget(self._build_look_group())
        layout.addWidget(self._build_grade_group())
        layout.addWidget(self._build_retouch_group())
        layout.addWidget(self._build_overlay_group())
        layout.addStretch()
        # Long option texts must not dictate the panel width; the popup list
        # still shows them in full.
        for box in panel.findChildren(QComboBox):
            box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            box.setMinimumContentsLength(10)
        scroll.setWidget(panel)
        # Never narrower than the content: with sideways scrolling off, a
        # narrower panel would clip the value read-outs at its right edge.
        scroll.setMinimumWidth(max(420, panel.minimumSizeHint().width()
                                   + scroll.verticalScrollBar().sizeHint().width() + 6))
        splitter.addWidget(scroll)
        splitter.addWidget(self._build_viewer())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        splitter.setSizes([480, 1000])
        root.addWidget(splitter, 1)

        bar = QFrame()
        bar.setObjectName("StatusStrip")
        bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        bottom = QHBoxLayout(bar)
        bottom.setContentsMargins(10, 6, 10, 6)
        self.lbl_status = QLabel("Načtěte snímek úplného zatmění a pak fotky částečných fází.")
        self.lbl_status.setObjectName("StatusLabel")
        bottom.addWidget(self.lbl_status, 1)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(220)
        self.progress.setVisible(False)
        bottom.addWidget(self.progress)
        self.btn_close = QPushButton("Zavřít")
        self.btn_close.clicked.connect(self.close)
        bottom.addWidget(self.btn_close)
        self.btn_export = QPushButton("💾  Exportovat kompozit…")
        self.btn_export.setObjectName("ExportButton")
        self.btn_export.setMinimumHeight(38)
        self.btn_export.clicked.connect(self.export_composite)
        bottom.addWidget(self.btn_export)
        root.addWidget(bar, 0)

    def _build_background_group(self) -> QGroupBox:
        group = QGroupBox("1 · Pozadí — snímek úplného zatmění")
        grid = QGridLayout(group)
        grid.setVerticalSpacing(6)

        self.btn_bg = QPushButton("📂  Načíst fotku úplného zatmění (stack)…")
        self.btn_bg.setObjectName("AddButton")
        self.btn_bg.clicked.connect(self._choose_background)
        grid.addWidget(self.btn_bg, 0, 0, 1, 2)
        self.lbl_bg = QLabel("Není načteno.")
        self.lbl_bg.setObjectName("StatusHint")
        self.lbl_bg.setWordWrap(True)
        grid.addWidget(self.lbl_bg, 1, 0, 1, 2)

        grid.addWidget(QLabel("Čas snímku:"), 2, 0)
        self.bg_time = DateTimeRow()
        self.bg_time.setToolTip("Čas pořízení snímku úplné fáze podle hodin fotoaparátu\n"
                                "(čte se z EXIF). U stacku z tohoto programu EXIF chybí —\n"
                                "převezměte čas z některé původní expozice tlačítkem níže.")
        self.bg_time.changed.connect(self._on_bg_time_changed)
        grid.addWidget(self.bg_time, 2, 1)

        self.btn_bg_time = QPushButton("🕑  Převzít čas z EXIF jiné fotky…")
        self.btn_bg_time.setToolTip("Například z jedné z původních expozic úplné fáze.")
        self.btn_bg_time.clicked.connect(self._take_time_from_file)
        grid.addWidget(self.btn_bg_time, 3, 0, 1, 2)

        grid.addWidget(QLabel("Časové pásmo:"), 4, 0)
        self.spin_utc = _spin(-12.0, 14.0, 0.5, 2, " h",
                              "Posun místního času fotoaparátu od UTC.\n"
                              "Letní čas ve Španělsku i v Česku = +2 h.")
        self.spin_utc.setPrefix("UTC ")
        self.spin_utc.valueChanged.connect(self._on_location_changed)
        grid.addWidget(self.spin_utc, 4, 1)

        grid.addWidget(QLabel("Korekce hodin:"), 5, 0)
        clock = QHBoxLayout()
        self.spin_camera_clock = _spin(-86400.0, 86400.0, 1.0, 1, " s",
                                       "O kolik opravit čas všech snímků, když hodiny fotoaparátu\n"
                                       "šly napřed (−) nebo pozadu (+). Platí pro pozadí i srpky.")
        self.spin_camera_clock.valueChanged.connect(self._on_camera_clock_changed)
        clock.addWidget(self.spin_camera_clock, 1)
        self.btn_clock_sync = QPushButton("⏱  Seřídit…")
        self.btn_clock_sync.setToolTip("Spočítá korekci ze snímku, jehož přesný čas znáte\n"
                                       "(např. začátek úplné fáze).")
        self.btn_clock_sync.clicked.connect(self._open_clock_sync)
        clock.addWidget(self.btn_clock_sync)
        grid.addLayout(clock, 5, 1)
        self.lbl_clock = QLabel("")
        self.lbl_clock.setObjectName("StatusHint")
        self.lbl_clock.setWordWrap(True)
        grid.addWidget(self.lbl_clock, 6, 0, 1, 2)

        grid.addWidget(QLabel("Místo:"), 7, 0)
        self.combo_place = QComboBox()
        for name, coords in LOCATION_PRESETS:
            self.combo_place.addItem(name, coords)
        self.combo_place.currentIndexChanged.connect(self._on_place_preset)
        grid.addWidget(self.combo_place, 7, 1)

        coords = QHBoxLayout()
        self.spin_lat = _spin(-90.0, 90.0, 0.01, 5, "°", "Zeměpisná šířka (+ sever)")
        self.spin_lon = _spin(-180.0, 180.0, 0.01, 5, "°", "Zeměpisná délka (+ východ, − západ)")
        self.spin_lat.valueChanged.connect(self._on_coords_edited)
        self.spin_lon.valueChanged.connect(self._on_coords_edited)
        coords.addWidget(QLabel("š."))
        coords.addWidget(self.spin_lat, 1)
        coords.addWidget(QLabel("d."))
        coords.addWidget(self.spin_lon, 1)
        grid.addWidget(QLabel("Souřadnice:"), 8, 0)
        grid.addLayout(coords, 8, 1)

        self.lbl_sunpos = QLabel("")
        self.lbl_sunpos.setObjectName("StatusHint")
        self.lbl_sunpos.setWordWrap(True)
        grid.addWidget(self.lbl_sunpos, 9, 0, 1, 2)
        grid.setColumnStretch(1, 1)
        return group

    def _build_calibration_group(self) -> QGroupBox:
        group = QGroupBox("2 · Kalibrace oblohy na pozadí")
        grid = QGridLayout(group)
        grid.setVerticalSpacing(6)

        self.btn_tool_horizon = QPushButton("〰  Vyznačit horizont")
        self.btn_tool_horizon.setCheckable(True)
        self.btn_tool_horizon.setToolTip("Táhněte myší podél vzdáleného obzoru (co nejdelší úsečka).")
        self.btn_tool_horizon.toggled.connect(lambda on: self._set_tool(CompositeCanvas.TOOL_HORIZON, on))
        tools = QHBoxLayout()
        tools.addWidget(self.btn_tool_horizon, 1)

        self.btn_tool_sun = QPushButton("☀  Vyznačit Slunce")
        self.btn_tool_sun.setCheckable(True)
        self.btn_tool_sun.setToolTip("Klikněte do středu disku a táhněte k jeho okraji.\n"
                                     "Pouhé kliknutí jen přesune střed.")
        self.btn_tool_sun.toggled.connect(lambda on: self._set_tool(CompositeCanvas.TOOL_SUN, on))
        tools.addWidget(self.btn_tool_sun, 1)
        # Its own row: sharing the grid's columns with the X/Y pair below would
        # add the two buttons' widths to the spin boxes' and widen the panel.
        grid.addLayout(tools, 0, 0, 1, 2)

        self.btn_find_disc = QPushButton("🔍  Najít disk Měsíce automaticky")
        self.btn_find_disc.clicked.connect(self._auto_find_disc)
        grid.addWidget(self.btn_find_disc, 1, 0, 1, 2)

        grid.addWidget(QLabel("Slunce X / Y:"), 2, 0)
        row = QHBoxLayout()
        self.spin_sun_x = _spin(0.0, 100000.0, 0.5, 1, " px")
        self.spin_sun_y = _spin(0.0, 100000.0, 0.5, 1, " px")
        row.addWidget(self.spin_sun_x, 1)
        row.addWidget(self.spin_sun_y, 1)
        grid.addLayout(row, 2, 1)

        grid.addWidget(QLabel("Průměr Slunce:"), 3, 0)
        self.spin_sun_d = _spin(0.0, 20000.0, 0.2, 1, " px",
                                "Průměr slunečního (měsíčního) disku v pixelech.\n"
                                "Určuje měřítko — kolik pixelů odpovídá jednomu stupni.")
        grid.addWidget(self.spin_sun_d, 3, 1)
        for spin in (self.spin_sun_x, self.spin_sun_y, self.spin_sun_d):
            spin.valueChanged.connect(self._on_calibration_spins)

        grid.addWidget(QLabel("Výška horizontu:"), 4, 0)
        self.spin_hor_alt = _spin(-5.0, 20.0, 0.05, 2, "°",
                                  "Skutečná výška vyznačeného obzoru nad matematickým horizontem.\n"
                                  "Z vyvýšeného místa je obzor níž: 100 m ≈ −0,3°, 400 m ≈ −0,6°.\n"
                                  "Vzdálené hory naopak obzor zvedají.")
        self.spin_hor_alt.valueChanged.connect(self._on_calibration_spins)
        grid.addWidget(self.spin_hor_alt, 4, 1)

        grid.addWidget(QLabel("Měřítko z:"), 5, 0)
        self.combo_scale = QComboBox()
        for text, key in (("Automaticky (horizont + průměr)", "auto"),
                          ("Jen průměr Slunce", "diameter"),
                          ("Jen výška Slunce nad horizontem", "horizon")):
            self.combo_scale.addItem(text, key)
        self.combo_scale.currentIndexChanged.connect(self._on_calibration_spins)
        grid.addWidget(self.combo_scale, 5, 1)

        self.lbl_calib = QLabel("")
        self.lbl_calib.setWordWrap(True)
        self.lbl_calib.setTextFormat(Qt.TextFormat.RichText)
        grid.addWidget(self.lbl_calib, 6, 0, 1, 2)
        grid.setColumnStretch(1, 1)
        return group

    def _build_frames_group(self) -> QGroupBox:
        group = QGroupBox("3 · Částečné fáze (fotky přes filtr)")
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        self.btn_add = QPushButton("+  Přidat snímky s filtrem…")
        self.btn_add.setObjectName("AddButton")
        self.btn_add.clicked.connect(self._choose_partials)
        row.addWidget(self.btn_add, 1)
        self.btn_remove = QPushButton("Odebrat")
        self.btn_remove.clicked.connect(self._remove_selected)
        row.addWidget(self.btn_remove)
        layout.addLayout(row)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["", "Čas", "Soubor", "Jas", "Stav"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setMinimumHeight(170)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        self.table.itemChanged.connect(self._on_table_item_changed)
        layout.addWidget(self.table)

        grid = QGridLayout()
        grid.addWidget(QLabel("Srpky navíc:"), 0, 0)
        self.spin_clock = _spin(-7200.0, 7200.0, 1.0, 1, " s",
                                "Jen když srpky fotil jiný přístroj než pozadí: o kolik\n"
                                "se jeho hodiny liší navíc. Korekce hodin fotoaparátu\n"
                                "(v části 1) se přičítá ke všem snímkům.")
        self.spin_clock.valueChanged.connect(self._on_clock_changed)
        grid.addWidget(self.spin_clock, 0, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        return group

    def _build_selected_group(self) -> QGroupBox:
        group = QGroupBox("Vybraný snímek")
        self.selected_group = group
        grid = QGridLayout(group)
        grid.setVerticalSpacing(6)

        grid.addWidget(QLabel("Čas pořízení:"), 0, 0)
        self.frame_time = DateTimeRow()
        self.frame_time.changed.connect(self._on_frame_time_changed)
        grid.addWidget(self.frame_time, 0, 1)
        self.lbl_frame_time = QLabel("")
        self.lbl_frame_time.setObjectName("StatusHint")
        self.lbl_frame_time.setWordWrap(True)
        grid.addWidget(self.lbl_frame_time, 1, 0, 1, 2)

        grid.addWidget(QLabel("Ruční posun X / Y:"), 2, 0)
        row = QHBoxLayout()
        self.spin_off_x = _spin(-20000.0, 20000.0, 0.5, 1, " px")
        self.spin_off_y = _spin(-20000.0, 20000.0, 0.5, 1, " px")
        self.spin_off_x.valueChanged.connect(self._on_frame_edit)
        self.spin_off_y.valueChanged.connect(self._on_frame_edit)
        row.addWidget(self.spin_off_x, 1)
        row.addWidget(self.spin_off_y, 1)
        grid.addLayout(row, 2, 1)

        grid.addWidget(QLabel("Otočení / velikost:"), 3, 0)
        row2 = QHBoxLayout()
        self.spin_rot = _spin(-180.0, 180.0, 1.0, 1, "°", "Otočení srpku po směru hodin.")
        self.spin_scale = _spin(0.2, 5.0, 0.02, 2, "×", "Velikost tohoto Slunce navíc.")
        self.spin_rot.valueChanged.connect(self._on_frame_edit)
        self.spin_scale.valueChanged.connect(self._on_frame_edit)
        row2.addWidget(self.spin_rot, 1)
        row2.addWidget(self.spin_scale, 1)
        grid.addLayout(row2, 3, 1)

        self.btn_reset_frame = QPushButton("↺  Vrátit na vypočtenou polohu")
        self.btn_reset_frame.setToolTip("Zruší ruční posun, otočení a změnu velikosti.")
        self.btn_reset_frame.clicked.connect(self._reset_frame_manual)
        grid.addWidget(self.btn_reset_frame, 4, 0, 1, 2)
        grid.setColumnStretch(1, 1)
        return group

    def _build_look_group(self) -> QGroupBox:
        group = QGroupBox("4 · Vzhled Sluncí")
        grid = QGridLayout(group)
        grid.setVerticalSpacing(6)

        self.slider_size = SliderRow("Velikost Sluncí", 0.5, 4.0, 1.0, step=0.05, suffix="×",
                                     tip="1× = skutečná velikost Slunce vůči krajině.")
        self.slider_size.valueChanged.connect(self._on_look_changed)
        grid.addWidget(self.slider_size, 0, 0, 1, 2)

        self.slider_soft = SliderRow("Měkkost okraje", 0.0, 0.2, 0.04, step=0.01,
                                     tip="Prolnutí okraje vyříznutého disku s oblohou.")
        self.slider_soft.valueChanged.connect(self._on_look_changed)
        grid.addWidget(self.slider_soft, 1, 0, 1, 2)

        def combo(label, row, entries, tip=""):
            grid.addWidget(QLabel(label), row, 0)
            box = QComboBox()
            for text, key in entries:
                box.addItem(text, key)
            if tip:
                box.setToolTip(tip)
            box.currentIndexChanged.connect(self._on_look_changed)
            grid.addWidget(box, row, 1)
            return box

        self.combo_color = combo("Barva:", 2, (
            ("Původní (jak je vyfoceno)", "original"),
            ("Sjednotit podle většiny snímků", "unify"),
            ("Neutrální bílá", "neutral"),
            ("Zlatavá", "golden"),
        ), "Sjednocení přebarví jas pixelů jedním odstínem — nezesiluje šum.")
        self.combo_blend = combo("Prolnutí:", 3, (
            ("Zesvětlit (Měsíc průhledný)", "lighten"),
            ("Závoj (screen)", "screen"),
            ("Normální (černý Měsíc)", "normal"),
        ), "Zesvětlit: zakrytá část ukazuje oblohu pozadí.\n"
           "Normální: zakrytá část je černá silueta Měsíce.")
        self.combo_orient = combo("Natočení srpků:", 4, (
            ("Jak byly vyfoceny", "as_shot"),
            ("Srovnat na horizont (fotky byly vodorovně)", "level"),
        ), "Srovnat: otočí srpky podle místního směru k zenitu v pozadí.\n"
           "Použijte, když byl fotoaparát při focení srpků vodorovně.")

        self.chk_clip = QCheckBox("Nekreslit Slunce pod horizontem")
        self.chk_clip.setToolTip("Zapadající Slunce se schová za vyznačený obzor.")
        self.chk_clip.toggled.connect(self._on_look_changed)
        grid.addWidget(self.chk_clip, 5, 0, 1, 2)
        grid.setColumnStretch(1, 1)
        return group

    def _build_grade_group(self) -> QGroupBox:
        group = QGroupBox("5 · Barvy a tóny")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(6, 10, 6, 6)
        self.grade_tabs = QTabWidget()
        self.grade_tabs.setUsesScrollButtons(True)

        self.grade_master = GradeEditor("Platí pro všechna Slunce najednou. Úpravy jednotlivých "
                                        "snímků se k nim přičítají.")
        self.grade_master.changed.connect(self._on_master_grade)
        self.grade_tabs.addTab(self.grade_master, "Všechna Slunce")

        self.frame_grade_tab = QWidget()
        frame_layout = QVBoxLayout(self.frame_grade_tab)
        frame_layout.setContentsMargins(0, 6, 0, 0)
        frame_layout.setSpacing(2)
        top = QVBoxLayout()
        top.setContentsMargins(8, 0, 8, 0)
        self.lbl_grade_frame = QLabel("")
        self.lbl_grade_frame.setObjectName("StatusHint")
        self.lbl_grade_frame.setWordWrap(True)
        top.addWidget(self.lbl_grade_frame)
        self.chk_auto_bright = QCheckBox("Automaticky vyrovnat jas povrchu Slunce")
        self.chk_auto_bright.setToolTip("Srovná jas povrchu tohoto Slunce s ostatními.\n"
                                        "Posuvník Jas pak doladí rozdíl navíc.")
        self.chk_auto_bright.toggled.connect(self._on_frame_grade)
        top.addWidget(self.chk_auto_bright)
        frame_layout.addLayout(top)
        self.grade_frame = GradeEditor()
        self.grade_frame.changed.connect(self._on_frame_grade)
        frame_layout.addWidget(self.grade_frame)
        self.grade_tabs.addTab(self.frame_grade_tab, "Vybraný snímek")

        self.grade_background = GradeEditor("Snímek úplné fáze pod Slunci: korona, obloha "
                                            "i krajina.")
        self.grade_background.changed.connect(self._on_background_grade)
        self.grade_tabs.addTab(self.grade_background, "Pozadí")

        layout.addWidget(self.grade_tabs)
        return group

    def _build_retouch_group(self) -> QGroupBox:
        group = QGroupBox("6 · Retuš pozadí (stébla, ptáci…)")
        layout = QVBoxLayout(group)
        hint = QLabel("Přetřete stéblo trávy nebo jiný předmět na obloze: zmizí a doplní se "
                      "okolní obloha i se zrnem. Štětec má měkký okraj (čárkovaný kruh). "
                      "Slunce se kreslí až na retušované pozadí.")
        hint.setObjectName("StatusHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.btn_retouch = QPushButton("🩹  Retušovat štětcem")
        self.btn_retouch.setCheckable(True)
        self.btn_retouch.setToolTip("Levé tlačítko maluje, pravé posouvá pohled,\n"
                                    "[ a ] mění velikost štětce, Ctrl+Z vrátí tah.")
        self.btn_retouch.toggled.connect(self._on_retouch_toggled)
        layout.addWidget(self.btn_retouch)
        self.slider_brush = SliderRow("Velikost štětce", 1.0, 200.0, 12.0, step=1.0, suffix=" px",
                                      tip="Poloměr štětce v pixelech plného rozlišení pozadí.",
                                      decimals=0)
        self.slider_brush.valueChanged.connect(lambda v: self.canvas.set_brush_radius(v))
        layout.addWidget(self.slider_brush)
        row = QHBoxLayout()
        self.btn_retouch_undo = QPushButton("↶  Zpět")
        self.btn_retouch_undo.setToolTip("Vrátit poslední tah (Ctrl+Z)")
        self.btn_retouch_undo.clicked.connect(self.undo_retouch)
        row.addWidget(self.btn_retouch_undo, 1)
        self.btn_retouch_clear = QPushButton("🗑  Smazat vše")
        self.btn_retouch_clear.clicked.connect(self.clear_retouch)
        row.addWidget(self.btn_retouch_clear, 1)
        layout.addLayout(row)
        self.lbl_retouch = QLabel("")
        self.lbl_retouch.setObjectName("StatusHint")
        layout.addWidget(self.lbl_retouch)
        return group

    def _build_overlay_group(self) -> QGroupBox:
        group = QGroupBox("7 · Pomocné čáry (jen v náhledu)")
        grid = QGridLayout(group)
        self.chk_path = QCheckBox("Denní dráha Slunce")
        self.chk_ticks = QCheckBox("Časové značky po")
        self.combo_ticks = QComboBox()
        for m in (1, 2, 5, 10, 15, 30):
            self.combo_ticks.addItem(f"{m} min", m)
        self.chk_ecliptic = QCheckBox("Ekliptika")
        self.chk_ecliptic.setToolTip("Rovina oběhu Země — Slunce na ní v okamžiku úplné fáze leží.")
        self.chk_calib = QCheckBox("Horizont a kalibrační Slunce")
        self.chk_markers = QCheckBox("Značky a časy snímků")
        for w in (self.chk_path, self.chk_ticks, self.chk_ecliptic, self.chk_calib, self.chk_markers):
            w.toggled.connect(self._on_overlay_toggled)
        self.combo_ticks.currentIndexChanged.connect(self._on_overlay_toggled)
        grid.addWidget(self.chk_path, 0, 0, 1, 2)
        grid.addWidget(self.chk_ticks, 1, 0)
        grid.addWidget(self.combo_ticks, 1, 1)
        grid.addWidget(self.chk_ecliptic, 2, 0, 1, 2)
        grid.addWidget(self.chk_calib, 3, 0, 1, 2)
        grid.addWidget(self.chk_markers, 4, 0, 1, 2)
        grid.setColumnStretch(1, 1)
        return group

    def _build_viewer(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        toolbar = QFrame()
        toolbar.setObjectName("ViewerToolbar")
        bar = QHBoxLayout(toolbar)
        bar.setContentsMargins(8, 5, 8, 5)
        for text, slot, tip in (
            ("Přizpůsobit", lambda: self.canvas.fit_to_window(), "Celý snímek (dvojklik)"),
            ("1:1", lambda: self.canvas.actual_size_100(), "Skutečná velikost pixelů"),
            ("🔍 Na Slunce", self._zoom_to_sun, "Přiblížit na kalibrační Slunce"),
            ("🔍 Na vybraný", self._zoom_to_selected, "Přiblížit na vybraný srpek"),
        ):
            btn = QPushButton(text)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            bar.addWidget(btn)
        bar.addStretch()
        self.chk_hide_overlays = QCheckBox("Skrýt vše pomocné")
        self.chk_hide_overlays.setToolTip("Náhled přesně tak, jak se vyexportuje.")
        self.chk_hide_overlays.toggled.connect(self._update_overlay)
        bar.addWidget(self.chk_hide_overlays)
        layout.addWidget(toolbar)

        self.canvas = CompositeCanvas(self)
        self.canvas.horizon_drawn.connect(self._on_horizon_drawn)
        self.canvas.sun_marked.connect(self._on_sun_marked)
        self.canvas.frame_clicked.connect(self._select_frame)
        self.canvas.frame_moved.connect(self._on_frame_dragged)
        self.canvas.retouch_stroke.connect(self._on_retouch_stroke)
        self.canvas.brush_radius_changed.connect(lambda r: _set_quiet(self.slider_brush, r))
        self.canvas.set_brush_radius(12.0)
        layout.addWidget(self.canvas, 1)
        return container

    # ================================================================ Sync

    def _load_settings_into_ui(self):
        """Pushes every value of self.settings into the widgets, silently."""
        s = self.settings
        self._syncing = True
        try:
            self.lbl_bg.setText(os.path.basename(s.background_path) if s.background_path
                                else "Není načteno.")
            if parse_time(s.background_time) is not None:
                self.bg_time.set_value(parse_time(s.background_time))
            _set_quiet(self.spin_utc, s.utc_offset_hours)
            _set_quiet(self.spin_camera_clock, s.camera_clock_offset_s)
            _set_quiet(self.spin_lat, s.latitude)
            _set_quiet(self.spin_lon, s.longitude)
            self.combo_place.blockSignals(True)
            self.combo_place.setCurrentIndex(0)
            self.combo_place.blockSignals(False)
            _set_quiet(self.spin_sun_x, s.sun_x)
            _set_quiet(self.spin_sun_y, s.sun_y)
            _set_quiet(self.spin_sun_d, s.sun_diameter)
            _set_quiet(self.spin_hor_alt, s.horizon_altitude)
            _set_quiet(self.spin_clock, s.clock_offset_s)
            for combo, value in ((self.combo_scale, s.scale_mode), (self.combo_color, s.color_mode),
                                 (self.combo_blend, s.blend_mode),
                                 (self.combo_orient, s.orientation_mode),
                                 (self.combo_ticks, s.tick_minutes)):
                combo.blockSignals(True)
                idx = combo.findData(value)
                combo.setCurrentIndex(max(0, idx))
                combo.blockSignals(False)
            self.grade_master.set_grade(s.sun_grade)
            self.grade_background.set_grade(s.background_grade)
            _set_quiet(self.slider_size, s.size_multiplier)
            _set_quiet(self.slider_soft, s.edge_softness)
            _set_quiet(self.chk_clip, s.clip_below_horizon)
            _set_quiet(self.chk_path, s.show_path)
            _set_quiet(self.chk_ticks, s.show_ticks)
            _set_quiet(self.chk_ecliptic, s.show_ecliptic)
            _set_quiet(self.chk_calib, s.show_calibration)
            _set_quiet(self.chk_markers, s.show_markers)
        finally:
            self._syncing = False
        self._update_clock_hint()
        self._update_retouch_label()
        self._rebuild_table()
        self._load_selected_into_editor()

    def _reload_inputs(self):
        """
        Loads whatever the settings name but this window does not hold yet: the
        background and cut-outs of a reopened project, or anything a close
        interrupted mid-load.
        """
        if self.busy():
            return
        s = self.settings
        if s.background_path and self._bg_proxy is None:
            self._start_background_load(s.background_path, adopt_metadata=False)
        pending = [f.path for f in s.frames if f.path not in self._cutouts]
        if pending:
            known = {f.path: f.disc for f in s.frames if f.disc}
            self._start_partial_load(pending, known, adopt=False)
        self._schedule_render()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._reload_inputs)

    # ============================================================ Threads

    def _run_task(self, fn: Callable[[_Task], Any], on_done: Callable[[Any], None],
                  label: str) -> _Task:
        task = _Task(fn, self)
        task.progress.connect(self._on_task_progress)
        task.done.connect(on_done)
        task.failed.connect(self._on_task_failed)
        task.finished.connect(lambda t=task: self._on_task_finished(t))
        self._tasks.append(task)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.lbl_status.setText(label)
        task.start()
        return task

    def _on_task_progress(self, pct: int, msg: str):
        self.progress.setValue(pct)
        self.lbl_status.setText(msg)

    def _on_task_failed(self, msg: str):
        self.lbl_status.setText(f"❌ {msg}")
        QMessageBox.warning(self, "Časosběrný kompozit", msg)

    def _on_task_finished(self, task: _Task):
        if task in self._tasks:
            self._tasks.remove(task)
        if task is self._export_task:
            self._export_task = None
            self.btn_export.setEnabled(True)
        if not any(t.isRunning() for t in self._tasks):
            self.progress.setVisible(False)

    def busy(self) -> bool:
        return any(t.isRunning() for t in self._tasks)

    def stop_tasks(self, msec: int = 30000):
        """Asks every task to stop and waits for it; each checks between steps."""
        for task in list(self._tasks):
            if task.isRunning():
                task.requestInterruption()
        for task in list(self._tasks):
            if task.isRunning():
                task.wait(msec)

    def closeEvent(self, event: QCloseEvent):
        if self._export_task is not None and self._export_task.isRunning():
            reply = QMessageBox.question(
                self, "Probíhá export",
                "Export kompozitu ještě běží. Opravdu okno zavřít a export zrušit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self._render_timer.stop()
        self.stop_tasks()
        super().closeEvent(event)

    def done(self, result: int):
        # Escape and accept/reject all funnel through here: no thread may outlive us.
        self._render_timer.stop()
        self.stop_tasks()
        super().done(result)

    # ========================================================= Background

    def _choose_background(self):
        start = os.path.dirname(self.settings.background_path) if self.settings.background_path else ""
        path, _ = QFileDialog.getOpenFileName(self, "Snímek úplného zatmění (stack)", start, IMAGE_FILTER)
        if path:
            self.set_background(path)

    def set_background(self, path: str):
        """Loads a new background; its EXIF time and GPS are adopted when present."""
        self.settings.background_path = path
        self.lbl_bg.setText(os.path.basename(path))
        self._start_background_load(path, adopt_metadata=True)

    def _start_background_load(self, path: str, adopt_metadata: bool):
        self._run_task(lambda task: _load_background(path, task),
                       lambda result: self._on_background_loaded(result, adopt_metadata),
                       f"Načítám {os.path.basename(path)}…")

    def _on_background_loaded(self, result: Dict[str, Any], adopt_metadata: bool):
        if result.get("path") != self.settings.background_path:
            return  # a newer background was chosen meanwhile
        self._bg_proxy = result["proxy"]
        self._bg_graded = None
        self._bg_scale = float(result["scale"])
        self._bg_size = (int(result["width"]), int(result["height"]))
        s = self.settings
        notes = [f"{self._bg_size[0]}×{self._bg_size[1]} px"]

        if adopt_metadata:
            if result.get("moment") is not None:
                s.background_time = format_time(result["moment"])
                self.bg_time.set_value(result["moment"])
                notes.append("čas z EXIF")
            else:
                notes.append("⚠ bez času v EXIF — zadejte ho")
            if result.get("offset") is not None:
                s.utc_offset_hours = float(result["offset"])
                _set_quiet(self.spin_utc, s.utc_offset_hours)
            if result.get("gps") is not None:
                s.latitude, s.longitude = result["gps"]
                s.location_set = True
                _set_quiet(self.spin_lat, s.latitude)
                _set_quiet(self.spin_lon, s.longitude)
                notes.append("poloha z GPS")
            disc = result.get("disc")
            if disc is not None:
                s.sun_x, s.sun_y, s.sun_diameter = disc[0], disc[1], 2.0 * disc[2]
                _set_quiet(self.spin_sun_x, s.sun_x)
                _set_quiet(self.spin_sun_y, s.sun_y)
                _set_quiet(self.spin_sun_d, s.sun_diameter)
                notes.append("disk Měsíce nalezen")
            if not s.background_time:
                # Something sensible to start editing from.
                s.background_time = format_time(self.bg_time.value())
            self._update_clock_hint()

        self.lbl_bg.setText(f"{os.path.basename(s.background_path)}  ·  " + "  ·  ".join(notes))
        self.canvas.set_original_size(*self._bg_size)
        self._first_image = True
        self.lbl_status.setText("Pozadí načteno. Vyznačte horizont (a zkontrolujte Slunce).")
        self._schedule_render()

    def _take_time_from_file(self):
        start = os.path.dirname(self.settings.background_path) if self.settings.background_path else ""
        path, _ = QFileDialog.getOpenFileName(self, "Fotka s časem úplné fáze", start, IMAGE_FILTER)
        if not path:
            return
        moment, offset = extract_capture_time(path)
        if moment is None:
            QMessageBox.information(self, "Čas nenalezen",
                                    f"{os.path.basename(path)} nemá v EXIF čas pořízení.")
            return
        self.settings.background_time = format_time(moment)
        self.bg_time.set_value(moment)
        self._update_clock_hint()
        if offset is not None:
            self.settings.utc_offset_hours = offset
            _set_quiet(self.spin_utc, offset)
        gps = extract_gps_position(path)
        if gps is not None and not self.settings.location_set:
            self._apply_coords(*gps)
        self.lbl_status.setText(f"🕑 Čas pozadí převzat z {os.path.basename(path)}: "
                                f"{moment.strftime('%d.%m.%Y %H:%M:%S')}")
        self._schedule_render()

    def _on_bg_time_changed(self):
        if self._syncing:
            return
        self.settings.background_time = format_time(self.bg_time.value())
        self._update_clock_hint()
        self._schedule_render()

    def _on_camera_clock_changed(self):
        if self._syncing:
            return
        self.settings.camera_clock_offset_s = self.spin_camera_clock.value()
        self._after_clock_change()

    def set_camera_clock_offset(self, seconds: float):
        self.settings.camera_clock_offset_s = float(seconds)
        _set_quiet(self.spin_camera_clock, float(seconds))
        self._after_clock_change()

    def _after_clock_change(self):
        self._update_clock_hint()
        self._rebuild_table()
        self._schedule_render()

    def _open_clock_sync(self):
        dialog = ClockSyncDialog(self.settings, self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.offset() is not None:
            self.set_camera_clock_offset(dialog.offset())
            self.lbl_status.setText(f"⏱ Hodiny seřízeny: {self.settings.camera_clock_offset_s:+.1f} s "
                                    f"({describe_clock_offset(self.settings.camera_clock_offset_s)}).")

    def _update_clock_hint(self):
        s = self.settings
        text = describe_clock_offset(s.camera_clock_offset_s).capitalize()
        moment = s.background_moment
        if moment is not None and abs(s.camera_clock_offset_s) >= 0.05:
            text = f"Skutečný čas pozadí: {moment:%H:%M:%S} · {describe_clock_offset(s.camera_clock_offset_s)}"
        self.lbl_clock.setText(text + ".")

    def _on_place_preset(self, idx: int):
        coords = self.combo_place.itemData(idx)
        if coords:
            self._apply_coords(*coords)

    def _apply_coords(self, lat: float, lon: float):
        self.settings.latitude, self.settings.longitude = float(lat), float(lon)
        self.settings.location_set = True
        _set_quiet(self.spin_lat, lat)
        _set_quiet(self.spin_lon, lon)
        self._schedule_render()

    def _on_coords_edited(self):
        if self._syncing:
            return
        self.settings.latitude = self.spin_lat.value()
        self.settings.longitude = self.spin_lon.value()
        self.settings.location_set = True
        self.combo_place.blockSignals(True)
        self.combo_place.setCurrentIndex(0)
        self.combo_place.blockSignals(False)
        self._schedule_render()

    def _on_location_changed(self):
        if self._syncing:
            return
        self.settings.utc_offset_hours = self.spin_utc.value()
        self._schedule_render()

    # ========================================================= Calibration

    def _set_tool(self, tool: int, on: bool):
        other = self.btn_tool_sun if tool == CompositeCanvas.TOOL_HORIZON else self.btn_tool_horizon
        if on:
            other.blockSignals(True)
            other.setChecked(False)
            other.blockSignals(False)
            self.canvas.set_tool(tool)
            if tool == CompositeCanvas.TOOL_SUN:
                self.lbl_status.setText("☀ Klikněte do středu disku a táhněte k okraji. "
                                        "Pro přesnost si Slunce nejdřív přibližte kolečkem.")
            else:
                self.lbl_status.setText("〰 Táhněte podél vzdáleného obzoru — čím delší úsečka, "
                                        "tím přesnější náklon.")
        elif self.canvas.tool() == tool:
            self.canvas.set_tool(CompositeCanvas.TOOL_NONE)

    def _on_horizon_drawn(self, x1: float, y1: float, x2: float, y2: float):
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        self.settings.horizon = (x1, y1, x2, y2)
        self.btn_tool_horizon.setChecked(False)
        tilt = math.degrees(math.atan2(y2 - y1, x2 - x1))
        self.lbl_status.setText(f"〰 Horizont vyznačen (sklon {tilt:+.2f}°).")
        self._schedule_render()

    def _on_sun_marked(self, x: float, y: float, diameter: float):
        s = self.settings
        s.sun_x, s.sun_y = x, y
        if diameter > 0:
            s.sun_diameter = diameter
        _set_quiet(self.spin_sun_x, s.sun_x)
        _set_quiet(self.spin_sun_y, s.sun_y)
        _set_quiet(self.spin_sun_d, s.sun_diameter)
        self.btn_tool_sun.setChecked(False)
        self.lbl_status.setText(f"☀ Slunce: [{x:.1f}, {y:.1f}], průměr {s.sun_diameter:.1f} px.")
        self._schedule_render()

    def _auto_find_disc(self):
        if self._bg_proxy is None:
            self.lbl_status.setText("Nejdřív načtěte pozadí.")
            return
        disc = detect_totality_disc(self._bg_proxy)
        if disc is None:
            QMessageBox.information(self, "Disk nenalezen",
                                    "Disk Měsíce se nepodařilo najít. Vyznačte Slunce ručně "
                                    "tlačítkem ☀ Vyznačit Slunce.")
            return
        k = 1.0 / self._bg_scale
        self._on_sun_marked((disc[0] + 0.5) * k - 0.5, (disc[1] + 0.5) * k - 0.5, 2.0 * disc[2] * k)

    def _on_calibration_spins(self):
        if self._syncing:
            return
        s = self.settings
        s.sun_x, s.sun_y = self.spin_sun_x.value(), self.spin_sun_y.value()
        s.sun_diameter = self.spin_sun_d.value()
        s.horizon_altitude = self.spin_hor_alt.value()
        s.scale_mode = self.combo_scale.currentData() or "auto"
        self._schedule_render()

    # ========================================================== Partials

    def _choose_partials(self):
        start = ""
        if self.settings.frames:
            start = os.path.dirname(self.settings.frames[-1].path)
        elif self.settings.background_path:
            start = os.path.dirname(self.settings.background_path)
        paths, _ = QFileDialog.getOpenFileNames(self, "Snímky částečných fází (přes filtr)",
                                                start, IMAGE_FILTER)
        if paths:
            self.add_partials(paths)

    def add_partials(self, paths: List[str]):
        known = {f.path for f in self.settings.frames}
        fresh = [p for p in paths if p not in known]
        if not fresh:
            self.lbl_status.setText("Tyto snímky už v seznamu jsou.")
            return
        self._start_partial_load(fresh, {}, adopt=True)

    def _start_partial_load(self, paths: List[str], known_discs: Dict[str, Tuple[float, float, float]],
                            adopt: bool):
        self._run_task(lambda task: _load_partials(paths, known_discs, task),
                       lambda results: self._on_partials_loaded(results, adopt),
                       f"Načítám {len(paths)} snímků částečných fází…")

    def _on_partials_loaded(self, results: List[Dict[str, Any]], adopt: bool):
        s = self.settings
        found, missing = 0, []
        for entry in results:
            path = entry["path"]
            self._cutouts[path] = entry.get("cutout")
            self._frame_errors[path] = entry.get("error", "")
            if entry.get("cutout") is not None:
                found += 1
            else:
                missing.append(os.path.basename(path))
            if not adopt:
                continue
            if any(f.path == path for f in s.frames):
                continue
            moment = entry.get("moment")
            s.frames.append(PartialFrame(
                path=path, filename=os.path.basename(path),
                time=format_time(moment), time_from_exif=moment is not None,
                enabled=entry.get("cutout") is not None, disc=entry.get("disc")))
            if entry.get("gps") is not None and not s.location_set:
                self._apply_coords(*entry["gps"])

        if adopt:
            # Chronological order: it reads naturally and paints later Suns on top.
            s.frames.sort(key=lambda f: (f.moment is None, f.moment or datetime.min))
        timeless = sum(1 for f in s.frames if f.moment is None)
        msg = f"☀ Slunce nalezeno v {found} z {len(results)} snímků."
        if missing:
            msg += "  Bez Slunce: " + ", ".join(missing[:4]) + ("…" if len(missing) > 4 else "")
        if timeless:
            msg += (f"  ⚠ Bez času: {timeless} — vyberte snímek v tabulce a zadejte čas "
                    "v části „Vybraný snímek“.")
        self.lbl_status.setText(msg)
        if self._selected < 0 and s.frames:
            self._selected = 0
        self._rebuild_table()
        self._load_selected_into_editor()
        self._schedule_render()

    def _remove_selected(self):
        if not (0 <= self._selected < len(self.settings.frames)):
            return
        frame = self.settings.frames.pop(self._selected)
        self._cutouts.pop(frame.path, None)
        self._selected = min(self._selected, len(self.settings.frames) - 1)
        self._rebuild_table()
        self._load_selected_into_editor()
        self._schedule_render()

    def _on_clock_changed(self):
        if self._syncing:
            return
        self.settings.clock_offset_s = self.spin_clock.value()
        self._rebuild_table()
        self._schedule_render()

    # ---------------------------------------------------------- Table

    def _frame_status(self, idx: int) -> str:
        frame = self.settings.frames[idx]
        error = self._frame_errors.get(frame.path, "")
        if error:
            return error
        if frame.path not in self._cutouts:
            return "načítám…"
        if frame.moment is None:
            return "⚠ bez času"
        place = self._placements[idx] if idx < len(self._placements) else None
        if frame.enabled and place is None and self._report is not None:
            return "mimo záběr"
        if place is not None and self._bg_size[0] > 0:
            w, h = self._bg_size
            if not (0 <= place.x < w and 0 <= place.y < h):
                return "mimo záběr"
        return "✓" if frame.enabled else "vypnuto"

    def _gain_text(self, idx: int, gains) -> str:
        """Total brightness change applied to a Sun, in EV ("—" without a Sun)."""
        frame = self.settings.frames[idx]
        if self._cutouts.get(frame.path) is None or idx >= len(gains):
            return "—"
        gain = gains[idx][0]
        return f"{(math.log2(gain) if gain > 0 else 0.0):+.1f} EV"

    def _rebuild_table(self):
        s = self.settings
        gains = frame_gains(s, [self._cutouts.get(f.path) for f in s.frames])
        self.table.blockSignals(True)
        self.table.setRowCount(len(s.frames))
        for i, frame in enumerate(s.frames):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
                           | Qt.ItemFlag.ItemIsSelectable)
            check.setCheckState(Qt.CheckState.Checked if frame.enabled else Qt.CheckState.Unchecked)
            self.table.setItem(i, 0, check)
            moment = s.frame_moment(frame)
            self.table.setItem(i, 1, QTableWidgetItem(moment.strftime("%H:%M:%S") if moment else "—"))
            name = QTableWidgetItem(frame.filename or os.path.basename(frame.path))
            name.setToolTip(frame.path)
            self.table.setItem(i, 2, name)
            self.table.setItem(i, 3, QTableWidgetItem(self._gain_text(i, gains)))
            self.table.setItem(i, 4, QTableWidgetItem(self._frame_status(i)))
        if 0 <= self._selected < len(s.frames):
            self.table.selectRow(self._selected)
        self.table.blockSignals(False)

    def _refresh_table_status(self):
        self.table.blockSignals(True)
        gains = frame_gains(self.settings, [self._cutouts.get(f.path) for f in self.settings.frames])
        for i in range(min(self.table.rowCount(), len(self.settings.frames))):
            item = self.table.item(i, 4)
            if item is not None:
                item.setText(self._frame_status(i))
            gain_item = self.table.item(i, 3)
            if gain_item is not None:
                gain_item.setText(self._gain_text(i, gains))
        self.table.blockSignals(False)

    def _on_table_selection(self):
        rows = {i.row() for i in self.table.selectedIndexes()}
        if rows:
            self._select_frame(min(rows), from_table=True)

    def _on_table_item_changed(self, item: QTableWidgetItem):
        if item.column() != 0 or not (0 <= item.row() < len(self.settings.frames)):
            return
        self.settings.frames[item.row()].enabled = item.checkState() == Qt.CheckState.Checked
        self._schedule_render()

    def _select_frame(self, idx: int, from_table: bool = False):
        if not (0 <= idx < len(self.settings.frames)):
            return
        self._selected = idx
        if not from_table:
            self.table.blockSignals(True)
            self.table.selectRow(idx)
            self.table.blockSignals(False)
        self._load_selected_into_editor()
        self._update_overlay()

    # ------------------------------------------------------ Selected frame

    def _current_frame(self) -> Optional[PartialFrame]:
        if 0 <= self._selected < len(self.settings.frames):
            return self.settings.frames[self._selected]
        return None

    def _load_selected_into_editor(self):
        frame = self._current_frame()
        self.selected_group.setEnabled(frame is not None)
        self.frame_grade_tab.setEnabled(frame is not None)
        if frame is None:
            self.selected_group.setTitle("Vybraný snímek")
            self.lbl_grade_frame.setText("Vyberte snímek v tabulce nebo klikněte na jeho Slunce.")
            return
        self.selected_group.setTitle(f"Vybraný snímek #{self._selected + 1}: {frame.filename}")
        self.lbl_grade_frame.setText(f"Snímek #{self._selected + 1}: {frame.filename} — "
                                     f"přičítá se k úpravám všech Sluncí.")
        self._syncing = True
        try:
            moment = frame.moment
            self.frame_time.set_value(moment or parse_time(self.settings.background_time))
            if moment is None:
                self.lbl_frame_time.setText("⚠ Snímek nemá čas v EXIF — nastavte ho, jinak se "
                                            "nedá umístit.")
            elif frame.time_from_exif:
                self.lbl_frame_time.setText("Čas z EXIF. Korekce hodin se přičítá navíc.")
            else:
                self.lbl_frame_time.setText("Čas zadán ručně.")
            _set_quiet(self.chk_auto_bright, frame.auto_brightness)
            self.grade_frame.set_grade(frame.grade)
            _set_quiet(self.spin_off_x, frame.offset_x)
            _set_quiet(self.spin_off_y, frame.offset_y)
            _set_quiet(self.spin_rot, frame.rotation)
            _set_quiet(self.spin_scale, frame.scale)
        finally:
            self._syncing = False

    def _on_frame_time_changed(self):
        frame = self._current_frame()
        if frame is None or self._syncing:
            return
        frame.time = format_time(self.frame_time.value())
        frame.time_from_exif = False
        self.lbl_frame_time.setText("Čas zadán ručně.")
        self._rebuild_table()
        self._schedule_render()

    def _on_frame_edit(self, *_):
        frame = self._current_frame()
        if frame is None or self._syncing:
            return
        frame.offset_x = self.spin_off_x.value()
        frame.offset_y = self.spin_off_y.value()
        frame.rotation = self.spin_rot.value()
        frame.scale = self.spin_scale.value()
        self._schedule_render()

    def _reset_frame_manual(self):
        frame = self._current_frame()
        if frame is None:
            return
        frame.offset_x = frame.offset_y = frame.rotation = 0.0
        frame.scale = 1.0
        self._load_selected_into_editor()
        self._schedule_render()

    def _on_frame_dragged(self, idx: int, x: float, y: float):
        if not (0 <= idx < len(self.settings.frames)) or idx >= len(self._placements):
            return
        place = self._placements[idx]
        if place is None:
            return
        frame = self.settings.frames[idx]
        # Relative to the computed position, not to the last drawn one: the
        # preview re-renders on a debounce, so during a fast drag the drawn
        # position lags behind offsets already stored.
        frame.offset_x = round(x - place.base_x, 1)
        frame.offset_y = round(y - place.base_y, 1)
        if idx == self._selected:
            _set_quiet(self.spin_off_x, frame.offset_x)
            _set_quiet(self.spin_off_y, frame.offset_y)
        self._schedule_render()

    def nudge_selected(self, dx: float, dy: float):
        frame = self._current_frame()
        if frame is None:
            return
        frame.offset_x = round(frame.offset_x + dx, 2)
        frame.offset_y = round(frame.offset_y + dy, 2)
        _set_quiet(self.spin_off_x, frame.offset_x)
        _set_quiet(self.spin_off_y, frame.offset_y)
        self._schedule_render()

    def keyPressEvent(self, event: QKeyEvent):
        if event.matches(QKeySequence.StandardKey.Undo):
            self.undo_retouch()
            return
        step = 1.0
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            step = 5.0
        elif event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier):
            step = 0.2
        moves = {Qt.Key.Key_Left: (-step, 0), Qt.Key.Key_Right: (step, 0),
                 Qt.Key.Key_Up: (0, -step), Qt.Key.Key_Down: (0, step)}
        key = event.key()
        if key in moves and self._current_frame() is not None:
            self.nudge_selected(*moves[key])
            return
        if key == Qt.Key.Key_PageUp:
            self._select_frame(max(0, self._selected - 1))
            return
        if key == Qt.Key.Key_PageDown:
            self._select_frame(min(len(self.settings.frames) - 1, self._selected + 1))
            return
        if key == Qt.Key.Key_Escape and self.canvas.tool() != CompositeCanvas.TOOL_NONE:
            self.btn_tool_horizon.setChecked(False)
            self.btn_tool_sun.setChecked(False)
            return
        super().keyPressEvent(event)

    # ============================================================= Grades

    def _on_master_grade(self):
        if self._syncing:
            return
        self.grade_master.write_into(self.settings.sun_grade)
        self._schedule_render()

    def _on_frame_grade(self, *_):
        frame = self._current_frame()
        if frame is None or self._syncing:
            return
        frame.auto_brightness = self.chk_auto_bright.isChecked()
        self.grade_frame.write_into(frame.grade)
        self._schedule_render()

    def _on_background_grade(self):
        if self._syncing:
            return
        self.grade_background.write_into(self.settings.background_grade)
        self._schedule_render()

    def _graded_background(self) -> Optional[np.ndarray]:
        """
        The preview background retouched and graded. Each stage is redone only
        when its own input changes: a new brush stroke heals just its own
        neighbourhood, a grade change never re-heals anything.
        """
        if self._bg_proxy is None:
            return None
        healed = self._retouch_cache.result(self._bg_proxy, self.settings.retouch,
                                            scale=self._bg_scale)
        grade = self.settings.background_grade
        key = (self._retouch_cache.version,) + astuple(grade)
        if self._bg_graded is None or self._bg_graded_key != key:
            self._bg_graded = apply_grade(healed, grade)
            self._bg_graded_key = key
        return self._bg_graded

    # ============================================================ Retouch

    def _on_retouch_toggled(self, on: bool):
        if on:
            self.btn_tool_horizon.setChecked(False)
            self.btn_tool_sun.setChecked(False)
            self.canvas.set_brush_radius(self.slider_brush.value())
            self.lbl_status.setText("🩹 Retuš: přetřete stéblo nebo jiný předmět na obloze. "
                                    "Pravým tlačítkem posun, [ ] velikost štětce, Ctrl+Z zpět.")
        self.canvas.set_retouch_mode(on)
        self.canvas.setFocus()

    def _on_retouch_stroke(self, points, radius: float):
        if self._bg_proxy is None:
            return
        # The canvas works in full-resolution background pixels already.
        self.settings.retouch.append(
            RetouchStroke([(float(x), float(y)) for x, y in points], float(radius)))
        self._update_retouch_label()
        self._refresh()

    def undo_retouch(self):
        if not self.settings.retouch:
            return
        self.settings.retouch.pop()
        self._update_retouch_label()
        self._schedule_render()

    def clear_retouch(self):
        if not self.settings.retouch:
            return
        reply = QMessageBox.question(self, "Smazat retuš",
                                     f"Smazat celou retuš pozadí ({count_strokes(len(self.settings.retouch))})?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.settings.retouch.clear()
        self._update_retouch_label()
        self._schedule_render()

    def _update_retouch_label(self):
        count = len(self.settings.retouch)
        self.lbl_retouch.setText(f"Retuš: {count_strokes(count)}" if count else "Zatím bez retuše.")

    # ============================================================== Look

    def _on_look_changed(self, *_):
        if self._syncing:
            return
        s = self.settings
        s.size_multiplier = self.slider_size.value()
        s.edge_softness = self.slider_soft.value()
        s.color_mode = self.combo_color.currentData() or "original"
        s.blend_mode = self.combo_blend.currentData() or "lighten"
        s.orientation_mode = self.combo_orient.currentData() or "as_shot"
        s.clip_below_horizon = self.chk_clip.isChecked()
        self._schedule_render()

    def _on_overlay_toggled(self, *_):
        if self._syncing:
            return
        s = self.settings
        s.show_path = self.chk_path.isChecked()
        s.show_ticks = self.chk_ticks.isChecked()
        s.tick_minutes = int(self.combo_ticks.currentData() or 10)
        s.show_ecliptic = self.chk_ecliptic.isChecked()
        s.show_calibration = self.chk_calib.isChecked()
        s.show_markers = self.chk_markers.isChecked()
        self._update_overlay()

    # ============================================================ Render

    def _schedule_render(self):
        self._render_timer.start()

    def _recalibrate(self):
        self._report = None
        self._calib_error = ""
        if self._bg_size[0] <= 0:
            self._calib_error = "Načtěte pozadí."
            return
        try:
            self._report = calibrate(self.settings, *self._bg_size)
        except CompositeError as e:
            self._calib_error = str(e)

    def _describe_calibration(self) -> str:
        s = self.settings
        lines = []
        moment = s.background_moment
        if moment is not None and s.location_set:
            az, alt = sun_position(moment, s.latitude, s.longitude, s.utc_offset_hours)
            self.lbl_sunpos.setText(
                f"Slunce v {moment.strftime('%H:%M:%S')}: azimut {az:.2f}°, výška {alt:.2f}° "
                f"(s refrakcí)")
        else:
            self.lbl_sunpos.setText("Zadejte čas a polohu — pak se spočítá poloha Slunce.")

        if self._report is None:
            return f"<span style='color:#fbbf24'>⚠ {self._calib_error}</span>"
        rep, cam = self._report, self._report.camera
        lines.append(f"✅ Měřítko: <b>{cam.pixels_per_degree:.1f} px/°</b> "
                     f"(ohnisko {cam.focal:.0f} px, podle: {rep.scale_source})")
        lines.append(f"Náklon fotoaparátu {cam.roll:+.2f}°, osa: azimut {cam.yaw:.1f}°, "
                     f"výška {cam.pitch:+.1f}°")
        if rep.horizon_rms_px is not None:
            lines.append(f"Horizont sedí na {rep.horizon_rms_px:.1f} px")
        else:
            lines.append("<span style='color:#fbbf24'>Horizont nevyznačen — předpokládám "
                         "vodorovný fotoaparát.</span>")
        if rep.focal_from_diameter and rep.diameter_measured_px > 0:
            lines.append(f"Průměr Slunce: změřeno {rep.diameter_measured_px:.1f} px, "
                         f"model {rep.diameter_model_px:.1f} px")
        for warning in rep.warnings:
            lines.append(f"<span style='color:#fbbf24'>⚠ {warning}</span>")
        return "<br>".join(lines)

    def _refresh(self):
        """Re-solves the camera, re-places every Sun and re-renders the preview."""
        self._recalibrate()
        self.lbl_calib.setText(self._describe_calibration())
        s = self.settings
        cutouts = [self._cutouts.get(f.path) for f in s.frames]

        if self._report is not None:
            self._placements = compute_placements(s, self._report.camera)
        else:
            self._placements = [None] * len(s.frames)

        background = self._graded_background()
        if background is not None:
            if self._report is not None:
                image = render_composite(background, s, cutouts, self._report.camera,
                                         scale=self._bg_scale, background_graded=True)
            else:
                image = background
            self.canvas.set_original_size(*self._bg_size)
            self.canvas.set_base_image_bgr_float(image, keep_view=not self._first_image)
            self._first_image = False

        self._refresh_table_status()
        self._update_overlay()

    def _update_overlay(self, *_):
        s = self.settings
        hide = self.chk_hide_overlays.isChecked()
        overlay: Dict[str, Any] = dict(
            show_path=s.show_path and not hide, show_ticks=s.show_ticks and not hide,
            show_ecliptic=s.show_ecliptic and not hide,
            show_calibration=s.show_calibration and not hide,
            # Markers stay hit-testable (for dragging) unless everything is hidden.
            show_markers=s.show_markers and not hide,
        )
        if s.horizon:
            overlay["horizon"] = s.horizon
        if s.sun_x > 0 or s.sun_y > 0:
            overlay["sun"] = (s.sun_x, s.sun_y, s.sun_diameter)
            moment = s.background_moment
            if moment is not None:
                overlay["sun_label"] = f"totalita {moment.strftime('%H:%M:%S')}"

        if self._report is not None:
            cam = self._report.camera
            if not hide and s.show_path:
                path = path_polyline(s, cam, step_minutes=1.0)
                overlay["path"] = path
                tick = max(1, int(s.tick_minutes))
                w, h = self._bg_size
                overlay["ticks"] = [
                    (t.strftime("%H:%M"), x, y) for t, x, y in path
                    if t.second == 0 and t.minute % tick == 0 and np.isfinite(x)
                    and 0 <= x < w and 0 <= y < h
                ]
            if not hide and s.show_ecliptic:
                overlay["ecliptic"] = ecliptic_polyline(s, cam)
            if not hide and s.show_calibration and s.horizon is not None:
                overlay["model_horizon"] = horizon_polyline(s, cam)
            markers = []
            for place in self._placements:
                if place is None:
                    continue
                frame = s.frames[place.index]
                moment = s.frame_moment(frame)
                markers.append(dict(index=place.index, x=place.x, y=place.y, r=place.radius,
                                    selected=place.index == self._selected,
                                    label=f"#{place.index + 1}  {moment.strftime('%H:%M:%S')}"))
            overlay["markers"] = markers
        self.canvas.set_overlay(overlay)

    def _zoom_to_sun(self):
        s = self.settings
        if s.sun_x > 0 or s.sun_y > 0:
            d = max(s.sun_diameter, 10.0)
            self.canvas.zoom_to(s.sun_x, s.sun_y, min(12.0, self.canvas.height() * 0.35 / d))

    def _zoom_to_selected(self):
        if 0 <= self._selected < len(self._placements) and self._placements[self._selected]:
            place = self._placements[self._selected]
            self.canvas.zoom_to(place.x, place.y,
                                min(12.0, self.canvas.height() * 0.25 / max(place.radius * 2, 10.0)))

    # ============================================================ Export

    def export_composite(self):
        s = self.settings
        if not s.background_path or self._bg_size[0] <= 0:
            QMessageBox.warning(self, "Chybí pozadí", "Nejdřív načtěte snímek úplného zatmění.")
            return
        self._recalibrate()
        if self._report is None:
            QMessageBox.warning(self, "Kalibrace není hotová", self._calib_error)
            return
        if self._export_task is not None and self._export_task.isRunning():
            return
        base = os.path.splitext(s.background_path)[0] + "_casosber.tif"
        path, _ = QFileDialog.getSaveFileName(
            self, "Uložit časosběrný kompozit", base,
            "16-bit TIFF (*.tif *.tiff);;JPEG vysoká kvalita (*.jpg);;16-bit PNG (*.png)")
        if path:
            self.start_export(path)

    def start_export(self, path: str) -> Optional[_Task]:
        """Renders at full resolution and saves; returns the running task."""
        if not os.path.splitext(path)[1]:
            path += ".tif"
        if self._report is None:
            self._recalibrate()
            if self._report is None:
                return None
        settings = copy.deepcopy(self.settings)
        camera = copy.deepcopy(self._report.camera)
        cutouts = [self._cutouts.get(f.path) for f in settings.frames]

        def work(task: _Task):
            task.report(5, "Načítám pozadí v plném rozlišení…")
            full = load_image_float(settings.background_path)
            if full is None:
                raise CompositeError(f"Pozadí nelze načíst: {settings.background_path}")
            if task.cancelled():
                return None
            task.report(45, "Skládám Slunce do plného rozlišení…")
            out = render_composite(full, settings, cutouts, camera, scale=1.0,
                                   should_cancel=task.cancelled)
            full = None
            if task.cancelled():
                return None
            task.report(85, f"Ukládám {os.path.basename(path)}…")
            if not save_image(path, out, jpeg_quality=98):
                raise CompositeError("Soubor se nepodařilo zapsat. Zkontrolujte, že složka "
                                     "existuje a soubor není otevřený jinde.")
            return path

        self.btn_export.setEnabled(False)
        self._export_task = self._run_task(work, self._on_export_done, "Export kompozitu…")
        return self._export_task

    def _on_export_done(self, path: Optional[str]):
        if not path:
            return
        self.lbl_status.setText(f"✅ Kompozit uložen: {os.path.basename(path)}")
        QMessageBox.information(self, "Export dokončen", f"Časosběrný kompozit byl uložen do:\n{path}")
