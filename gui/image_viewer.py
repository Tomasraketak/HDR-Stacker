"""
Interactive high-performance image viewer with zoom, pan, pixel inspector,
split comparison, a live histogram, and a seamless real-time ROI crop overlay.

All heavy pixel work happens once per update; painting only ever blits cached
QPixmaps, so panning and zooming stay smooth on large eclipse frames.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRectF, QRect, QPointF, pyqtSignal
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QWheelEvent, QMouseEvent, QPainterPath,
    QPen, QColor, QFont, QResizeEvent, QShowEvent, QBrush, QLinearGradient
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QComboBox, QSizePolicy, QFrame
)

# Palette shared with styles.py so the canvas and the chrome agree.
CANVAS_BG = QColor("#0d0f14")
ACCENT = QColor("#38bdf8")
ACCENT_DIM = QColor("#0ea5e9")
TEXT_DIM = QColor("#94a3b8")
TEXT_BRIGHT = QColor("#e2e8f0")


def _numpy_rgb_to_pixmap(rgb: np.ndarray) -> QPixmap:
    """
    Builds a QPixmap from an RGB uint8 array.

    The array must be C-contiguous and the QImage must be copied before the
    numpy buffer can go out of scope — skipping either is the classic way to
    get a garbled image or a hard crash.
    """
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class InteractiveImageViewer(QWidget):
    """Smooth panning, zooming, 1:1 view, live pixel readout, and ROI placement."""

    pixel_hovered = pyqtSignal(int, int, int, int, int)   # x, y, r, g, b
    roi_selected = pyqtSignal(int, int, int, int)          # x, y, w, h in scene coords

    MIN_ZOOM = 0.02
    MAX_ZOOM = 40.0

    def __init__(self, parent=None):
        super().__init__(parent)

        # Full-image dimensions define the coordinate space everything else uses.
        self._orig_w: int = 0
        self._orig_h: int = 0

        self._base_image_rgb: Optional[np.ndarray] = None
        self._base_pixmap: Optional[QPixmap] = None

        self._roi_crop_pixmap: Optional[QPixmap] = None
        self._roi_rect: Optional[QRect] = None

        self._zoom: float = 1.0
        self._pan_pos: QPointF = QPointF(0, 0)
        self._is_panning: bool = False
        self._last_mouse_pos: QPointF = QPointF(0, 0)
        self._needs_fit: bool = True

        self._compare_mode: bool = False
        self._compare_pixmap: Optional[QPixmap] = None
        self._compare_split: float = 0.5

        self._roi_enabled: bool = False
        self._roi_size: int = 300
        self._is_dragging_roi: bool = False

        self._show_histogram: bool = False
        self._histogram: Optional[np.ndarray] = None   # shape (3, 128), normalised

        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(260, 200)

    # ------------------------------------------------------------ Scene image

    def set_original_size(self, w: int, h: int):
        if w > 0 and h > 0:
            self._orig_w, self._orig_h = int(w), int(h)

    def set_base_image_bgr_float(self, img_bgr_f32: np.ndarray, keep_view: bool = True):
        u8 = (np.clip(np.nan_to_num(img_bgr_f32), 0.0, 1.0) * 255.0).astype(np.uint8)
        self.set_base_image_bgr_uint8(u8, keep_view=keep_view)

    def set_base_image_bgr_uint8(self, img_bgr: np.ndarray, keep_view: bool = True):
        self.set_base_image_rgb_uint8(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), keep_view=keep_view)

    def set_base_image_rgb_uint8(self, img_rgb: np.ndarray, keep_view: bool = True):
        if img_rgb is None or img_rgb.size == 0:
            return
        self._base_image_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)
        h, w = self._base_image_rgb.shape[:2]

        if self._orig_w == 0 or self._orig_h == 0:
            self._orig_w, self._orig_h = w, h

        self._base_pixmap = _numpy_rgb_to_pixmap(self._base_image_rgb)
        self._recompute_histogram()

        if self._roi_rect is None and self._orig_w > 50 and self._orig_h > 50:
            self.set_roi_center(self._orig_w // 2, self._orig_h // 2, emit_signal=False)

        if not keep_view or self._needs_fit:
            self.fit_to_window()
        else:
            self.update()

    # Legacy aliases kept so existing callers keep working.
    def set_image_bgr_float(self, img_bgr_f32: np.ndarray):
        self.set_base_image_bgr_float(img_bgr_f32, keep_view=True)

    def set_image_rgb_uint8(self, img_rgb: np.ndarray, keep_view: bool = True):
        self.set_base_image_rgb_uint8(img_rgb, keep_view=keep_view)

    # -------------------------------------------------------------- Histogram

    def set_histogram_visible(self, visible: bool):
        self._show_histogram = bool(visible)
        self.update()

    def _recompute_histogram(self):
        """Computes a 128-bin per-channel histogram from a subsampled copy."""
        if self._base_image_rgb is None:
            self._histogram = None
            return
        # Subsample: a histogram of every 4th pixel is visually identical and
        # keeps this well under a millisecond even on full-resolution frames.
        sample = self._base_image_rgb[::4, ::4]
        bins = 128
        hist = np.empty((3, bins), dtype=np.float32)
        for c in range(3):
            counts = np.bincount((sample[:, :, c].ravel() >> 1), minlength=bins)[:bins]
            hist[c] = counts
        # Log scale: coronal images are dominated by dark sky and a linear plot
        # would be a single spike at zero.
        hist = np.log1p(hist)
        peak = float(hist.max())
        self._histogram = hist / peak if peak > 0 else hist

    # -------------------------------------------------------- ROI crop overlay

    def set_roi_enabled(self, enabled: bool):
        self._roi_enabled = bool(enabled)
        if enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)
            if self._roi_rect is None and self._orig_w > 50 and self._orig_h > 50:
                self.set_roi_center(self._orig_w // 2, self._orig_h // 2, emit_signal=False)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self._roi_crop_pixmap = None
        self.update()

    def set_roi_size(self, size: int):
        self._roi_size = max(32, int(size))
        if self._roi_rect is not None:
            self.set_roi_center(self._roi_rect.center().x(), self._roi_rect.center().y(),
                                emit_signal=True)

    def set_roi_center(self, cx: int, cy: int, emit_signal: bool = True):
        """Centres the ROI box on (cx, cy) in scene coordinates."""
        w = self._orig_w if self._orig_w > 0 else (
            self._base_image_rgb.shape[1] if self._base_image_rgb is not None else 0)
        h = self._orig_h if self._orig_h > 0 else (
            self._base_image_rgb.shape[0] if self._base_image_rgb is not None else 0)
        if w <= 0 or h <= 0:
            return

        rw = min(self._roi_size, w)
        rh = min(self._roi_size, h)
        x0 = int(np.clip(cx - rw // 2, 0, max(0, w - rw)))
        y0 = int(np.clip(cy - rh // 2, 0, max(0, h - rh)))

        self._roi_rect = QRect(x0, y0, rw, rh)
        self.update()
        if emit_signal:
            self.roi_selected.emit(x0, y0, rw, rh)

    def set_roi_crop_bgr_float(self, crop_bgr_f32: np.ndarray, x: int, y: int, w: int, h: int):
        """Updates just the processed ROI patch drawn on top of the full scene."""
        if crop_bgr_f32 is None or crop_bgr_f32.size == 0:
            return
        u8 = (np.clip(np.nan_to_num(crop_bgr_f32), 0.0, 1.0) * 255.0).astype(np.uint8)
        self._roi_crop_pixmap = _numpy_rgb_to_pixmap(cv2.cvtColor(u8, cv2.COLOR_BGR2RGB))
        self._roi_rect = QRect(int(x), int(y), int(w), int(h))
        self.update()

    def get_roi_rect(self) -> Optional[Tuple[int, int, int, int]]:
        if self._roi_rect is None:
            return None
        r = self._roi_rect
        return (r.x(), r.y(), r.width(), r.height())

    def zoom_to_roi(self):
        """Zooms and centres the viewport on the current ROI."""
        if self._roi_rect is None or self._base_pixmap is None:
            return
        vw, vh = self.width(), self.height()
        rw = max(1, self._roi_rect.width())
        rh = max(1, self._roi_rect.height())
        self._zoom = float(np.clip(min((vw * 0.8) / rw, (vh * 0.8) / rh),
                                   self.MIN_ZOOM, self.MAX_ZOOM))
        cx, cy = self._roi_rect.center().x(), self._roi_rect.center().y()
        self._pan_pos = QPointF(vw / 2.0 - cx * self._zoom, vh / 2.0 - cy * self._zoom)
        self._needs_fit = False
        self.update()

    # -------------------------------------------------------- Comparison mode

    def set_compare_image_bgr_float(self, img_bgr_f32: np.ndarray):
        u8 = (np.clip(np.nan_to_num(img_bgr_f32), 0.0, 1.0) * 255.0).astype(np.uint8)
        self.set_compare_image_bgr_uint8(u8)

    def set_compare_image_bgr_uint8(self, img_bgr: np.ndarray):
        if img_bgr is None or img_bgr.size == 0:
            return
        self._compare_pixmap = _numpy_rgb_to_pixmap(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        self.update()

    def set_compare_mode(self, enabled: bool):
        self._compare_mode = bool(enabled)
        self.update()

    def set_compare_split(self, split: float):
        self._compare_split = float(np.clip(split, 0.0, 1.0))
        self.update()

    # --------------------------------------------------------- Layout & paint

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        if self._needs_fit and self._base_pixmap is not None and not self._base_pixmap.isNull():
            self.fit_to_window()

    def showEvent(self, event: QShowEvent):
        super().showEvent(event)
        if self._needs_fit and self._base_pixmap is not None and not self._base_pixmap.isNull():
            self.fit_to_window()

    def _scene_size(self) -> Tuple[int, int]:
        if self._orig_w > 0 and self._orig_h > 0:
            return self._orig_w, self._orig_h
        if self._base_pixmap is not None and not self._base_pixmap.isNull():
            return self._base_pixmap.width(), self._base_pixmap.height()
        return 0, 0

    def fit_to_window(self):
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return
        vw, vh = self.width(), self.height()
        iw, ih = self._scene_size()
        if iw <= 0 or ih <= 0 or vw <= 20 or vh <= 20:
            self._needs_fit = True
            return

        self._zoom = float(np.clip(min((vw - 24) / iw, (vh - 24) / ih),
                                   self.MIN_ZOOM, self.MAX_ZOOM))
        self._pan_pos = QPointF((vw - iw * self._zoom) / 2.0, (vh - ih * self._zoom) / 2.0)
        self._needs_fit = False
        self.update()

    def actual_size_100(self):
        """Zooms to 1:1, keeping whatever is currently at the centre centred."""
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return
        vw, vh = self.width(), self.height()
        centre = QPointF(vw / 2.0, vh / 2.0)
        scene_centre = (centre - self._pan_pos) / max(self._zoom, 1e-6)
        self._zoom = 1.0
        self._pan_pos = centre - scene_centre
        self._needs_fit = False
        self.update()

    def _dest_rect(self) -> QRectF:
        iw, ih = self._scene_size()
        return QRectF(self._pan_pos.x(), self._pan_pos.y(), iw * self._zoom, ih * self._zoom)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        self._paint_background(painter)

        if self._base_pixmap is None or self._base_pixmap.isNull():
            self._paint_empty_state(painter)
            return

        dest = self._dest_rect()
        self._paint_image(painter, dest)

        if self._roi_enabled and self._roi_rect is not None:
            self._paint_roi(painter, dest)

        if self._show_histogram and self._histogram is not None:
            self._paint_histogram(painter)

        self._paint_status_pill(painter)

    def _paint_background(self, painter: QPainter):
        """A subtle vertical gradient reads better than a flat fill behind photos."""
        grad = QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0.0, QColor("#141822"))
        grad.setColorAt(1.0, CANVAS_BG)
        painter.fillRect(self.rect(), QBrush(grad))

    def _paint_empty_state(self, painter: QPainter):
        painter.setPen(QColor("#3b4256"))
        painter.setFont(QFont("Segoe UI", 40))
        painter.drawText(QRectF(0, self.height() / 2 - 90, self.width(), 70),
                         Qt.AlignmentFlag.AlignCenter, "🌞")

        painter.setPen(QColor("#8492ac"))
        painter.setFont(QFont("Segoe UI", 14, QFont.Weight.DemiBold))
        painter.drawText(QRectF(0, self.height() / 2 - 12, self.width(), 30),
                         Qt.AlignmentFlag.AlignCenter,
                         "Přetáhněte sem sérii fotografií zatmění")

        painter.setPen(QColor("#5a657d"))
        painter.setFont(QFont("Segoe UI", 10))
        painter.drawText(QRectF(0, self.height() / 2 + 20, self.width(), 26),
                         Qt.AlignmentFlag.AlignCenter,
                         "nebo klikněte na „+ Přidat fotky“   ·   Ctrl+O")

    def _paint_image(self, painter: QPainter, dest: QRectF):
        target = dest.toRect()
        if not self._compare_mode or self._compare_pixmap is None:
            painter.drawPixmap(target, self._base_pixmap)
            return

        split_x = dest.left() + dest.width() * self._compare_split

        painter.save()
        painter.setClipRect(QRectF(dest.left(), dest.top(),
                                   dest.width() * self._compare_split, dest.height()))
        painter.drawPixmap(target, self._base_pixmap)
        painter.restore()

        painter.save()
        painter.setClipRect(QRectF(split_x, dest.top(),
                                   dest.width() * (1.0 - self._compare_split), dest.height()))
        painter.drawPixmap(target, self._compare_pixmap)
        painter.restore()

        # Divider with a soft glow so it stays visible over both halves.
        painter.setPen(QPen(QColor(0, 0, 0, 140), 4))
        painter.drawLine(QPointF(split_x, dest.top()), QPointF(split_x, dest.bottom()))
        painter.setPen(QPen(ACCENT, 1.6))
        painter.drawLine(QPointF(split_x, dest.top()), QPointF(split_x, dest.bottom()))

        self._draw_tag(painter, QPointF(dest.left() + 12, dest.top() + 14), "HDR VÝSLEDEK", ACCENT)
        self._draw_tag(painter, QPointF(split_x + 12, dest.top() + 14), "ORIGINÁL", QColor("#f59e0b"))

    def _draw_tag(self, painter: QPainter, pos: QPointF, text: str, colour: QColor):
        """A small pill label that stays legible over any image content."""
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        metrics = painter.fontMetrics()
        w = metrics.horizontalAdvance(text) + 18
        h = metrics.height() + 8
        rect = QRectF(pos.x(), pos.y(), w, h)

        path = QPainterPath()
        path.addRoundedRect(rect, h / 2.0, h / 2.0)
        painter.fillPath(path, QColor(10, 12, 18, 205))
        painter.setPen(QPen(colour, 1.0))
        painter.drawPath(path)
        painter.setPen(colour)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_roi(self, painter: QPainter, dest: QRectF):
        r = self._roi_rect
        screen = QRectF(
            dest.left() + r.x() * self._zoom,
            dest.top() + r.y() * self._zoom,
            r.width() * self._zoom,
            r.height() * self._zoom,
        )

        if self._roi_crop_pixmap is not None and not self._roi_crop_pixmap.isNull():
            painter.drawPixmap(screen.toRect(), self._roi_crop_pixmap)

        # Vignette everything outside the ROI so the eye goes straight to it.
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(4, 6, 10, 130))
        outside = QPainterPath()
        outside.addRect(dest)
        inner = QPainterPath()
        inner.addRect(screen)
        painter.drawPath(outside.subtracted(inner))
        painter.restore()

        # Halo + crisp frame.
        painter.setPen(QPen(QColor(56, 189, 248, 70), 6))
        painter.drawRect(screen)
        painter.setPen(QPen(ACCENT, 1.5))
        painter.drawRect(screen)

        self._paint_corner_brackets(painter, screen)

        # Centre crosshair with a gap, so the Sun itself stays visible.
        c = screen.center()
        painter.setPen(QPen(QColor(255, 255, 255, 210), 1.4))
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            painter.drawLine(QPointF(c.x() + dx * 5, c.y() + dy * 5),
                             QPointF(c.x() + dx * 16, c.y() + dy * 16))

        self._draw_tag(painter,
                       QPointF(screen.left(), max(2.0, screen.top() - 26)),
                       f"🎯 {r.width()}×{r.height()} px — klikněte nebo táhněte",
                       ACCENT)

    @staticmethod
    def _paint_corner_brackets(painter: QPainter, rect: QRectF):
        """Camera-style corner brackets — a clear focus affordance."""
        arm = min(22.0, rect.width() * 0.22, rect.height() * 0.22)
        painter.setPen(QPen(ACCENT, 2.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        for cx, cy, sx, sy in (
            (rect.left(), rect.top(), 1, 1),
            (rect.right(), rect.top(), -1, 1),
            (rect.left(), rect.bottom(), 1, -1),
            (rect.right(), rect.bottom(), -1, -1),
        ):
            painter.drawLine(QPointF(cx, cy), QPointF(cx + arm * sx, cy))
            painter.drawLine(QPointF(cx, cy), QPointF(cx, cy + arm * sy))

    def _paint_histogram(self, painter: QPainter):
        """Log-scaled RGB histogram — essential for judging coronal clipping."""
        hist = self._histogram
        bins = hist.shape[1]
        w, h = 220.0, 96.0
        panel = QRectF(self.width() - w - 14, 14.0, w, h)

        path = QPainterPath()
        path.addRoundedRect(panel, 8, 8)
        painter.fillPath(path, QColor(8, 10, 16, 215))
        painter.setPen(QPen(QColor(56, 189, 248, 90), 1))
        painter.drawPath(path)

        plot = panel.adjusted(8, 8, -8, -20)
        colours = (QColor(248, 113, 113, 190), QColor(74, 222, 128, 190), QColor(96, 165, 250, 190))

        painter.save()
        painter.setClipRect(plot)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        for c in range(3):
            poly = QPainterPath()
            poly.moveTo(plot.left(), plot.bottom())
            for i in range(bins):
                poly.lineTo(plot.left() + plot.width() * i / (bins - 1),
                            plot.bottom() - plot.height() * float(hist[c, i]))
            poly.lineTo(plot.right(), plot.bottom())
            poly.closeSubpath()
            painter.fillPath(poly, colours[c])
        painter.restore()

        # The caption belongs inside the panel: drawn below it, it lands on the
        # photograph itself and becomes unreadable over bright corona.
        painter.setPen(QColor(148, 163, 184, 190))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(QRectF(panel.left(), plot.bottom() + 2, panel.width(), 14),
                         Qt.AlignmentFlag.AlignCenter, "Histogram (log)   stíny → světla")

    def _paint_status_pill(self, painter: QPainter):
        mode = "Režim výřezu — klikněte na Slunce" if self._roi_enabled else "Levé tlačítko: posun"
        text = f"Zoom {int(self._zoom * 100)} %   ·   {mode}   ·   Kolečko: zoom   ·   Dvojklik: přizpůsobit"

        painter.setFont(QFont("Segoe UI", 9))
        metrics = painter.fontMetrics()
        rect = QRectF(12, self.height() - metrics.height() - 18,
                      metrics.horizontalAdvance(text) + 22, metrics.height() + 10)

        path = QPainterPath()
        path.addRoundedRect(rect, rect.height() / 2.0, rect.height() / 2.0)
        painter.fillPath(path, QColor(8, 10, 16, 190))
        painter.setPen(TEXT_DIM)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    # ----------------------------------------------------- Mouse & navigation

    def _screen_to_image_coords(self, pos: QPointF) -> Optional[Tuple[int, int]]:
        """Converts a widget position to scene (full-image) coordinates."""
        if self._base_image_rgb is None or self._zoom <= 0:
            return None
        img_x = int((pos.x() - self._pan_pos.x()) / self._zoom)
        img_y = int((pos.y() - self._pan_pos.y()) / self._zoom)
        w, h = self._scene_size()
        if 0 <= img_x < w and 0 <= img_y < h:
            return img_x, img_y
        return None

    def wheelEvent(self, event: QWheelEvent):
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return
        cursor_pos = event.position()
        old_zoom = self._zoom
        factor = 1.15 if event.angleDelta().y() > 0 else (1.0 / 1.15)
        new_zoom = float(np.clip(old_zoom * factor, self.MIN_ZOOM, self.MAX_ZOOM))
        if abs(new_zoom - old_zoom) < 1e-6:
            return
        # Keep the point under the cursor fixed while zooming.
        self._pan_pos = cursor_pos - (cursor_pos - self._pan_pos) * (new_zoom / old_zoom)
        self._zoom = new_zoom
        self._needs_fit = False
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        pos = event.position()

        if event.button() == Qt.MouseButton.LeftButton and self._roi_enabled:
            coords = self._screen_to_image_coords(pos)
            if coords is not None:
                self._is_dragging_roi = True
                self.set_roi_center(coords[0], coords[1], emit_signal=True)
                return

        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton,
                              Qt.MouseButton.MiddleButton):
            self._is_panning = True
            self._last_mouse_pos = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()

        if self._is_dragging_roi and self._roi_enabled:
            coords = self._screen_to_image_coords(pos)
            if coords is not None:
                self.set_roi_center(coords[0], coords[1], emit_signal=True)
        elif self._is_panning:
            self._pan_pos += pos - self._last_mouse_pos
            self._last_mouse_pos = pos
            self.update()

        self._emit_hover(pos)

    def _emit_hover(self, pos: QPointF):
        coords = self._screen_to_image_coords(pos)
        if coords is None or self._base_image_rgb is None:
            return
        img_x, img_y = coords
        scene_w, scene_h = self._scene_size()
        act_h, act_w = self._base_image_rgb.shape[:2]

        # The displayed array may be a proxy: map scene coords into its pixels.
        read_x = int(img_x * (act_w / scene_w)) if scene_w > 0 else img_x
        read_y = int(img_y * (act_h / scene_h)) if scene_h > 0 else img_y
        read_x = int(np.clip(read_x, 0, act_w - 1))
        read_y = int(np.clip(read_y, 0, act_h - 1))

        r, g, b = self._base_image_rgb[read_y, read_x]
        self.pixel_hovered.emit(img_x, img_y, int(r), int(g), int(b))

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging_roi = False
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton,
                              Qt.MouseButton.MiddleButton):
            self._is_panning = False
            self.setCursor(Qt.CursorShape.CrossCursor if self._roi_enabled
                           else Qt.CursorShape.ArrowCursor)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if self._roi_enabled and self._roi_rect is not None:
            self.zoom_to_roi()
        else:
            self.fit_to_window()


class ImageViewerContainer(QWidget):
    """The viewer plus its toolbar (fit, 1:1, compare, histogram, ROI mode)."""

    roi_mode_toggled = pyqtSignal(bool, int, int, int, int)
    center_sun_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        toolbar = QFrame()
        toolbar.setObjectName("ViewerToolbar")
        top_bar = QHBoxLayout(toolbar)
        top_bar.setContentsMargins(10, 6, 10, 6)
        top_bar.setSpacing(6)

        self.lbl_info = QLabel("Připraveno")
        self.lbl_info.setObjectName("PixelReadout")
        self.lbl_info.setMinimumWidth(230)
        top_bar.addWidget(self.lbl_info)
        top_bar.addStretch()

        self.btn_roi_toggle = self._tool_button(
            "🎯 ROI", checkable=True,
            tip="Rychlý výřez kolem Slunce pro okamžitou editaci.\n"
                "Po zapnutí stačí kliknout do fotky tam, kde je Slunce.")
        self.btn_roi_toggle.setObjectName("RoiToggle")
        self.btn_roi_toggle.toggled.connect(self._on_roi_toggled)
        top_bar.addWidget(self.btn_roi_toggle)

        self.combo_roi_size = QComboBox()
        for size in (300, 450, 600, 800, 1200):
            self.combo_roi_size.addItem(f"{size}×{size}", size)
        self.combo_roi_size.setFixedHeight(30)
        self.combo_roi_size.setVisible(False)
        self.combo_roi_size.currentIndexChanged.connect(self._on_roi_size_changed)
        top_bar.addWidget(self.combo_roi_size)

        self.btn_find_sun = self._tool_button(
            "☀️ Najít", tip="Automaticky najde Slunce / disk Měsíce a umístí na něj výřez.")
        self.btn_find_sun.setVisible(False)
        self.btn_find_sun.clicked.connect(self.center_sun_requested.emit)
        top_bar.addWidget(self.btn_find_sun)

        self.btn_zoom_roi = self._tool_button("🔍 Přiblížit", tip="Přiblíží pohled na vybraný výřez.")
        self.btn_zoom_roi.setVisible(False)
        self.btn_zoom_roi.clicked.connect(lambda: self.viewer.zoom_to_roi())
        top_bar.addWidget(self.btn_zoom_roi)

        top_bar.addWidget(self._separator())

        self.btn_fit = self._tool_button("Přizpůsobit", tip="Přizpůsobit oknu (Ctrl+0)")
        self.btn_fit.clicked.connect(lambda: self.viewer.fit_to_window())
        top_bar.addWidget(self.btn_fit)

        self.btn_100 = self._tool_button("1:1", tip="Skutečná velikost 100 % (Ctrl+1)")
        self.btn_100.clicked.connect(lambda: self.viewer.actual_size_100())
        top_bar.addWidget(self.btn_100)

        self.btn_hist = self._tool_button("📊", checkable=True, tip="Zobrazit histogram")
        self.btn_hist.setFixedWidth(38)
        self.btn_hist.toggled.connect(lambda c: self.viewer.set_histogram_visible(c))
        top_bar.addWidget(self.btn_hist)

        self.btn_split = self._tool_button("Srovnání", checkable=True,
                                           tip="Porovnání HDR výsledku s původní expozicí")
        self.btn_split.toggled.connect(self._on_split_toggled)
        top_bar.addWidget(self.btn_split)

        self.split_slider = QSlider(Qt.Orientation.Horizontal)
        self.split_slider.setRange(0, 100)
        self.split_slider.setValue(50)
        self.split_slider.setFixedWidth(90)
        self.split_slider.setVisible(False)
        self.split_slider.valueChanged.connect(lambda v: self.viewer.set_compare_split(v / 100.0))
        top_bar.addWidget(self.split_slider)

        layout.addWidget(toolbar)

        self.viewer = InteractiveImageViewer(self)
        self.viewer.pixel_hovered.connect(self._on_pixel_hovered)
        self.viewer.roi_selected.connect(self._on_viewer_roi_selected)
        layout.addWidget(self.viewer, 1)

    @staticmethod
    def _tool_button(text: str, checkable: bool = False, tip: str = "") -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedHeight(30)
        btn.setCheckable(checkable)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        if tip:
            btn.setToolTip(tip)
        return btn

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setObjectName("ToolbarSeparator")
        line.setFixedHeight(22)
        return line

    def _on_split_toggled(self, checked: bool):
        self.split_slider.setVisible(checked)
        self.viewer.set_compare_mode(checked)

    def _on_roi_toggled(self, checked: bool):
        for widget in (self.combo_roi_size, self.btn_find_sun, self.btn_zoom_roi):
            widget.setVisible(checked)
        self.viewer.set_roi_enabled(checked)

        rect = self.viewer.get_roi_rect()
        if rect:
            self.roi_mode_toggled.emit(checked, *rect)
        else:
            self.roi_mode_toggled.emit(checked, 0, 0, self.combo_roi_size.currentData() or 300,
                                       self.combo_roi_size.currentData() or 300)

    def _on_roi_size_changed(self):
        size = self.combo_roi_size.currentData()
        if size:
            self.viewer.set_roi_size(int(size))

    def _on_viewer_roi_selected(self, x: int, y: int, w: int, h: int):
        if self.btn_roi_toggle.isChecked():
            self.roi_mode_toggled.emit(True, x, y, w, h)

    def _on_pixel_hovered(self, x: int, y: int, r: int, g: int, b: int):
        luma = int(0.299 * r + 0.587 * g + 0.114 * b)
        self.lbl_info.setText(f"[{x}, {y}]   RGB {r:3d} {g:3d} {b:3d}   Jas {luma:3d}")
