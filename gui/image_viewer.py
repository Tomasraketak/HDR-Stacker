"""
Interactive High-Performance Image Viewer Widget with Zoom, Pan, Pixel Inspector, Split Preview,
and Seamless Real-Time ROI (Region of Interest) Crop Overlay.
"""

from typing import Optional, Tuple
import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRectF, QRect, QPointF, QPoint, pyqtSignal
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QWheelEvent, QMouseEvent,
    QPen, QColor, QFont, QResizeEvent, QShowEvent, QBrush
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QComboBox, QSizePolicy
)


class InteractiveImageViewer(QWidget):
    """
    Custom widget providing smooth panning, zooming, 1:1 view, fit-to-view,
    live pixel information, and interactive ROI crop box positioning.
    """
    pixel_hovered = pyqtSignal(int, int, int, int, int)  # x, y, r, g, b
    roi_selected = pyqtSignal(int, int, int, int)         # x, y, w, h in base scene coordinates

    def __init__(self, parent=None):
        super().__init__(parent)

        # Base full scene image
        self._base_image_rgb: Optional[np.ndarray] = None
        self._base_pixmap: Optional[QPixmap] = None

        # Processed ROI Crop overlay
        self._roi_crop_rgb: Optional[np.ndarray] = None
        self._roi_crop_pixmap: Optional[QPixmap] = None
        self._roi_rect: Optional[QRect] = None  # in base image coordinates (x, y, w, h)

        # Transform / view state
        self._zoom: float = 1.0
        self._pan_pos: QPointF = QPointF(0, 0)
        self._is_panning: bool = False
        self._last_mouse_pos: QPointF = QPointF(0, 0)
        self._needs_fit: bool = True
        
        # Split comparison mode (Original vs Result)
        self._compare_mode: bool = False
        self._compare_pixmap: Optional[QPixmap] = None
        self._compare_split: float = 0.5

        # Interactive ROI Mode
        self._roi_enabled: bool = False
        self._roi_size: int = 300
        self._is_dragging_roi: bool = False

        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ------------------ Base Scene Image ------------------
    def set_base_image_bgr_float(self, img_bgr_f32: np.ndarray, keep_view: bool = True):
        """Sets full scene image from float32 BGR [0.0, 1.0]."""
        u8 = (np.clip(img_bgr_f32, 0.0, 1.0) * 255.0).astype(np.uint8)
        rgb = cv2.cvtColor(u8, cv2.COLOR_BGR2RGB)
        self.set_base_image_rgb_uint8(rgb, keep_view=keep_view)

    def set_base_image_rgb_uint8(self, img_rgb: np.ndarray, keep_view: bool = True):
        """Sets full scene image from uint8 RGB array."""
        self._base_image_rgb = img_rgb.copy()
        h, w, ch = self._base_image_rgb.shape
        bytes_per_line = ch * w
        
        qimg = QImage(self._base_image_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        self._base_pixmap = QPixmap.fromImage(qimg)

        # If ROI is enabled and not yet set, place default at center
        if self._roi_rect is None and w > 50 and h > 50:
            self.set_roi_center(w // 2, h // 2, emit_signal=False)

        if not keep_view or self._needs_fit:
            self.fit_to_window()
        else:
            self.update()

    # Legacy compatibility methods
    def set_image_bgr_float(self, img_bgr_f32: np.ndarray):
        self.set_base_image_bgr_float(img_bgr_f32, keep_view=True)

    def set_image_rgb_uint8(self, img_rgb: np.ndarray, keep_view: bool = True):
        self.set_base_image_rgb_uint8(img_rgb, keep_view=keep_view)

    # ------------------ ROI Crop Overlay ------------------
    def set_roi_enabled(self, enabled: bool):
        self._roi_enabled = enabled
        if enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self._roi_crop_pixmap = None
        self.update()

    def set_roi_size(self, size: int):
        self._roi_size = size
        if self._roi_rect is not None and self._base_image_rgb is not None:
            cx = self._roi_rect.center().x()
            cy = self._roi_rect.center().y()
            self.set_roi_center(cx, cy, emit_signal=True)

    def set_roi_center(self, cx: int, cy: int, emit_signal: bool = True):
        """Centers the ROI box on (cx, cy) in base image coordinates."""
        if self._base_image_rgb is None:
            return
        h, w = self._base_image_rgb.shape[:2]
        half = self._roi_size // 2
        x0 = max(0, min(w - self._roi_size, cx - half))
        y0 = max(0, min(h - self._roi_size, cy - half))
        rw = min(self._roi_size, w)
        rh = min(self._roi_size, h)
        self._roi_rect = QRect(int(x0), int(y0), int(rw), int(rh))
        self.update()
        if emit_signal:
            self.roi_selected.emit(int(x0), int(y0), int(rw), int(rh))

    def set_roi_crop_bgr_float(self, crop_bgr_f32: np.ndarray, x: int, y: int, w: int, h: int):
        """Updates only the HDR processed ROI patch to draw directly on top of the full scene."""
        u8 = (np.clip(crop_bgr_f32, 0.0, 1.0) * 255.0).astype(np.uint8)
        rgb = cv2.cvtColor(u8, cv2.COLOR_BGR2RGB)
        ch_h, ch_w, ch_c = rgb.shape
        qimg = QImage(rgb.data, ch_w, ch_h, ch_c * ch_w, QImage.Format.Format_RGB888).copy()
        self._roi_crop_pixmap = QPixmap.fromImage(qimg)
        self._roi_rect = QRect(x, y, w, h)
        self.update()

    def get_roi_rect(self) -> Optional[Tuple[int, int, int, int]]:
        if self._roi_rect is not None:
            return (self._roi_rect.x(), self._roi_rect.y(), self._roi_rect.width(), self._roi_rect.height())
        return None

    def zoom_to_roi(self):
        """Smoothly zooms and centers the viewport on the current ROI crop."""
        if self._roi_rect is None or self._base_image_rgb is None:
            return
        
        vw, vh = self.width(), self.height()
        rw, rh = self._roi_rect.width(), self._roi_rect.height()
        cx, cy = self._roi_rect.center().x(), self._roi_rect.center().y()

        scale_w = (vw * 0.75) / float(rw)
        scale_h = (vh * 0.75) / float(rh)
        self._zoom = max(0.2, min(8.0, min(scale_w, scale_h)))

        self._pan_pos = QPointF(
            vw / 2.0 - cx * self._zoom,
            vh / 2.0 - cy * self._zoom
        )
        self._needs_fit = False
        self.update()

    # ------------------ Comparison Mode ------------------
    def set_compare_image_bgr_float(self, img_bgr_f32: np.ndarray):
        u8 = (np.clip(img_bgr_f32, 0.0, 1.0) * 255.0).astype(np.uint8)
        rgb = cv2.cvtColor(u8, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        self._compare_pixmap = QPixmap.fromImage(qimg)
        self.update()

    def set_compare_mode(self, enabled: bool):
        self._compare_mode = enabled
        self.update()

    def set_compare_split(self, split: float):
        self._compare_split = max(0.0, min(1.0, split))
        self.update()

    # ------------------ Layout & Paint ------------------
    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        if self._needs_fit and self._base_pixmap and not self._base_pixmap.isNull():
            self.fit_to_window()

    def showEvent(self, event: QShowEvent):
        super().showEvent(event)
        if self._base_pixmap and not self._base_pixmap.isNull():
            self.fit_to_window()

    def fit_to_window(self):
        """Fits the image to current widget dimensions."""
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return
        
        vw, vh = self.width(), self.height()
        iw, ih = self._base_pixmap.width(), self._base_pixmap.height()
        
        if iw <= 0 or ih <= 0 or vw <= 20 or vh <= 20:
            self._needs_fit = True
            return

        scale_w = (vw - 20) / float(iw)
        scale_h = (vh - 20) / float(ih)
        self._zoom = max(0.01, min(scale_w, scale_h, 1.0))
        
        self._pan_pos = QPointF(
            (vw - iw * self._zoom) / 2.0,
            (vh - ih * self._zoom) / 2.0
        )
        self._needs_fit = False
        self.update()

    def actual_size_100(self):
        """Sets zoom to 100% (1:1 pixel scale)."""
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return
        vw, vh = self.width(), self.height()
        iw, ih = self._base_pixmap.width(), self._base_pixmap.height()
        self._zoom = 1.0
        self._pan_pos = QPointF((vw - iw) / 2.0, (vh - ih) / 2.0)
        self._needs_fit = False
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Dark Canvas Background
        painter.fillRect(self.rect(), QColor("#12141a"))

        if self._base_pixmap is None or self._base_pixmap.isNull():
            painter.setPen(QColor("#555e70"))
            painter.setFont(QFont("Segoe UI", 13))
            text = "Přetáhněte sem sérii fotografií nebo klikněte na '+ Přidat fotky'"
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, text)
            return

        iw = self._base_pixmap.width() * self._zoom
        ih = self._base_pixmap.height() * self._zoom
        dest_rect = QRectF(self._pan_pos.x(), self._pan_pos.y(), iw, ih)

        # 1. Base image rendering
        if not self._compare_mode or self._compare_pixmap is None:
            painter.drawPixmap(dest_rect.toRect(), self._base_pixmap)
        else:
            split_x = dest_rect.left() + iw * self._compare_split
            painter.save()
            painter.setClipRect(QRectF(dest_rect.left(), dest_rect.top(), iw * self._compare_split, ih))
            painter.drawPixmap(dest_rect.toRect(), self._base_pixmap)
            painter.restore()

            painter.save()
            painter.setClipRect(QRectF(split_x, dest_rect.top(), iw * (1.0 - self._compare_split), ih))
            painter.drawPixmap(dest_rect.toRect(), self._compare_pixmap)
            painter.restore()

            painter.setPen(QPen(QColor("#4da6ff"), 2))
            painter.drawLine(int(split_x), int(dest_rect.top()), int(split_x), int(dest_rect.bottom()))

            painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(int(dest_rect.left() + 10), int(dest_rect.top() + 25), "HDR VÝSLEDEK")
            painter.drawText(int(split_x + 10), int(dest_rect.top() + 25), "ORIGINÁL")

        # 2. Draw Processed ROI Crop overlay if active
        if self._roi_enabled and self._roi_rect is not None and self._zoom > 0:
            rx = dest_rect.left() + self._roi_rect.x() * self._zoom
            ry = dest_rect.top() + self._roi_rect.y() * self._zoom
            rw = self._roi_rect.width() * self._zoom
            rh = self._roi_rect.height() * self._zoom
            screen_roi = QRectF(rx, ry, rw, rh)

            # Draw the live processed HDR crop if available
            if self._roi_crop_pixmap is not None and not self._roi_crop_pixmap.isNull():
                painter.drawPixmap(screen_roi.toRect(), self._roi_crop_pixmap)

            # Darken outside background slightly to make ROI stand out
            painter.save()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 90))
            painter.drawRect(QRectF(dest_rect.left(), dest_rect.top(), iw, screen_roi.top() - dest_rect.top()))
            painter.drawRect(QRectF(dest_rect.left(), screen_roi.bottom(), iw, dest_rect.bottom() - screen_roi.bottom()))
            painter.drawRect(QRectF(dest_rect.left(), screen_roi.top(), screen_roi.left() - dest_rect.left(), screen_roi.height()))
            painter.drawRect(QRectF(screen_roi.right(), screen_roi.top(), dest_rect.right() - screen_roi.right(), screen_roi.height()))
            painter.restore()

            # Glowing neon bounding box
            pen = QPen(QColor("#00d2ff"), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(screen_roi)

            # Crosshairs at center
            painter.setPen(QPen(QColor("#ffffff"), 2))
            center = screen_roi.center()
            painter.drawLine(int(center.x() - 10), int(center.y()), int(center.x() + 10), int(center.y()))
            painter.drawLine(int(center.x()), int(center.y() - 10), int(center.x()), int(center.y() + 10))

            # Floating badge
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.setPen(QColor("#00d2ff"))
            badge_text = f"🎯 Rychlý výřez: {self._roi_rect.width()}x{self._roi_rect.height()} px — Klikněte nebo táhněte pro přesun"
            painter.drawText(int(rx + 4), max(16, int(ry - 6)), badge_text)

        # Bottom info overlay
        painter.setFont(QFont("Segoe UI", 9))
        painter.setPen(QColor("#8c9ba5"))
        zoom_pct = int(self._zoom * 100)
        mode_text = "Režim výřezu: Klikněte na Slunce" if self._roi_enabled else "Levé tlačítko: Posun"
        painter.drawText(10, self.height() - 10, f"Zoom: {zoom_pct}% | {mode_text} | Kolečko: Zoom")

    # ------------------ Mouse & Navigation Events ------------------
    def _screen_to_image_coords(self, pos: QPointF) -> Optional[Tuple[int, int]]:
        """Converts screen pixel position to base image coordinates."""
        if self._base_image_rgb is None or self._zoom <= 0:
            return None
        img_x = int((pos.x() - self._pan_pos.x()) / self._zoom)
        img_y = int((pos.y() - self._pan_pos.y()) / self._zoom)
        h, w = self._base_image_rgb.shape[:2]
        if 0 <= img_x < w and 0 <= img_y < h:
            return img_x, img_y
        return None

    def wheelEvent(self, event: QWheelEvent):
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return

        cursor_pos = event.position()
        old_zoom = self._zoom
        
        factor = 1.15 if event.angleDelta().y() > 0 else (1.0 / 1.15)
        new_zoom = max(0.02, min(30.0, old_zoom * factor))
        
        if abs(new_zoom - old_zoom) < 1e-6:
            return

        self._pan_pos = cursor_pos - (cursor_pos - self._pan_pos) * (new_zoom / old_zoom)
        self._zoom = new_zoom
        self._needs_fit = False
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        pos = event.position()
        
        # Left button in ROI mode: immediately center/drag ROI on click
        if event.button() == Qt.MouseButton.LeftButton and self._roi_enabled:
            coords = self._screen_to_image_coords(pos)
            if coords is not None:
                self._is_dragging_roi = True
                self.set_roi_center(coords[0], coords[1], emit_signal=True)
                return

        # Left button in normal mode or Right/Middle button: pan canvas
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._is_panning = True
            self._last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
        
        # Dragging ROI
        if self._is_dragging_roi and self._roi_enabled:
            coords = self._screen_to_image_coords(pos)
            if coords is not None:
                self.set_roi_center(coords[0], coords[1], emit_signal=True)

        # Panning
        elif self._is_panning:
            delta = pos - self._last_mouse_pos
            self._pan_pos += delta
            self._last_mouse_pos = pos
            self.update()

        # Hover pixel readout
        coords = self._screen_to_image_coords(pos)
        if coords is not None and self._base_image_rgb is not None:
            img_x, img_y = coords
            r, g, b = self._base_image_rgb[img_y, img_x]
            self.pixel_hovered.emit(img_x, img_y, int(r), int(g), int(b))

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging_roi = False

        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._is_panning = False
            if self._roi_enabled:
                self.setCursor(Qt.CursorShape.CrossCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if self._roi_enabled and self._roi_rect is not None:
            self.zoom_to_roi()
        else:
            self.fit_to_window()


class ImageViewerContainer(QWidget):
    """
    Encapsulates InteractiveImageViewer with top-right overlay controls (Fit, 100%, Compare, ROI Mode).
    """
    roi_mode_toggled = pyqtSignal(bool, int, int, int, int)  # enabled, x, y, w, h
    center_sun_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Top toolbar
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(8, 4, 8, 4)
        
        self.lbl_info = QLabel("Připraveno")
        self.lbl_info.setStyleSheet("color: #8c9ba5; font-size: 11px;")
        top_bar.addWidget(self.lbl_info)
        top_bar.addStretch()

        # ROI Mode Selector
        self.btn_roi_toggle = QPushButton("🎯 Rychlý výřez Slunce (ROI)")
        self.btn_roi_toggle.setCheckable(True)
        self.btn_roi_toggle.setFixedHeight(28)
        self.btn_roi_toggle.setToolTip("Aktivuje rychlý výřez pro bleskovou editaci. Po zapnutí stačí kliknout myší do fotky, kde se nachází Slunce!")
        self.btn_roi_toggle.toggled.connect(self._on_roi_toggled)
        top_bar.addWidget(self.btn_roi_toggle)

        self.combo_roi_size = QComboBox()
        self.combo_roi_size.addItem("300x300 px (Bleskové ⚡)", 300)
        self.combo_roi_size.addItem("450x450 px", 450)
        self.combo_roi_size.addItem("600x600 px", 600)
        self.combo_roi_size.addItem("800x800 px", 800)
        self.combo_roi_size.setFixedHeight(28)
        self.combo_roi_size.setVisible(False)
        self.combo_roi_size.currentIndexChanged.connect(self._on_roi_size_changed)
        top_bar.addWidget(self.combo_roi_size)

        self.btn_find_sun = QPushButton("☀️ Najít Slunce")
        self.btn_find_sun.setFixedHeight(28)
        self.btn_find_sun.setVisible(False)
        self.btn_find_sun.setToolTip("Automaticky detekuje polohu Slunce / disku Měsíce a umístí na něj výřez.")
        self.btn_find_sun.clicked.connect(self.center_sun_requested.emit)
        top_bar.addWidget(self.btn_find_sun)

        self.btn_zoom_roi = QPushButton("🔍 Zaměřit výřez")
        self.btn_zoom_roi.setFixedHeight(28)
        self.btn_zoom_roi.setVisible(False)
        self.btn_zoom_roi.setToolTip("Přiblíží a vycentruje pohled přímo na vybraný výřez.")
        self.btn_zoom_roi.clicked.connect(self._on_zoom_roi)
        top_bar.addWidget(self.btn_zoom_roi)

        self.btn_fit = QPushButton("Přizpůsobit oknu")
        self.btn_fit.setFixedHeight(28)
        self.btn_fit.clicked.connect(self._on_fit)
        top_bar.addWidget(self.btn_fit)

        self.btn_100 = QPushButton("100% (1:1)")
        self.btn_100.setFixedHeight(28)
        self.btn_100.clicked.connect(self._on_100)
        top_bar.addWidget(self.btn_100)

        self.btn_split = QPushButton("Srovnání")
        self.btn_split.setFixedHeight(28)
        self.btn_split.setCheckable(True)
        self.btn_split.toggled.connect(self._on_split_toggled)
        top_bar.addWidget(self.btn_split)

        self.split_slider = QSlider(Qt.Orientation.Horizontal)
        self.split_slider.setRange(0, 100)
        self.split_slider.setValue(50)
        self.split_slider.setFixedWidth(100)
        self.split_slider.setVisible(False)
        self.split_slider.valueChanged.connect(self._on_split_slider)
        top_bar.addWidget(self.split_slider)

        layout.addLayout(top_bar)

        self.viewer = InteractiveImageViewer(self)
        self.viewer.pixel_hovered.connect(self._on_pixel_hovered)
        self.viewer.roi_selected.connect(self._on_viewer_roi_selected)
        layout.addWidget(self.viewer)

    def _on_fit(self):
        self.viewer.fit_to_window()

    def _on_100(self):
        self.viewer.actual_size_100()

    def _on_zoom_roi(self):
        self.viewer.zoom_to_roi()

    def _on_split_toggled(self, checked: bool):
        self.split_slider.setVisible(checked)
        self.viewer.set_compare_mode(checked)

    def _on_split_slider(self, val: int):
        self.viewer.set_compare_split(val / 100.0)

    def _on_roi_toggled(self, checked: bool):
        self.combo_roi_size.setVisible(checked)
        self.btn_find_sun.setVisible(checked)
        self.btn_zoom_roi.setVisible(checked)
        self.viewer.set_roi_enabled(checked)
        
        if checked:
            self.btn_roi_toggle.setStyleSheet("background-color: #0099ff; color: white; font-weight: bold;")
        else:
            self.btn_roi_toggle.setStyleSheet("")

        rect = self.viewer.get_roi_rect()
        if rect:
            self.roi_mode_toggled.emit(checked, rect[0], rect[1], rect[2], rect[3])
        else:
            self.roi_mode_toggled.emit(checked, 0, 0, 300, 300)

    def _on_roi_size_changed(self):
        size = self.combo_roi_size.currentData()
        self.viewer.set_roi_size(size)

    def _on_viewer_roi_selected(self, x: int, y: int, w: int, h: int):
        if self.btn_roi_toggle.isChecked():
            self.roi_mode_toggled.emit(True, x, y, w, h)

    def _on_pixel_hovered(self, x: int, y: int, r: int, g: int, b: int):
        self.lbl_info.setText(f"Pozice: [{x}, {y}] | RGB: ({r}, {g}, {b})")
