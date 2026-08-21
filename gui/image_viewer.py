"""
Interactive High-Performance Image Viewer Widget with Zoom, Pan, Pixel Inspector and Split Preview.
"""

from typing import Optional
import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QWheelEvent, QMouseEvent, QPen, QColor, QFont, QResizeEvent, QShowEvent
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QSizePolicy


class InteractiveImageViewer(QWidget):
    """
    Custom widget providing smooth panning, zooming, 1:1 view, fit-to-view,
    and live pixel information.
    """
    pixel_hovered = pyqtSignal(int, int, int, int, int)  # x, y, r, g, b

    def __init__(self, parent=None):
        super().__init__(parent)

        self._image_rgb: Optional[np.ndarray] = None
        self._pixmap: Optional[QPixmap] = None

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

        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_image_bgr_float(self, img_bgr_f32: np.ndarray):
        """Sets preview image from float32 BGR [0.0, 1.0]."""
        u8 = (np.clip(img_bgr_f32, 0.0, 1.0) * 255.0).astype(np.uint8)
        rgb = cv2.cvtColor(u8, cv2.COLOR_BGR2RGB)
        self.set_image_rgb_uint8(rgb)

    def set_image_rgb_uint8(self, img_rgb: np.ndarray, keep_view: bool = True):
        """Sets preview image from uint8 RGB array."""
        self._image_rgb = img_rgb.copy()
        h, w, ch = self._image_rgb.shape
        bytes_per_line = ch * w
        
        # Explicit copy ensures Qt manages lifetime safely
        qimg = QImage(self._image_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimg)

        if not keep_view or self._needs_fit:
            self.fit_to_window()
        else:
            self.update()

    def set_compare_image_bgr_float(self, img_bgr_f32: np.ndarray):
        """Sets a secondary image for side-by-side split comparison."""
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

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        if self._needs_fit and self._pixmap and not self._pixmap.isNull():
            self.fit_to_window()

    def showEvent(self, event: QShowEvent):
        super().showEvent(event)
        if self._pixmap and not self._pixmap.isNull():
            self.fit_to_window()

    def fit_to_window(self):
        """Fits the image to current widget dimensions."""
        if self._pixmap is None or self._pixmap.isNull():
            return
        
        vw, vh = self.width(), self.height()
        iw, ih = self._pixmap.width(), self._pixmap.height()
        
        if iw <= 0 or ih <= 0 or vw <= 20 or vh <= 20:
            self._needs_fit = True
            return

        scale_w = (vw - 20) / float(iw)
        scale_h = (vh - 20) / float(ih)
        self._zoom = max(0.01, min(scale_w, scale_h, 1.0))
        
        # Center image
        self._pan_pos = QPointF(
            (vw - iw * self._zoom) / 2.0,
            (vh - ih * self._zoom) / 2.0
        )
        self._needs_fit = False
        self.update()

    def actual_size_100(self):
        """Sets zoom to 100% (1:1 pixel scale)."""
        if self._pixmap is None or self._pixmap.isNull():
            return
        vw, vh = self.width(), self.height()
        iw, ih = self._pixmap.width(), self._pixmap.height()
        self._zoom = 1.0
        self._pan_pos = QPointF((vw - iw) / 2.0, (vh - ih) / 2.0)
        self._needs_fit = False
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Background
        painter.fillRect(self.rect(), QColor("#12141a"))

        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(QColor("#555e70"))
            painter.setFont(QFont("Segoe UI", 13))
            text = "Náhled není k dispozici"
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, text)
            return

        iw = self._pixmap.width() * self._zoom
        ih = self._pixmap.height() * self._zoom
        dest_rect = QRectF(self._pan_pos.x(), self._pan_pos.y(), iw, ih)

        if not self._compare_mode or self._compare_pixmap is None:
            painter.drawPixmap(dest_rect.toRect(), self._pixmap)
        else:
            split_x = dest_rect.left() + iw * self._compare_split

            painter.save()
            painter.setClipRect(QRectF(dest_rect.left(), dest_rect.top(), iw * self._compare_split, ih))
            painter.drawPixmap(dest_rect.toRect(), self._pixmap)
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

        painter.setFont(QFont("Segoe UI", 9))
        painter.setPen(QColor("#8c9ba5"))
        zoom_pct = int(self._zoom * 100)
        painter.drawText(10, self.height() - 10, f"Zoom: {zoom_pct}% | Kolečko: Zoom | Myš: Posun")

    def wheelEvent(self, event: QWheelEvent):
        if self._pixmap is None or self._pixmap.isNull():
            return

        cursor_pos = event.position()
        old_zoom = self._zoom
        
        factor = 1.15 if event.angleDelta().y() > 0 else (1.0 / 1.15)
        new_zoom = max(0.02, min(25.0, old_zoom * factor))
        
        if abs(new_zoom - old_zoom) < 1e-6:
            return

        self._pan_pos = cursor_pos - (cursor_pos - self._pan_pos) * (new_zoom / old_zoom)
        self._zoom = new_zoom
        self._needs_fit = False
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_panning = True
            self._last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
        if self._is_panning:
            delta = pos - self._last_mouse_pos
            self._pan_pos += delta
            self._last_mouse_pos = pos
            self.update()

        if self._image_rgb is not None and self._zoom > 0:
            img_x = int((pos.x() - self._pan_pos.x()) / self._zoom)
            img_y = int((pos.y() - self._pan_pos.y()) / self._zoom)
            h, w = self._image_rgb.shape[:2]
            if 0 <= img_x < w and 0 <= img_y < h:
                r, g, b = self._image_rgb[img_y, img_x]
                self.pixel_hovered.emit(img_x, img_y, int(r), int(g), int(b))

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        self.fit_to_window()


class ImageViewerContainer(QWidget):
    """
    Encapsulates InteractiveImageViewer with top-right overlay controls (Fit, 100%, Compare).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(8, 4, 8, 4)
        
        self.lbl_info = QLabel("Připraveno")
        self.lbl_info.setStyleSheet("color: #8c9ba5; font-size: 11px;")
        top_bar.addWidget(self.lbl_info)
        top_bar.addStretch()

        self.btn_fit = QPushButton("Přizpůsobit oknu")
        self.btn_fit.setFixedHeight(26)
        self.btn_fit.clicked.connect(self._on_fit)
        top_bar.addWidget(self.btn_fit)

        self.btn_100 = QPushButton("100% (1:1)")
        self.btn_100.setFixedHeight(26)
        self.btn_100.clicked.connect(self._on_100)
        top_bar.addWidget(self.btn_100)

        self.btn_split = QPushButton("Srovnání Původní / HDR")
        self.btn_split.setFixedHeight(26)
        self.btn_split.setCheckable(True)
        self.btn_split.toggled.connect(self._on_split_toggled)
        top_bar.addWidget(self.btn_split)

        self.split_slider = QSlider(Qt.Orientation.Horizontal)
        self.split_slider.setRange(0, 100)
        self.split_slider.setValue(50)
        self.split_slider.setFixedWidth(120)
        self.split_slider.setVisible(False)
        self.split_slider.valueChanged.connect(self._on_split_slider)
        top_bar.addWidget(self.split_slider)

        layout.addLayout(top_bar)

        self.viewer = InteractiveImageViewer(self)
        self.viewer.pixel_hovered.connect(self._on_pixel_hovered)
        layout.addWidget(self.viewer)

    def _on_fit(self):
        self.viewer.fit_to_window()

    def _on_100(self):
        self.viewer.actual_size_100()

    def _on_split_toggled(self, checked: bool):
        self.split_slider.setVisible(checked)
        self.viewer.set_compare_mode(checked)

    def _on_split_slider(self, val: int):
        self.viewer.set_compare_split(val / 100.0)

    def _on_pixel_hovered(self, x: int, y: int, r: int, g: int, b: int):
        self.lbl_info.setText(f"Pozice: [{x}, {y}] | RGB: ({r}, {g}, {b})")
