"""
Shared window-geometry helpers.

A window sized with a hardcoded pixel height is a trap on laptops: Windows
scales a 15" 1080p panel to 125 % by default, leaving roughly 1536x826 of usable
desktop. Anything taller opens with its bottom row — the action buttons —
below the screen edge, where it cannot be clicked.
"""

from PyQt6.QtCore import QSize
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QWidget


def available_screen_size(widget: QWidget = None) -> QSize:
    """
    Usable desktop size (taskbar excluded) of the screen the widget is on,
    falling back to the primary screen and finally to a conservative guess.
    """
    screen = None
    if widget is not None and widget.window().windowHandle() is not None:
        screen = widget.window().windowHandle().screen()
    if screen is None:
        screen = QGuiApplication.primaryScreen()
    if screen is None:
        return QSize(1280, 720)
    geo = screen.availableGeometry()
    return QSize(geo.width(), geo.height())


def fit_window_to_screen(
    widget: QWidget,
    preferred_w: int,
    preferred_h: int,
    margin_w: int = 80,
    margin_h: int = 80,
) -> QSize:
    """
    Resizes `widget` to its preferred size, shrunk to fit the usable desktop.

    The margins leave room for the window frame and the taskbar. The result is
    never smaller than the widget's own minimum, so this can only ever prevent
    an oversized window — it cannot squash a valid layout.
    """
    available = available_screen_size(widget)
    width = min(preferred_w, max(320, available.width() - margin_w))
    height = min(preferred_h, max(240, available.height() - margin_h))

    minimum = widget.minimumSizeHint()
    width = max(width, minimum.width())
    height = max(height, minimum.height())

    widget.resize(width, height)
    return QSize(width, height)


def center_on_screen(widget: QWidget):
    """Centres the widget on its screen, clamped so no edge lands off-screen."""
    available = available_screen_size(widget)
    size = widget.frameGeometry().size()
    x = max(0, (available.width() - size.width()) // 2)
    y = max(0, (available.height() - size.height()) // 2)
    widget.move(x, y)
