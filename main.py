"""
Astro HDR Stacker — Entry Point.
Launches the PyQt6 Desktop Application.
"""

import sys
import os

# Suppress noisy OpenCV terminal messages
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
try:
    cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
except Exception:
    pass

# Ensure package root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Astro HDR Stacker")
    app.setApplicationDisplayName("Astro HDR Stacker — Solar Eclipse & Exposure Fusion Studio")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
