"""
Astro HDR Stacker — Entry Point.
Launches the PyQt6 Desktop Application.
"""

import sys
import os

# Ensure package root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from gui.main_window import MainWindow


def main():
    # Enable High DPI scaling attributes
    app = QApplication(sys.argv)
    app.setApplicationName("Astro HDR Stacker")
    app.setApplicationDisplayName("Astro HDR Stacker — Solar Eclipse & Exposure Fusion Studio")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
