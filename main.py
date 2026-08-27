"""
Astro HDR Stacker — entry point.

Launches the PyQt6 desktop application and installs the crash guards that keep
it alive on Windows: an exception hook (without one, PyQt6 aborts the process on
any unhandled exception inside a slot), High-DPI-aware pixmaps, and a bounded
OpenCV thread pool.
"""

import os
import sys
import traceback

# Quiet OpenCV's terminal chatter before it is imported.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

# Make the package root importable whether this is run as a script or a module.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def _configure_opencv():
    import cv2
    try:
        cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
    except Exception:
        pass
    try:
        # OpenCV defaults to one thread per logical core. On a laptop that
        # oversubscribes the CPU next to the Qt GUI thread and makes the UI
        # stutter during a merge; leaving a core free keeps it responsive.
        cores = os.cpu_count() or 4
        cv2.setNumThreads(max(1, min(cores - 1, 8)))
    except Exception:
        pass


def _install_excepthook(app):
    """
    Shows unexpected errors in a dialog instead of aborting the process.

    PyQt6 calls qFatal() when a Python exception escapes a slot, which kills the
    application instantly with no message. Catching it here means an unforeseen
    bug costs the user one dialog, not their whole editing session.
    """
    from PyQt6.QtWidgets import QMessageBox

    def hook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return

        details = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(details, file=sys.stderr)

        try:
            box = QMessageBox()
            box.setIcon(QMessageBox.Icon.Critical)
            box.setWindowTitle("Neočekávaná chyba")
            box.setText(
                "V aplikaci došlo k neočekávané chybě, ale běží dál.\n\n"
                f"{exc_type.__name__}: {exc_value}"
            )
            box.setInformativeText(
                "Vaše rozpracovaná práce zůstala zachována. Pokud se chyba opakuje, "
                "zkuste snížit pracovní rozlišení nebo zapnout režim výřezu 🎯 ROI."
            )
            box.setDetailedText(details)
            box.exec()
        except Exception:
            pass  # never let the error handler itself take the app down

    sys.excepthook = hook


def main() -> int:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QGuiApplication
    from PyQt6.QtWidgets import QApplication

    # Crisp rendering on the scaled displays typical of modern laptops.
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    _configure_opencv()

    app = QApplication(sys.argv)
    app.setApplicationName("Astro HDR Stacker")
    app.setApplicationDisplayName("Astro HDR Stacker — Solar Eclipse & Exposure Fusion Studio")
    app.setOrganizationName("Astro HDR Stacker")

    _install_excepthook(app)

    from gui.main_window import MainWindow
    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
