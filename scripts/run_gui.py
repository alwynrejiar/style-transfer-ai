"""
GUI entry point for Style Transfer AI.
Ensures dependencies are present and launches the CustomTkinter app.
Works both in development and when frozen by PyInstaller for MS Store.
"""

import os
import sys


# When frozen by PyInstaller, _MEIPASS is the temp extraction folder.
# In development, use the script's parent directory (project root).
if getattr(sys, 'frozen', False):
    ROOT = os.path.dirname(sys.executable)
    # PyInstaller bundles data relative to _MEIPASS
    BUNDLE_DIR = sys._MEIPASS
else:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    BUNDLE_DIR = ROOT

SRC_DIR = os.path.join(BUNDLE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if BUNDLE_DIR not in sys.path:
    sys.path.insert(0, BUNDLE_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import customtkinter  # noqa: F401
    import matplotlib  # noqa: F401
except ImportError:
    print("Error: GUI dependencies missing.")
    print("Install with: pip install customtkinter matplotlib pillow")
    sys.exit(1)

from src.gui.app import StyleTransferApp


def main():
    app = StyleTransferApp()
    app.run()


if __name__ == "__main__":
    main()
