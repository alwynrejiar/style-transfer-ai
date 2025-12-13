"""
GUI entry point for Style Transfer AI.
Ensures dependencies are present and launches the CustomTkinter app.
"""

import os
import sys


ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
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
