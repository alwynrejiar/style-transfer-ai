# run.py
"""
Entry point to run the Style Transfer AI CLI analyzer.
Updated to use the new modular architecture.
Works both in development and when frozen by PyInstaller for MS Store.
"""

import os
import sys

# When frozen by PyInstaller, _MEIPASS is the temp extraction folder.
if getattr(sys, 'frozen', False):
    ROOT = os.path.dirname(sys.executable)
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

from src.main import cli_entry_point

if __name__ == "__main__":
    cli_entry_point()
