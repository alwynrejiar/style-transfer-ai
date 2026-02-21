"""
Streamlit GUI launcher for Style Transfer AI (Stylomex).
Used by the MS Store MSIX package and for direct execution.

When frozen by PyInstaller, this script:
  1. Starts Streamlit as a subprocess on a free port
  2. Opens the user's default browser
  3. Waits for the Streamlit process to exit
"""

import os
import sys
import socket
import subprocess
import webbrowser
import time
import signal

# When frozen by PyInstaller, _MEIPASS is the temp extraction folder.
if getattr(sys, 'frozen', False):
    ROOT = os.path.dirname(sys.executable)
    BUNDLE_DIR = sys._MEIPASS
else:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    BUNDLE_DIR = ROOT

# Ensure src is importable
SRC_DIR = os.path.join(BUNDLE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if BUNDLE_DIR not in sys.path:
    sys.path.insert(0, BUNDLE_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def find_free_port():
    """Find an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    port = find_free_port()
    app_py = os.path.join(BUNDLE_DIR, "app.py")

    if not os.path.isfile(app_py):
        print(f"ERROR: app.py not found at {app_py}")
        sys.exit(1)

    url = f"http://localhost:{port}"
    print(f"Starting Stylomex on {url} ...")

    # Build Streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run", app_py,
        "--server.port", str(port),
        "--server.headless", "true",
        "--server.address", "127.0.0.1",
        "--browser.gatherUsageStats", "false",
        "--theme.base", "dark",
        "--theme.primaryColor", "#6C5CE7",
    ]

    # Start Streamlit server
    proc = subprocess.Popen(cmd)

    # Wait briefly then open browser
    time.sleep(2)
    webbrowser.open(url)

    print("Stylomex is running. Close this window or press Ctrl+C to stop.")

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down Stylomex...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
