"""
Streamlit GUI launcher for Style Transfer AI (Stylomex).
Used by the MS Store MSIX package and for direct execution.

When frozen by PyInstaller:
  - Runs Streamlit **in-process** via bootstrap.run() to avoid infinite
    recursion (sys.executable is the frozen .exe, not python.exe).
  - Opens the browser once via a background thread timer.

When running from source:
  - Spawns Streamlit as a subprocess and opens the browser.
"""

import os
import sys
import socket
import webbrowser
import multiprocessing

# Needed for PyInstaller frozen multiprocessing support
multiprocessing.freeze_support()

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


def _open_browser_once(url, port):
    """Wait for the Streamlit server to accept connections, then open browser."""
    import time
    for _ in range(30):  # Try for up to 15 seconds
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                webbrowser.open(url)
                return
        except OSError:
            time.sleep(0.5)


def main():
    port = find_free_port()
    app_py = os.path.join(BUNDLE_DIR, "app.py")

    if not os.path.isfile(app_py):
        print(f"ERROR: app.py not found at {app_py}")
        sys.exit(1)

    url = f"http://localhost:{port}"
    print(f"Starting Stylomex on {url} ...")

    flag_options = {
        "server.port": port,
        "server.headless": True,
        "server.address": "127.0.0.1",
        "browser.gatherUsageStats": False,
        "theme.base": "dark",
        "theme.primaryColor": "#6C5CE7",
    }

    if getattr(sys, 'frozen', False):
        # ── Frozen (PyInstaller / MSIX) ──────────────────────────────
        # Run Streamlit in-process.  Using subprocess here would
        # re-launch this .exe, causing infinite recursion + tab spam.
        import threading
        threading.Thread(
            target=_open_browser_once, args=(url, port), daemon=True
        ).start()

        from streamlit.web import bootstrap
        bootstrap.run(app_py, False, [], flag_options)
    else:
        # ── Development (running from source) ────────────────────────
        import subprocess
        import time

        cmd = [
            sys.executable, "-m", "streamlit", "run", app_py,
            "--server.port", str(port),
            "--server.headless", "true",
            "--server.address", "127.0.0.1",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "dark",
            "--theme.primaryColor", "#6C5CE7",
        ]

        proc = subprocess.Popen(cmd)
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
