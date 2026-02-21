# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Style Transfer AI (Stylomex) — Microsoft Store build.

Usage:
    pyinstaller msstore/stylomex.spec

Produces:
    dist/Stylomex/Stylomex.exe       (Streamlit launcher — opens browser)
    dist/Stylomex/StylomexCLI.exe    (CLI — runs in terminal)
"""

import os
import sys

block_cipher = None

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

# ---------- Locate spaCy model data ----------
import spacy
spacy_path = os.path.dirname(spacy.__file__)

try:
    import en_core_web_sm
    spacy_model_path = os.path.dirname(en_core_web_sm.__file__)
    spacy_model_datas = [(spacy_model_path, "en_core_web_sm")]
except ImportError:
    spacy_model_datas = []
    print("WARNING: en_core_web_sm not found — deep stylometry will be unavailable in the build.")

# ---------- Locate streamlit data ----------
import streamlit
streamlit_path = os.path.dirname(streamlit.__file__)

# ---------- Analysis --------------------------
a = Analysis(
    [
        os.path.join(ROOT, "scripts", "run_streamlit.py"),
        os.path.join(ROOT, "scripts", "run.py"),
    ],
    pathex=[ROOT],
    binaries=[],
    datas=[
        # Application source
        (os.path.join(ROOT, "src"), "src"),
        (os.path.join(ROOT, "gui"), "gui"),
        (os.path.join(ROOT, "app.py"), "."),
        (os.path.join(ROOT, "config"), "config"),
        (os.path.join(ROOT, "data"), "data"),
        (os.path.join(ROOT, "assets"), "assets"),

        # Streamlit static files (required for the web UI)
        (streamlit_path, "streamlit"),

        # spaCy data
        (spacy_path, "spacy"),
    ] + spacy_model_datas,
    hiddenimports=[
        # Core app
        "src", "src.main",
        "src.config", "src.config.settings",
        "src.analysis", "src.analysis.analyzer", "src.analysis.metrics",
        "src.generation", "src.generation.content_generator", "src.generation.style_transfer",
        "src.storage", "src.storage.local_storage",
        "src.models", "src.models.ollama_client", "src.models.openai_client", "src.models.gemini_client",
        "src.menu", "src.menu.main_menu", "src.menu.model_selection", "src.menu.navigation",
        "src.utils",

        # Streamlit GUI modules
        "gui", "gui.home", "gui.analyze", "gui.transfer", "gui.profiles", "gui.settings",

        # Streamlit and web dependencies
        "streamlit", "streamlit.web", "streamlit.web.cli",
        "streamlit.runtime", "streamlit.runtime.scriptrunner",
        "altair", "plotly",
        "tornado", "tornado.web", "tornado.websocket",

        # NLP / analysis
        "spacy", "spacy.lang.en",
        "en_core_web_sm",
        "numpy",

        # Network / API
        "requests", "urllib3", "certifi", "charset_normalizer", "idna",

        # Optional API clients
        "openai", "google.generativeai",

        # Stdlib commonly missed
        "json", "re", "collections", "statistics", "math", "datetime",
        "socket", "webbrowser", "subprocess", "signal",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest", "sphinx", "notebook", "jupyter",
        "IPython", "ipykernel", "ipywidgets",
        "customtkinter",
        "matplotlib",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ---------- Streamlit launcher (no console window) ----------
gui_exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Stylomex",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,              # No console — opens browser with Streamlit
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(ROOT, "msstore", "icons", "app.ico"),
    version_file=os.path.join(ROOT, "msstore", "version_info.txt"),
)

# ---------- CLI executable (with console) ----------
cli_exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="StylomexCLI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,               # Console window for CLI
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(ROOT, "msstore", "icons", "app.ico"),
    version_file=os.path.join(ROOT, "msstore", "version_info.txt"),
)

coll = COLLECT(
    gui_exe,
    cli_exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Stylomex",
)
