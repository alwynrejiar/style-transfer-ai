# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Style Transfer AI (Stylomex) — Microsoft Store build.

Usage:
    pyinstaller msstore/stylomex.spec

Produces:
    dist/Stylomex/Stylomex.exe       (GUI — no console window)
    dist/Stylomex/StylomexCLI.exe    (CLI — runs in terminal)
"""

import os
import sys
import importlib

block_cipher = None

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

# ---------- Locate spaCy model data ----------
import spacy
spacy_path = os.path.dirname(spacy.__file__)

# Try to find en_core_web_sm
try:
    import en_core_web_sm
    spacy_model_path = os.path.dirname(en_core_web_sm.__file__)
    spacy_model_datas = [(spacy_model_path, "en_core_web_sm")]
except ImportError:
    spacy_model_datas = []
    print("WARNING: en_core_web_sm not found — deep stylometry will be unavailable in the build.")

# ---------- Locate customtkinter theme data ----------
import customtkinter
ctk_path = os.path.dirname(customtkinter.__file__)

# ---------- Analysis --------------------------
# Shared analysis for both GUI and CLI entry points
a = Analysis(
    [
        os.path.join(ROOT, "scripts", "run_gui.py"),
        os.path.join(ROOT, "scripts", "run.py"),
    ],
    pathex=[ROOT],
    binaries=[],
    datas=[
        # Application source
        (os.path.join(ROOT, "src"), "src"),
        (os.path.join(ROOT, "config"), "config"),
        (os.path.join(ROOT, "default text"), "default text"),
        (os.path.join(ROOT, "assets"), "assets"),

        # CustomTkinter themes
        (ctk_path, "customtkinter"),

        # spaCy data
        (spacy_path, "spacy"),
    ] + spacy_model_datas,
    hiddenimports=[
        # Core app
        "src", "src.main", "src.gui", "src.gui.app", "src.gui.styles", "src.gui.utils",
        "src.config", "src.config.settings",
        "src.analysis", "src.analysis.analyzer", "src.analysis.metrics",
        "src.generation", "src.generation.content_generator", "src.generation.style_transfer",
        "src.storage", "src.storage.local_storage",
        "src.models", "src.models.ollama_client", "src.models.openai_client", "src.models.gemini_client",
        "src.menu", "src.menu.main_menu", "src.menu.model_selection", "src.menu.navigation",
        "src.utils",

        # GUI frameworks
        "customtkinter", "tkinter", "tkinter.filedialog",
        "matplotlib", "matplotlib.backends.backend_tkagg",
        "PIL", "PIL.Image", "PIL.ImageTk",

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
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest", "sphinx", "notebook", "jupyter",
        "IPython", "ipykernel", "ipywidgets",
        "streamlit",  # exclude web GUI — desktop only
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ---------- GUI executable (no console) ----------
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
    console=False,              # NO console window for GUI app
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
