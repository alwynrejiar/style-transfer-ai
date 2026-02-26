"""
Desktop GUI application for Style Transfer AI.

Provides a full-featured CustomTkinter interface covering:
- Style analysis with cognitive-load optimization
- Style transfer / content restyling
- Profile management (load, view, delete)
- Cognitive bridging (standalone analogy engine)
- Settings (model selection, analogy domain)

Launched via ``scripts/run_gui.py`` or directly:
    python -m src.gui.app
"""

from __future__ import annotations

import json
import os
import sys
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox
from typing import Optional

import customtkinter as ctk

# ---------------------------------------------------------------------------
# Resolve project root so relative imports work even when frozen
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config.settings import (
    ANALOGY_AUGMENTATION_ENABLED,
    ANALOGY_DOMAINS,
    APPLICATION_NAME,
    AVAILABLE_MODELS,
    CONCEPTUAL_DENSITY_THRESHOLD,
    DEFAULT_ANALOGY_DOMAIN,
    VERSION,
)
from src.utils.text_processing import extract_basic_stats, read_text_file

# ---------------------------------------------------------------------------
# Appearance defaults
# ---------------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

_PAD = 10
_FONT_HEADING = ("Segoe UI", 18, "bold")
_FONT_SUBHEADING = ("Segoe UI", 14, "bold")
_FONT_BODY = ("Segoe UI", 12)
_FONT_MONO = ("Consolas", 11)


# ═══════════════════════════════════════════════════════════════════════════
# Main Application
# ═══════════════════════════════════════════════════════════════════════════

class StyleTransferApp:
    """CustomTkinter desktop application for Style Transfer AI."""

    def __init__(self) -> None:
        self.root = ctk.CTk()
        self.root.title(f"{APPLICATION_NAME} v{VERSION}")
        self.root.geometry("1100x750")
        self.root.minsize(900, 600)

        # Shared state -------------------------------------------------------
        self._model_var = ctk.StringVar(value="gemma3:1b")
        self._use_local_var = ctk.BooleanVar(value=True)
        self._analogy_enabled_var = ctk.BooleanVar(value=ANALOGY_AUGMENTATION_ENABLED)
        self._analogy_domain_var = ctk.StringVar(value=DEFAULT_ANALOGY_DOMAIN)

        self._profiles_dir = os.path.join(_PROJECT_ROOT, "stylometry fingerprints")
        os.makedirs(self._profiles_dir, exist_ok=True)

        # Build UI -----------------------------------------------------------
        self._build_sidebar()
        self._build_main_area()

        # Start on the Home tab
        self._show_page("home")

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_sidebar(self) -> None:
        self.sidebar = ctk.CTkFrame(self.root, width=200, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        logo = ctk.CTkLabel(self.sidebar, text="Stylomex", font=_FONT_HEADING)
        logo.pack(pady=(20, 5))
        ver = ctk.CTkLabel(self.sidebar, text=f"v{VERSION}", font=("Segoe UI", 10))
        ver.pack(pady=(0, 20))

        self._nav_buttons: dict[str, ctk.CTkButton] = {}
        pages = [
            ("Home", "home"),
            ("Analyze Style", "analyze"),
            ("Cognitive Bridging", "bridging"),
            ("Style Transfer", "transfer"),
            ("Profiles", "profiles"),
            ("Settings", "settings"),
        ]
        for label, key in pages:
            btn = ctk.CTkButton(
                self.sidebar,
                text=label,
                command=lambda k=key: self._show_page(k),
                anchor="w",
                font=_FONT_BODY,
            )
            btn.pack(fill="x", padx=_PAD, pady=3)
            self._nav_buttons[key] = btn

    def _build_main_area(self) -> None:
        self.main_frame = ctk.CTkFrame(self.root, corner_radius=0, fg_color="transparent")
        self.main_frame.pack(side="right", fill="both", expand=True)

        # Container that swaps page content
        self.page_container = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.page_container.pack(fill="both", expand=True, padx=_PAD, pady=_PAD)

    def _clear_page(self) -> None:
        for w in self.page_container.winfo_children():
            w.destroy()

    def _show_page(self, key: str) -> None:
        # Highlight active nav button
        for k, btn in self._nav_buttons.items():
            if k == key:
                btn.configure(fg_color=("gray75", "gray25"))
            else:
                btn.configure(fg_color=("gray90", "gray14"))

        self._clear_page()
        builder = {
            "home": self._page_home,
            "analyze": self._page_analyze,
            "bridging": self._page_bridging,
            "transfer": self._page_transfer,
            "profiles": self._page_profiles,
            "settings": self._page_settings,
        }.get(key, self._page_home)
        builder()

    # ==================================================================
    # PAGE: Home
    # ==================================================================

    def _page_home(self) -> None:
        ctk.CTkLabel(self.page_container, text="Welcome to Stylomex", font=_FONT_HEADING).pack(pady=(10, 5))
        ctk.CTkLabel(
            self.page_container,
            text="Advanced Stylometry & Text Rewrite Engine",
            font=_FONT_SUBHEADING,
        ).pack(pady=(0, 20))

        info = ctk.CTkTextbox(self.page_container, height=200, font=_FONT_BODY, wrap="word")
        info.pack(fill="x", padx=20)
        info.insert("1.0", (
            "Stylomex lets you:\n\n"
            "  \u2022 Analyze Style — Discover the linguistic DNA of any text.\n"
            "  \u2022 Cognitive Bridging — Generate contextual analogies for dense passages.\n"
            "  \u2022 Style Transfer — Rewrite content to match a target style profile.\n"
            "  \u2022 Profiles — Manage saved stylometric fingerprints.\n"
            "  \u2022 Settings — Choose model, analogy domain, and appearance.\n\n"
            "Use the sidebar to navigate."
        ))
        info.configure(state="disabled")

        # Quick stats
        profile_count = len([f for f in os.listdir(self._profiles_dir) if f.endswith(".json")])
        stats_frame = ctk.CTkFrame(self.page_container)
        stats_frame.pack(pady=20)
        ctk.CTkLabel(stats_frame, text=f"Saved Profiles: {profile_count}", font=_FONT_BODY).pack(padx=20, pady=5)
        ctk.CTkLabel(stats_frame, text=f"Model: {self._model_var.get()}", font=_FONT_BODY).pack(padx=20, pady=5)

    # ==================================================================
    # PAGE: Analyze Style
    # ==================================================================

    def _page_analyze(self) -> None:
        ctk.CTkLabel(self.page_container, text="Analyze Style", font=_FONT_HEADING).pack(pady=(5, 10))

        # Scrollable content
        scroll = ctk.CTkScrollableFrame(self.page_container)
        scroll.pack(fill="both", expand=True)

        # --- Input ---
        ctk.CTkLabel(scroll, text="1. Input Text", font=_FONT_SUBHEADING).pack(anchor="w", pady=(5, 2))
        self._analyze_text = ctk.CTkTextbox(scroll, height=160, font=_FONT_MONO, wrap="word")
        self._analyze_text.pack(fill="x", pady=(0, 5))

        btn_load = ctk.CTkButton(scroll, text="Load from file\u2026", command=self._load_text_for_analysis)
        btn_load.pack(anchor="w", pady=(0, 10))

        # --- Cognitive Load Options ---
        ctk.CTkLabel(scroll, text="2. Cognitive Load Optimization", font=_FONT_SUBHEADING).pack(anchor="w", pady=(5, 2))
        opt_frame = ctk.CTkFrame(scroll)
        opt_frame.pack(fill="x", pady=(0, 10))

        self._analyze_analogy_sw = ctk.CTkSwitch(
            opt_frame,
            text="Auto-Inject Contextual Analogies",
            variable=self._analogy_enabled_var,
            font=_FONT_BODY,
        )
        self._analyze_analogy_sw.pack(anchor="w", padx=10, pady=5)

        domain_labels = [v["label"] for v in ANALOGY_DOMAINS.values()]
        self._analyze_domain_menu = ctk.CTkOptionMenu(
            opt_frame,
            values=domain_labels,
            variable=self._analogy_domain_var,
            font=_FONT_BODY,
        )
        # Initialise to current domain label
        current_label = ANALOGY_DOMAINS.get(self._analogy_domain_var.get(), {}).get("label", "General Simplification")
        self._analogy_domain_var.set(current_label)
        self._analyze_domain_menu.pack(anchor="w", padx=10, pady=(0, 5))

        # --- Run ---
        self._analyze_btn = ctk.CTkButton(
            scroll, text="Run Analysis", font=_FONT_BODY, command=self._run_analysis
        )
        self._analyze_btn.pack(pady=10)

        self._analyze_progress = ctk.CTkProgressBar(scroll, mode="indeterminate")
        self._analyze_progress.pack(fill="x", padx=20, pady=(0, 5))
        self._analyze_progress.pack_forget()  # hidden initially

        # --- Results ---
        ctk.CTkLabel(scroll, text="3. Results", font=_FONT_SUBHEADING).pack(anchor="w", pady=(10, 2))
        self._analyze_result_box = ctk.CTkTextbox(scroll, height=300, font=_FONT_MONO, wrap="word")
        self._analyze_result_box.pack(fill="both", expand=True, pady=(0, 10))

    def _load_text_for_analysis(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt *.md"), ("All files", "*.*")]
        )
        if path:
            content = read_text_file(path)
            self._analyze_text.delete("1.0", "end")
            self._analyze_text.insert("1.0", content)

    def _run_analysis(self) -> None:
        text = self._analyze_text.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("No text", "Please enter or load text to analyze.")
            return
        self._analyze_btn.configure(state="disabled")
        self._analyze_progress.pack(fill="x", padx=20, pady=(0, 5))
        self._analyze_progress.start()
        self._analyze_result_box.delete("1.0", "end")
        self._analyze_result_box.insert("1.0", "Analyzing\u2026 please wait.\n")

        # Resolve analogy domain key from label
        domain_key = self._resolve_domain_key(self._analogy_domain_var.get())

        threading.Thread(
            target=self._analysis_worker,
            args=(text, self._analogy_enabled_var.get(), domain_key),
            daemon=True,
        ).start()

    def _analysis_worker(self, text: str, analogy_on: bool, domain: str) -> None:
        try:
            from src.analysis.analyzer import create_enhanced_style_profile

            input_data = {
                "type": "custom_text",
                "text": text,
                "word_count": len(text.split()),
            }
            profile = create_enhanced_style_profile(
                input_data,
                use_local=self._use_local_var.get(),
                model_name=self._model_var.get() if self._use_local_var.get() else None,
                processing_mode="enhanced",
                analogy_augmentation=analogy_on,
                analogy_domain=domain,
            )
            self.root.after(0, self._display_analysis_result, profile)
        except Exception as exc:
            self.root.after(0, self._display_analysis_error, str(exc))

    def _display_analysis_result(self, profile: dict) -> None:
        self._analyze_progress.stop()
        self._analyze_progress.pack_forget()
        self._analyze_btn.configure(state="normal")

        self._analyze_result_box.delete("1.0", "end")
        lines: list[str] = []

        lines.append("=" * 60)
        lines.append("ANALYSIS COMPLETE")
        lines.append("=" * 60)

        meta = profile.get("metadata", {})
        lines.append(f"Model: {meta.get('model_used', 'N/A')}")
        lines.append(f"Mode:  {meta.get('processing_mode', 'N/A')}")
        lines.append(f"Date:  {meta.get('analysis_date', 'N/A')}")
        lines.append("")

        # Text statistics
        stats = profile.get("text_statistics", {})
        if stats:
            lines.append("--- Text Statistics ---")
            lines.append(f"Word count:        {stats.get('word_count', 'N/A')}")
            lines.append(f"Sentence count:    {stats.get('sentence_count', 'N/A')}")
            lines.append(f"Lexical diversity: {stats.get('lexical_diversity', 'N/A')}")
            lines.append("")

        # Readability
        rm = profile.get("readability_metrics", {})
        if rm:
            lines.append("--- Readability ---")
            lines.append(f"Flesch Reading Ease: {rm.get('flesch_reading_ease', 'N/A')}")
            lines.append(f"Grade Level:         {rm.get('flesch_kincaid_grade', 'N/A')}")
            lines.append(f"Coleman-Liau Index:  {rm.get('coleman_liau_index', 'N/A')}")
            lines.append("")

        # Deep stylometry highlights
        ds = profile.get("deep_stylometry", {})
        if ds:
            lines.append("--- Deep Stylometry ---")
            sl = ds.get("sentence_length_distribution", {})
            if sl.get("mean"):
                lines.append(f"Avg sentence length: {sl['mean']} words")
            lines.append(f"Passive voice:  {round(ds.get('passive_voice_ratio', 0) * 100, 1)}%")
            lines.append(f"Contraction rate: {round(ds.get('contraction_rate', 0) * 100, 1)}%")
            lines.append("")

        # Consolidated analysis (LLM output)
        ca = profile.get("consolidated_analysis", "")
        if ca:
            lines.append("--- Stylometric Profile ---")
            lines.append(str(ca))
            lines.append("")

        # Conceptual density
        cd = profile.get("conceptual_density", {})
        if cd:
            lines.append("--- Cognitive Load Optimization ---")
            lines.append(f"Overall density:      {cd.get('overall_density', 0):.3f}")
            lines.append(f"High-density passages: {cd.get('high_density_count', 0)}")
            lines.append("")

        # Cognitive bridging
        cb = profile.get("cognitive_bridging", {})
        if cb:
            count = cb.get("analogy_count", 0)
            lines.append(f"--- Cognitive Bridging Notes ({count}) ---")
            for i, item in enumerate(cb.get("analogies", []), 1):
                preview = item["source_sentence"][:70]
                if len(item["source_sentence"]) > 70:
                    preview += "..."
                lines.append(f"  {i}. [{item['density_score']:.2f}] \"{preview}\"")
                lines.append(f"     Analogy: {item['analogy']}")
            lines.append("")

        lines.append("=" * 60)

        self._analyze_result_box.insert("1.0", "\n".join(lines))

        # Store for later saving
        self._last_profile = profile

    def _display_analysis_error(self, msg: str) -> None:
        self._analyze_progress.stop()
        self._analyze_progress.pack_forget()
        self._analyze_btn.configure(state="normal")
        self._analyze_result_box.delete("1.0", "end")
        self._analyze_result_box.insert("1.0", f"Analysis failed:\n{msg}")

    # ==================================================================
    # PAGE: Cognitive Bridging (standalone)
    # ==================================================================

    def _page_bridging(self) -> None:
        ctk.CTkLabel(self.page_container, text="Cognitive Bridging", font=_FONT_HEADING).pack(pady=(5, 10))

        scroll = ctk.CTkScrollableFrame(self.page_container)
        scroll.pack(fill="both", expand=True)

        ctk.CTkLabel(scroll, text="Paste or load text to analyze for conceptual density and generate analogies.", font=_FONT_BODY, wraplength=600).pack(anchor="w", pady=(0, 5))

        # Text input
        self._bridge_text = ctk.CTkTextbox(scroll, height=140, font=_FONT_MONO, wrap="word")
        self._bridge_text.pack(fill="x", pady=(0, 5))

        btn_load = ctk.CTkButton(scroll, text="Load from file\u2026", command=self._load_text_for_bridging)
        btn_load.pack(anchor="w", pady=(0, 10))

        # Domain selection
        domain_frame = ctk.CTkFrame(scroll)
        domain_frame.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(domain_frame, text="Analogy Domain:", font=_FONT_BODY).pack(side="left", padx=(10, 5))
        domain_labels = [v["label"] for v in ANALOGY_DOMAINS.values()]
        self._bridge_domain_menu = ctk.CTkOptionMenu(
            domain_frame, values=domain_labels, font=_FONT_BODY
        )
        current_label = ANALOGY_DOMAINS.get(DEFAULT_ANALOGY_DOMAIN, {}).get("label", "General Simplification")
        self._bridge_domain_menu.set(current_label)
        self._bridge_domain_menu.pack(side="left", padx=5)

        # Run button + progress
        self._bridge_btn = ctk.CTkButton(scroll, text="Analyze & Generate Analogies", command=self._run_bridging)
        self._bridge_btn.pack(pady=10)

        self._bridge_progress = ctk.CTkProgressBar(scroll, mode="indeterminate")
        self._bridge_progress.pack(fill="x", padx=20, pady=(0, 5))
        self._bridge_progress.pack_forget()

        # Results
        ctk.CTkLabel(scroll, text="Results", font=_FONT_SUBHEADING).pack(anchor="w", pady=(10, 2))
        self._bridge_result_box = ctk.CTkTextbox(scroll, height=300, font=_FONT_MONO, wrap="word")
        self._bridge_result_box.pack(fill="both", expand=True, pady=(0, 10))

    def _load_text_for_bridging(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt *.md"), ("All files", "*.*")]
        )
        if path:
            content = read_text_file(path)
            self._bridge_text.delete("1.0", "end")
            self._bridge_text.insert("1.0", content)

    def _run_bridging(self) -> None:
        text = self._bridge_text.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("No text", "Please enter or load text first.")
            return

        self._bridge_btn.configure(state="disabled")
        self._bridge_progress.pack(fill="x", padx=20, pady=(0, 5))
        self._bridge_progress.start()
        self._bridge_result_box.delete("1.0", "end")
        self._bridge_result_box.insert("1.0", "Computing density and generating analogies\u2026\n")

        domain_key = self._resolve_domain_key(self._bridge_domain_menu.get())

        threading.Thread(
            target=self._bridging_worker,
            args=(text, domain_key),
            daemon=True,
        ).start()

    def _bridging_worker(self, text: str, domain: str) -> None:
        try:
            from src.analysis.analogy_engine import detect_conceptual_density, AnalogyInjector

            density = detect_conceptual_density(text)
            analogy_result: Optional[dict] = None

            if density["high_density_count"] > 0:
                injector = AnalogyInjector(domain=domain)
                analogy_result = injector.augment_text(
                    text,
                    use_local=self._use_local_var.get(),
                    model_name=self._model_var.get() if self._use_local_var.get() else None,
                )

            self.root.after(0, self._display_bridging_result, density, analogy_result)
        except Exception as exc:
            self.root.after(0, self._display_bridging_error, str(exc))

    def _display_bridging_result(self, density: dict, analogy_result: Optional[dict]) -> None:
        self._bridge_progress.stop()
        self._bridge_progress.pack_forget()
        self._bridge_btn.configure(state="normal")

        self._bridge_result_box.delete("1.0", "end")
        lines: list[str] = []

        lines.append("=" * 60)
        lines.append("CONCEPTUAL DENSITY REPORT")
        lines.append("=" * 60)
        lines.append(f"Overall density:      {density['overall_density']:.3f}")
        lines.append(f"Total sentences:      {len(density['sentence_scores'])}")
        lines.append(f"High-density passages: {density['high_density_count']}")
        lines.append("")

        # Per-sentence breakdown (top dense ones)
        dense = [s for s in density["sentence_scores"] if s["density"] >= CONCEPTUAL_DENSITY_THRESHOLD]
        if dense:
            lines.append("--- Dense Passages ---")
            for i, s in enumerate(dense, 1):
                preview = s["text"][:80] + ("..." if len(s["text"]) > 80 else "")
                lines.append(f"  {i}. [{s['density']:.3f}] \"{preview}\"")
            lines.append("")

        if analogy_result and analogy_result["analogy_count"] > 0:
            lines.append("=" * 60)
            lines.append("COGNITIVE BRIDGING NOTES")
            lines.append("=" * 60)
            for i, item in enumerate(analogy_result["analogies"], 1):
                lines.append(f"  {i}. Dense passage (score {item['density_score']:.2f}):")
                preview = item["source_sentence"][:80]
                if len(item["source_sentence"]) > 80:
                    preview += "..."
                lines.append(f"     \"{preview}\"")
                lines.append(f"     Analogy: {item['analogy']}")
                lines.append("")
        elif density["high_density_count"] == 0:
            lines.append("No passages exceed the density threshold \u2014 text is already accessible.")

        lines.append("=" * 60)
        self._bridge_result_box.insert("1.0", "\n".join(lines))

    def _display_bridging_error(self, msg: str) -> None:
        self._bridge_progress.stop()
        self._bridge_progress.pack_forget()
        self._bridge_btn.configure(state="normal")
        self._bridge_result_box.delete("1.0", "end")
        self._bridge_result_box.insert("1.0", f"Error:\n{msg}")

    # ==================================================================
    # PAGE: Style Transfer
    # ==================================================================

    def _page_transfer(self) -> None:
        ctk.CTkLabel(self.page_container, text="Style Transfer", font=_FONT_HEADING).pack(pady=(5, 10))

        scroll = ctk.CTkScrollableFrame(self.page_container)
        scroll.pack(fill="both", expand=True)

        # Source text
        ctk.CTkLabel(scroll, text="1. Source Content", font=_FONT_SUBHEADING).pack(anchor="w", pady=(5, 2))
        self._transfer_text = ctk.CTkTextbox(scroll, height=140, font=_FONT_MONO, wrap="word")
        self._transfer_text.pack(fill="x", pady=(0, 5))

        btn_load = ctk.CTkButton(scroll, text="Load from file\u2026", command=self._load_text_for_transfer)
        btn_load.pack(anchor="w", pady=(0, 10))

        # Target style profile
        ctk.CTkLabel(scroll, text="2. Target Style Profile", font=_FONT_SUBHEADING).pack(anchor="w", pady=(5, 2))
        profile_files = [f for f in os.listdir(self._profiles_dir) if f.endswith(".json")]
        if not profile_files:
            profile_files = ["(no profiles saved)"]
        self._transfer_profile_menu = ctk.CTkOptionMenu(
            scroll, values=profile_files, font=_FONT_BODY
        )
        self._transfer_profile_menu.pack(anchor="w", pady=(0, 10))

        # Transfer settings
        ctk.CTkLabel(scroll, text="3. Transfer Intensity", font=_FONT_SUBHEADING).pack(anchor="w", pady=(5, 2))
        self._transfer_intensity = ctk.CTkSlider(scroll, from_=0.1, to=1.0, number_of_steps=9)
        self._transfer_intensity.set(0.8)
        self._transfer_intensity.pack(fill="x", pady=(0, 10))

        # Run
        self._transfer_btn = ctk.CTkButton(scroll, text="Transfer Style", command=self._run_transfer)
        self._transfer_btn.pack(pady=10)

        self._transfer_progress = ctk.CTkProgressBar(scroll, mode="indeterminate")
        self._transfer_progress.pack(fill="x", padx=20, pady=(0, 5))
        self._transfer_progress.pack_forget()

        # Results
        ctk.CTkLabel(scroll, text="4. Result", font=_FONT_SUBHEADING).pack(anchor="w", pady=(10, 2))
        self._transfer_result_box = ctk.CTkTextbox(scroll, height=300, font=_FONT_MONO, wrap="word")
        self._transfer_result_box.pack(fill="both", expand=True, pady=(0, 10))

    def _load_text_for_transfer(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt *.md"), ("All files", "*.*")]
        )
        if path:
            content = read_text_file(path)
            self._transfer_text.delete("1.0", "end")
            self._transfer_text.insert("1.0", content)

    def _run_transfer(self) -> None:
        text = self._transfer_text.get("1.0", "end").strip()
        profile_file = self._transfer_profile_menu.get()
        if not text:
            messagebox.showwarning("No text", "Please enter or load source content.")
            return
        if profile_file == "(no profiles saved)":
            messagebox.showwarning("No profile", "Save a style profile first via Analyze Style.")
            return

        self._transfer_btn.configure(state="disabled")
        self._transfer_progress.pack(fill="x", padx=20, pady=(0, 5))
        self._transfer_progress.start()
        self._transfer_result_box.delete("1.0", "end")
        self._transfer_result_box.insert("1.0", "Transferring style\u2026 please wait.\n")

        profile_path = os.path.join(self._profiles_dir, profile_file)
        intensity = self._transfer_intensity.get()

        threading.Thread(
            target=self._transfer_worker,
            args=(text, profile_path, intensity),
            daemon=True,
        ).start()

    def _transfer_worker(self, text: str, profile_path: str, intensity: float) -> None:
        try:
            from src.generation.style_transfer import StyleTransfer

            with open(profile_path, "r", encoding="utf-8") as f:
                target_profile = json.load(f)

            engine = StyleTransfer()
            result = engine.transfer_style(
                original_content=text,
                target_style_profile=target_profile,
                transfer_type="direct_transfer",
                intensity=intensity,
                use_local=self._use_local_var.get(),
                model_name=self._model_var.get() if self._use_local_var.get() else None,
            )
            self.root.after(0, self._display_transfer_result, result)
        except Exception as exc:
            self.root.after(0, self._display_transfer_error, str(exc))

    def _display_transfer_result(self, result: dict) -> None:
        self._transfer_progress.stop()
        self._transfer_progress.pack_forget()
        self._transfer_btn.configure(state="normal")
        self._transfer_result_box.delete("1.0", "end")

        if isinstance(result, dict):
            transferred = result.get("transferred_content", result.get("content", json.dumps(result, indent=2)))
            self._transfer_result_box.insert("1.0", str(transferred))
        else:
            self._transfer_result_box.insert("1.0", str(result))

    def _display_transfer_error(self, msg: str) -> None:
        self._transfer_progress.stop()
        self._transfer_progress.pack_forget()
        self._transfer_btn.configure(state="normal")
        self._transfer_result_box.delete("1.0", "end")
        self._transfer_result_box.insert("1.0", f"Transfer failed:\n{msg}")

    # ==================================================================
    # PAGE: Profiles
    # ==================================================================

    def _page_profiles(self) -> None:
        ctk.CTkLabel(self.page_container, text="Saved Profiles", font=_FONT_HEADING).pack(pady=(5, 10))

        toolbar = ctk.CTkFrame(self.page_container)
        toolbar.pack(fill="x", pady=(0, 10))
        ctk.CTkButton(toolbar, text="Refresh", command=lambda: self._show_page("profiles")).pack(side="left", padx=5)
        ctk.CTkButton(toolbar, text="Open Folder", command=self._open_profiles_folder).pack(side="left", padx=5)

        scroll = ctk.CTkScrollableFrame(self.page_container)
        scroll.pack(fill="both", expand=True)

        profiles = sorted(
            [f for f in os.listdir(self._profiles_dir) if f.endswith(".json")],
            key=lambda f: os.path.getmtime(os.path.join(self._profiles_dir, f)),
            reverse=True,
        )
        if not profiles:
            ctk.CTkLabel(scroll, text="No profiles saved yet. Run an analysis first!", font=_FONT_BODY).pack(pady=20)
            return

        for fname in profiles:
            fpath = os.path.join(self._profiles_dir, fname)
            size_kb = os.path.getsize(fpath) / 1024
            mod_time = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M")

            row = ctk.CTkFrame(scroll)
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=fname, font=_FONT_BODY, anchor="w").pack(side="left", padx=10, fill="x", expand=True)
            ctk.CTkLabel(row, text=f"{size_kb:.1f} KB", font=("Segoe UI", 10)).pack(side="left", padx=5)
            ctk.CTkLabel(row, text=mod_time, font=("Segoe UI", 10)).pack(side="left", padx=5)
            ctk.CTkButton(
                row, text="View", width=60,
                command=lambda p=fpath: self._view_profile(p),
            ).pack(side="left", padx=2)
            ctk.CTkButton(
                row, text="Delete", width=60, fg_color="darkred",
                command=lambda p=fpath, n=fname: self._delete_profile(p, n),
            ).pack(side="left", padx=2)

    def _open_profiles_folder(self) -> None:
        os.startfile(self._profiles_dir)

    def _view_profile(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Show in a top-level viewer window
            win = ctk.CTkToplevel(self.root)
            win.title(os.path.basename(path))
            win.geometry("700x500")
            box = ctk.CTkTextbox(win, font=_FONT_MONO, wrap="word")
            box.pack(fill="both", expand=True, padx=10, pady=10)
            box.insert("1.0", json.dumps(data, indent=2))
            box.configure(state="disabled")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load profile:\n{exc}")

    def _delete_profile(self, path: str, name: str) -> None:
        if messagebox.askyesno("Confirm Delete", f"Delete profile '{name}'?"):
            try:
                os.remove(path)
                # Also remove accompanying .txt if present
                txt_path = path.rsplit(".", 1)[0] + ".txt"
                if os.path.exists(txt_path):
                    os.remove(txt_path)
                self._show_page("profiles")
            except Exception as exc:
                messagebox.showerror("Error", f"Failed to delete:\n{exc}")

    # ==================================================================
    # PAGE: Settings
    # ==================================================================

    def _page_settings(self) -> None:
        ctk.CTkLabel(self.page_container, text="Settings", font=_FONT_HEADING).pack(pady=(5, 10))

        scroll = ctk.CTkScrollableFrame(self.page_container)
        scroll.pack(fill="both", expand=True)

        # Model selection
        ctk.CTkLabel(scroll, text="Model Configuration", font=_FONT_SUBHEADING).pack(anchor="w", pady=(10, 2))

        model_frame = ctk.CTkFrame(scroll)
        model_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(model_frame, text="Use Local (Ollama):", font=_FONT_BODY).pack(anchor="w", padx=10, pady=(5, 0))
        ctk.CTkSwitch(model_frame, text="Local inference", variable=self._use_local_var, font=_FONT_BODY).pack(anchor="w", padx=10, pady=2)

        ctk.CTkLabel(model_frame, text="Model Name:", font=_FONT_BODY).pack(anchor="w", padx=10, pady=(5, 0))
        model_names = list(AVAILABLE_MODELS.keys())
        ctk.CTkOptionMenu(
            model_frame, values=model_names, variable=self._model_var, font=_FONT_BODY
        ).pack(anchor="w", padx=10, pady=2)

        for name, info in AVAILABLE_MODELS.items():
            ctk.CTkLabel(
                model_frame,
                text=f"  {name}: {info['description']} ({info['type']})",
                font=("Segoe UI", 10),
            ).pack(anchor="w", padx=20)

        # Analogy defaults
        ctk.CTkLabel(scroll, text="Analogy Engine Defaults", font=_FONT_SUBHEADING).pack(anchor="w", pady=(15, 2))
        analogy_frame = ctk.CTkFrame(scroll)
        analogy_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkSwitch(
            analogy_frame,
            text="Enable analogy augmentation by default",
            variable=self._analogy_enabled_var,
            font=_FONT_BODY,
        ).pack(anchor="w", padx=10, pady=5)

        ctk.CTkLabel(analogy_frame, text="Default Domain:", font=_FONT_BODY).pack(anchor="w", padx=10, pady=(5, 0))
        domain_labels = [v["label"] for v in ANALOGY_DOMAINS.values()]
        self._settings_domain_menu = ctk.CTkOptionMenu(
            analogy_frame, values=domain_labels, font=_FONT_BODY,
            command=self._on_settings_domain_change,
        )
        current_label = ANALOGY_DOMAINS.get(self._analogy_domain_var.get(), {}).get("label", "General Simplification")
        self._settings_domain_menu.set(current_label)
        self._settings_domain_menu.pack(anchor="w", padx=10, pady=2)

        # Appearance
        ctk.CTkLabel(scroll, text="Appearance", font=_FONT_SUBHEADING).pack(anchor="w", pady=(15, 2))
        appear_frame = ctk.CTkFrame(scroll)
        appear_frame.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(appear_frame, text="Theme:", font=_FONT_BODY).pack(anchor="w", padx=10, pady=(5, 0))
        ctk.CTkOptionMenu(
            appear_frame,
            values=["Dark", "Light", "System"],
            command=lambda v: ctk.set_appearance_mode(v.lower()),
            font=_FONT_BODY,
        ).pack(anchor="w", padx=10, pady=2)

    def _on_settings_domain_change(self, label: str) -> None:
        key = self._resolve_domain_key(label)
        self._analogy_domain_var.set(key)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_domain_key(label: str) -> str:
        """Reverse-lookup an ``ANALOGY_DOMAINS`` key from its display label."""
        for k, v in ANALOGY_DOMAINS.items():
            if v["label"] == label:
                return k
        return DEFAULT_ANALOGY_DOMAIN

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the main event loop."""
        self.root.mainloop()


# ═══════════════════════════════════════════════════════════════════════════
# Direct launch
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = StyleTransferApp()
    app.run()
