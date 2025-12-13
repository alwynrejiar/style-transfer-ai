"""
Full GUI application for Style Transfer AI.
Implements dashboard analysis, generation studio, profile hub, and settings.
"""
import os
import json
import customtkinter as ctk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

from src.config.settings import AVAILABLE_MODELS, PROCESSING_MODES
from src.analysis.analyzer import analyze_style
from src.analysis.metrics import analyze_text_statistics, calculate_readability_metrics
from src.storage.local_storage import list_local_profiles, load_local_profile, save_style_profile_locally, cleanup_old_reports
from src.generation.content_generator import ContentGenerator
from src.generation.style_transfer import StyleTransfer
from src.models.ollama_client import check_ollama_connection
from src.models.openai_client import setup_openai_client
from src.models.gemini_client import setup_gemini_client
from .styles import COLORS, FONTS, DIMENSIONS
from .utils import ThreadedTask, safe_read_text, compute_metrics, radar_from_metrics

# Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")


class StyleTransferApp(ctk.CTk):
    """Main GUI window."""

    def __init__(self):
        super().__init__()
        self.title("Style Transfer AI - GUI")
        self.geometry(f"{DIMENSIONS['window_width']}x{DIMENSIONS['window_height']}")
        self.minsize(1100, 760)

        # State
        self.model_var = ctk.StringVar(value="gpt-oss:20b")
        self.turbo_var = ctk.BooleanVar(value=False)
        self.api_keys = {"openai": "", "gemini": ""}
        self.active_profile = None
        self.active_profile_path = None
        self.generator = ContentGenerator()
        self.transfer = StyleTransfer()

        # Layout grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._setup_sidebar()
        self._setup_main_container()
        self.show_dashboard()

    # ---------------- Sidebar -----------------
    def _setup_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=DIMENSIONS["sidebar_width"], fg_color=COLORS["bg_darker"], corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(5, weight=1)

        ctk.CTkLabel(self.sidebar, text="Style Transfer AI", font=FONTS["header_lg"], text_color=COLORS["accent_primary"]).grid(row=0, column=0, padx=20, pady=(20, 5), sticky="w")
        ctk.CTkLabel(self.sidebar, text="Deep Stylometry", font=FONTS["body_sm"], text_color=COLORS["text_secondary"]).grid(row=1, column=0, padx=20, pady=(0, 20), sticky="w")

        self.nav_items = ["Dashboard", "Generation", "Profiles", "Settings"]
        self.nav_buttons = []
        for idx, label in enumerate(self.nav_items):
            btn = ctk.CTkButton(
                self.sidebar,
                text=label,
                anchor="w",
                height=42,
                corner_radius=0,
                fg_color="transparent",
                hover_color=COLORS["bg_card"],
                text_color=COLORS["text_secondary"],
                font=FONTS["body"],
                command=lambda x=label: self.select_nav(x),
            )
            btn.grid(row=idx + 2, column=0, sticky="ew")
            self.nav_buttons.append(btn)

        self.status_label = ctk.CTkLabel(self.sidebar, text="Ready", font=FONTS["body_sm"], text_color=COLORS["text_secondary"])
        self.status_label.grid(row=6, column=0, padx=20, pady=20, sticky="sw")

    def select_nav(self, label):
        for btn in self.nav_buttons:
            btn.configure(fg_color="transparent", text_color=COLORS["text_secondary"])
            if btn.cget("text") == label:
                btn.configure(fg_color=COLORS["bg_card"], text_color=COLORS["text_primary"])

        if label == "Dashboard":
            self.show_dashboard()
        elif label == "Generation":
            self.show_generation()
        elif label == "Profiles":
            self.show_profiles()
        elif label == "Settings":
            self.show_settings()

    # ---------------- Main container -----------------
    def _setup_main_container(self):
        self.main = ctk.CTkFrame(self, fg_color=COLORS["bg_dark"], corner_radius=0)
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.grid_rowconfigure(0, weight=1)
        self.main.grid_columnconfigure(0, weight=1)

    def clear_main(self):
        for child in self.main.winfo_children():
            child.destroy()

    # ---------------- Dashboard -----------------
    def show_dashboard(self):
        self.clear_main()
        container = ctk.CTkFrame(self.main, fg_color=COLORS["bg_dark"])
        container.grid(row=0, column=0, sticky="nsew")
        container.grid_rowconfigure(1, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Top bar
        top = ctk.CTkFrame(container, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=16, pady=12)
        top.grid_columnconfigure(4, weight=1)

        ctk.CTkLabel(top, text="Model Selection", font=FONTS["body"], text_color=COLORS["text_secondary"]).grid(row=0, column=0, padx=(0, 8))
        model_options = list(AVAILABLE_MODELS.keys())
        self.model_menu = ctk.CTkOptionMenu(
            top,
            variable=self.model_var,
            values=model_options,
            width=180,
            fg_color=COLORS["bg_card"],
            button_color=COLORS["accent_primary"],
            button_hover_color=COLORS["success"],
            text_color=COLORS["text_primary"],
        )
        self.model_menu.grid(row=0, column=1)

        self.turbo_switch = ctk.CTkSwitch(top, text="Turbo Mode", variable=self.turbo_var, progress_color=COLORS["accent_primary"], font=FONTS["body"])
        self.turbo_switch.grid(row=0, column=2, padx=12)

        self.load_btn = ctk.CTkButton(top, text="Load Text File", command=self.load_text_file, fg_color=COLORS["bg_card"], hover_color=COLORS["bg_card"], text_color=COLORS["text_primary"], font=FONTS["body"])
        self.load_btn.grid(row=0, column=3, padx=8)

        self.analyze_btn = ctk.CTkButton(top, text="Analyze Text", fg_color=COLORS["accent_primary"], hover_color=COLORS["success"], text_color=COLORS["bg_dark"], font=FONTS["header_md"], command=self.start_analysis)
        self.analyze_btn.grid(row=0, column=4, padx=(8, 0))

        # Split content
        content = ctk.CTkFrame(container, fg_color=COLORS["bg_dark"])
        content.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(1, weight=1)

        # Left panel
        left = ctk.CTkFrame(content, fg_color=COLORS["bg_card"], corner_radius=10)
        left.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 12))
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(left, text="Input Text", font=FONTS["header_md"], text_color=COLORS["text_primary"]).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 6))
        self.input_box = ctk.CTkTextbox(left, font=FONTS["mono"], fg_color=COLORS["bg_dark"], text_color=COLORS["text_primary"], border_width=1, border_color=COLORS["border_subtle"])
        self.input_box.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 10))
        self.input_box.insert("1.0", "Paste your text here to begin analysis...")

        # Right panel
        right = ctk.CTkFrame(content, fg_color=COLORS["bg_card"], corner_radius=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(right, text="Deep Analysis Engine", font=FONTS["header_md"], text_color=COLORS["text_primary"]).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 6))

        # Chart placeholder
        self.chart_frame = ctk.CTkFrame(right, fg_color=COLORS["bg_dark"], corner_radius=8)
        self.chart_frame.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 10))
        self.chart_frame.grid_rowconfigure(0, weight=1)
        self.chart_frame.grid_columnconfigure(0, weight=1)
        self.chart_canvas = None
        self._draw_empty_radar()

        # Metrics bars
        metrics = ctk.CTkFrame(right, fg_color="transparent")
        metrics.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 12))
        self.metric_bars = {}
        self._add_metric_bar(metrics, "Flesch Reading Ease")
        self._add_metric_bar(metrics, "Flesch-Kincaid Grade")
        self._add_metric_bar(metrics, "Coleman-Liau Index")

        # Deep analysis text box
        self.analysis_output = ctk.CTkTextbox(right, height=140, fg_color=COLORS["bg_dark"], text_color=COLORS["text_primary"], font=FONTS["body_sm"], wrap="word")
        self.analysis_output.grid(row=3, column=0, sticky="nsew", padx=16, pady=(0, 16))
        self.analysis_output.insert("1.0", "Run analysis to view results here...")

        # Store references for later updates
        self.dashboard_widgets = {
            "metrics_frame": metrics,
            "right_panel": right,
        }

    def _draw_empty_radar(self):
        if self.chart_canvas:
            self.chart_canvas.get_tk_widget().destroy()
        fig = plt.Figure(figsize=(3.8, 3.0), facecolor=COLORS["bg_dark"])
        ax = fig.add_subplot(111, projection="polar")
        ax.set_facecolor(COLORS["bg_dark"])
        ax.grid(True, color=COLORS["border_subtle"])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        self.chart_canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _add_metric_bar(self, parent, label):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=4)
        lbl = ctk.CTkLabel(frame, text=label, font=FONTS["body_sm"], text_color=COLORS["text_primary"])
        lbl.pack(side="left")
        bar = ctk.CTkProgressBar(frame, height=6, progress_color=COLORS["accent_primary"], fg_color=COLORS["border_subtle"])
        bar.pack(side="left", fill="x", expand=True, padx=10)
        val = ctk.CTkLabel(frame, text="--", font=FONTS["body_sm"], text_color=COLORS["text_secondary"])
        val.pack(side="right")
        self.metric_bars[label] = (bar, val)

    def load_text_file(self):
        path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if not path:
            return
        content = safe_read_text(path)
        if content:
            self.input_box.delete("1.0", "end")
            self.input_box.insert("1.0", content)
            self.status_label.configure(text=f"Loaded {os.path.basename(path)}")
        else:
            self.status_label.configure(text="Could not read file")

    def start_analysis(self):
        text = self.input_box.get("1.0", "end").strip()
        if not text or text.startswith("Paste your text"):
            self.status_label.configure(text="Please enter text to analyze")
            return
        self.analyze_btn.configure(state="disabled", text="Analyzing...")
        self.status_label.configure(text="Running analysis...")
        task = ThreadedTask(self._run_analysis_task, self._on_analysis_complete, text)
        task.start()

    def _run_analysis_task(self, text):
        stats = analyze_text_statistics(text)
        readability = calculate_readability_metrics(text)
        use_local, model_name, api_type, api_client = self._resolve_model()
        mode = "enhanced" if not self.turbo_var.get() else "statistical"
        deep_result = analyze_style(
            text_to_analyze=text,
            use_local=use_local,
            model_name=model_name,
            api_type=api_type,
            api_client=api_client,
            processing_mode=mode,
            user_profile=None,
        )
        return {"stats": stats, "readability": readability, "deep": deep_result}

    def _on_analysis_complete(self, result):
        self.analyze_btn.configure(state="normal", text="Analyze Text")
        if isinstance(result, dict) and result.get("error"):
            self.status_label.configure(text=f"Error: {result['error']}")
            return
        stats = result.get("stats", {})
        readability = result.get("readability", {})
        deep = result.get("deep", "No analysis result.")
        self._update_metrics(readability)
        self._update_radar(stats, readability)
        self.analysis_output.delete("1.0", "end")
        self.analysis_output.insert("1.0", deep)
        self.status_label.configure(text="Analysis complete")

    def _update_metrics(self, readability):
        mapping = {
            "Flesch Reading Ease": readability.get("flesch_reading_ease"),
            "Flesch-Kincaid Grade": readability.get("flesch_kincaid_grade"),
            "Coleman-Liau Index": readability.get("coleman_liau_index"),
        }
        for label, value in mapping.items():
            bar, val = self.metric_bars[label]
            if value is None:
                val.configure(text="--")
                bar.set(0)
            else:
                bar.set(max(0, min(1, value / 20 if "Grade" not in label else value / 12)))
                val.configure(text=f"{value:.1f}")

    def _update_radar(self, stats, readability):
        labels, values = radar_from_metrics(stats, readability)
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values_cycle = values + values[:1]
        angles_cycle = angles + angles[:1]

        fig = plt.Figure(figsize=(3.8, 3.0), facecolor=COLORS["bg_dark"])
        ax = fig.add_subplot(111, projection="polar")
        ax.set_facecolor(COLORS["bg_dark"])
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.grid(True, color=COLORS["border_subtle"])
        ax.set_yticklabels([])
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, color=COLORS["text_secondary"], fontsize=8)
        ax.plot(angles_cycle, values_cycle, color=COLORS["accent_primary"], linewidth=2)
        ax.fill(angles_cycle, values_cycle, color=COLORS["accent_primary"], alpha=0.25)

        if self.chart_canvas:
            self.chart_canvas.get_tk_widget().destroy()
        self.chart_canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _resolve_model(self):
        key = self.model_var.get()
        info = AVAILABLE_MODELS.get(key, {})
        model_type = info.get("type")
        if model_type == "ollama":
            available, message = check_ollama_connection(key)
            if not available:
                self.status_label.configure(text=message)
            return True, key, None, None
        if model_type == "openai":
            client, _ = setup_openai_client(self.api_keys.get("openai"))
            return False, None, "openai", client
        if model_type == "gemini":
            client, _ = setup_gemini_client(self.api_keys.get("gemini"))
            return False, None, "gemini", client
        return True, "gpt-oss:20b", None, None

    # ---------------- Generation -----------------
    def show_generation(self):
        self.clear_main()
        container = ctk.CTkFrame(self.main, fg_color=COLORS["bg_dark"])
        container.grid(row=0, column=0, sticky="nsew")
        container.grid_columnconfigure(1, weight=1)
        container.grid_rowconfigure(0, weight=1)

        # Left controls
        left = ctk.CTkFrame(container, fg_color=COLORS["bg_card"], corner_radius=10)
        left.grid(row=0, column=0, sticky="nsw", padx=16, pady=16)
        left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(left, text="Active Profile", font=FONTS["body"], text_color=COLORS["text_secondary"]).grid(row=0, column=0, sticky="w", padx=14, pady=(14, 4))
        self.profile_label = ctk.CTkLabel(left, text="None loaded", font=FONTS["header_md"], text_color=COLORS["text_primary"])
        self.profile_label.grid(row=1, column=0, sticky="w", padx=14)

        ctk.CTkButton(left, text="Load Profile", fg_color=COLORS["bg_dark"], hover_color=COLORS["bg_card"], command=self._load_profile).grid(row=2, column=0, sticky="ew", padx=14, pady=(10, 10))

        ctk.CTkLabel(left, text="Content Type", font=FONTS["body"], text_color=COLORS["text_secondary"]).grid(row=3, column=0, sticky="w", padx=14)
        self.content_type_var = ctk.StringVar(value=self.generator.supported_content_types[0])
        ctk.CTkOptionMenu(left, variable=self.content_type_var, values=self.generator.supported_content_types, fg_color=COLORS["bg_dark"], button_color=COLORS["accent_primary"], text_color=COLORS["text_primary"]).grid(row=4, column=0, sticky="ew", padx=14, pady=(4, 10))

        ctk.CTkLabel(left, text="Topic or Prompt", font=FONTS["body"], text_color=COLORS["text_secondary"]).grid(row=5, column=0, sticky="w", padx=14)
        self.topic_box = ctk.CTkTextbox(left, height=100, fg_color=COLORS["bg_dark"], text_color=COLORS["text_primary"], border_width=1, border_color=COLORS["border_subtle"])
        self.topic_box.grid(row=6, column=0, sticky="ew", padx=14, pady=(4, 10))

        self.length_var = ctk.IntVar(value=500)
        self.creativity_var = ctk.DoubleVar(value=0.5)
        self.adherence_var = ctk.DoubleVar(value=0.5)
        self._add_slider(left, "Style Adherence", self.adherence_var, 0, 1, row=7)
        self._add_slider(left, "Creativity", self.creativity_var, 0, 1, row=8)
        self._add_slider(left, "Length (words)", self.length_var, 100, 1200, row=9, step=50)

        ctk.CTkButton(left, text="Generate Content", fg_color=COLORS["accent_primary"], hover_color=COLORS["success"], text_color=COLORS["bg_dark"], command=self.start_generation).grid(row=10, column=0, sticky="ew", padx=14, pady=(12, 14))

        # Right editor
        right = ctk.CTkFrame(container, fg_color=COLORS["bg_card"], corner_radius=10)
        right.grid(row=0, column=1, sticky="nsew", padx=16, pady=16)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(right, text="Output", font=FONTS["header_md"], text_color=COLORS["text_primary"]).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 6))
        self.generation_output = ctk.CTkTextbox(right, fg_color=COLORS["bg_dark"], text_color=COLORS["text_primary"], wrap="word", font=("Georgia", 14))
        self.generation_output.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))

    def _add_slider(self, parent, label, variable, from_, to, row, step=None):
        ctk.CTkLabel(parent, text=label, font=FONTS["body"], text_color=COLORS["text_secondary"]).grid(row=row, column=0, sticky="w", padx=14, pady=(8, 0))
        slider = ctk.CTkSlider(parent, variable=variable, from_=from_, to=to, progress_color=COLORS["accent_primary"], button_color=COLORS["accent_primary"], button_hover_color=COLORS["success"])
        slider.grid(row=row + 1, column=0, sticky="ew", padx=14)
        if step:
            slider.configure(number_of_steps=int((to - from_) / step))

    def _load_profile(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        loaded = load_local_profile(path)
        if loaded.get("success"):
            self.active_profile = loaded["profile"]
            self.active_profile_path = path
            self.profile_label.configure(text=os.path.basename(path))
            self.status_label.configure(text="Profile loaded")
        else:
            self.status_label.configure(text=loaded.get("error", "Failed to load profile"))

    def start_generation(self):
        if not self.active_profile:
            self.status_label.configure(text="Load a profile first")
            return
        topic = self.topic_box.get("1.0", "end").strip()
        if not topic:
            self.status_label.configure(text="Enter a topic or prompt")
            return
        self.generation_output.delete("1.0", "end")
        self.generation_output.insert("1.0", "Generating...")
        task = ThreadedTask(self._run_generation_task, self._on_generation_complete, topic)
        task.start()

    def _run_generation_task(self, topic):
        use_local, model_name, api_type, api_client = self._resolve_model()
        return self.generator.generate_content(
            style_profile=self.active_profile,
            content_type=self.content_type_var.get(),
            topic_or_prompt=topic,
            target_length=self.length_var.get(),
            tone="neutral",
            additional_context=f"Style adherence: {self.adherence_var.get():.2f}; Creativity: {self.creativity_var.get():.2f}",
            use_local=use_local,
            model_name=model_name,
            api_type=api_type,
            api_client=api_client,
        )

    def _on_generation_complete(self, result):
        if result.get("error"):
            self.generation_output.delete("1.0", "end")
            self.generation_output.insert("1.0", f"Error: {result['error']}")
            self.status_label.configure(text="Generation failed")
            return
        text = result.get("generated_content", "")
        self.generation_output.delete("1.0", "end")
        self.generation_output.insert("1.0", text)
        self.status_label.configure(text="Generation complete")

    # ---------------- Profiles -----------------
    def show_profiles(self):
        self.clear_main()
        container = ctk.CTkFrame(self.main, fg_color=COLORS["bg_dark"])
        container.grid(row=0, column=0, sticky="nsew")
        container.grid_columnconfigure(1, weight=1)
        container.grid_rowconfigure(0, weight=1)

        sidebar = ctk.CTkFrame(container, fg_color=COLORS["bg_card"], corner_radius=10, width=260)
        sidebar.grid(row=0, column=0, sticky="nsw", padx=16, pady=16)
        sidebar.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(sidebar, text="Saved Profiles", font=FONTS["header_md"], text_color=COLORS["text_primary"]).grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))
        refresh = ctk.CTkButton(sidebar, text="Refresh", command=self._refresh_profiles, fg_color=COLORS["bg_dark"], hover_color=COLORS["bg_card"], text_color=COLORS["text_primary"])
        refresh.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))

        self.profile_list = ctk.CTkScrollableFrame(sidebar, fg_color=COLORS["bg_card"], height=500)
        self.profile_list.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 12))

        self.profile_preview = ctk.CTkTextbox(container, fg_color=COLORS["bg_card"], text_color=COLORS["text_primary"], font=FONTS["body_sm"], wrap="word")
        self.profile_preview.grid(row=0, column=1, sticky="nsew", padx=16, pady=16)
        self.profile_preview.insert("1.0", "Select a profile to view details")

        self._refresh_profiles()

    def _refresh_profiles(self):
        for child in self.profile_list.winfo_children():
            child.destroy()
        profiles = list_local_profiles()
        if not profiles:
            ctk.CTkLabel(self.profile_list, text="No profiles found", font=FONTS["body"], text_color=COLORS["text_secondary"]).pack(anchor="w", padx=8, pady=8)
            return
        for prof in profiles:
            btn = ctk.CTkButton(
                self.profile_list,
                text=f"{prof['filename']}\n{prof['modified']}",
                anchor="w",
                fg_color=COLORS["bg_dark"],
                hover_color=COLORS["bg_card"],
                command=lambda p=prof: self._load_preview(p["filename"]),
            )
            btn.pack(fill="x", pady=4, padx=6)

    def _load_preview(self, filename):
        loaded = load_local_profile(filename)
        if not loaded.get("success"):
            self.profile_preview.delete("1.0", "end")
            self.profile_preview.insert("1.0", loaded.get("error", "Failed to load profile"))
            return
        profile = loaded["profile"]
        self.profile_preview.delete("1.0", "end")
        self.profile_preview.insert("1.0", json.dumps(profile, indent=2))
        self.active_profile = profile
        self.active_profile_path = filename
        self.status_label.configure(text=f"Profile ready: {os.path.basename(filename)}")

    # ---------------- Settings -----------------
    def show_settings(self):
        self.clear_main()
        container = ctk.CTkFrame(self.main, fg_color=COLORS["bg_dark"])
        container.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        container.grid_columnconfigure(0, weight=1)

        api_frame = ctk.CTkFrame(container, fg_color=COLORS["bg_card"], corner_radius=10)
        api_frame.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        api_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(api_frame, text="API Keys", font=FONTS["header_md"], text_color=COLORS["text_primary"]).grid(row=0, column=0, columnspan=2, sticky="w", padx=16, pady=(16, 8))

        ctk.CTkLabel(api_frame, text="OpenAI", font=FONTS["body"], text_color=COLORS["text_secondary"]).grid(row=1, column=0, sticky="w", padx=16)
        self.openai_entry = ctk.CTkEntry(api_frame, placeholder_text="sk-...", show="*", fg_color=COLORS["bg_dark"], text_color=COLORS["text_primary"])
        self.openai_entry.grid(row=1, column=1, sticky="ew", padx=16, pady=6)

        ctk.CTkLabel(api_frame, text="Gemini", font=FONTS["body"], text_color=COLORS["text_secondary"]).grid(row=2, column=0, sticky="w", padx=16)
        self.gemini_entry = ctk.CTkEntry(api_frame, placeholder_text="gem-...", show="*", fg_color=COLORS["bg_dark"], text_color=COLORS["text_primary"])
        self.gemini_entry.grid(row=2, column=1, sticky="ew", padx=16, pady=(0, 12))

        save_btn = ctk.CTkButton(api_frame, text="Save Keys", fg_color=COLORS["accent_primary"], hover_color=COLORS["success"], text_color=COLORS["bg_dark"], command=self._save_keys)
        save_btn.grid(row=3, column=1, sticky="e", padx=16, pady=(0, 16))

        housekeeping = ctk.CTkFrame(container, fg_color=COLORS["bg_card"], corner_radius=10)
        housekeeping.grid(row=1, column=0, sticky="ew")
        ctk.CTkLabel(housekeeping, text="Housekeeping", font=FONTS["header_md"], text_color=COLORS["text_primary"]).grid(row=0, column=0, sticky="w", padx=16, pady=(16, 8))
        ctk.CTkButton(housekeeping, text="Cleanup Old Reports", fg_color=COLORS["bg_dark"], hover_color=COLORS["bg_card"], command=self._cleanup_reports).grid(row=1, column=0, sticky="w", padx=16, pady=(0, 16))

    def _save_keys(self):
        self.api_keys["openai"] = self.openai_entry.get().strip()
        self.api_keys["gemini"] = self.gemini_entry.get().strip()
        self.status_label.configure(text="API keys stored for this session")

    def _cleanup_reports(self):
        res = cleanup_old_reports()
        if res.get("success"):
            self.status_label.configure(text=res.get("message", "Cleanup done"))
        else:
            self.status_label.configure(text=res.get("error", "Cleanup failed"))

    # ---------------- Mainloop -----------------
    def run(self):
        self.mainloop()


if __name__ == "__main__":
    app = StyleTransferApp()
    app.run()
