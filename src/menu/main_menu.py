"""
Main menu module for Style Transfer AI.
Handles primary navigation and menu display.
"""

import sys
import subprocess
import os
from ..config.settings import (
    AVAILABLE_MODELS,
    ANALOGY_DOMAINS,
    DEFAULT_ANALOGY_DOMAIN,
    CONCEPTUAL_DENSITY_THRESHOLD,
)
from ..analysis.analyzer import create_enhanced_style_profile
from ..analysis.metrics import calculate_style_similarity, extract_deep_stylometry
from ..analysis.analogy_engine import detect_conceptual_density, AnalogyInjector
from ..storage.local_storage import list_local_profiles, load_local_profile, save_style_profile_locally, delete_profile
from .model_selection import (
    select_model_interactive,
    reset_model_selection,
    get_current_model_info,
    SELECTED_API_TYPE,
    SELECTED_API_CLIENT,
)
from ..models.ollama_client import list_ollama_models, pull_ollama_model, is_ollama_installed
from ..utils.user_profile import get_file_paths
from ..utils.text_processing import read_text_file
from ..utils.formatters import format_human_readable_output
from ..generation import ContentGenerator, StyleTransfer, QualityController


def display_main_menu():
    """Display the main menu options."""
    print("\n" + "="*60)
    print("STYLE TRANSFER AI - ADVANCED STYLOMETRY ANALYSIS")
    print("="*60)

    model_info = get_current_model_info()
    if model_info['has_model']:
        print(f"Current Model: {model_info['selected_model']} (will be reset before next analysis)")
    else:
        print("Current Model: None (will be selected before analysis)")
    print("-" * 60)

    print("STYLE ANALYSIS:")
    print("1. Analyze Writing Style (Fast Mode - Recommended)")
    print("2. Balanced Analysis (More Detail, Slower)")
    print("3. View Existing Style Profiles")
    print("")
    print("CONTENT GENERATION:")
    print("4. Generate Content with Style Profile")
    print("5. Transfer Content to Different Style")
    print("")
    print("COMPARISON & EVALUATION:")
    print("6. Style Comparison & Analysis (Profiles / Text / File)")
    print("")
    print("COGNITIVE LOAD OPTIMIZATION:")
    print("7. Cognitive Bridging / Analogy Engine")
    print("")
    print("DATA MANAGEMENT:")
    print("8. Delete Style Profiles")
    print("9. Manage Local Models (Ollama)")
    print("10. Switch Analysis Model")
    print("11. Check Configuration")
    print("12. Launch GUI")
    print("13. Run Scripts (Utilities/Install)")
    print("0. Exit")
    print("="*60)


def _ask_analogy_options():
    """Prompt user for Cognitive Bridging / Analogy settings.
    Returns (enable: bool, domain: str|None).
    """
    print("\n--- Cognitive Load Optimization ---")
    toggle = input("Enable auto-inject contextual analogies? (y/n) [n]: ").strip().lower()
    if toggle != 'y':
        return False, None

    print("\nSelect base domain for analogies:")
    domains = list(ANALOGY_DOMAINS.items())
    for i, (key, info) in enumerate(domains, 1):
        print(f"  {i}. {info['label']}")
    choice = input(f"Enter choice (1-{len(domains)}) [{len(domains)}]: ").strip()
    try:
        idx = int(choice) - 1
        domain = domains[idx][0] if 0 <= idx < len(domains) else DEFAULT_ANALOGY_DOMAIN
    except (ValueError, IndexError):
        domain = DEFAULT_ANALOGY_DOMAIN
    print(f"  Domain set to: {ANALOGY_DOMAINS[domain]['label']}")
    return True, domain


def handle_analyze_style(processing_mode='fast'):
    """Handle style analysis workflow."""
    try:
        print(f"\nStarting {processing_mode} style analysis...")

        print("\nPlease select a model for this analysis:")
        select_model_interactive()

        model_info = get_current_model_info()
        if not model_info['has_model']:
            print("No model selected. Analysis cancelled.")
            return

        input_data = get_file_paths()
        if not input_data:
            print("No input provided. Analysis cancelled.")
            return

        analogy_enabled, analogy_domain = False, None
        if processing_mode in ['statistical', 'enhanced']:
            analogy_enabled, analogy_domain = _ask_analogy_options()

        # ── FIXED: use model_info for all four routing params ──────────────
        style_profile = create_enhanced_style_profile(
            input_data,
            use_local=model_info['use_local_model'],
            model_name=model_info['selected_model'],
            api_type=model_info['api_type'],
            api_client=model_info['api_client'],
            processing_mode=processing_mode,
            analogy_augmentation=analogy_enabled,
            analogy_domain=analogy_domain,
        )

        if style_profile:
            print("\nAnalysis complete.")

            # Ask who this profile belongs to so saved filenames are personalized.
            profile_name = input("Enter a name for this profile (leave blank for Anonymous_User): ").strip()
            if profile_name:
                if 'user_profile' not in style_profile or not isinstance(style_profile.get('user_profile'), dict):
                    style_profile['user_profile'] = {}
                style_profile['user_profile']['name'] = profile_name

            print("\nSaving analysis results...")
            save_result = save_style_profile_locally(style_profile)
            if save_result['success']:
                print(f"✔ {save_result['message']}")
            else:
                print(f"✗ Failed to save results: {save_result.get('error', 'Unknown error')}")

        print("\nAnalysis completed successfully!")
        reset_model_selection()
        input("\nPress Enter to continue...")

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        reset_model_selection()
    except Exception as e:
        print(f"\nError during analysis: {e}")
        reset_model_selection()


def handle_view_profiles():
    """Handle viewing existing style profiles."""
    profiles = list_local_profiles()

    if not profiles:
        print("\nNo style profiles found.")
        return

    print(f"\nFound {len(profiles)} style profiles:")
    print("-" * 80)

    for i, profile in enumerate(profiles, 1):
        filename_only = os.path.basename(profile['filename'])
        user_name = filename_only.split('_stylometric_profile_')[0] if '_stylometric_profile_' in filename_only else 'Unknown'
        size_mb = profile['size'] / (1024 * 1024)
        print(f"{i:2d}. {filename_only} | {user_name} | {profile['modified']} | {size_mb:.2f} MB")

    print("-" * 80)

    try:
        while True:
            choice = input("\nEnter profile number to view (0 to return): ").strip()
            if choice == "0":
                break
            try:
                profile_num = int(choice)
                if 1 <= profile_num <= len(profiles):
                    selected_profile = profiles[profile_num - 1]
                    result = load_local_profile(selected_profile['filename'])
                    if result['success']:
                        profile_data = result['profile']

                        # Show full human-readable report immediately after profile selection.
                        report_text = None
                        json_path = selected_profile['filename']
                        txt_path = os.path.splitext(json_path)[0] + ".txt"

                        if os.path.isfile(txt_path):
                            try:
                                with open(txt_path, 'r', encoding='utf-8') as f:
                                    report_text = f.read()
                            except Exception:
                                report_text = None

                        if not report_text and isinstance(profile_data.get('human_readable_report'), str):
                            report_text = profile_data.get('human_readable_report')

                        if not report_text:
                            report_text = format_human_readable_output(profile_data)

                        print("\n" + "="*80)
                        print("FULL HUMAN-READABLE STYLE PROFILE")
                        print("="*80)
                        print(report_text)
                    else:
                        print(f"Error loading profile: {result['error']}")
                else:
                    print("Invalid profile number.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    except KeyboardInterrupt:
        print("\n\nReturning to main menu...")


def handle_delete_profiles():
    """Handle deleting unwanted style profiles."""
    profiles = list_local_profiles()

    if not profiles:
        print("\nNo style profiles found.")
        return

    print(f"\nFound {len(profiles)} style profiles:")
    print("-" * 80)

    for i, profile in enumerate(profiles, 1):
        filename_only = os.path.basename(profile['filename'])
        user_name = filename_only.split('_stylometric_profile_')[0] if '_stylometric_profile_' in filename_only else 'Unknown'
        size_mb = profile['size'] / (1024 * 1024)
        print(f"{i:2d}. {filename_only} | {user_name} | {profile['modified']} | {size_mb:.2f} MB")

    print("-" * 80)
    print("Enter profile numbers to delete (comma-separated), 'all' to delete all, or 0 to cancel.")

    try:
        choice = input("\nProfiles to delete: ").strip()
        if choice == "0":
            return

        if choice.lower() == "all":
            confirm = input(f"Delete ALL {len(profiles)} profiles? This cannot be undone. (yes/no): ").strip().lower()
            if confirm != "yes":
                print("Cancelled.")
                return
            indices = range(len(profiles))
        else:
            try:
                indices = [int(n.strip()) - 1 for n in choice.split(",")]
                if any(i < 0 or i >= len(profiles) for i in indices):
                    print("One or more invalid profile numbers.")
                    return
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas.")
                return

        deleted_total = 0
        for idx in indices:
            result = delete_profile(profiles[idx]['filename'])
            if result['success']:
                print(f"  Deleted: {result['message']}")
                deleted_total += 1
            else:
                print(f"  Failed: {result['error']}")

        print(f"\nDeleted {deleted_total} profile(s).")

    except KeyboardInterrupt:
        print("\n\nReturning to main menu...")


def display_profile_summary(profile_data):
    """Display a summary of a style profile."""
    print("\n" + "="*80)
    print("STYLE PROFILE SUMMARY")
    print("="*80)

    user_profile = profile_data.get('user_profile', {})
    print(f"User: {user_profile.get('name', 'Unknown')}")
    print(f"Background: {user_profile.get('background', 'Not specified')}")
    print(f"Writing Goals: {user_profile.get('writing_goals', 'Not specified')}")

    metadata = profile_data.get('metadata', {})
    print(f"\nAnalysis Date: {metadata.get('analysis_date', 'Unknown')}")
    print(f"Model Used: {metadata.get('model_used', 'Unknown')}")
    print(f"Processing Mode: {metadata.get('processing_mode', 'Unknown')}")
    print(f"Files Analyzed: {metadata.get('total_samples', 0)}")
    print(f"Total Words: {metadata.get('total_words', 0):,}")

    statistics = profile_data.get('statistical_analysis', {})
    if statistics:
        print(f"\nReadability Scores:")
        print(f"  Flesch Reading Ease: {statistics.get('flesch_reading_ease', 'N/A')}")
        print(f"  Flesch-Kincaid Grade: {statistics.get('flesch_kincaid_grade', 'N/A')}")
        print(f"  Coleman-Liau Index: {statistics.get('coleman_liau_index', 'N/A')}")
        print(f"  Lexical Diversity: {statistics.get('lexical_diversity', 'N/A')}")

    print("="*80)


def handle_check_configuration():
    """Handle configuration checking with fallback mechanisms."""
    try:
        print("\nRunning configuration check...")

        try:
            from ..main import check_system_requirements
            results = check_system_requirements()

            print("\n" + "="*60)
            print("SYSTEM CONFIGURATION CHECK RESULTS")
            print("="*60)

            if results['success']:
                print("✔ All system requirements are met!")
            else:
                print("✗ Some issues were found:")
                for issue in results.get('issues', []):
                    print(f"  • {issue}")

            if results.get('warnings'):
                print("\n⚠  Warnings:")
                for warning in results['warnings']:
                    print(f"  • {warning}")

            if results.get('recommendations'):
                print("\n💡 Recommendations:")
                for rec in results['recommendations']:
                    print(f"  • {rec}")

            print("="*60)
            print("Configuration check completed successfully!")

        except ImportError:
            print("Running basic configuration checks...")
            print("\n" + "="*60)
            print("BASIC SYSTEM CHECK")
            print("="*60)
            print(f"✔ Python version: {sys.version.split()[0]}")
            for module in ['sys', 'os', 'json', 'datetime', 'requests']:
                try:
                    __import__(module)
                    print(f"✔ {module}: Available")
                except ImportError:
                    print(f"✗ {module}: Missing")
            print("="*60)
            print("Basic configuration check completed!")

        input("\nPress Enter to continue...")

    except KeyboardInterrupt:
        print("\n\nReturning to main menu...")
    except Exception as e:
        print(f"\nError running configuration check: {e}")
        input("\nPress Enter to continue...")


def handle_model_management():
    """Manage local Ollama models from the CLI."""
    try:
        if not is_ollama_installed():
            if not handle_ollama_installation():
                return
        while True:
            print("\n" + "="*60)
            print("LOCAL MODEL MANAGEMENT (OLLAMA)")
            print("="*60)
            print("1. List installed models")
            print("2. Download a model")
            print("0. Return to main menu")

            choice = input("\nEnter your choice (0-2): ").strip()
            if choice == "0":
                return
            if choice == "1":
                models, error = list_ollama_models()
                if error:
                    print(f"\n✗ {error}")
                elif not models:
                    print("\nNo models found. Use option 2 to download.")
                else:
                    print("\nInstalled models:")
                    for model in models:
                        print(f"• {model}")
                input("\nPress Enter to continue...")
            elif choice == "2":
                available_ollama = [
                    key for key, info in AVAILABLE_MODELS.items()
                    if info.get("type") == "ollama"
                ]
                if not available_ollama:
                    print("\nNo Ollama models configured.")
                    input("\nPress Enter to continue...")
                    continue

                print("\nAvailable Ollama models:")
                for idx, model_name in enumerate(available_ollama, 1):
                    print(f"{idx}. {model_name}")

                selection = input("\nSelect a model to download (0 to cancel): ").strip()
                if selection == "0":
                    continue
                try:
                    model_index = int(selection) - 1
                    if 0 <= model_index < len(available_ollama):
                        model_name = available_ollama[model_index]
                        success, message = pull_ollama_model(model_name)
                        print(f"\n{'✔' if success else '✗'} {message}")
                    else:
                        print("\nInvalid model selection.")
                except ValueError:
                    print("\nInvalid input. Please enter a number.")
                input("\nPress Enter to continue...")
            else:
                print("Invalid choice. Please enter 0-2.")
    except KeyboardInterrupt:
        print("\n\nReturning to main menu...")


def handle_ollama_installation():
    """Offer CLI installation options for Ollama on Windows."""
    import webbrowser

    print("\n✗ Ollama is not installed.")
    print("You can install it via the official PowerShell script or a portable CLI zip.")

    while True:
        print("\nInstallation Options:")
        print("1. Install via PowerShell script (admin required)")
        print("2. Install portable CLI zip (manual path setup)")
        print("3. Open download page")
        print("4. Open a new PowerShell window")
        print("0. Return to main menu")

        choice = input("\nEnter your choice (0-4): ").strip()
        if choice == "0":
            return False
        if choice == "1":
            command = (
                "Start-Process powershell -Verb RunAs -ArgumentList "
                "'-ExecutionPolicy Bypass -Command irm https://ollama.com/install.ps1 | iex'"
            )
            subprocess.Popen(["powershell", "-ExecutionPolicy", "Bypass", "-Command", command])
            input("\nPress Enter to return to the main menu...")
            return False
        if choice == "2":
            default_url = "https://github.com/ollama/ollama/releases/latest/download/ollama-windows-amd64.zip"
            destination = input("Install folder (default: C:\\Ollama): ").strip() or "C:\\Ollama"
            download_url = input(f"Zip URL (default: {default_url}): ").strip() or default_url
            install_cmd = (
                f"$dest='{destination}'; $zip='ollama.zip'; "
                f"curl.exe -L -o $zip '{download_url}'; "
                "if (-Not (Test-Path $dest)) { New-Item -ItemType Directory -Force -Path $dest | Out-Null }; "
                "Expand-Archive -Path $zip -DestinationPath $dest -Force; "
                "Remove-Item $zip; "
                "[System.Environment]::SetEnvironmentVariable('Path', $env:Path + ';' + $dest, [System.EnvironmentVariableTarget]::User)"
            )
            subprocess.Popen(["powershell", "-ExecutionPolicy", "Bypass", "-Command", install_cmd])
            input("\nPress Enter to return to the main menu...")
            return False
        if choice == "3":
            webbrowser.open("https://ollama.ai/download")
            input("\nPress Enter to return to the main menu...")
            return False
        if choice == "4":
            subprocess.Popen(["powershell", "-NoExit"])
            input("\nPress Enter to return to the main menu...")
            return False
        print("Invalid choice. Please enter 0-4.")


def handle_launch_gui():
    """Launch the Streamlit web GUI from the CLI."""
    try:
        try:
            import streamlit  # noqa: F401
        except ImportError:
            print("\n✗ Streamlit is not installed. Run: pip install streamlit")
            input("\nPress Enter to continue...")
            return

        app_path = "app.py"
        if not os.path.exists(app_path):
            app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "app.py")

        subprocess.Popen([sys.executable, "-m", "streamlit", "run", app_path])
        print("\n✔ Streamlit GUI launched in your browser.")
        input("\nPress Enter to continue...")
    except KeyboardInterrupt:
        print("\n\nReturning to main menu...")


def handle_run_scripts():
    """Run helper and installer scripts from the CLI."""
    scripts = [
        {"label": "Run CLI (scripts/run.py)", "path": os.path.join("scripts", "run.py")},
        {"label": "Run Streamlit GUI (scripts/run_streamlit.py)", "path": os.path.join("scripts", "run_streamlit.py")},
        {"label": "Setup PATH (add_to_path.ps1)", "path": "add_to_path.ps1"},
        {"label": "Setup PATH (setup_path.bat)", "path": "setup_path.bat"},
        {"label": "Run Config Check (check_config.py)", "path": "check_config.py"},
        {"label": "Install CLI (install/install_cli.bat)", "path": os.path.join("install", "install_cli.bat")},
        {"label": "Quick Install (install/quick_install.bat)", "path": os.path.join("install", "quick_install.bat")},
        {"label": "Launch Streamlit UI (app.py)", "path": "app.py", "streamlit": True},
    ]

    try:
        while True:
            print("\n" + "="*60)
            print("RUN SCRIPTS")
            print("="*60)
            for idx, script in enumerate(scripts, 1):
                print(f"{idx}. {script['label']}")
            print("0. Return to main menu")

            choice = input(f"\nEnter your choice (0-{len(scripts)}): ").strip()
            if choice == "0":
                return
            try:
                script_index = int(choice) - 1
                if 0 <= script_index < len(scripts):
                    selected = scripts[script_index]
                    script_path = selected["path"]
                    if not os.path.exists(script_path):
                        print(f"\n✗ Script not found: {script_path}")
                        input("\nPress Enter to continue...")
                        continue

                    if selected.get("streamlit"):
                        try:
                            import streamlit  # noqa: F401
                        except ImportError:
                            print("\n✗ Streamlit is not installed. Run: pip install streamlit")
                            input("\nPress Enter to continue...")
                            continue
                        subprocess.Popen([sys.executable, "-m", "streamlit", "run", script_path])
                    elif script_path.endswith(".ps1"):
                        subprocess.Popen(["powershell", "-ExecutionPolicy", "Bypass", "-File", script_path])
                    elif script_path.endswith(".bat"):
                        subprocess.Popen(["cmd", "/c", script_path])
                    elif script_path.endswith(".py"):
                        subprocess.Popen([sys.executable, script_path])
                    else:
                        print("\n✗ Unsupported script type.")
                        input("\nPress Enter to continue...")
                        continue

                    print(f"\n✔ Launched: {selected['label']}")
                    return
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    except KeyboardInterrupt:
        print("\n\nReturning to main menu...")


def handle_generate_content():
    """Handle content generation with style profiles."""
    try:
        print("\n" + "="*50)
        print("CONTENT GENERATION WITH STYLE PROFILE")
        print("="*50)

        profiles = list_local_profiles()
        if not profiles:
            print("No style profiles found. Please analyze some writing samples first.")
            input("\nPress Enter to continue...")
            return

        print("\nAvailable Style Profiles:")
        for i, profile in enumerate(profiles, 1):
            print(f"{i}. {profile['filename']} ({profile['modified']})")

        choice = input(f"\nSelect a profile (1-{len(profiles)}): ").strip()

        try:
            profile_index = int(choice) - 1
            if 0 <= profile_index < len(profiles):
                selected_profile = profiles[profile_index]
                profile_data = load_local_profile(selected_profile['filename'])

                if not profile_data:
                    print("Failed to load profile data.")
                    return

                generator = ContentGenerator()

                print("\nContent Generation Parameters:")
                content_type = input("Content type (email/article/story/essay/letter/review/blog/social/academic/creative): ").strip().lower()
                if not content_type:
                    content_type = "general"

                content_type_mapping = {'blog': 'blog_post', 'social': 'social_media', 'general': 'creative'}
                if content_type in content_type_mapping:
                    content_type = content_type_mapping[content_type]

                topic = input("Topic/Subject: ").strip()
                if not topic:
                    print("Topic is required for content generation.")
                    return

                target_length = input("Target length in words (default: 300): ").strip()
                try:
                    target_length = int(target_length) if target_length else 300
                except ValueError:
                    target_length = 300

                tone = input("Desired tone (formal/casual/professional/creative/persuasive): ").strip()
                if not tone:
                    tone = "neutral"

                context = input("Additional context/requirements (optional): ").strip()

                print("\nPlease select a model for content generation:")
                select_model_interactive()
                model_info = get_current_model_info()

                if not model_info['has_model']:
                    print("No model selected. Generation cancelled.")
                    return

                print(f"\nGenerating {content_type} content...")

                # ── FIXED ──────────────────────────────────────────────────
                result = generator.generate_content(
                    style_profile=profile_data,
                    content_type=content_type,
                    topic_or_prompt=topic,
                    target_length=target_length,
                    tone=tone,
                    additional_context=context,
                    use_local=model_info['use_local_model'],
                    model_name=model_info['selected_model'],
                    api_type=model_info['api_type'],
                    api_client=model_info['api_client'],
                )

                if 'error' in result:
                    print(f"\nGeneration failed: {result['error']}")
                else:
                    print("\n" + "="*60)
                    print("GENERATED CONTENT")
                    print("="*60)
                    print(result['generated_content'])
                    print("\n" + "="*60)

                    metadata = result.get('generation_metadata', {})
                    print(f"\nContent Type: {metadata.get('content_type', 'Unknown')}")
                    print(f"Word Count: {metadata.get('actual_length', 'Unknown')}")
                    print(f"Model Used: {metadata.get('model_used', 'Unknown')}")

                    style_score = result.get('style_adherence_score', result.get('style_match_score', 'N/A'))
                    if isinstance(style_score, (int, float)):
                        bar = '\u2588' * int(style_score * 30) + '\u2591' * (30 - int(style_score * 30))
                        print(f"Style Adherence:  [{bar}] {style_score:.1%}")

                    save_choice = input("\nSave this generated content? (y/n): ").strip().lower()
                    if save_choice == 'y':
                        from ..utils.text_processing import sanitize_topic_for_filename
                        generated_dir = "generated content"
                        os.makedirs(generated_dir, exist_ok=True)
                        timestamp = metadata.get('timestamp', 'unknown')
                        topic_clean = sanitize_topic_for_filename(metadata.get('topic_prompt', topic))
                        filename = os.path.join(generated_dir, f"{topic_clean}_{content_type}_{timestamp}.txt")
                        try:
                            with open(filename, 'w', encoding='utf-8') as f:
                                f.write("="*60 + "\nSTYLE TRANSFER AI - GENERATED CONTENT\n" + "="*60 + "\n\n")
                                for k, v in metadata.items():
                                    f.write(f"{k}: {v}\n")
                                f.write("\n" + "="*60 + "\nGENERATED CONTENT\n" + "="*60 + "\n\n")
                                f.write(result['generated_content'])
                            print(f"Content saved as: {filename}")
                        except Exception as e:
                            print(f"Failed to save content: {e}")

                reset_model_selection()
            else:
                print("Invalid selection.")

        except ValueError:
            print("Invalid selection.")

        input("\nPress Enter to continue...")

    except KeyboardInterrupt:
        print("\n\nReturning to main menu...")
    except Exception as e:
        print(f"\nError in content generation: {e}")


def handle_style_transfer():
    """Handle style transfer between different writing styles."""
    try:
        print("\n" + "="*50)
        print("STYLE TRANSFER AND CONTENT RESTYLING")
        print("="*50)

        content_choice = input("Enter content (1) or file path (2)? (1/2): ").strip()
        original_content = ""
        if content_choice == "1":
            print("Enter your content (press Enter twice to finish):")
            lines = []
            while True:
                line = input()
                if line == "" and len(lines) > 0 and lines[-1] == "":
                    break
                lines.append(line)
            original_content = "\n".join(lines[:-1])
        elif content_choice == "2":
            file_path = input("Enter file path: ").strip()
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()
            except Exception as e:
                print(f"Failed to read file: {e}")
                return
        else:
            print("Invalid choice.")
            return

        if not original_content.strip():
            print("No content provided.")
            return

        profiles = list_local_profiles()
        if not profiles:
            print("No style profiles found.")
            input("\nPress Enter to continue...")
            return

        print("\nAvailable Style Profiles:")
        for i, profile in enumerate(profiles, 1):
            print(f"{i}. {profile['filename']} ({profile['modified']})")

        choice = input(f"\nSelect target style profile (1-{len(profiles)}): ").strip()

        try:
            profile_index = int(choice) - 1
            if 0 <= profile_index < len(profiles):
                selected_profile = profiles[profile_index]
                profile_data = load_local_profile(selected_profile['filename'])

                if not profile_data:
                    print("Failed to load profile data.")
                    return

                transfer = StyleTransfer()

                print("\nStyle Transfer Parameters:")
                transfer_type = input("Transfer type (default: direct_transfer): ").strip() or "direct_transfer"

                intensity = input("Transfer intensity (0.1-1.0, default: 0.8): ").strip()
                try:
                    intensity = float(intensity) if intensity else 0.8
                    intensity = max(0.1, min(1.0, intensity))
                except ValueError:
                    intensity = 0.8

                preserve_elements = input("Elements to preserve (comma-separated, e.g., facts,names): ").strip()
                preserve_list = [item.strip() for item in preserve_elements.split(',')] if preserve_elements else []

                print("\nPlease select a model for style transfer:")
                select_model_interactive()
                model_info = get_current_model_info()

                if not model_info['has_model']:
                    print("No model selected. Transfer cancelled.")
                    return

                print(f"\nTransferring content...")

                # ── FIXED ──────────────────────────────────────────────────
                result = transfer.transfer_style(
                    original_content=original_content,
                    target_style_profile=profile_data,
                    transfer_type=transfer_type,
                    intensity=intensity,
                    preserve_elements=preserve_list,
                    use_local=model_info['use_local_model'],
                    model_name=model_info['selected_model'],
                    api_type=model_info['api_type'],
                    api_client=model_info['api_client'],
                )

                if 'error' in result:
                    print(f"\nStyle transfer failed: {result['error']}")
                else:
                    print("\n" + "="*60)
                    print("TRANSFERRED CONTENT")
                    print("="*60)
                    print(result['transferred_content'])
                    print("\n" + "="*60)

                    metadata = result.get('transfer_metadata', {})
                    style_score = result.get('style_match_score', 'N/A')
                    if isinstance(style_score, (int, float)):
                        bar = '\u2588' * int(style_score * 30) + '\u2591' * (30 - int(style_score * 30))
                        print(f"Style Match:      [{bar}] {style_score:.1%}")

                    save_choice = input("\nSave the transferred content? (y/n): ").strip().lower()
                    if save_choice == 'y':
                        from ..utils.text_processing import sanitize_topic_for_filename
                        generated_dir = "generated content"
                        os.makedirs(generated_dir, exist_ok=True)
                        timestamp = metadata.get('timestamp', 'unknown')
                        topic_clean = sanitize_topic_for_filename(original_content[:50])
                        filename = os.path.join(generated_dir, f"{topic_clean}_transferred_{transfer_type}_{timestamp}.txt")
                        try:
                            with open(filename, 'w', encoding='utf-8') as f:
                                f.write("="*60 + "\nSTYLE TRANSFER AI - STYLE TRANSFERRED CONTENT\n" + "="*60 + "\n\n")
                                for k, v in metadata.items():
                                    f.write(f"{k}: {v}\n")
                                f.write("\n" + "="*60 + "\nORIGINAL CONTENT\n" + "="*60 + "\n\n")
                                f.write(original_content)
                                f.write("\n\n" + "="*60 + "\nTRANSFERRED CONTENT\n" + "="*60 + "\n\n")
                                f.write(result['transferred_content'])
                            print(f"Transferred content saved as: {filename}")
                        except Exception as e:
                            print(f"Failed to save content: {e}")

                reset_model_selection()
            else:
                print("Invalid selection.")

        except ValueError:
            print("Invalid selection.")

        input("\nPress Enter to continue...")

    except KeyboardInterrupt:
        print("\n\nReturning to main menu...")
    except Exception as e:
        print(f"\nError in style transfer: {e}")


def _get_person_input(label):
    """Prompt the user to provide input for one side of a comparison."""
    print(f"\n--- {label} ---")
    print("  1. Load a saved profile")
    print("  2. Paste text")
    print("  3. Load text from file")
    choice = input("Choose input method (1/2/3): ").strip()

    if choice == "1":
        profiles = list_local_profiles()
        if not profiles:
            print("No saved profiles found.")
            return None, None
        print(f"\nFound {len(profiles)} saved profiles:\n")
        for i, p in enumerate(profiles, 1):
            filename = os.path.basename(p['filename'])
            user_name = filename.split('_stylometric_profile_')[0] if '_stylometric_profile_' in filename else filename.replace('.json', '')
            size_kb = p['size'] / 1024
            print(f"  {i:2d}. {filename}  ({user_name}, {size_kb:.1f} KB, {p['modified']})")
        sel = input("Enter profile number: ").strip()
        try:
            idx = int(sel) - 1
            if not (0 <= idx < len(profiles)):
                print("Invalid selection.")
                return None, None
        except ValueError:
            print("Invalid input.")
            return None, None
        result = load_local_profile(profiles[idx]['filename'])
        if not result['success']:
            print(f"Failed to load profile: {result.get('message', 'Unknown error')}")
            return None, None
        ds = result['profile'].get('deep_stylometry', {})
        if not ds:
            print("This profile has no deep stylometry data. Re-analyze with the current version.")
            return None, None
        name = os.path.basename(profiles[idx]['filename']).replace('.json', '')
        return ds, name

    elif choice == "2":
        print(f"Paste text for {label} (press Enter twice on an empty line to finish):")
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        text = "\n".join(lines[:-1]) if lines and lines[-1] == "" else "\n".join(lines)
        if not text.strip():
            print("No text entered.")
            return None, None
        print("Computing deep stylometry from text...")
        ds = extract_deep_stylometry(text)
        return ds, label

    elif choice == "3":
        file_path = input("Enter file path: ").strip()
        text = read_text_file(file_path)
        if text.startswith("Error:"):
            print(text)
            return None, None
        print("Computing deep stylometry from file...")
        ds = extract_deep_stylometry(text)
        name = os.path.basename(file_path)
        return ds, name

    else:
        print("Invalid choice.")
        return None, None


def handle_style_comparison():
    """Unified style comparison — accepts saved profiles or raw text for each side."""
    try:
        print("\n" + "="*60)
        print("STYLE COMPARISON & ANALYSIS")
        print("="*60)

        ds1, name1 = _get_person_input("Person 1")
        if ds1 is None:
            input("\nPress Enter to continue...")
            return

        ds2, name2 = _get_person_input("Person 2")
        if ds2 is None:
            input("\nPress Enter to continue...")
            return

        print("\nCalculating deep stylometry similarity...\n")
        similarity = calculate_style_similarity(ds1, ds2)

        print("="*60)
        print("STYLE COMPARISON RESULTS")
        print("="*60)
        print(f"\n  A: {name1}")
        print(f"  B: {name2}")
        print()

        combined = similarity.get('combined_score', 0.0)
        cosine   = similarity.get('cosine_similarity', 0.0)
        burrows  = similarity.get('burrows_delta', 0.0)
        ngram    = similarity.get('ngram_overlap', 0.0)

        bar = '\u2588' * int(combined * 40) + '\u2591' * (40 - int(combined * 40))
        print(f"  COMBINED SCORE:     [{bar}] {combined:.1%}")
        print(f"  Cosine Similarity:  {cosine:.4f}  (30% weight)")
        print(f"  Burrows' Delta:     {burrows:.4f}  (45% weight — gold standard)")
        print(f"  N-gram Overlap:     {ngram:.4f}  (25% weight)")

        print()
        if combined >= 0.85:
            print("  ✓ Very high similarity — likely the same author or very similar style.")
        elif combined >= 0.65:
            print("  ~ Moderate similarity — noticeable stylistic overlap.")
        elif combined >= 0.40:
            print("  △ Low similarity — some shared features, mostly different styles.")
        else:
            print("  ✗ Very low similarity — distinctly different writing styles.")

        print("\n" + "-"*60)
        print("KEY METRIC COMPARISON")
        print("-"*60)

        vr1 = ds1.get('vocabulary_richness', {})
        vr2 = ds2.get('vocabulary_richness', {})
        print(f"\n  {'Metric':<30}  {'A':>12}  {'B':>12}")
        print(f"  {'─'*30}  {'─'*12}  {'─'*12}")
        print(f"  {'Avg Word Length':<30}  {ds1.get('avg_word_length', 0):>12.4f}  {ds2.get('avg_word_length', 0):>12.4f}")
        print(f"  {'Contraction Rate':<30}  {ds1.get('contraction_rate', 0):>12.4f}  {ds2.get('contraction_rate', 0):>12.4f}")
        print(f"  {'Passive Voice Ratio':<30}  {ds1.get('passive_voice_ratio', 0):>12.4f}  {ds2.get('passive_voice_ratio', 0):>12.4f}")
        print(f"  {'Punctuation Density':<30}  {ds1.get('punctuation_density', 0):>12.4f}  {ds2.get('punctuation_density', 0):>12.4f}")
        print(f"  {'Question Ratio':<30}  {ds1.get('question_ratio', 0):>12.4f}  {ds2.get('question_ratio', 0):>12.4f}")
        print(f"  {'Hapax Legomena Ratio':<30}  {vr1.get('hapax_legomena_ratio', 0):>12.4f}  {vr2.get('hapax_legomena_ratio', 0):>12.4f}")
        print("  {:<30}  {:>12.4f}  {:>12.4f}".format("Yule's K", vr1.get('yules_k', 0), vr2.get('yules_k', 0)))
        print("  {:<30}  {:>12.4f}  {:>12.4f}".format("Simpson's Diversity", vr1.get('simpsons_diversity', 0), vr2.get('simpsons_diversity', 0)))

        print("\n" + "="*60)

        save_choice = input("\nSave comparison results to file? (y/n): ").strip().lower()
        if save_choice == 'y':
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"profile_comparison_{timestamp}.txt"
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"STYLOMETRIC COMPARISON\n{'='*60}\n")
                    f.write(f"A: {name1}\nB: {name2}\n\n")
                    f.write(f"Combined Score:     {combined:.4f}\n")
                    f.write(f"Cosine Similarity:  {cosine:.4f}\n")
                    f.write(f"Burrows' Delta:     {burrows:.4f}\n")
                    f.write(f"N-gram Overlap:     {ngram:.4f}\n")
                print(f"✓ Comparison saved as: {filename}")
            except Exception as e:
                print(f"✗ Failed to save: {e}")

        input("\nPress Enter to continue...")

    except KeyboardInterrupt:
        print("\n\nReturning to main menu...")
    except Exception as e:
        print(f"\nError in style comparison: {e}")


def handle_cognitive_bridging():
    """Standalone Cognitive Bridging / Analogy Engine."""
    try:
        print("\n" + "="*60)
        print("COGNITIVE BRIDGING / ANALOGY ENGINE")
        print("="*60)

        print("Select a model for analogy generation:")
        select_model_interactive()
        model_info = get_current_model_info()
        if not model_info['has_model']:
            print("No model selected. Returning.")
            return

        print("\nHow would you like to provide text?")
        print("  1. Type / paste content")
        print("  2. Load from file path")
        choice = input("Choice (1/2): ").strip()
        text = ""
        if choice == "2":
            path = input("File path: ").strip().strip('"').strip("'")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
                print(f"Loaded {len(text)} characters from file.")
            except Exception as e:
                print(f"Cannot read file: {e}")
                return
        else:
            print("Paste or type your text below.")
            print("When finished, press Enter on an empty line:")
            lines = []
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                if line == "" and lines:
                    break
                lines.append(line)
            text = "\n".join(lines)

        if not text.strip():
            print("No text provided.")
            return

        density = detect_conceptual_density(text)
        print(f"\nOverall conceptual density: {density['overall_density']:.3f}")
        print(f"High-density passages (>={CONCEPTUAL_DENSITY_THRESHOLD}): {density['high_density_count']}")

        if density['high_density_count'] == 0:
            print("\nNo passages exceed the density threshold — no analogies needed.")
            input("\nPress Enter to continue...")
            return

        print("\nSelect analogy domain:")
        domains = list(ANALOGY_DOMAINS.items())
        for i, (key, info) in enumerate(domains, 1):
            print(f"  {i}. {info['label']}")
        d_choice = input(f"Choice (1-{len(domains)}) [{len(domains)}]: ").strip()
        try:
            d_idx = int(d_choice) - 1
            domain = domains[d_idx][0] if 0 <= d_idx < len(domains) else DEFAULT_ANALOGY_DOMAIN
        except (ValueError, IndexError):
            domain = DEFAULT_ANALOGY_DOMAIN

        print(f"\nGenerating analogies (domain: {ANALOGY_DOMAINS[domain]['label']})...")
        injector = AnalogyInjector(domain=domain)

        # ── FIXED ──────────────────────────────────────────────────────────
        result = injector.augment_text(
            text,
            use_local=model_info['use_local_model'],
            model_name=model_info['selected_model'],
            api_type=model_info['api_type'],
            api_client=model_info['api_client'],
        )

        print(f"\nGenerated {result['analogy_count']} cognitive note(s).")
        if result['analogy_count'] > 0:
            print("\n" + "="*60)
            print("ANALOGY BREAKDOWN")
            print("="*60)
            for i, item in enumerate(result.get('analogies', []), 1):
                preview = item['source_sentence'][:100]
                if len(item['source_sentence']) > 100:
                    preview += "..."
                print(f"\n  {i}. [{item['density_score']:.2f}] \"{preview}\"")
                if item.get('concept'):
                    print(f"     Concept:  {item['concept']}")
                print(f"     Analogy:  {item['analogy']}")
                if item.get('example'):
                    print(f"     Example:  {item['example']}")
            print()
            print("="*60)
            print("AUGMENTED TEXT (original + cognitive notes)")
            print("="*60)
            print(result['augmented_text'])
            print("="*60)

        reset_model_selection()
        input("\nPress Enter to continue...")

    except KeyboardInterrupt:
        print("\n\nReturning to main menu...")
        reset_model_selection()
    except Exception as e:
        print(f"\nError in cognitive bridging: {e}")
        reset_model_selection()


def run_main_menu():
    """Run the main menu loop."""
    while True:
        try:
            display_main_menu()
            choice = input("\nEnter your choice (0-13): ").strip()

            if choice == "0":
                print("\nThank you for using Style Transfer AI!")
                sys.exit(0)
            elif choice == "1":
                handle_analyze_style(processing_mode='fast')
            elif choice == "2":
                handle_analyze_style(processing_mode='statistical')
            elif choice == "3":
                handle_view_profiles()
            elif choice == "4":
                handle_generate_content()
            elif choice == "5":
                handle_style_transfer()
            elif choice == "6":
                handle_style_comparison()
            elif choice == "7":
                handle_cognitive_bridging()
            elif choice == "8":
                handle_delete_profiles()
            elif choice == "9":
                handle_model_management()
            elif choice == "10":
                reset_model_selection()
                print("\nModel selection reset. Please choose a new model:")
                select_model_interactive()
            elif choice == "11":
                handle_check_configuration()
            elif choice == "12":
                handle_launch_gui()
            elif choice == "13":
                handle_run_scripts()
            else:
                print("Invalid choice. Please enter 0-13.")

        except KeyboardInterrupt:
            print("\n\nExiting Style Transfer AI...")
            sys.exit(0)
        except Exception as e:
            print(f"Unexpected error: {e}")


if __name__ == "__main__":
    run_main_menu()