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
    get_current_model_info
)
from ..models.ollama_client import list_ollama_models, pull_ollama_model, is_ollama_installed
from ..utils.user_profile import get_file_paths
from ..utils.text_processing import read_text_file
from ..generation import ContentGenerator, StyleTransfer, QualityController


def display_main_menu():
    """Display the main menu options."""
    
    print("\n" + "="*60)
    print("STYLE TRANSFER AI - ADVANCED STYLOMETRY ANALYSIS")
    print("="*60)
    
    # Show current model status
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
        
        # Force model selection for each analysis
        print("\nPlease select a model for this analysis:")
        select_model_interactive()
        
        # Get current model info
        model_info = get_current_model_info()
        if not model_info['has_model']:
            print("No model selected. Analysis cancelled.")
            return
        
        # Get input from user (file paths or custom text)
        input_data = get_file_paths()
        
        if not input_data:
            print("No input provided. Analysis cancelled.")
            return
        
        # Ask about analogy augmentation (not available in fast mode)
        analogy_enabled, analogy_domain = False, None
        if processing_mode in ['statistical', 'enhanced']:
            analogy_enabled, analogy_domain = _ask_analogy_options()
        
        # Prepare model parameters based on selection
        style_profile = create_enhanced_style_profile(
            input_data, 
            use_local=True, 
            model_name=model_info['selected_model'], 
            processing_mode=processing_mode,
            analogy_augmentation=analogy_enabled,
            analogy_domain=analogy_domain,
        )
        
        # Save the analysis results locally
        if style_profile:
            print("\nSaving analysis results...")
            save_result = save_style_profile_locally(style_profile)
            if save_result['success']:
                print(f"Γ£ô {save_result['message']}")
                
            else:
                print(f"Γ£ù Failed to save results: {save_result.get('error', 'Unknown error')}")
        
        print("\nAnalysis completed successfully!")
        
        # Reset model selection to force re-selection next time
        reset_model_selection()
        
        input("\nPress Enter to continue...")
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        # Reset model selection even if interrupted
        reset_model_selection()
    except Exception as e:
        print(f"\nError during analysis: {e}")
        # Reset model selection even if error occurred
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
        # Extract user name from filename (remove path and extension parts)
        filename_only = os.path.basename(profile['filename'])
        user_name = filename_only.split('_stylometric_profile_')[0] if '_stylometric_profile_' in filename_only else 'Unknown'
        size_mb = profile['size'] / (1024 * 1024)  # Convert bytes to MB
        
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
                    result = load_local_profile(selected_profile['filename'])  # Use 'filename' not 'filepath'
                    
                    if result['success']:
                        profile_data = result['profile']
                        display_profile_summary(profile_data)
                        
                        # Ask if user wants to see full report
                        view_full = input("\nView full human-readable report? (y/n): ").strip().lower()
                        if view_full == 'y':
                            if 'human_readable_report' in profile_data:
                                print("\n" + "="*80)
                                print("FULL STYLE ANALYSIS REPORT")
                                print("="*80)
                                print(profile_data['human_readable_report'])
                            else:
                                print("Human-readable report not available in this profile.")
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
    
    # User information
    user_profile = profile_data.get('user_profile', {})
    print(f"User: {user_profile.get('name', 'Unknown')}")
    print(f"Background: {user_profile.get('background', 'Not specified')}")
    print(f"Writing Goals: {user_profile.get('writing_goals', 'Not specified')}")
    
    # Analysis metadata
    metadata = profile_data.get('metadata', {})
    print(f"\nAnalysis Date: {metadata.get('analysis_date', 'Unknown')}")
    print(f"Model Used: {metadata.get('model_used', 'Unknown')}")
    print(f"Processing Mode: {metadata.get('processing_mode', 'Unknown')}")
    print(f"Files Analyzed: {metadata.get('total_samples', 0)}")
    print(f"Total Words: {metadata.get('total_words', 0):,}")
    
    # Statistical summary
    statistics = profile_data.get('statistical_analysis', {})
    if statistics:
        print(f"\nReadability Scores:")
        print(f"  Flesch Reading Ease: {statistics.get('flesch_reading_ease', 'N/A')}")
        print(f"  Flesch-Kincaid Grade: {statistics.get('flesch_kincaid_grade', 'N/A')}")
        print(f"  Coleman-Liau Index: {statistics.get('coleman_liau_index', 'N/A')}")
        print(f"  Lexical Diversity: {statistics.get('lexical_diversity', 'N/A')}")
    
    # Deep analysis summary (if available)
    deep_analysis = profile_data.get('deep_stylometry_analysis', {})
    if deep_analysis and 'analysis_summary' in deep_analysis:
        print(f"\nDeep Analysis Summary:")
        summary = deep_analysis['analysis_summary']
        # Show first 200 characters of summary
        print(f"  {summary[:200]}...")
    
    print("="*80)


def handle_check_configuration():
    """Handle configuration checking with fallback mechanisms."""
    try:
        print("\nRunning configuration check...")
        
        # Method 1: Try to import and run the integrated checker
        try:
            from ..main import check_system_requirements
            results = check_system_requirements()
            
            print("\n" + "="*60)
            print("SYSTEM CONFIGURATION CHECK RESULTS")
            print("="*60)
            
            if results['success']:
                print("Γ£à All system requirements are met!")
            else:
                print("Γ¥î Some issues were found:")
                for issue in results.get('issues', []):
                    print(f"  ΓÇó {issue}")
            
            if results.get('warnings'):
                print("\nΓÜá∩╕Å  Warnings:")
                for warning in results['warnings']:
                    print(f"  ΓÇó {warning}")
            
            if results.get('recommendations'):
                print("\n≡ƒÆí Recommendations:")
                for rec in results['recommendations']:
                    print(f"  ΓÇó {rec}")
            
            print("="*60)
            print("Configuration check completed successfully!")
            
        except ImportError:
            # Method 2: Try to run standalone check_config.py script
            print("Trying standalone configuration checker...")
            
            # Look for check_config.py in various locations
            possible_paths = [
                # In the same directory as the package
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "check_config.py"),
                # In the package root
                os.path.join(sys.prefix, "check_config.py"),
                # In site-packages
                os.path.join(sys.prefix, "lib", "python" + sys.version[:3], "site-packages", "check_config.py"),
            ]
            
            config_script_found = False
            for check_config_path in possible_paths:
                if os.path.exists(check_config_path):
                    print(f"Found configuration checker at: {check_config_path}")
                    result = subprocess.run([sys.executable, check_config_path], 
                                         capture_output=False, text=True)
                    print(f"\nConfiguration check completed with exit code: {result.returncode}")
                    config_script_found = True
                    break
            
            if not config_script_found:
                # Method 3: Basic manual checks
                print("Running basic configuration checks...")
                print("\n" + "="*60)
                print("BASIC SYSTEM CHECK")
                print("="*60)
                
                # Check Python version
                print(f"Γ£à Python version: {sys.version.split()[0]}")
                
                # Check basic imports
                basic_modules = ['sys', 'os', 'json', 'datetime']
                for module in basic_modules:
                    try:
                        __import__(module)
                        print(f"Γ£à {module}: Available")
                    except ImportError:
                        print(f"Γ¥î {module}: Missing")
                
                # Check for requests
                try:
                    import requests
                    print("Γ£à requests: Available")
                except ImportError:
                    print("Γ¥î requests: Missing (required for API calls)")
                    print("  Install with: pip install requests")
                
                print("="*60)
                print("Basic configuration check completed!")
        
        input("\nPress Enter to continue...")
        
    except KeyboardInterrupt:
        print("\n\nReturning to main menu...")
    except Exception as e:
        print(f"\nError running configuration check: {e}")
        print("This might indicate installation issues.")
        print("Try reinstalling with: pip install --force-reinstall style-transfer-ai")
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
                    print(f"\nΓ£ù {error}")
                elif not models:
                    print("\nNo models found. Use option 2 to download.")
                else:
                    print("\nInstalled models:")
                    for model in models:
                        print(f"ΓÇó {model}")
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
                        status_symbol = "Γ£ô" if success else "Γ£ù"
                        print(f"\n{status_symbol} {message}")
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

    print("\nΓ£ù Ollama is not installed.")
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
            print("\nLaunching elevated installer...")
            command = (
                "Start-Process powershell -Verb RunAs -ArgumentList "
                "'-ExecutionPolicy Bypass -Command irm https://ollama.com/install.ps1 | iex'"
            )
            subprocess.Popen(["powershell", "-ExecutionPolicy", "Bypass", "-Command", command])
            print("\nWhen installation finishes, reopen the menu to manage models.")
            input("\nPress Enter to return to the main menu...")
            return False
        if choice == "2":
            default_url = "https://github.com/ollama/ollama/releases/latest/download/ollama-windows-amd64.zip"
            destination = input("Install folder (default: C:\\Ollama): ").strip() or "C:\\Ollama"
            download_url = input(f"Zip URL (default: {default_url}): ").strip() or default_url

            print("\nDownloading and extracting Ollama...")
            install_cmd = (
                f"$dest='{destination}'; "
                f"$zip='ollama.zip'; "
                f"curl.exe -L -o $zip '{download_url}'; "
                "if (-Not (Test-Path $dest)) { New-Item -ItemType Directory -Force -Path $dest | Out-Null }; "
                "Expand-Archive -Path $zip -DestinationPath $dest -Force; "
                "Remove-Item $zip; "
                "[System.Environment]::SetEnvironmentVariable('Path', $env:Path + ';' + $dest, [System.EnvironmentVariableTarget]::User)"
            )
            subprocess.Popen(["powershell", "-ExecutionPolicy", "Bypass", "-Command", install_cmd])
            print("\nPortable install started. Restart your terminal after it completes.")
            input("\nPress Enter to return to the main menu...")
            return False
        if choice == "3":
            webbrowser.open("https://ollama.ai/download")
            input("\nPress Enter to return to the main menu...")
            return False
        if choice == "4":
            subprocess.Popen(["powershell", "-NoExit"])
            print("\nΓ£ô Opened a new PowerShell window.")
            input("\nPress Enter to return to the main menu...")
            return False

        print("Invalid choice. Please enter 0-4.")


def handle_launch_gui():
    """Launch the Streamlit web GUI from the CLI."""
    try:
        print("\n" + "="*60)
        print("LAUNCH STREAMLIT GUI")
        print("="*60)

        try:
            import streamlit  # noqa: F401
        except ImportError:
            print("\nΓ£ù Streamlit is not installed. Run: pip install streamlit")
            input("\nPress Enter to continue...")
            return

        app_path = "app.py"
        if not os.path.exists(app_path):
            app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "app.py")

        subprocess.Popen([sys.executable, "-m", "streamlit", "run", app_path])
        print("\nΓ£ô Streamlit GUI launched in your browser.")
        input("\nPress Enter to continue...")
    except KeyboardInterrupt:
        print("\n\nReturning to main menu...")


def handle_run_scripts():
    """Run helper and installer scripts from the CLI."""
    scripts = [
        {"label": "Run CLI (scripts/run.py)", "path": os.path.join("scripts", "run.py")},
        {"label": "Run Streamlit GUI (scripts/run_streamlit.py)", "path": os.path.join("scripts", "run_streamlit.py")},
        {"label": "Run Legacy Analyzer (scripts/style_analyzer_enhanced.py)", "path": os.path.join("scripts", "style_analyzer_enhanced.py")},
        {"label": "Setup PATH (add_to_path.ps1)", "path": "add_to_path.ps1"},
        {"label": "Setup PATH (setup_path.bat)", "path": "setup_path.bat"},
        {"label": "Setup PATH (setup_auto_path.py)", "path": "setup_auto_path.py"},
        {"label": "Run Config Check (check_config.py)", "path": "check_config.py"},
        {"label": "Install CLI (install/install_cli.bat)", "path": os.path.join("install", "install_cli.bat")},
        {"label": "Quick Install (install/quick_install.bat)", "path": os.path.join("install", "quick_install.bat")},
        {"label": "Install One-Line (install/install_one_line.ps1)", "path": os.path.join("install", "install_one_line.ps1")},
        {"label": "Install Full (install/install_style_transfer_ai.ps1)", "path": os.path.join("install", "install_style_transfer_ai.ps1")},
        {"label": "Launch Streamlit UI (app.py)", "path": "app.py", "streamlit": True}
    ]

    try:
        while True:
            print("\n" + "="*60)
            print("RUN SCRIPTS")
            print("="*60)
            for idx, script in enumerate(scripts, 1):
                print(f"{idx}. {script['label']}")
            print("0. Return to main menu")

            choice = input("\nEnter your choice (0-{0}): ".format(len(scripts))).strip()
            if choice == "0":
                return
            try:
                script_index = int(choice) - 1
                if 0 <= script_index < len(scripts):
                    selected = scripts[script_index]
                    script_path = selected["path"]
                    if not os.path.exists(script_path):
                        print(f"\nΓ£ù Script not found: {script_path}")
                        input("\nPress Enter to continue...")
                        continue

                    if selected.get("streamlit"):
                        try:
                            import streamlit  # noqa: F401
                        except ImportError:
                            print("\nΓ£ù Streamlit is not installed. Run: pip install streamlit")
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
                        print("\nΓ£ù Unsupported script type.")
                        input("\nPress Enter to continue...")
                        continue

                    print(f"\nΓ£ô Launched: {selected['label']}")
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
        
        # List available style profiles
        profiles = list_local_profiles()
        if not profiles:
            print("No style profiles found. Please analyze some writing samples first.")
            input("\nPress Enter to continue...")
            return
        
        # Let user select a style profile
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
                
                # Initialize content generator
                generator = ContentGenerator()
                
                # Get content generation parameters
                print("\nContent Generation Parameters:")
                content_type = input("Content type (email/article/story/essay/letter/review/blog/social/academic/creative): ").strip().lower()
                if not content_type:
                    content_type = "general"
                
                # Map common user inputs to actual content types
                content_type_mapping = {
                    'blog': 'blog_post',
                    'social': 'social_media',
                    'general': 'creative'  # Default fallback
                }
                
                # Apply mapping if needed
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
                
                # Additional context
                context = input("Additional context/requirements (optional): ").strip()
                
                # Select model for generation
                print("\nPlease select a model for content generation:")
                select_model_interactive()
                model_info = get_current_model_info()
                
                if not model_info['has_model']:
                    print("No model selected. Generation cancelled.")
                    return
                
                print(f"\nGenerating {content_type} content with {selected_profile['filename']} style...")
                
                # Generate content based on model type
                result = generator.generate_content(
                    style_profile=profile_data,
                    content_type=content_type,
                    topic_or_prompt=topic,
                    target_length=target_length,
                    tone=tone,
                    additional_context=context,
                    use_local=True,
                    model_name=model_info['selected_model']
                )
                
                # Display results
                if 'error' in result:
                    print(f"\nGeneration failed: {result['error']}")
                else:
                    print("\n" + "="*60)
                    print("GENERATED CONTENT")
                    print("="*60)
                    print(result['generated_content'])
                    print("\n" + "="*60)
                    
                    # Show metadata
                    print("\nGeneration Metadata:")
                    metadata = result.get('generation_metadata', {})
                    print(f"Content Type: {metadata.get('content_type', 'Unknown')}")
                    print(f"Word Count: {metadata.get('actual_length', 'Unknown')}")
                    print(f"Model Used: {metadata.get('model_used', 'Unknown')}")
                    
                    # Handle style adherence score (deep stylometry comparison)
                    style_score = result.get('style_adherence_score', result.get('style_match_score', 'N/A'))
                    if isinstance(style_score, (int, float)):
                        bar_len = 30
                        filled = int(style_score * bar_len)
                        bar = '\u2588' * filled + '\u2591' * (bar_len - filled)
                        print(f"Style Adherence:  [{bar}] {style_score:.1%}")
                    else:
                        print(f"Style Adherence: {style_score}")
                    
                    # Quality analysis if available
                    if 'quality_analysis' in result:
                        quality = result['quality_analysis']
                        quality_score = quality.get('overall_score', 'N/A')
                        if isinstance(quality_score, (int, float)):
                            print(f"Quality Score: {quality_score:.2f}")
                        else:
                            print(f"Quality Score: {quality_score}")
                    
                    # Save generated content
                    save_choice = input("\nSave this generated content? (y/n): ").strip().lower()
                    if save_choice == 'y':
                        import os
                        from ..utils.text_processing import sanitize_topic_for_filename
                        
                        # Create generated content directory if it doesn't exist
                        generated_dir = "generated content"
                        if not os.path.exists(generated_dir):
                            os.makedirs(generated_dir)
                        
                        # Get metadata for filename
                        metadata = result.get('generation_metadata', {})
                        timestamp = metadata.get('timestamp', 'unknown')
                        topic_raw = metadata.get('topic_prompt', topic if 'topic' in locals() else 'general')
                        
                        # Sanitize topic for filename
                        topic_clean = sanitize_topic_for_filename(topic_raw)
                        
                        # Create filename with topic-based naming
                        filename = os.path.join(generated_dir, f"{topic_clean}_{content_type}_{timestamp}.txt")
                        
                        try:
                            with open(filename, 'w', encoding='utf-8') as f:
                                # Write metadata header
                                f.write("="*60 + "\n")
                                f.write("STYLE TRANSFER AI - GENERATED CONTENT\n")
                                f.write("="*60 + "\n\n")
                                
                                # Write generation details
                                f.write(f"Content Type: {metadata.get('content_type', 'Unknown')}\n")
                                f.write(f"Topic/Prompt: {metadata.get('topic_prompt', 'Unknown')}\n")
                                f.write(f"Target Length: {metadata.get('target_length', 'Unknown')} words\n")
                                f.write(f"Actual Length: {metadata.get('actual_length', 'Unknown')} words\n")
                                f.write(f"Tone: {metadata.get('tone', 'Unknown')}\n")
                                f.write(f"Model Used: {metadata.get('model_used', 'Unknown')}\n")
                                f.write(f"Generated: {metadata.get('timestamp', 'Unknown')}\n")
                                f.write(f"Style Profile: {metadata.get('style_profile_source', 'Unknown')}\n")
                                
                                if metadata.get('additional_context'):
                                    f.write(f"Additional Context: {metadata.get('additional_context')}\n")
                                
                                # Add quality metrics if available
                                if 'style_match_score' in result:
                                    if isinstance(result['style_match_score'], (int, float)):
                                        f.write(f"Style Match Score: {result['style_match_score']:.2f}\n")
                                    else:
                                        f.write(f"Style Match Score: {result['style_match_score']}\n")
                                
                                f.write("\n" + "="*60 + "\n")
                                f.write("GENERATED CONTENT\n")
                                f.write("="*60 + "\n\n")
                                
                                # Write the actual content
                                f.write(result['generated_content'])
                                
                            print(f"Content saved as: {filename}")
                        except Exception as e:
                            print(f"Failed to save content: {e}")
                
                # Reset model selection
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
        
        # Get original content
        print("1. Provide content to transfer:")
        content_choice = input("Enter content (1) or file path (2)? (1/2): ").strip()
        
        original_content = ""
        if content_choice == "1":
            print("Enter your content (press Enter twice to finish):")
            lines = []
            while True:
                line = input()
                if line == "" and len(lines) > 0:
                    if lines[-1] == "":
                        break
                lines.append(line)
            original_content = "\n".join(lines[:-1])  # Remove last empty line
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
        
        # List available style profiles
        profiles = list_local_profiles()
        if not profiles:
            print("No style profiles found. Please analyze some writing samples first.")
            input("\nPress Enter to continue...")
            return
        
        # Let user select target style profile
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
                
                # Initialize style transfer
                transfer = StyleTransfer()
                
                # Get transfer parameters
                print("\nStyle Transfer Parameters:")
                print("Transfer types: direct_transfer, style_blend, gradual_transform, tone_shift, formality_adjust, audience_adapt")
                transfer_type = input("Transfer type (default: direct_transfer): ").strip()
                if not transfer_type:
                    transfer_type = "direct_transfer"
                
                intensity = input("Transfer intensity (0.1-1.0, default: 0.8): ").strip()
                try:
                    intensity = float(intensity) if intensity else 0.8
                    intensity = max(0.1, min(1.0, intensity))
                except ValueError:
                    intensity = 0.8
                
                preserve_elements = input("Elements to preserve (comma-separated, e.g., facts,names): ").strip()
                preserve_list = [item.strip() for item in preserve_elements.split(',')] if preserve_elements else []
                
                # Select model for transfer
                print("\nPlease select a model for style transfer:")
                select_model_interactive()
                model_info = get_current_model_info()
                
                if not model_info['has_model']:
                    print("No model selected. Transfer cancelled.")
                    return
                
                print(f"\nTransferring content to {selected_profile['filename']} style...")
                
                # Perform style transfer based on model type
                result = transfer.transfer_style(
                    original_content=original_content,
                    target_style_profile=profile_data,
                    transfer_type=transfer_type,
                    intensity=intensity,
                    preserve_elements=preserve_list,
                    use_local=True,
                    model_name=model_info['selected_model']
                )
                
                # Display results
                if 'error' in result:
                    print(f"\nStyle transfer failed: {result['error']}")
                else:
                    print("\n" + "="*60)
                    print("ORIGINAL CONTENT")
                    print("="*60)
                    print(original_content[:500] + "..." if len(original_content) > 500 else original_content)
                    
                    print("\n" + "="*60)
                    print("TRANSFERRED CONTENT")
                    print("="*60)
                    print(result['transferred_content'])
                    print("\n" + "="*60)
                    
                    # Show transfer metadata
                    print("\nTransfer Analysis:")
                    metadata = result.get('transfer_metadata', {})
                    print(f"Transfer Type: {metadata.get('transfer_type', 'Unknown')}")
                    print(f"Intensity: {metadata.get('intensity', 'Unknown')}")
                    print(f"Model Used: {metadata.get('model_used', 'Unknown')}")
                    
                    # Style match score (now powered by deep stylometry)
                    style_score = result.get('style_match_score', 'N/A')
                    if isinstance(style_score, (int, float)):
                        bar_len = 30
                        filled = int(style_score * bar_len)
                        bar = '\u2588' * filled + '\u2591' * (bar_len - filled)
                        print(f"Style Match:      [{bar}] {style_score:.1%}")
                    else:
                        print(f"Style Match Score: {style_score}")
                    
                    # Quality analysis if available
                    if 'quality_analysis' in result:
                        quality = result['quality_analysis']
                        print(f"Content Preservation: {quality.get('content_preservation', 'N/A'):.2f}")
                        print(f"Style Transformation: {quality.get('style_transformation', 'N/A'):.2f}")
                    
                    # Save transferred content
                    save_choice = input("\nSave the transferred content? (y/n): ").strip().lower()
                    if save_choice == 'y':
                        import os
                        from ..utils.text_processing import sanitize_topic_for_filename
                        
                        # Create generated content directory if it doesn't exist
                        generated_dir = "generated content"
                        if not os.path.exists(generated_dir):
                            os.makedirs(generated_dir)
                        
                        timestamp = metadata.get('timestamp', 'unknown')
                        
                        # Create topic name from original content (first few words) or transfer type
                        topic_from_content = original_content.strip()[:50] if original_content.strip() else transfer_type
                        topic_clean = sanitize_topic_for_filename(topic_from_content)
                        
                        # Create filename with topic-based naming for transfers
                        filename = os.path.join(generated_dir, f"{topic_clean}_transferred_{transfer_type}_{timestamp}.txt")
                        
                        try:
                            with open(filename, 'w', encoding='utf-8') as f:
                                # Write metadata header
                                f.write("="*60 + "\n")
                                f.write("STYLE TRANSFER AI - STYLE TRANSFERRED CONTENT\n")
                                f.write("="*60 + "\n\n")
                                
                                # Write transfer details
                                f.write(f"Transfer Type: {metadata.get('transfer_type', 'Unknown')}\n")
                                f.write(f"Intensity: {metadata.get('intensity', 'Unknown')}\n")
                                f.write(f"Model Used: {metadata.get('model_used', 'Unknown')}\n")
                                f.write(f"Target Style Profile: {metadata.get('target_style_source', 'Unknown')}\n")
                                f.write(f"Transferred: {metadata.get('timestamp', 'Unknown')}\n")
                                
                                if 'style_match_score' in result:
                                    f.write(f"Style Match Score: {result['style_match_score']:.2f}\n")
                                
                                # Add quality metrics if available
                                if 'quality_analysis' in result:
                                    quality = result['quality_analysis']
                                    f.write(f"Content Preservation: {quality.get('content_preservation', 'N/A'):.2f}\n")
                                    f.write(f"Style Transformation: {quality.get('style_transformation', 'N/A'):.2f}\n")
                                
                                # Preserve elements if any
                                preserve_elements = metadata.get('preserve_elements', [])
                                if preserve_elements:
                                    f.write(f"Preserved Elements: {', '.join(preserve_elements)}\n")
                                
                                f.write("\n" + "="*60 + "\n")
                                f.write("ORIGINAL CONTENT\n")
                                f.write("="*60 + "\n\n")
                                f.write(original_content)
                                
                                f.write("\n\n" + "="*60 + "\n")
                                f.write("TRANSFERRED CONTENT\n")
                                f.write("="*60 + "\n\n")
                                f.write(result['transferred_content'])
                                
                            print(f"Transferred content saved as: {filename}")
                        except Exception as e:
                            print(f"Failed to save content: {e}")
                
                # Reset model selection
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
    """Prompt the user to provide input for one side of a comparison.

    Returns (deep_stylometry_dict, display_name) or (None, None) on failure.
    """
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
            print("This profile has no deep stylometry data. Re-analyze the text with the current version.")
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
        print("Compare two writing samples using deep stylometry.")
        print("For each person you can load a saved profile, paste text,")
        print("or load text from a file.\n")

        ds1, name1 = _get_person_input("Person 1")
        if ds1 is None:
            input("\nPress Enter to continue...")
            return

        ds2, name2 = _get_person_input("Person 2")
        if ds2 is None:
            input("\nPress Enter to continue...")
            return

        # Calculate similarity
        print("\nCalculating deep stylometry similarity...\n")
        similarity = calculate_style_similarity(ds1, ds2)

        # Display results
        print("="*60)
        print("STYLE COMPARISON RESULTS")
        print("="*60)
        print(f"\n  A: {name1}")
        print(f"  B: {name2}")
        print()

        combined = similarity.get('combined_score', 0.0)
        cosine = similarity.get('cosine_similarity', 0.0)
        burrows = similarity.get('burrows_delta', 0.0)
        ngram = similarity.get('ngram_overlap', 0.0)

        # Combined score bar
        bar_len = 40
        filled = int(combined * bar_len)
        bar = '\u2588' * filled + '\u2591' * (bar_len - filled)
        print(f"  COMBINED SCORE:     [{bar}] {combined:.1%}")
        print(f"  Cosine Similarity:  {cosine:.4f}  (30% weight)")
        print(f"  Burrows' Delta:     {burrows:.4f}  (45% weight \u2014 gold standard)")
        print(f"  N-gram Overlap:     {ngram:.4f}  (25% weight)")

        # Interpretation
        print()
        if combined >= 0.85:
            print("  \u2713 Very high similarity \u2014 likely the same author or very similar style.")
        elif combined >= 0.65:
            print("  ~ Moderate similarity \u2014 noticeable stylistic overlap.")
        elif combined >= 0.40:
            print("  \u25b3 Low similarity \u2014 some shared features, mostly different styles.")
        else:
            print("  \u2717 Very low similarity \u2014 distinctly different writing styles.")

        # Side-by-side key metrics comparison
        print("\n" + "-"*60)
        print("KEY METRIC COMPARISON")
        print("-"*60)

        vr1 = ds1.get('vocabulary_richness', {})
        vr2 = ds2.get('vocabulary_richness', {})
        header = f"\n  {'Metric':<30}  {'A':>12}  {'B':>12}"
        print(header)
        divider_char = '\u2500'
        print(f"  {divider_char*30}  {divider_char*12}  {divider_char*12}")
        print(f"  {'Avg Word Length':<30}  {ds1.get('avg_word_length', 0):>12.4f}  {ds2.get('avg_word_length', 0):>12.4f}")
        print(f"  {'Contraction Rate':<30}  {ds1.get('contraction_rate', 0):>12.4f}  {ds2.get('contraction_rate', 0):>12.4f}")
        print(f"  {'Passive Voice Ratio':<30}  {ds1.get('passive_voice_ratio', 0):>12.4f}  {ds2.get('passive_voice_ratio', 0):>12.4f}")
        print(f"  {'Punctuation Density':<30}  {ds1.get('punctuation_density', 0):>12.4f}  {ds2.get('punctuation_density', 0):>12.4f}")
        print(f"  {'Question Ratio':<30}  {ds1.get('question_ratio', 0):>12.4f}  {ds2.get('question_ratio', 0):>12.4f}")
        print(f"  {'Exclamation Ratio':<30}  {ds1.get('exclamation_ratio', 0):>12.4f}  {ds2.get('exclamation_ratio', 0):>12.4f}")
        print(f"  {'Hapax Legomena Ratio':<30}  {vr1.get('hapax_legomena_ratio', 0):>12.4f}  {vr2.get('hapax_legomena_ratio', 0):>12.4f}")
        yules_label = "Yule's K"
        simpsons_label = "Simpson's Diversity"
        print(f"  {yules_label:<30}  {vr1.get('yules_k', 0):>12.4f}  {vr2.get('yules_k', 0):>12.4f}")
        print(f"  {simpsons_label:<30}  {vr1.get('simpsons_diversity', 0):>12.4f}  {vr2.get('simpsons_diversity', 0):>12.4f}")

        # Top POS differences
        from ..analysis.metrics import POS_TAGS
        pos1 = ds1.get('pos_ratios', {})
        pos2 = ds2.get('pos_ratios', {})
        if pos1 and pos2:
            pos_diffs = sorted(
                [(tag, abs(pos1.get(tag, 0) - pos2.get(tag, 0))) for tag in POS_TAGS],
                key=lambda x: x[1],
                reverse=True
            )
            print(f"\n  Top POS Tag Differences:")
            for tag, diff in pos_diffs[:5]:
                print(f"    {tag:<8}  A={pos1.get(tag, 0):.4f}  B={pos2.get(tag, 0):.4f}  \u0394={diff:.4f}")

        # Sentence length distribution
        sl1 = ds1.get('sentence_length_distribution', {})
        sl2 = ds2.get('sentence_length_distribution', {})
        if sl1 and sl2:
            print(f"\n  Sentence Length Distribution:")
            print(f"    {'':>12}  {'A':>12}  {'B':>12}")
            for key in ['mean', 'median', 'std_dev', 'min', 'max']:
                print(f"    {key:>12}  {sl1.get(key, 0):>12.2f}  {sl2.get(key, 0):>12.2f}")

        print("\n" + "="*60)

        # Option to save comparison
        save_choice = input("\nSave comparison results to file? (y/n): ").strip().lower()
        if save_choice == 'y':
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"profile_comparison_{timestamp}.txt"
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("STYLOMETRIC COMPARISON\n")
                    f.write("="*60 + "\n")
                    f.write(f"A: {name1}\n")
                    f.write(f"B: {name2}\n\n")
                    f.write(f"Combined Score:     {combined:.4f}\n")
                    f.write(f"Cosine Similarity:  {cosine:.4f}\n")
                    f.write(f"Burrows' Delta:     {burrows:.4f}\n")
                    f.write(f"N-gram Overlap:     {ngram:.4f}\n")
                print(f"\u2713 Comparison saved as: {filename}")
            except Exception as e:
                print(f"\u2717 Failed to save: {e}")

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
        print("Educational analogy generation agent.")
        print("Analyzes content for abstract or technical concepts and")
        print("generates simple, accurate real-world analogies.")
        print()
        print("Rules:")
        print("  - Preserves original meaning (no distortion)")
        print("  - Concise & structured output")
        print("  - Practical examples when helpful")
        print("  - Supplements, never replaces, the original text")
        print()

        # --- Model ---
        print("Select a model for analogy generation:")
        select_model_interactive()
        model_info = get_current_model_info()
        if not model_info['has_model']:
            print("No model selected. Returning.")
            return

        # --- Input ---
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

        # --- Density analysis (instant, no LLM) ---
        density = detect_conceptual_density(text)
        print(f"\nOverall conceptual density: {density['overall_density']:.3f}")
        print(f"High-density passages (>={CONCEPTUAL_DENSITY_THRESHOLD}): "
              f"{density['high_density_count']}")

        if density['high_density_count'] == 0:
            print("\nNo passages exceed the density threshold ΓÇö no analogies needed.")
            input("\nPress Enter to continue...")
            return

        # --- Domain selection ---
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

        # --- Generate analogies ---
        print(f"\nGenerating analogies (domain: {ANALOGY_DOMAINS[domain]['label']})...")
        print("This may take 30-60 seconds while the model processes...")
        injector = AnalogyInjector(domain=domain)

        result = injector.augment_text(
            text,
            use_local=True,
            model_name=model_info['selected_model'],
        )

        # --- Display ---
        print(f"\nGenerated {result['analogy_count']} cognitive note(s).")
        if result['analogy_count'] > 0:
            # Show structured analogy breakdown first
            print("\n" + "="*60)
            print("ANALOGY BREAKDOWN")
            print("="*60)
            for i, item in enumerate(result.get('analogies', []), 1):
                print(f"\n  {i}. Original passage (density {item['density_score']:.2f}):")
                preview = item['source_sentence'][:100]
                if len(item['source_sentence']) > 100:
                    preview += "..."
                print(f"     \"{preview}\"")
                if item.get('concept'):
                    print(f"     Concept:  {item['concept']}")
                print(f"     Analogy:  {item['analogy']}")
                if item.get('example'):
                    print(f"     Example:  {item['example']}")
            print()

            # Then show full augmented text
            print("="*60)
            print("AUGMENTED TEXT (original + cognitive notes)")
            print("="*60)
            print(result['augmented_text'])
            print("="*60)
        else:
            print("\nThe model returned a response but no analogies could be")
            print("parsed. This can happen with very short text or if the")
            print("model output was in an unexpected format.")
            print(f"\nDense passages found: {result['density_report']['high_density_count']}")

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