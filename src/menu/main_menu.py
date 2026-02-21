"""
Main menu module for Style Transfer AI.
Handles primary navigation and menu display.
"""

import sys
import subprocess
import os
from ..config.settings import AVAILABLE_MODELS
from ..analysis.analyzer import create_enhanced_style_profile
from ..analysis.metrics import calculate_style_similarity, extract_deep_stylometry
from ..storage.local_storage import list_local_profiles, load_local_profile, cleanup_old_reports, save_style_profile_locally
from .model_selection import (
    select_model_interactive, 
    reset_model_selection, 
    get_current_model_info
)
from ..models.openai_client import setup_openai_client
from ..models.gemini_client import setup_gemini_client
from ..models.ollama_client import list_ollama_models, pull_ollama_model, is_ollama_installed
from ..utils.user_profile import get_file_paths
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
    print("1. Analyze Writing Style (Complete Analysis)")
    print("2. Quick Style Analysis (Statistical Only)")
    print("3. View Existing Style Profiles")
    print("")
    print("CONTENT GENERATION:")
    print("4. Generate Content with Style Profile")
    print("5. Transfer Content to Different Style")
    print("6. Style Comparison & Analysis")
    print("")
    print("COMPARISON & EVALUATION:")
    print("7. Compare Saved Profiles (Deep Stylometry)")
    print("")
    print("DATA MANAGEMENT:")
    print("8. Cleanup Old Reports")
    print("9. Manage Local Models (Ollama)")
    print("10. Switch Analysis Model")
    print("11. Check Configuration")
    print("12. Launch GUI")
    print("13. Run Scripts (Utilities/Install)")
    print("0. Exit")
    print("="*60)


def handle_analyze_style(processing_mode='enhanced'):
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
        
        # Prepare model parameters based on selection
        if model_info['use_local_model']:
            # Local Ollama model
            style_profile = create_enhanced_style_profile(
                input_data, 
                use_local=True, 
                model_name=model_info['selected_model'], 
                processing_mode=processing_mode
            )
        else:
            # Cloud API model
            model_type = None
            api_client = None
            
            if 'gpt' in model_info['selected_model'].lower():
                model_type = 'openai'
                client, message = setup_openai_client(model_info['user_chosen_api_key'])
                if client:
                    api_client = client
                    print(f"✓ {message}")
                else:
                    print(f"✗ Failed to setup OpenAI client: {message}")
                    return
            elif 'gemini' in model_info['selected_model'].lower():
                model_type = 'gemini'
                client, message = setup_gemini_client()
                if client:
                    api_client = client
                    print(f"✓ {message}")
                else:
                    print(f"✗ Failed to setup Gemini client: {message}")
                    return
            
            if not api_client:
                print("✗ Failed to initialize API client")
                return
            
            style_profile = create_enhanced_style_profile(
                input_data,
                use_local=False,
                api_type=model_type,
                api_client=api_client,
                processing_mode=processing_mode
            )
        
        # Save the analysis results locally
        if style_profile:
            print("\nSaving analysis results...")
            save_result = save_style_profile_locally(style_profile)
            if save_result['success']:
                print(f"✓ {save_result['message']}")
                
            else:
                print(f"✗ Failed to save results: {save_result.get('error', 'Unknown error')}")
        
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


def handle_cleanup_reports():
    """Handle cleanup of old reports."""
    print("\nCleaning up old reports...")
    
    # Get cleanup preferences
    try:
        days = input("Keep reports from last how many days? (default: 30): ").strip()
        if not days:
            days = 30
        else:
            days = int(days)
        
        result = cleanup_old_reports(days_to_keep=days)
        
        if result['success']:
            print(f"SUCCESS: {result['message']}")
            if result['deleted_files']:
                print("Deleted files:")
                for file in result['deleted_files']:
                    print(f"  - {file}")
        else:
            print(f"ERROR: {result['error']}")
            
    except ValueError:
        print("Invalid input. Using default 30 days.")
        result = cleanup_old_reports(days_to_keep=30)
        if result['success']:
            print(f"SUCCESS: {result['message']}")
    except Exception as e:
        print(f"Error during cleanup: {e}")
    
    input("\nPress Enter to continue...")


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
                print("✅ All system requirements are met!")
            else:
                print("❌ Some issues were found:")
                for issue in results.get('issues', []):
                    print(f"  • {issue}")
            
            if results.get('warnings'):
                print("\n⚠️  Warnings:")
                for warning in results['warnings']:
                    print(f"  • {warning}")
            
            if results.get('recommendations'):
                print("\n💡 Recommendations:")
                for rec in results['recommendations']:
                    print(f"  • {rec}")
            
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
                print(f"✅ Python version: {sys.version.split()[0]}")
                
                # Check basic imports
                basic_modules = ['sys', 'os', 'json', 'datetime']
                for module in basic_modules:
                    try:
                        __import__(module)
                        print(f"✅ {module}: Available")
                    except ImportError:
                        print(f"❌ {module}: Missing")
                
                # Check for requests
                try:
                    import requests
                    print("✅ requests: Available")
                except ImportError:
                    print("❌ requests: Missing (required for API calls)")
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
                        status_symbol = "✓" if success else "✗"
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
            print("\n✓ Opened a new PowerShell window.")
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
            print("\n✗ Streamlit is not installed. Run: pip install streamlit")
            input("\nPress Enter to continue...")
            return

        app_path = "app.py"
        if not os.path.exists(app_path):
            app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "app.py")

        subprocess.Popen([sys.executable, "-m", "streamlit", "run", app_path])
        print("\n✓ Streamlit GUI launched in your browser.")
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

                    print(f"\n✓ Launched: {selected['label']}")
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
                if model_info['use_local_model']:
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
                else:
                    # Setup API client
                    api_client = None
                    api_type = None
                    
                    if 'gpt' in model_info['selected_model'].lower():
                        api_type = 'openai'
                        client, message = setup_openai_client(model_info['user_chosen_api_key'])
                        if client:
                            api_client = client
                        else:
                            print(f"Failed to setup OpenAI client: {message}")
                            return
                    elif 'gemini' in model_info['selected_model'].lower():
                        api_type = 'gemini'
                        client, message = setup_gemini_client()
                        if client:
                            api_client = client
                        else:
                            print(f"Failed to setup Gemini client: {message}")
                            return
                    
                    result = generator.generate_content(
                        style_profile=profile_data,
                        content_type=content_type,
                        topic_or_prompt=topic,
                        target_length=target_length,
                        tone=tone,
                        additional_context=context,
                        use_local=False,
                        api_type=api_type,
                        api_client=api_client
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
                if model_info['use_local_model']:
                    result = transfer.transfer_style(
                        original_content=original_content,
                        target_style_profile=profile_data,
                        transfer_type=transfer_type,
                        intensity=intensity,
                        preserve_elements=preserve_list,
                        use_local=True,
                        model_name=model_info['selected_model']
                    )
                else:
                    # Setup API client
                    api_client = None
                    api_type = None
                    
                    if 'gpt' in model_info['selected_model'].lower():
                        api_type = 'openai'
                        client, message = setup_openai_client(model_info['user_chosen_api_key'])
                        if client:
                            api_client = client
                        else:
                            print(f"Failed to setup OpenAI client: {message}")
                            return
                    elif 'gemini' in model_info['selected_model'].lower():
                        api_type = 'gemini'
                        client, message = setup_gemini_client()
                        if client:
                            api_client = client
                        else:
                            print(f"Failed to setup Gemini client: {message}")
                            return
                    
                    result = transfer.transfer_style(
                        original_content=original_content,
                        target_style_profile=profile_data,
                        transfer_type=transfer_type,
                        intensity=intensity,
                        preserve_elements=preserve_list,
                        use_local=False,
                        api_type=api_type,
                        api_client=api_client
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


def handle_compare_profiles():
    """Compare two saved stylometric profiles using deep stylometry similarity."""
    try:
        print("\n" + "="*60)
        print("COMPARE SAVED STYLOMETRIC PROFILES")
        print("="*60)

        profiles = list_local_profiles()
        if not profiles or len(profiles) < 2:
            print("\nYou need at least 2 saved profiles to compare.")
            print("Run a style analysis first to create profiles.")
            input("\nPress Enter to continue...")
            return

        # Display available profiles
        print(f"\nFound {len(profiles)} saved profiles:\n")
        for i, p in enumerate(profiles, 1):
            filename = os.path.basename(p['filename'])
            user_name = filename.split('_stylometric_profile_')[0] if '_stylometric_profile_' in filename else filename.replace('.json', '')
            size_kb = p['size'] / 1024
            print(f"  {i:2d}. {filename}  ({user_name}, {size_kb:.1f} KB, {p['modified']})")

        # Select first profile
        print("\n--- Select the FIRST profile ---")
        choice1 = input("Enter profile number: ").strip()
        try:
            idx1 = int(choice1) - 1
            if not (0 <= idx1 < len(profiles)):
                print("Invalid selection.")
                return
        except ValueError:
            print("Invalid input.")
            return

        # Select second profile
        print("--- Select the SECOND profile ---")
        choice2 = input("Enter profile number: ").strip()
        try:
            idx2 = int(choice2) - 1
            if not (0 <= idx2 < len(profiles)):
                print("Invalid selection.")
                return
        except ValueError:
            print("Invalid input.")
            return

        if idx1 == idx2:
            print("Please select two different profiles.")
            return

        # Load both profiles
        print("\nLoading profiles...")
        result1 = load_local_profile(profiles[idx1]['filename'])
        result2 = load_local_profile(profiles[idx2]['filename'])

        if not result1['success']:
            print(f"Failed to load profile 1: {result1.get('message', 'Unknown error')}")
            return
        if not result2['success']:
            print(f"Failed to load profile 2: {result2.get('message', 'Unknown error')}")
            return

        profile_data1 = result1['profile']
        profile_data2 = result2['profile']

        name1 = os.path.basename(profiles[idx1]['filename']).replace('.json', '')
        name2 = os.path.basename(profiles[idx2]['filename']).replace('.json', '')

        # Extract deep stylometry from saved profiles
        ds1 = profile_data1.get('deep_stylometry', {})
        ds2 = profile_data2.get('deep_stylometry', {})

        if not ds1 or not ds2:
            print("\nOne or both profiles lack deep stylometry data.")
            print("Re-analyze the text with v1.3.0+ to include deep stylometry.")
            input("\nPress Enter to continue...")
            return

        # Calculate similarity
        print("\nCalculating deep stylometry similarity...\n")
        similarity = calculate_style_similarity(ds1, ds2)

        # Display results
        print("="*60)
        print("PROFILE COMPARISON RESULTS")
        print("="*60)
        print(f"\n  Profile A: {name1}")
        print(f"  Profile B: {name2}")
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

        # Vocabulary richness
        vr1 = ds1.get('vocabulary_richness', {})
        vr2 = ds2.get('vocabulary_richness', {})
        header = f"\n  {'Metric':<30}  {'Profile A':>12}  {'Profile B':>12}"
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
            print(f"    {'':>12}  {'Profile A':>12}  {'Profile B':>12}")
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
                    f.write("STYLOMETRIC PROFILE COMPARISON\n")
                    f.write("="*60 + "\n")
                    f.write(f"Profile A: {name1}\n")
                    f.write(f"Profile B: {name2}\n\n")
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
        print(f"\nError comparing profiles: {e}")
        input("\nPress Enter to continue...")


def handle_style_comparison():
    """Handle style comparison between two content samples."""
    try:
        print("\n" + "="*50)
        print("STYLE COMPARISON & ANALYSIS")
        print("="*50)
        
        # Initialize style transfer for comparison functionality
        transfer = StyleTransfer()
        
        # Get first content sample
        print("1. First content sample:")
        content1_choice = input("Enter content (1) or file path (2)? (1/2): ").strip()
        
        content1 = ""
        if content1_choice == "1":
            print("Enter first content (press Enter twice to finish):")
            lines = []
            while True:
                line = input()
                if line == "" and len(lines) > 0:
                    if lines[-1] == "":
                        break
                lines.append(line)
            content1 = "\n".join(lines[:-1])
        elif content1_choice == "2":
            file_path = input("Enter file path: ").strip()
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content1 = f.read()
            except Exception as e:
                print(f"Failed to read file: {e}")
                return
        
        # Get second content sample
        print("\n2. Second content sample:")
        content2_choice = input("Enter content (1) or file path (2)? (1/2): ").strip()
        
        content2 = ""
        if content2_choice == "1":
            print("Enter second content (press Enter twice to finish):")
            lines = []
            while True:
                line = input()
                if line == "" and len(lines) > 0:
                    if lines[-1] == "":
                        break
                lines.append(line)
            content2 = "\n".join(lines[:-1])
        elif content2_choice == "2":
            file_path = input("Enter file path: ").strip()
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content2 = f.read()
            except Exception as e:
                print(f"Failed to read file: {e}")
                return
        
        if not content1.strip() or not content2.strip():
            print("Both content samples are required.")
            return
        
        print("\nAnalyzing style differences...")
        
        # Perform style comparison
        comparison = transfer.compare_styles(content1, content2)
        
        if 'error' in comparison:
            print(f"Comparison failed: {comparison['error']}")
        else:
            print("\n" + "="*60)
            print("STYLE COMPARISON RESULTS")
            print("="*60)
            
            # Display basic statistics
            analysis1 = comparison.get('content1_analysis', {})
            analysis2 = comparison.get('content2_analysis', {})
            
            print("\nCONTENT 1 ANALYSIS:")
            print(f"Word Count: {analysis1.get('word_count', 'N/A')}")
            print(f"Sentence Count: {analysis1.get('sentence_count', 'N/A')}")
            print(f"Avg Sentence Length: {analysis1.get('avg_sentence_length', 'N/A'):.1f}")
            print(f"Lexical Diversity: {analysis1.get('lexical_diversity', 'N/A'):.3f}")
            print(f"Formality Level: {analysis1.get('formality_level', 'N/A')}")
            
            print("\nCONTENT 2 ANALYSIS:")
            print(f"Word Count: {analysis2.get('word_count', 'N/A')}")
            print(f"Sentence Count: {analysis2.get('sentence_count', 'N/A')}")
            print(f"Avg Sentence Length: {analysis2.get('avg_sentence_length', 'N/A'):.1f}")
            print(f"Lexical Diversity: {analysis2.get('lexical_diversity', 'N/A'):.3f}")
            print(f"Formality Level: {analysis2.get('formality_level', 'N/A')}")
            
            # Display comparison metrics
            print(f"\nSIMILARITY SCORE: {comparison.get('similarity_score', 'N/A'):.3f}")
            
            differences = comparison.get('style_differences', {})
            print(f"\nSTYLE DIFFERENCES:")
            print(f"Sentence Length Difference: {differences.get('sentence_length_diff', 'N/A'):.1f} words")
            print(f"Formality Difference: {'Yes' if differences.get('formality_diff', False) else 'No'}")
            
            # Display recommendations
            recommendations = comparison.get('recommendations', [])
            if recommendations:
                print(f"\nRECOMMENDATIONS:")
                for rec in recommendations:
                    print(f"• {rec}")
            
            print("="*60)
            
            # Save comparison results
            save_choice = input("\nSave comparison results? (y/n): ").strip().lower()
            if save_choice == 'y':
                timestamp = comparison.get('timestamp', 'unknown')
                filename = f"style_comparison_{timestamp}.txt"
                
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write("STYLE COMPARISON RESULTS\n")
                        f.write("="*60 + "\n\n")
                        f.write("CONTENT 1:\n")
                        f.write(content1[:300] + "...\n\n" if len(content1) > 300 else content1 + "\n\n")
                        f.write("CONTENT 2:\n")
                        f.write(content2[:300] + "...\n\n" if len(content2) > 300 else content2 + "\n\n")
                        f.write(f"Similarity Score: {comparison.get('similarity_score', 'N/A')}\n")
                        f.write(f"Differences: {differences}\n")
                        if recommendations:
                            f.write(f"Recommendations: {recommendations}\n")
                    
                    print(f"Comparison results saved as: {filename}")
                except Exception as e:
                    print(f"Failed to save results: {e}")
        
        input("\nPress Enter to continue...")
        
    except KeyboardInterrupt:
        print("\n\nReturning to main menu...")
    except Exception as e:
        print(f"\nError in style comparison: {e}")


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
                handle_analyze_style(processing_mode='enhanced')
                
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
                handle_compare_profiles()
                
            elif choice == "8":
                handle_cleanup_reports()
                
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