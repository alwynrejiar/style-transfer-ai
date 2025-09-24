"""
Main menu module for Style Transfer AI.
Handles primary navigation and menu display.
"""

import sys
import subprocess
import os
from ..config.settings import PROCESSING_MODES
from ..analysis.analyzer import create_enhanced_style_profile
from ..storage.local_storage import list_local_profiles, load_local_profile, cleanup_old_reports, save_style_profile_locally
from .model_selection import (
    select_model_interactive, 
    reset_model_selection, 
    get_current_model_info
)
from ..models.openai_client import setup_openai_client
from ..models.gemini_client import setup_gemini_client
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
    print("DATA MANAGEMENT:")
    print("7. Cleanup Old Reports")
    print("8. Switch Analysis Model")
    print("9. Check Configuration")
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
                    
                    # Handle style match score safely
                    style_score = result.get('style_match_score', 'N/A')
                    if isinstance(style_score, (int, float)):
                        print(f"Style Match Score: {style_score:.2f}")
                    else:
                        print(f"Style Match Score: {style_score}")
                    
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
                    print(f"Style Match Score: {result.get('style_match_score', 'N/A'):.2f}")
                    
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
            choice = input("\nEnter your choice (0-10): ").strip()
            
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
                handle_cleanup_reports()
                
            elif choice == "8":
                reset_model_selection()
                print("\nModel selection reset. Please choose a new model:")
                select_model_interactive()
                
            elif choice == "9":
                handle_check_configuration()
                
            else:
                print("Invalid choice. Please enter 0-10.")
                
        except KeyboardInterrupt:
            print("\n\nExiting Style Transfer AI...")
            sys.exit(0)
        except Exception as e:
            print(f"Unexpected error: {e}")


if __name__ == "__main__":
    run_main_menu()