"""
Model selection module for Style Transfer AI.
Handles interactive model selection and validation.
"""

from ..config.settings import AVAILABLE_MODELS, OLLAMA_BASE_URL, PROCESSING_MODES
from ..models.ollama_client import check_ollama_connection, pull_ollama_model, list_ollama_models
from ..models.openai_client import setup_openai_client, get_api_key as get_openai_api_key
from ..models.gemini_client import setup_gemini_client, get_api_key as get_gemini_api_key

# Global model selection state
USE_LOCAL_MODEL = False
SELECTED_MODEL = None
USER_CHOSEN_API_KEY = None


def reset_model_selection():
    """Reset the model selection state to force new selection."""
    global USE_LOCAL_MODEL, SELECTED_MODEL, USER_CHOSEN_API_KEY
    USE_LOCAL_MODEL = False
    SELECTED_MODEL = None
    USER_CHOSEN_API_KEY = None


def display_model_menu():
    """Display available models and selection options."""
    print("\n" + "="*60)
    print("MODEL SELECTION - STYLE TRANSFER AI")
    print("="*60)
    print("Available Models:")
    print("-" * 60)
    
    for i, (model_key, model_info) in enumerate(AVAILABLE_MODELS.items(), 1):
        print(f"{i}. {model_key}")
        print(f"   Description: {model_info['description']}")
        print(f"   Type: {model_info['type']}")
        print()
    
    print("C. Choose from installed Ollama models")
    print("0. Exit model selection")
    print("="*60)


def select_installed_ollama_model():
    """Select from locally installed Ollama models."""
    global USE_LOCAL_MODEL, SELECTED_MODEL

    models, error = list_ollama_models()
    if error:
        print(f"\n✗ {error}")
        return False
    if not models:
        print("\nNo Ollama models found. Download one from the model menu first.")
        return False

    print("\nInstalled Ollama Models:")
    for idx, model in enumerate(models, 1):
        print(f"{idx}. {model}")
    print("0. Cancel")

    choice = input("\nSelect a model (0 to cancel): ").strip()
    if choice == "0":
        return False

    try:
        model_index = int(choice) - 1
        if 0 <= model_index < len(models):
            selected_model = models[model_index]
            USE_LOCAL_MODEL = True
            SELECTED_MODEL = selected_model
            print(f"\n✓ Model set to: {selected_model}")
            return True
        print("Invalid selection. Please try again.")
        return False
    except ValueError:
        print("Invalid input. Please enter a number.")
        return False


def validate_model_selection(model_key):
    """
    Validate that the selected model is available and properly configured.
    
    Args:
        model_key (str): The model key to validate
        
    Returns:
        dict: Validation result with success status and details
    """
    global USE_LOCAL_MODEL, SELECTED_MODEL, USER_CHOSEN_API_KEY
    
    if model_key not in AVAILABLE_MODELS:
        return {
            'success': False,
            'error': f"Unknown model: {model_key}"
        }
    
    model_info = AVAILABLE_MODELS[model_key]
    model_type = model_info['type']
    
    print(f"\nValidating {model_key}...")
    
    if model_type == 'ollama':
        # Test Ollama connection and model availability
        is_available, message = check_ollama_connection(model_key)
        if is_available:
            USE_LOCAL_MODEL = True
            SELECTED_MODEL = model_key
            return {
                'success': True,
                'message': f"✓ {message}"
            }
        download_choice = input(f"{message}\nDownload {model_key} now? (y/n): ").strip().lower()
        if download_choice == "y":
            success, pull_message = pull_ollama_model(model_key)
            if success:
                recheck_available, recheck_message = check_ollama_connection(model_key)
                if recheck_available:
                    USE_LOCAL_MODEL = True
                    SELECTED_MODEL = model_key
                    return {
                        'success': True,
                        'message': f"✓ {recheck_message}"
                    }
                return {
                    'success': False,
                    'error': recheck_message
                }
            return {
                'success': False,
                'error': pull_message
            }

        return {
            'success': False,
            'error': message,
            'suggestion': f"Run: ollama pull {model_key}"
        }
    
    elif model_type == 'openai':
        # Test OpenAI API
        client, message = setup_openai_client()
        if client:
            USE_LOCAL_MODEL = False
            SELECTED_MODEL = model_key
            # Get the API key that was used
            api_key = get_openai_api_key() if not USER_CHOSEN_API_KEY else USER_CHOSEN_API_KEY
            USER_CHOSEN_API_KEY = api_key
            return {
                'success': True,
                'message': f"✓ OpenAI API configured and ready"
            }
        else:
            return {
                'success': False,
                'error': message
            }
    
    elif model_type == 'gemini':
        # Test Gemini API
        client, message = setup_gemini_client()
        if client:
            USE_LOCAL_MODEL = False
            SELECTED_MODEL = model_key
            return {
                'success': True,
                'message': f"✓ Gemini API configured and ready"
            }
        else:
            return {
                'success': False,
                'error': message
            }
    
    else:
        return {
            'success': False,
            'error': f"Unknown model type: {model_type}"
        }


def select_model_interactive():
    """Interactive model selection with validation."""
    global USE_LOCAL_MODEL, SELECTED_MODEL, USER_CHOSEN_API_KEY
    
    while True:
        display_model_menu()
        
        try:
            choice = input("\nSelect a model (0 to exit): ").strip()
            
            if choice == "0":
                break

            if choice.lower() == "c":
                if select_installed_ollama_model():
                    show_processing_modes()
                    input("Press Enter to continue...")
                    break
                input("Press Enter to continue...")
                continue
            
            try:
                model_index = int(choice) - 1
                model_keys = list(AVAILABLE_MODELS.keys())
                
                if 0 <= model_index < len(model_keys):
                    selected_model_key = model_keys[model_index]
                    
                    print(f"\nSelected: {selected_model_key}")
                    validation_result = validate_model_selection(selected_model_key)
                    
                    if validation_result['success']:
                        print(f"SUCCESS: {validation_result['message']}")
                        
                        # Show processing mode options for the selected model
                        show_processing_modes()
                        
                        # Confirm selection
                        confirm = input(f"\nUse {selected_model_key} for analysis? (y/n): ").strip().lower()
                        if confirm == 'y':
                            print(f"\n✓ Model set to: {selected_model_key}")
                            print(f"✓ Local processing: {'Yes' if USE_LOCAL_MODEL else 'No'}")
                            input("Press Enter to continue...")
                            break
                    else:
                        print(f"ERROR: {validation_result['error']}")
                        if 'suggestion' in validation_result:
                            print(f"SUGGESTION: {validation_result['suggestion']}")
                        input("Press Enter to continue...")
                else:
                    print("Invalid selection. Please try again.")
                    
            except ValueError:
                print("Invalid input. Please enter a number.")
                
        except KeyboardInterrupt:
            print("\n\nReturning to main menu...")
            break


def show_processing_modes():
    """Display available processing modes for the selected model."""
    print("\nProcessing Modes:")
    print("-" * 30)
    
    for mode_key, mode_info in PROCESSING_MODES.items():
        print(f"• {mode_key.upper()}: {mode_info['description']}")
        print(f"  Features: {', '.join(mode_info['features'])}")
    
    print("\nNote: Processing mode will be selected during analysis.")


def get_current_model_info():
    """
    Get information about the currently selected model.
    
    Returns:
        dict: Current model configuration
    """
    global USE_LOCAL_MODEL, SELECTED_MODEL, USER_CHOSEN_API_KEY
    
    return {
        'use_local_model': USE_LOCAL_MODEL,
        'selected_model': SELECTED_MODEL,
        'user_chosen_api_key': USER_CHOSEN_API_KEY,
        'has_model': SELECTED_MODEL is not None,
        'model_type': AVAILABLE_MODELS.get(SELECTED_MODEL, {}).get('type', 'unknown') if SELECTED_MODEL else 'none'
    }


def set_model_configuration(use_local=False, model_name=None, api_key=None):
    """
    Programmatically set model configuration.
    
    Args:
        use_local (bool): Whether to use local Ollama models
        model_name (str): Name of the model to use
        api_key (str): API key for cloud models
    """
    global USE_LOCAL_MODEL, SELECTED_MODEL, USER_CHOSEN_API_KEY
    
    USE_LOCAL_MODEL = use_local
    SELECTED_MODEL = model_name
    USER_CHOSEN_API_KEY = api_key


def auto_select_best_available_model():
    """
    Automatically select the best available model based on system capabilities.
    
    Returns:
        dict: Selection result with model information
    """
    print("Auto-detecting best available model...")
    
    # Try Ollama models first (for privacy and speed)
    for model_key, model_info in AVAILABLE_MODELS.items():
        if model_info['type'] == 'ollama':
            validation_result = validate_model_selection(model_key)
            if validation_result['success']:
                print(f"✓ Auto-selected: {model_key} (Local Ollama)")
                return {
                    'success': True,
                    'model': model_key,
                    'type': 'ollama',
                    'message': f"Auto-selected local model: {model_key}"
                }
    
    # Try OpenAI API as fallback
    for model_key, model_info in AVAILABLE_MODELS.items():
        if model_info['type'] == 'openai':
            validation_result = validate_model_selection(model_key)
            if validation_result['success']:
                print(f"✓ Auto-selected: {model_key} (OpenAI API)")
                return {
                    'success': True,
                    'model': model_key,
                    'type': 'openai',
                    'message': f"Auto-selected OpenAI model: {model_key}"
                }
    
    # Try Gemini API as last resort
    for model_key, model_info in AVAILABLE_MODELS.items():
        if model_info['type'] == 'gemini':
            validation_result = validate_model_selection(model_key)
            if validation_result['success']:
                print(f"✓ Auto-selected: {model_key} (Gemini API)")
                return {
                    'success': True,
                    'model': model_key,
                    'type': 'gemini',
                    'message': f"Auto-selected Gemini model: {model_key}"
                }
    
    return {
        'success': False,
        'error': 'No models available. Please check your configuration.'
    }


def require_model_selection():
    """
    Ensure a model is selected before proceeding with analysis.
    If no model is selected, prompt for selection.
    
    Returns:
        bool: True if model is selected, False otherwise
    """
    global SELECTED_MODEL
    
    if SELECTED_MODEL:
        return True
    
    print("\nNo model selected. Please choose a model for analysis.")
    
    # Manual model selection required
    print("\nManual model selection required.")
    select_model_interactive()
    
    return SELECTED_MODEL is not None


if __name__ == "__main__":
    select_model_interactive()