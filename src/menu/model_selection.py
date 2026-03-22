"""
Model selection module for Style Transfer AI.
Handles interactive model selection and validation.
"""

from ..config.settings import AVAILABLE_MODELS, OLLAMA_BASE_URL, PROCESSING_MODES, GEMINI_API_KEY
from ..models.ollama_client import check_ollama_connection, pull_ollama_model
from ..models.remote_ollama_client import (
    setup_remote_ollama,
    select_remote_model,
    check_remote_connection,
    get_selected_remote_model,
)
from ..models.gemini_client import setup_gemini_client

# Global model selection state
USE_LOCAL_MODEL = False
SELECTED_MODEL = None
SELECTED_API_TYPE = None      # "gemini" | "openai" | None
SELECTED_API_CLIENT = None    # API key string for cloud models
USER_CHOSEN_API_KEY = None    # kept for backwards compatibility


def reset_model_selection():
    """Reset the model selection state to force new selection."""
    global USE_LOCAL_MODEL, SELECTED_MODEL, SELECTED_API_TYPE, SELECTED_API_CLIENT, USER_CHOSEN_API_KEY
    USE_LOCAL_MODEL = False
    SELECTED_MODEL = None
    SELECTED_API_TYPE = None
    SELECTED_API_CLIENT = None
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

    print("0. Exit model selection")
    print("="*60)


def validate_model_selection(model_key):
    """
    Validate that the selected model is available and properly configured.
    Sets USE_LOCAL_MODEL, SELECTED_MODEL, SELECTED_API_TYPE, SELECTED_API_CLIENT
    on success.
    """
    global USE_LOCAL_MODEL, SELECTED_MODEL, SELECTED_API_TYPE, SELECTED_API_CLIENT, USER_CHOSEN_API_KEY

    if model_key not in AVAILABLE_MODELS:
        return {'success': False, 'error': f"Unknown model: {model_key}"}

    model_info = AVAILABLE_MODELS[model_key]
    model_type = model_info['type']

    print(f"\nValidating {model_key}...")

    # ── Remote Ollama ──────────────────────────────────────────────────────
    if model_key == 'remote-ollama':
        success, message = setup_remote_ollama()
        if not success:
            return {'success': False, 'error': message}
        print(f"✔ {message}")
        selected, sel_msg = select_remote_model()
        if not selected:
            return {'success': False, 'error': sel_msg}
        USE_LOCAL_MODEL = True
        SELECTED_MODEL = model_key
        SELECTED_API_TYPE = None
        SELECTED_API_CLIENT = None
        return {'success': True, 'message': f"✔ {sel_msg}"}

    # ── Local Ollama ───────────────────────────────────────────────────────
    elif model_type == 'ollama':
        is_available, message = check_ollama_connection(model_key)
        if is_available:
            USE_LOCAL_MODEL = True
            SELECTED_MODEL = model_key
            SELECTED_API_TYPE = None
            SELECTED_API_CLIENT = None
            return {'success': True, 'message': f"✔ {message}"}

        download_choice = input(f"{message}\nDownload {model_key} now? (y/n): ").strip().lower()
        if download_choice == "y":
            success, pull_message = pull_ollama_model(model_key)
            if success:
                recheck_available, recheck_message = check_ollama_connection(model_key)
                if recheck_available:
                    USE_LOCAL_MODEL = True
                    SELECTED_MODEL = model_key
                    SELECTED_API_TYPE = None
                    SELECTED_API_CLIENT = None
                    return {'success': True, 'message': f"✔ {recheck_message}"}
                return {'success': False, 'error': recheck_message}
            return {'success': False, 'error': pull_message}

        return {
            'success': False,
            'error': message,
            'suggestion': f"Run: ollama pull {model_key}"
        }

    # ── Google Gemini API ──────────────────────────────────────────────────
    elif model_type == 'gemini':
        key = GEMINI_API_KEY
        if not key:
            print("\nGemini requires a Google API key.")
            print("Get one at: https://aistudio.google.com/apikey")
            key = input("Enter your Gemini API key: ").strip()
        else:
            print("  Using GEMINI_API_KEY from environment.")

        if not key:
            return {'success': False, 'error': "No API key provided."}

        success, message = setup_gemini_client(api_key=key)
        if not success:
            return {'success': False, 'error': message}

        USE_LOCAL_MODEL = False
        SELECTED_MODEL = model_key
        SELECTED_API_TYPE = "gemini"
        SELECTED_API_CLIENT = key
        USER_CHOSEN_API_KEY = key
        return {'success': True, 'message': f"✔ {message}"}

    # ── OpenAI API ─────────────────────────────────────────────────────────
    elif model_type == 'openai':
        print("\nOpenAI requires an API key.")
        print("Get one at: https://platform.openai.com/account/api-keys")
        key = input("Enter your OpenAI API key: ").strip()
        if not key:
            return {'success': False, 'error': "No API key provided."}

        USE_LOCAL_MODEL = False
        SELECTED_MODEL = model_key
        SELECTED_API_TYPE = "openai"
        SELECTED_API_CLIENT = key
        USER_CHOSEN_API_KEY = key
        return {'success': True, 'message': f"✔ OpenAI model '{model_key}' configured"}

    else:
        return {'success': False, 'error': f"Unknown model type: {model_type}"}


def select_model_interactive():
    """Interactive model selection with validation."""
    global USE_LOCAL_MODEL, SELECTED_MODEL

    while True:
        display_model_menu()

        try:
            choice = input("\nSelect a model (0 to exit): ").strip()

            if choice == "0":
                break

            try:
                model_index = int(choice) - 1
                model_keys = list(AVAILABLE_MODELS.keys())

                if 0 <= model_index < len(model_keys):
                    selected_model_key = model_keys[model_index]
                    print(f"\nSelected: {selected_model_key}")
                    validation_result = validate_model_selection(selected_model_key)

                    if validation_result['success']:
                        print(f"SUCCESS: {validation_result['message']}")
                        show_processing_modes()
                        confirm = input(f"\nUse {selected_model_key} for analysis? (y/n): ").strip().lower()
                        if confirm == 'y':
                            print(f"\n✔ Model set to : {selected_model_key}")
                            print(f"✔ Local        : {USE_LOCAL_MODEL}")
                            print(f"✔ API type     : {SELECTED_API_TYPE or 'n/a (Ollama)'}")
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

    Returns everything needed to call analyze_style() or
    create_enhanced_style_profile():
        use_local_model  → bool
        selected_model   → str | None
        api_type         → "gemini" | "openai" | None
        api_client       → API key string | None
        has_model        → bool
        model_type       → str
    """
    global USE_LOCAL_MODEL, SELECTED_MODEL, SELECTED_API_TYPE, SELECTED_API_CLIENT, USER_CHOSEN_API_KEY

    return {
        'use_local_model':     USE_LOCAL_MODEL,
        'selected_model':      SELECTED_MODEL,
        'api_type':            SELECTED_API_TYPE,
        'api_client':          SELECTED_API_CLIENT,
        'user_chosen_api_key': USER_CHOSEN_API_KEY,  # backwards compat
        'has_model':           SELECTED_MODEL is not None,
        'model_type': (
            AVAILABLE_MODELS.get(SELECTED_MODEL, {}).get('type', 'unknown')
            if SELECTED_MODEL else 'none'
        ),
    }


def set_model_configuration(use_local=False, model_name=None, api_key=None, api_type=None):
    """Programmatically set model configuration."""
    global USE_LOCAL_MODEL, SELECTED_MODEL, SELECTED_API_TYPE, SELECTED_API_CLIENT, USER_CHOSEN_API_KEY
    USE_LOCAL_MODEL = use_local
    SELECTED_MODEL = model_name
    SELECTED_API_TYPE = api_type
    SELECTED_API_CLIENT = api_key
    USER_CHOSEN_API_KEY = api_key


def auto_select_best_available_model():
    """Try Ollama models first; return first that connects."""
    print("Auto-detecting best available model...")
    for model_key, model_info in AVAILABLE_MODELS.items():
        if model_info['type'] == 'ollama':
            validation_result = validate_model_selection(model_key)
            if validation_result['success']:
                print(f"✔ Auto-selected: {model_key} (Local Ollama)")
                return {
                    'success': True,
                    'model': model_key,
                    'type': 'ollama',
                    'message': f"Auto-selected local model: {model_key}"
                }
    return {
        'success': False,
        'error': 'No Ollama models reachable. Select Gemini manually.'
    }


def require_model_selection():
    """Ensure a model is selected; prompt interactively if not."""
    global SELECTED_MODEL
    if SELECTED_MODEL:
        return True
    print("\nNo model selected. Please choose a model for analysis.")
    select_model_interactive()
    return SELECTED_MODEL is not None


if __name__ == "__main__":
    select_model_interactive()