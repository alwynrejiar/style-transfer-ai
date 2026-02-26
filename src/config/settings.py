"""
Configuration settings for Style Transfer AI.
Contains all constants, API endpoints, and model configurations.
"""

# Application Information
APPLICATION_NAME = "Style Transfer AI"
VERSION = "1.2.0"
AUTHOR = "Style Transfer AI Team"

# API Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OPENAI_API_KEY = "your-openai-api-key-here"  # Replace with your actual OpenAI API key
GEMINI_API_KEY = "your-gemini-api-key-here"  # Replace with your actual Gemini API key

# Available AI Models
AVAILABLE_MODELS = {
    "gpt-oss:20b": {
        "description": "GPT-OSS 20B (Advanced, Slower)",
        "type": "ollama"
    },
    "gemma3:1b": {
        "description": "Gemma 3:1B (Fast, Efficient)", 
        "type": "ollama"
    },
    "gpt-3.5-turbo": {
        "description": "OpenAI GPT-3.5 Turbo",
        "type": "openai"
    },
    "gemini-1.5-flash": {
        "description": "Google Gemini 1.5 Flash",
        "type": "gemini"
    }
}

# Processing Modes
PROCESSING_MODES = {
    "enhanced": {
        "description": "Complete 25-point stylometry analysis with statistical metrics",
        "features": ["Deep Analysis", "Statistical Metrics", "Readability Scores", "Style Profiling"],
        "temperature": 0.2,
        "timeout": 180,
        "gpt_oss_tokens": 3000,
        "gemma_tokens": 2000
    },
    "statistical": {
        "description": "Statistical analysis only (word count, readability, etc.)",
        "features": ["Statistical Metrics", "Readability Scores", "Basic Analysis"],
        "temperature": 0.3,
        "timeout": 120,
        "gpt_oss_tokens": 2000,
        "gemma_tokens": 1500
    }
}

# File Processing
DEFAULT_FILE_PATHS = [
    "data/samples/about_my_pet.txt",
    "data/samples/about_my_pet_1.txt",
    "data/samples/about_my_pet_2.txt"
]
SUPPORTED_ENCODINGS = ["utf-8", "latin-1"]
MAX_FILENAME_LENGTH = 30

# Output Configuration
DEFAULT_OUTPUT_BASE = "user_style_profile_enhanced"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# Menu Configuration
MAIN_MENU_WIDTH = 60
SUB_MENU_WIDTH = 40

# ---------------------------------------------------------------------------
# Analogy Engine / Cognitive Bridging Configuration
# ---------------------------------------------------------------------------

# Toggle analogy augmentation globally (can be overridden at request level)
ANALOGY_AUGMENTATION_ENABLED = False

# Supported analogy domains — each maps to a display label and a short
# description used inside LLM prompts so the model picks relevant metaphors.
ANALOGY_DOMAINS = {
    "sports": {
        "label": "Sports",
        "description": "Use sports metaphors and athletic analogies (football, basketball, track & field, etc.)"
    },
    "gaming": {
        "label": "Gaming",
        "description": "Use video-game and board-game analogies (levels, power-ups, strategy, respawn, etc.)"
    },
    "cooking": {
        "label": "Cooking",
        "description": "Use culinary and kitchen-based analogies (recipes, ingredients, seasoning, baking, etc.)"
    },
    "nature": {
        "label": "Nature",
        "description": "Use nature and ecology analogies (ecosystems, weather, rivers, growth, seasons, etc.)"
    },
    "daily_life": {
        "label": "Daily Life",
        "description": "Use everyday household and routine analogies (commuting, shopping, organising, etc.)"
    },
    "tech": {
        "label": "Tech",
        "description": "Use technology and software analogies (apps, networks, debugging, upgrades, etc.)"
    },
    "general_simplification": {
        "label": "General Simplification",
        "description": "Simplify using the clearest everyday analogy regardless of domain"
    },
}

DEFAULT_ANALOGY_DOMAIN = "general_simplification"

# Conceptual density threshold — sentences scoring above this trigger analogies.
# Range 0.0-1.0.  Higher = only the densest sentences get analogies.
# Most academic/technical sentences score 0.45-0.70; casual text scores 0.25-0.40.
CONCEPTUAL_DENSITY_THRESHOLD = 0.45