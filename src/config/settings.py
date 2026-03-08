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
REMOTE_OLLAMA_DEFAULT_URL = "https://myollamaapi2000.share.zrok.io"
OPENAI_API_KEY = ""  # Unused – reserved for future cloud providers
GEMINI_API_KEY = ""  # Unused – reserved for future cloud providers

# Available AI Models
AVAILABLE_MODELS = {
    "gemma3:1b": {
        "description": "Gemma 3:1B (Fast, Efficient)", 
        "type": "ollama"
    },
    "remote-ollama": {
        "description": "Remote Ollama (via tunnel — select model after connecting)",
        "type": "ollama"
    }
}

# Processing Modes
PROCESSING_MODES = {
    "fast": {
        "description": "Quick stylometry analysis (fastest, recommended)",
        "features": ["Fast Analysis", "Core Metrics", "Quick Profiling"],
        "temperature": 0.3,
        "timeout": 90,
        "gemma_tokens": 2500
    },
    "statistical": {
        "description": "Balanced analysis with statistical metrics",
        "features": ["Statistical Metrics", "Readability Scores", "Balanced Analysis"],
        "temperature": 0.3,
        "timeout": 120,
        "gemma_tokens": 2500
    },
    "enhanced": {
        "description": "Complete deep stylometry analysis (slowest, most thorough)",
        "features": ["Deep Analysis", "Statistical Metrics", "Readability Scores", "Style Profiling"],
        "temperature": 0.2,
        "timeout": 180,
        "gemma_tokens": 3000
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

# Supported analogy domains ΓÇö each maps to a display label and a short
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

# Conceptual density threshold ΓÇö sentences scoring above this trigger analogies.
# Range 0.0-1.0.  Higher = only the densest sentences get analogies.
# Most academic/technical sentences score 0.45-0.70; casual text scores 0.25-0.40.
CONCEPTUAL_DENSITY_THRESHOLD = 0.45