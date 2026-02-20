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