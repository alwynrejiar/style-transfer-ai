"""
Style Transfer AI - Enhanced Deep Stylometry Analyzer
===================================================
Advanced stylometry analysis system with comprehensive depth and dual output formats:
- Deep 25-point analysis framework with advanced metrics
- Dual output: JSON (machine-readable) + TXT (human-readable)
- Advanced linguistic computation and readability analysis
- Writing flow, coherence, and psychological profiling
- Statistical analysis with quantitative measures
- Automatic cleanup option for previous reports
- User profile collection for cultural and linguistic context

Enhanced Features:
• Deep Analysis: 25 comprehensive stylometric dimensions
• Dual Formats: JSON for data + TXT for human reading
• Advanced Metrics: Readability scores, complexity indices
• Psychological Profiling: Personality and cognitive markers
• Statistical Analysis: Word frequencies, pattern distributions
• Writing Flow: Coherence, transitions, narrative progression
• Report Cleanup: Optional deletion of previous analysis reports
• Session Management: Continue analyzing without restarting program
• User Profiling: Collect cultural and linguistic background information

Usage:
    python style_analyzer_enhanced.py

Author: Style Transfer AI Team
Version: 4.6 (Firebase-Free Local Edition)
"""

import os
import json
import re
import math
import sys
import glob
import requests
from datetime import datetime
from collections import Counter

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
AVAILABLE_MODELS = {
    "gpt-oss:20b": "GPT-OSS 20B (Advanced, Slower)",
    "gemma3:1b": "Gemma 3:1B (Fast, Efficient)",
}
OPENAI_API_KEY = "your-openai-api-key-here"  # Replace with your actual OpenAI API key
GEMINI_API_KEY = "your-gemini-api-key-here"  # Replace with your actual Gemini API key

# GitHub API Configuration
GITHUB_API_KEY = "your-github-personal-access-token-here"  # Replace with your GitHub Personal Access Token
GITHUB_USERNAME = "your-github-username"  # Replace with your GitHub username

def get_user_model_choice():
    """Get user's choice of operation with main menu."""
    while True:
        print("\n" + "="*60)
        print("STYLE TRANSFER AI - MAIN MENU")
        print("="*60)
        print("Choose your operation:")
        print("1. Analyze Writing Style (Local Processing)")
        print("2. Analyze Writing Style (Online Processing)")
        print("0. Exit")
        print("="*60)
        
        try:
            main_choice = input("\nEnter your choice (0-2): ").strip()
            if main_choice == "0":
                print("Goodbye!")
                sys.exit(0)
            elif main_choice == "1":
                return get_local_model_choice()
            elif main_choice == "2":
                return get_online_model_choice()
            else:
                print("Invalid choice. Please enter 0, 1, or 2.")
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user. Goodbye!")
            sys.exit(0)

def get_local_model_choice():
    """Get user's choice of local model."""
    print("\n" + "="*40)
    print("LOCAL MODELS")
    print("="*40)
    print("Choose your local AI model:")
    print("1. GPT-OSS 20B - Advanced analysis")
    print("2. Gemma 3:1B - Fast processing")
    print("0. Back to main menu")
    print("="*40)
    
    while True:
        try:
            choice = input("\nEnter your choice (0-2): ").strip()
            if choice == "0":
                return get_user_model_choice()
            elif choice == "1":
                # GPT-OSS selected, now ask for mode
                return get_gpt_oss_mode_choice()
            elif choice == "2":
                return True, "gemma3:1b", None, "normal"    # Use local model with Gemma
            else:
                print("Invalid choice. Please enter 0, 1, or 2.")
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user. Goodbye!")
            sys.exit(0)

def get_gpt_oss_mode_choice():
    """Get user's choice of GPT-OSS processing mode."""
    print("\n" + "="*50)
    print("GPT-OSS 20B PROCESSING MODE")
    print("="*50)
    print("Choose processing mode:")
    print("1. Normal Mode - Thorough analysis (slower, higher quality)")
    print("2. Turbo Mode - Fast analysis (faster, good quality)")
    print("0. Back to model selection")
    print("="*50)
    
    while True:
        try:
            choice = input("\nEnter your choice (0-2): ").strip()
            if choice == "0":
                return get_local_model_choice()
            elif choice == "1":
                return True, "gpt-oss:20b", None, "normal"  # Normal mode
            elif choice == "2":
                return True, "gpt-oss:20b", None, "turbo"   # Turbo mode
            else:
                print("Invalid choice. Please enter 0, 1, or 2.")
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user. Goodbye!")
            sys.exit(0)

def get_online_model_choice():
    """Get user's choice of online model."""
    print("\n" + "="*40)
    print("ONLINE MODELS")
    print("="*40)
    print("Choose your cloud AI model:")
    print("1. OpenAI GPT-3.5-turbo")
    print("2. Google Gemini-1.5-flash")
    print("0. Back to main menu")
    print("="*40)
    
    while True:
        try:
            choice = input("\nEnter your choice (0-2): ").strip()
            if choice == "0":
                return get_user_model_choice()
            elif choice == "1":
                return False, None, "openai", "normal"      # Use OpenAI API
            elif choice == "2":
                return False, None, "gemini", "normal"      # Use Gemini API
            else:
                print("Invalid choice. Please enter 0, 1, or 2.")
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user. Goodbye!")
            sys.exit(0)

def get_api_key(api_type="openai"):
    """Get API key from user with validation and existing key detection."""
    if api_type == "openai":
        existing_key = OPENAI_API_KEY
        service_name = "OpenAI"
        key_url = "https://platform.openai.com/api-keys"
    elif api_type == "gemini":
        existing_key = GEMINI_API_KEY
        service_name = "Google Gemini"
        key_url = "https://aistudio.google.com/app/apikey"
    else:
        raise ValueError(f"Unknown API type: {api_type}")
    
    # Check if there's an existing API key in the code
    if existing_key and len(existing_key) > 20:
        masked_key = f"{existing_key[:10]}...{existing_key[-10:]}"
        print(f"\nFound existing {service_name} API key: {masked_key}")
        use_existing = input(f"Do you want to use this {service_name} API key? (y/n): ").strip().lower()
        
        if use_existing == 'y':
            return existing_key
    
    # Get new API key from user
    print(f"\nPlease enter your {service_name} API key:")
    print(f"(You can find this at: {key_url})")
    
    while True:
        try:
            api_key = input("API Key: ").strip()
            
            if not api_key:
                print("API key cannot be empty. Please try again.")
                continue
                
            if len(api_key) < 20:
                print("API key seems too short. Please check and try again.")
                continue
                
            return api_key
            
        except KeyboardInterrupt:
            print("\n\nSetup cancelled by user.")
            return None

def check_ollama_connection(model_name):
    """Check if Ollama server is running and specified model is available."""
    try:
        # Check if Ollama server is running
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model.get('name', '') for model in models]
            
            if model_name in model_names:
                return True, f"Ollama server running with {model_name} model available"
            else:
                return False, f"Model {model_name} not found. Available models: {model_names}"
        else:
            return False, f"Ollama server responded with status {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to Ollama server. Please run 'ollama serve'"
    except Exception as e:
        return False, f"Error checking Ollama: {e}"

def setup_openai_client(api_key=None):
    """Initialize OpenAI client with provided API key."""
    try:
        from openai import OpenAI
        
        # Use provided key or get from user
        if not api_key:
            api_key = get_api_key("openai")
        
        if not api_key:
            return None, "No API key provided"
        
        client = OpenAI(api_key=api_key)
        return client, f"OpenAI client initialized successfully with key: {api_key[:10]}...{api_key[-10:]}"
    except ImportError:
        return None, "OpenAI library not installed. Run: pip install openai"
    except Exception as e:
        return None, f"Error initializing OpenAI client: {e}"

def setup_gemini_client(api_key=None):
    """Initialize Gemini client with provided API key."""
    try:
        import google.generativeai as genai
        
        # Use provided key or get from user
        if not api_key:
            api_key = get_api_key("gemini")
        
        if not api_key:
            return None, "No API key provided"
        
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Create the model instance
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        return model, f"Gemini client initialized successfully with key: {api_key[:10]}...{api_key[-10:]}"
    except ImportError:
        return None, "Google Generative AI library not installed. Run: pip install google-generativeai"
    except Exception as e:
        return None, f"Error initializing Gemini client: {e}"

def read_text_file(file_path):
    """
    Reads text content from a file.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        str: File content or error message
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            if len(content.strip()) == 0:
                return f"Error: File '{file_path}' is empty"
            return content
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found"
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as e:
            return f"Error: Cannot read file '{file_path}' - {e}"
    except Exception as e:
        return f"Error: Cannot read file '{file_path}' - {e}"

def calculate_readability_metrics(text):
    """Calculate various readability and complexity metrics."""
    # Input validation
    if not text or not text.strip():
        return {}
    
    # Basic text statistics
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = text.split()
    
    # Additional validation
    if not words or not sentences:
        return {}
    
    syllables = sum([count_syllables(word) for word in words])
    
    # Readability scores
    avg_sentence_length = len(words) / len(sentences)
    avg_syllables_per_word = syllables / len(words)
    
    # Flesch Reading Ease
    flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    
    # Flesch-Kincaid Grade Level
    fk_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
    
    # Coleman-Liau Index
    avg_letters_per_100_words = (sum(len(word) for word in words) / len(words)) * 100
    avg_sentences_per_100_words = (len(sentences) / len(words)) * 100
    coleman_liau = (0.0588 * avg_letters_per_100_words) - (0.296 * avg_sentences_per_100_words) - 15.8
    
    return {
        "flesch_reading_ease": round(flesch_score, 2),
        "flesch_kincaid_grade": round(fk_grade, 2),
        "coleman_liau_index": round(coleman_liau, 2),
        "avg_sentence_length": round(avg_sentence_length, 2),
        "avg_syllables_per_word": round(avg_syllables_per_word, 2)
    }

def count_syllables(word):
    """Count syllables in a word."""
    word = word.lower()
    vowels = "aeiouy"
    syllable_count = 0
    previous_was_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not previous_was_vowel:
            syllable_count += 1
        previous_was_vowel = is_vowel
    
    # Handle silent e
    if word.endswith('e') and syllable_count > 1:
        syllable_count -= 1
    
    return max(1, syllable_count)

def analyze_text_statistics(text):
    """Perform detailed statistical analysis of text."""
    # Input validation
    if not text or not text.strip():
        return {
            'word_count': 0,
            'sentence_count': 0,
            'paragraph_count': 0,
            'character_count': 0,
            'avg_words_per_sentence': 0,
            'avg_sentences_per_paragraph': 0,
            'word_frequency': {},
            'punctuation_counts': {},
            'sentence_types': {},
            'unique_words': 0,
            'lexical_diversity': 0
        }
    
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    # Additional validation for empty results
    if not words:
        return {
            'word_count': 0,
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'character_count': len(text),
            'avg_words_per_sentence': 0,
            'avg_sentences_per_paragraph': 0,
            'word_frequency': {},
            'punctuation_counts': {},
            'sentence_types': {},
            'unique_words': 0,
            'lexical_diversity': 0
        }
    
    # Word frequency analysis
    word_freq = Counter(word.lower().strip('.,!?";:()[]{}') for word in words)
    
    # Punctuation analysis
    punctuation_counts = {
        'commas': text.count(','),
        'periods': text.count('.'),
        'semicolons': text.count(';'),
        'colons': text.count(':'),
        'exclamations': text.count('!'),
        'questions': text.count('?'),
        'dashes': text.count('—') + text.count('--'),
        'parentheses': text.count('(')
    }
    
    # Sentence type analysis
    sentence_types = {
        'declarative': len([s for s in sentences if s.strip().endswith('.')]),
        'interrogative': len([s for s in sentences if s.strip().endswith('?')]),
        'exclamatory': len([s for s in sentences if s.strip().endswith('!')]),
        'imperative': 0  # Would need more sophisticated analysis
    }
    
    # Safe calculations with validation
    unique_words_count = len(set(word.lower() for word in words))
    
    return {
        'word_count': len(words),
        'sentence_count': len(sentences),
        'paragraph_count': len(paragraphs),
        'character_count': len(text),
        'avg_words_per_sentence': round(len(words) / len(sentences), 2) if sentences else 0,
        'avg_sentences_per_paragraph': round(len(sentences) / len(paragraphs), 2) if paragraphs else 0,
        'word_frequency': dict(word_freq.most_common(20)),
        'punctuation_counts': punctuation_counts,
        'sentence_types': sentence_types,
        'unique_words': unique_words_count,
        'lexical_diversity': round(unique_words_count / len(words), 3) if words else 0
    }

def create_enhanced_deep_prompt(text_to_analyze, user_profile=None):
    """Create the enhanced 25-point deep stylometry analysis prompt with user context."""
    
    # Build user context section if profile is provided
    user_context = ""
    if user_profile and user_profile.get('native_language', 'Not provided') != 'Not provided':
        user_context = f"""
**WRITER BACKGROUND CONTEXT:**
Consider this writer's background when analyzing their style:
- Native Language: {user_profile.get('native_language', 'Unknown')}
- English Fluency: {user_profile.get('english_fluency', 'Unknown')}
- Other Languages: {user_profile.get('other_languages', 'None specified')}
- Nationality/Culture: {user_profile.get('nationality', 'Unknown')}
- Cultural Background: {user_profile.get('cultural_background', 'Not specified')}
- Education Level: {user_profile.get('education_level', 'Unknown')}
- Field of Study: {user_profile.get('field_of_study', 'Unknown')}
- Writing Experience: {user_profile.get('writing_experience', 'Unknown')}
- Writing Frequency: {user_profile.get('writing_frequency', 'Unknown')}

Use this background to:
1. Interpret language transfer patterns from their native language
2. Understand cultural influences on writing style
3. Consider educational and professional writing conventions
4. Recognize multilingual writing characteristics
5. Account for non-native English patterns (if applicable)

"""
    
    return f"""
Perform an ENHANCED DEEP stylometry analysis of the following text for creating a comprehensive writing style profile. Provide specific, quantifiable insights with exact numbers, percentages, and examples:
{user_context}
**PART 1: LINGUISTIC ARCHITECTURE**
1. Sentence Structure Mastery: Calculate exact average sentence length, identify complex/compound/simple ratios with percentages, analyze syntactic patterns
2. Clause Choreography: Measure subordinate clause frequency, coordination vs subordination ratios, dependent clause patterns
3. Punctuation Symphony: Count and categorize ALL punctuation usage - commas, semicolons, dashes, parentheses with specific frequencies
4. Syntactic Sophistication: Identify sentence variety index, grammatical complexity scoring, parsing preferences

**PART 2: LEXICAL INTELLIGENCE**
5. Vocabulary Sophistication: Analyze word complexity levels, formal vs informal ratios, academic vocabulary percentage
6. Semantic Field Preferences: Categorize word choices by domain (abstract/concrete, emotional/logical, technical/general)
7. Lexical Diversity Metrics: Calculate type-token ratio, vocabulary richness index, word repetition patterns
8. Register Flexibility: Measure formality spectrum, colloquialisms vs standard usage, domain-specific terminology

**PART 3: STYLISTIC DNA**
9. Tone Architecture: Identify confidence indicators, emotional markers, certainty/uncertainty expressions with examples
10. Voice Consistency: Analyze person preference (1st/2nd/3rd percentages), active vs passive voice ratios
11. Rhetorical Weaponry: Count metaphors, similes, rhetorical questions, parallel structures, repetition patterns
12. Narrative Technique: Point of view consistency, perspective shifts, storytelling vs explanatory modes

**PART 4: COGNITIVE PATTERNS**
13. Logical Flow Design: Analyze argument structure, cause-effect patterns, sequential vs thematic organization
14. Transition Mastery: Count and categorize transition words, coherence mechanisms, paragraph linking strategies
15. Emphasis Engineering: Identify how key points are highlighted - repetition, positioning, linguistic intensity
16. Information Density: Measure concept-to-word ratios, information packaging efficiency, elaboration patterns

**PART 5: PSYCHOLOGICAL MARKERS**
17. Cognitive Processing Style: Analyze linear vs circular thinking, analytical vs intuitive patterns, detail vs big-picture focus
18. Emotional Intelligence: Identify empathy markers, emotional vocabulary richness, interpersonal awareness
19. Authority Positioning: Measure hedging language, assertiveness markers, expertise indicators
20. Risk Tolerance: Analyze certainty language, qualification usage, experimental vs conservative expressions

**PART 6: STRUCTURAL GENIUS**
21. Paragraph Architecture: Calculate paragraph length variance, topic development patterns, structural rhythm
22. Coherence Engineering: Measure text cohesion, referential chains, thematic progression strategies
23. Temporal Dynamics: Analyze tense usage patterns, time reference preferences, narrative temporality
24. Modal Expression: Count modal verbs, probability expressions, obligation vs possibility language

**PART 7: UNIQUE FINGERPRINT**
25. Personal Signature Elements: Identify unique phrases, idiosyncratic expressions, personal linguistic habits

PROVIDE YOUR ANALYSIS AS:
1. Quantitative metrics with exact numbers and percentages
2. Specific examples from the text for each point
3. Comparative assessments (high/medium/low with context)
4. Pattern recognition insights
5. Psychological and cognitive style indicators
6. Cultural/linguistic influence markers (based on writer background)

Text to analyze:
{text_to_analyze}
"""

def analyze_style(text_to_analyze, use_local=True, model_name="gpt-oss:20b", api_type=None, api_client=None, user_profile=None, processing_mode="normal"):
    """
    Performs enhanced deep stylometry analysis using local Ollama, OpenAI API, or Gemini API.
    
    Args:
        text_to_analyze (str): The text to be analyzed
        use_local (bool): Whether to use local Ollama model or cloud APIs
        model_name (str): The specific model to use (for local models)
        api_type (str): 'openai' or 'gemini' for cloud APIs
        api_client: Pre-initialized API client (for OpenAI or Gemini)
        user_profile (dict): User background information for context
        processing_mode (str): 'normal' for thorough analysis, 'turbo' for faster processing
        
    Returns:
        str: Structured deep stylometric analysis for style profiling
    """
    prompt = create_enhanced_deep_prompt(text_to_analyze, user_profile)
    
    if use_local:
        # Use local Ollama model
        try:
            mode_text = "turbo" if processing_mode == "turbo" else "normal"
            print(f"Sending request to local Ollama model ({model_name} - {mode_text} mode)...")
            
            # Configure parameters based on processing mode
            if processing_mode == "turbo":
                # Turbo mode: faster but still quality analysis
                temperature = 0.3
                num_predict = 2000 if "gpt-oss" in model_name else 1500
                timeout = 120
            else:
                # Normal mode: thorough analysis
                temperature = 0.2
                num_predict = 3000 if "gpt-oss" in model_name else 2000
                timeout = 180
            
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": num_predict,
                    "stop": ["Human:", "Assistant:"]
                }
            }
            
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                print("Deep analysis completed successfully!")
                return result.get('response', 'No response received')
            else:
                return f"Ollama Error: HTTP {response.status_code} - {response.text}"
                
        except requests.exceptions.Timeout:
            return "Timeout Error: Deep analysis took too long. Try with shorter text or cloud API."
        except Exception as e:
            return f"Ollama Error: {e}"
    
    elif api_type == "openai":
        # Use OpenAI API
        if not api_client:
            return "OpenAI Error: No client provided. Please ensure client is initialized."
        
        print("Sending request to OpenAI GPT-3.5-turbo model...")
        
        try:
            response = api_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Lower for more consistent analysis
                max_tokens=3000   # More tokens for deep analysis
            )
            
            print("Deep analysis completed successfully!")
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI API Error: {e}"
    
    elif api_type == "gemini":
        # Use Gemini API
        if not api_client:
            return "Gemini Error: No client provided. Please ensure client is initialized."
        
        print("Sending request to Google Gemini-1.5-flash model...")
        
        try:
            # Import for generation config
            import google.generativeai as genai
            
            # Configure generation settings for consistent analysis
            generation_config = genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=3000,
                candidate_count=1
            )
            
            response = api_client.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            print("Deep analysis completed successfully!")
            return response.text
        except Exception as e:
            return f"Gemini API Error: {e}"
    
    else:
        return "Error: Unknown API type or configuration"

def format_human_readable_output(style_profile):
    """Format the style profile into a human-readable text format."""
    
    output_lines = []
    
    # Header
    output_lines.append("=" * 80)
    output_lines.append("PERSONAL STYLOMETRIC FINGERPRINT ANALYSIS")
    output_lines.append("=" * 80)
    
    # Add user name prominently if available
    if 'user_profile' in style_profile and 'name' in style_profile['user_profile']:
        user_name = style_profile['user_profile']['name']
        output_lines.append(f"WRITER: {user_name.upper()}")
        output_lines.append("=" * 80)
    output_lines.append("")
    
    # User Profile Section
    if 'user_profile' in style_profile:
        user_profile = style_profile['user_profile']
        output_lines.append("WRITER PROFILE INFORMATION")
        output_lines.append("-" * 40)
        
        # Writer Identity
        output_lines.append("WRITER IDENTITY:")
        output_lines.append(f"  Name: {user_profile.get('name', 'Not provided')}")
        output_lines.append("")
        
        # Language Background
        output_lines.append("LANGUAGE BACKGROUND:")
        output_lines.append(f"  Native Language: {user_profile.get('native_language', 'Not provided')}")
        output_lines.append(f"  English Fluency: {user_profile.get('english_fluency', 'Not provided')}")
        output_lines.append(f"  Other Languages: {user_profile.get('other_languages', 'Not provided')}")
        output_lines.append("")
        
        # Cultural Context
        output_lines.append("CULTURAL CONTEXT:")
        output_lines.append(f"  Nationality: {user_profile.get('nationality', 'Not provided')}")
        output_lines.append(f"  Cultural Background: {user_profile.get('cultural_background', 'Not provided')}")
        output_lines.append("")
        
        # Educational Background
        output_lines.append("EDUCATIONAL BACKGROUND:")
        output_lines.append(f"  Education Level: {user_profile.get('education_level', 'Not provided')}")
        output_lines.append(f"  Field of Study: {user_profile.get('field_of_study', 'Not provided')}")
        output_lines.append("")
        
        # Writing Experience
        output_lines.append("WRITING EXPERIENCE:")
        output_lines.append(f"  Writing Experience: {user_profile.get('writing_experience', 'Not provided')}")
        output_lines.append(f"  Writing Frequency: {user_profile.get('writing_frequency', 'Not provided')}")
        output_lines.append("")
        output_lines.append("=" * 80)
        output_lines.append("")
    
    # Metadata section
    output_lines.append("ANALYSIS METADATA")
    output_lines.append("-" * 40)
    output_lines.append(f"Analysis Date: {style_profile['metadata']['analysis_date']}")
    output_lines.append(f"Analysis Method: {style_profile['metadata']['analysis_method']}")
    output_lines.append(f"Model Used: {style_profile['metadata']['model_used']}")
    output_lines.append(f"Total Samples Analyzed: {style_profile['metadata']['total_samples']}")
    output_lines.append(f"Combined Text Length: {style_profile['metadata']['combined_text_length']} characters")
    output_lines.append("")
    
    # File information
    if style_profile['metadata']['file_info']:
        output_lines.append("SOURCE FILES")
        output_lines.append("-" * 40)
        for file_info in style_profile['metadata']['file_info']:
            output_lines.append(f"• {file_info['filename']}: {file_info['word_count']} words, {file_info['character_count']} characters")
        output_lines.append("")
    
    # Statistical analysis
    if 'text_statistics' in style_profile:
        stats = style_profile['text_statistics']
        output_lines.append("STATISTICAL ANALYSIS")
        output_lines.append("-" * 40)
        output_lines.append(f"Total Words: {stats.get('word_count', 'N/A')}")
        output_lines.append(f"Total Sentences: {stats.get('sentence_count', 'N/A')}")
        output_lines.append(f"Total Paragraphs: {stats.get('paragraph_count', 'N/A')}")
        output_lines.append(f"Average Words per Sentence: {stats.get('avg_words_per_sentence', 'N/A')}")
        output_lines.append(f"Lexical Diversity Score: {stats.get('lexical_diversity', 'N/A')}")
        output_lines.append("")
        
        # Punctuation patterns
        if 'punctuation_counts' in stats:
            output_lines.append("PUNCTUATION PATTERNS")
            output_lines.append("-" * 25)
            punct = stats['punctuation_counts']
            for punct_type, count in punct.items():
                output_lines.append(f"• {punct_type.capitalize()}: {count}")
            output_lines.append("")
        
        # Most frequent words
        if 'word_frequency' in stats:
            output_lines.append("MOST FREQUENT WORDS")
            output_lines.append("-" * 25)
            for word, freq in list(stats['word_frequency'].items())[:10]:
                output_lines.append(f"• '{word}': {freq} times")
            output_lines.append("")
    
    # Readability metrics
    if 'readability_metrics' in style_profile:
        metrics = style_profile['readability_metrics']
        output_lines.append("READABILITY ANALYSIS")
        output_lines.append("-" * 40)
        output_lines.append(f"Flesch Reading Ease: {metrics.get('flesch_reading_ease', 'N/A')} (0-100, higher = easier)")
        output_lines.append(f"Flesch-Kincaid Grade Level: {metrics.get('flesch_kincaid_grade', 'N/A')}")
        output_lines.append(f"Coleman-Liau Index: {metrics.get('coleman_liau_index', 'N/A')}")
        output_lines.append(f"Average Syllables per Word: {metrics.get('avg_syllables_per_word', 'N/A')}")
        output_lines.append("")
    
    # Individual file analyses
    if style_profile['individual_analyses']:
        output_lines.append("INDIVIDUAL FILE ANALYSES")
        output_lines.append("=" * 50)
        for i, analysis in enumerate(style_profile['individual_analyses'], 1):
            output_lines.append(f"\nFILE {i}: {analysis['filename']}")
            output_lines.append("-" * (10 + len(analysis['filename'])))
            output_lines.append(f"Character Count: {analysis['character_count']}")
            output_lines.append(f"Word Count: {analysis['word_count']}")
            output_lines.append("")
            output_lines.append("STYLOMETRIC ANALYSIS:")
            # Add the analysis content with proper formatting
            analysis_lines = analysis['analysis'].split('\n')
            for line in analysis_lines:
                if line.strip():
                    output_lines.append(f"  {line}")
            output_lines.append("")
    
    # Consolidated analysis
    if 'consolidated_analysis' in style_profile:
        output_lines.append("CONSOLIDATED STYLOMETRIC PROFILE")
        output_lines.append("=" * 50)
        # Add the consolidated analysis content with proper formatting
        analysis_lines = style_profile['consolidated_analysis'].split('\n')
        for line in analysis_lines:
            if line.strip():
                output_lines.append(line)
        output_lines.append("")
    
    # Recommendations
    output_lines.append("STYLE PROFILE INSIGHTS")
    output_lines.append("-" * 40)
    output_lines.append("This comprehensive analysis provides quantitative and qualitative")
    output_lines.append("insights into your unique writing style. The statistical measures")
    output_lines.append("can be used for:")
    output_lines.append("• AI text generation that matches your style")
    output_lines.append("• Writing consistency analysis")
    output_lines.append("• Style evolution tracking over time")
    output_lines.append("• Comparative stylometric studies")
    output_lines.append("")
    
    # Footer
    output_lines.append("=" * 80)
    output_lines.append("End of Enhanced Deep Stylometry Analysis Report")
    output_lines.append("Generated by Style Transfer AI - Enhanced Deep Analysis v4.2")
    output_lines.append("=" * 80)
    
    return '\n'.join(output_lines)

def sanitize_filename(name):
    """
    Sanitize a name for use in filenames by removing or replacing invalid characters.
    
    Args:
        name (str): The user's name
        
    Returns:
        str: Sanitized filename-safe string
    """
    import re
    
    # Remove or replace invalid filename characters
    # Keep only alphanumeric, spaces, hyphens, and underscores
    sanitized = re.sub(r'[<>:"/\\|?*]', '', name)
    
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    
    # Remove any remaining problematic characters and limit length
    sanitized = re.sub(r'[^\w\-_]', '', sanitized)
    
    # Limit length to reasonable filename size
    sanitized = sanitized[:30] if len(sanitized) > 30 else sanitized
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = "User"
    
    return sanitized

def save_style_profile_dual_format(style_profile, base_filename="user_style_profile_enhanced", use_cloud_storage=False):
    """
    Save the style profile in both JSON and human-readable text formats locally.
    
    Args:
        style_profile (dict): The complete style profile data
        base_filename (str): Base filename without extension
        use_cloud_storage (bool): Legacy parameter (ignored)
    """
    
    try:
        # Extract user name from profile for personalized filename
        user_name = "Anonymous_User"
        if 'user_profile' in style_profile and 'name' in style_profile['user_profile']:
            user_name = sanitize_filename(style_profile['user_profile']['name'])
        
        # Generate timestamped filenames with user name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename_local = f"{user_name}_stylometric_profile_{timestamp}.json"
        txt_filename_local = f"{user_name}_stylometric_profile_{timestamp}.txt"
        
        # Save JSON format locally
        with open(json_filename_local, 'w', encoding='utf-8') as f:
            json.dump(style_profile, f, indent=2, ensure_ascii=False)
        
        # Generate human-readable content and save locally
        human_readable_content = format_human_readable_output(style_profile)
        with open(txt_filename_local, 'w', encoding='utf-8') as f:
            f.write(human_readable_content)
        
        return {
            'success': True,
            'json_file': json_filename_local,
            'txt_file': txt_filename_local,
            'message': f"Personal stylometric profile saved locally for {user_name}:\n• JSON: {json_filename_local}\n• TXT: {txt_filename_local}"
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Error saving files locally: {e}"
        }

def create_enhanced_style_profile(file_paths, use_local=True, model_name="gpt-oss:20b", api_type=None, api_client=None, processing_mode="normal"):
    """
    Creates an enhanced comprehensive style profile from multiple text samples.
    
    Args:
        file_paths (list): List of file paths containing user's writing samples
        use_local (bool): Whether to use local Ollama model or cloud APIs
        model_name (str): The specific model to use (for local models)
        api_type (str): 'openai' or 'gemini' for cloud APIs
        api_client: Pre-initialized API client (for OpenAI or Gemini)
        processing_mode (str): 'normal' for thorough analysis, 'turbo' for faster processing
        
    Returns:
        dict: Enhanced consolidated style profile with deep analysis
    """
    # Collect user profile information first
    user_profile = get_user_profile()
    
    all_analyses = []
    combined_text = ""
    file_info = []
    
    print(f"\nFound {len(file_paths)} text sample(s) for enhanced deep analysis")
    
    for file_path in file_paths:
        file_content = read_text_file(file_path)
        
        if "Error" not in file_content:
            print(f"  Processing: {file_path}")
            
            # Calculate basic statistics for this file
            words = len(file_content.split())
            chars = len(file_content)
            
            file_info.append({
                'filename': file_path,
                'word_count': words,
                'character_count': chars
            })
            
            # Individual analysis
            print(f"  Performing deep analysis...")
            individual_analysis = analyze_style(file_content, use_local, model_name, api_type, api_client, user_profile, processing_mode)
            
            all_analyses.append({
                'filename': file_path,
                'word_count': words,
                'character_count': chars,
                'analysis': individual_analysis
            })
            
            combined_text += f"\n\n--- From {file_path} ---\n{file_content}"
            print(f"  Analysis completed for {file_path}")
        else:
            print(f"  Error with {file_path}: {file_content}")
    
    if not all_analyses:
        return {
            'profile_created': False,
            'error': 'No valid files could be analyzed'
        }
    
    # Perform enhanced statistical analysis on combined text
    print("Calculating comprehensive statistics...")
    text_statistics = analyze_text_statistics(combined_text)
    
    # Calculate readability metrics
    print("Computing readability metrics...")
    readability_metrics = calculate_readability_metrics(combined_text)
    
    # Consolidated analysis of all texts combined
    print("Generating consolidated deep analysis...")
    consolidated_analysis = analyze_style(combined_text, use_local, model_name, api_type, api_client, user_profile, processing_mode)
    
    # Create metadata
    if use_local:
        analysis_method = f"Local Ollama ({model_name})"
        model_used = model_name
    elif api_type == "openai":
        analysis_method = "OpenAI GPT-3.5-turbo"
        model_used = "gpt-3.5-turbo"
    elif api_type == "gemini":
        analysis_method = "Google Gemini-1.5-flash"
        model_used = "gemini-1.5-flash"
    else:
        analysis_method = "Unknown"
        model_used = "unknown"
    
    enhanced_profile = {
        'profile_created': True,
        'user_profile': user_profile,  # Add user profile to the main profile
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'analysis_method': analysis_method,
            'model_used': model_used,
            'total_samples': len(file_paths),
            'combined_text_length': len(combined_text),
            'file_info': file_info,
            'version': '4.6_firebase_free_local'
        },
        'text_statistics': text_statistics,
        'readability_metrics': readability_metrics,
        'individual_analyses': all_analyses,
        'consolidated_analysis': consolidated_analysis,
        'analysis_depth': 'enhanced_deep_25_point'
    }
    
    return enhanced_profile

def get_user_profile():
    """Collect essential user profile information for enhanced stylometry analysis."""
    print("\nUSER PROFILE COLLECTION")
    print("=" * 50)
    print("Creating your personal stylometric fingerprint...")
    print("This helps understand linguistic and cultural influences on your writing style.")
    print("All fields are optional except name - press Enter to skip any question.")
    print("=" * 50)
    
    profile = {}
    
    # Name (Required for file naming)
    print("\nPERSONAL IDENTIFICATION:")
    print("-" * 25)
    while True:
        name = input("Your Name (required for file naming): ").strip()
        if name:
            profile['name'] = name
            break
        else:
            print("Name is required to create your personalized stylometric profile.")
            retry = input("Would you like to continue? (y/n): ").strip().lower()
            if retry != 'y':
                profile['name'] = "Anonymous_User"
                print("Using 'Anonymous_User' as filename.")
                break
    
    # Essential Language Information
    print("\nLANGUAGE BACKGROUND:")
    print("-" * 20)
    profile['native_language'] = input("Native/Mother Tongue Language: ").strip() or "Not provided"
    profile['english_fluency'] = input("English Fluency (Native/Advanced/Intermediate/Basic): ").strip() or "Not provided"
    profile['other_languages'] = input("Other Languages You Speak: ").strip() or "Not provided"
    
    # Essential Cultural Context
    print("\nCULTURAL CONTEXT:")
    print("-" * 17)
    profile['nationality'] = input("Nationality/Country: ").strip() or "Not provided"
    profile['cultural_background'] = input("Cultural Background (if different from nationality): ").strip() or "Not provided"
    
    # Essential Educational Context
    print("\nEDUCATIONAL BACKGROUND:")
    print("-" * 23)
    profile['education_level'] = input("Education Level (High School/Bachelor's/Master's/PhD/etc.): ").strip() or "Not provided"
    profile['field_of_study'] = input("Field of Study/Academic Focus: ").strip() or "Not provided"
    
    # Essential Writing Context
    print("\nWRITING EXPERIENCE:")
    print("-" * 18)
    profile['writing_experience'] = input("Writing Experience (Academic/Professional/Creative/Casual): ").strip() or "Not provided"
    profile['writing_frequency'] = input("How Often Do You Write? (Daily/Weekly/Monthly/Rarely): ").strip() or "Not provided"
    
    # Add timestamp
    profile['profile_created'] = datetime.now().isoformat()
    
    print(f"\nProfile collection complete for {profile['name']}!")
    print("This will enhance the analysis accuracy and create personalized files.")
    
    return profile

def cleanup_old_reports():
    """Ask user if they want to delete previously generated reports."""
    import glob
    
    # Find existing report files (both old and new naming patterns)
    old_json_files = glob.glob("user_style_profile_enhanced_*.json")
    old_txt_files = glob.glob("user_style_profile_enhanced_*.txt")
    new_json_files = glob.glob("*_stylometric_profile_*.json")
    new_txt_files = glob.glob("*_stylometric_profile_*.txt")
    
    all_report_files = old_json_files + old_txt_files + new_json_files + new_txt_files
    
    if not all_report_files:
        print("No previous stylometric reports found.")
        return
    
    print(f"\nFound {len(all_report_files)} previous stylometric report files:")
    for file in sorted(all_report_files):
        file_size = os.path.getsize(file)
        file_size_kb = file_size / 1024
        print(f"  • {file} ({file_size_kb:.1f} KB)")
    
    print("\nWould you like to delete these old stylometric reports before generating new ones?")
    print("This will help keep your workspace clean and organized.")
    
    while True:
        choice = input("Delete old reports? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            deleted_count = 0
            for file in all_report_files:
                try:
                    os.remove(file)
                    deleted_count += 1
                    print(f"  Deleted: {file}")
                except Exception as e:
                    print(f"  Error deleting {file}: {e}")
            
            print(f"\nSuccessfully deleted {deleted_count} old report files.")
            break
        elif choice in ['n', 'no']:
            print("Keeping existing reports.")
            break
        else:
            print("Please enter 'y' or 'n'.")

def get_file_paths():
    """Get file paths for analysis from user input."""
    print("\nFILE SELECTION")
    print("=" * 40)
    print("Enter the text files you want to analyze.")
    print("You can:")
    print("• Enter file paths one by one (press Enter after each)")
    print("• Use default sample files (type 'default')")
    print("• Finish by typing 'done' when you're finished")
    print("=" * 40)
    
    file_paths = []
    
    while True:
        file_input = input("\nEnter file path (or 'default'/'done'): ").strip()
        
        if file_input.lower() == 'done':
            break
        elif file_input.lower() == 'default':
            default_files = ["about_my_pet.txt", "about_my_pet_1.txt"]
            print(f"Using default files: {', '.join(default_files)}")
            file_paths.extend(default_files)
            break
        elif file_input:
            if os.path.exists(file_input):
                file_paths.append(file_input)
                print(f"Added: {file_input}")
            else:
                print(f"File not found: {file_input}")
                retry = input("Do you want to try again? (y/n): ").strip().lower()
                if retry != 'y':
                    continue
        else:
            print("Please enter a file path or 'done' to finish.")
    
    if not file_paths:
        print("No files selected. Using default files.")
        file_paths = ["about_my_pet.txt", "about_my_pet_1.txt"]
    
    print(f"\nSelected {len(file_paths)} file(s) for analysis:")
    for path in file_paths:
        print(f"  • {path}")
    
    return file_paths

def display_enhanced_results(style_profile):
    """Display enhanced results summary for the user."""
    print("\nEnhanced Deep Stylometry Analysis Complete!")
    print("=" * 60)
    
    if style_profile['profile_created']:
        print(f"• Analysis Method: {style_profile['metadata']['analysis_method']}")
        print(f"• Files Analyzed: {style_profile['metadata']['total_samples']}")
        print(f"• Total Text Length: {style_profile['metadata']['combined_text_length']} characters")
        
        # Show statistical highlights
        if 'text_statistics' in style_profile:
            stats = style_profile['text_statistics']
            print(f"• Total Words: {stats.get('word_count', 'N/A')}")
            print(f"• Total Sentences: {stats.get('sentence_count', 'N/A')}")
            print(f"• Lexical Diversity: {stats.get('lexical_diversity', 'N/A')}")
        
        # Show readability highlights
        if 'readability_metrics' in style_profile:
            metrics = style_profile['readability_metrics']
            print(f"• Reading Ease Score: {metrics.get('flesch_reading_ease', 'N/A')}")
            print(f"• Grade Level: {metrics.get('flesch_kincaid_grade', 'N/A')}")
        
        print("\nIndividual File Analysis Summary:")
        for analysis in style_profile['individual_analyses']:
            print(f"  • {analysis['filename']}: {analysis['word_count']} words")
    
    print(f"\nYour enhanced deep style profile is ready for advanced text generation!")

def calculate_text_complexity_score(text):
    """
    Calculate a comprehensive text complexity score based on multiple metrics.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        dict: Complexity metrics including overall score and component scores
    """
    if not text or not text.strip():
        return {
            'overall_score': 0,
            'sentence_complexity': 0,
            'vocabulary_complexity': 0,
            'readability_score': 0,
            'error': 'Empty text provided'
        }
    
    try:
        # Basic text statistics
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not words or not sentences:
            return {
                'overall_score': 0,
                'sentence_complexity': 0,
                'vocabulary_complexity': 0,
                'readability_score': 0,
                'error': 'Insufficient text for analysis'
            }
        
        # 1. Sentence Complexity (average words per sentence)
        avg_words_per_sentence = len(words) / len(sentences)
        sentence_complexity = min(avg_words_per_sentence / 20, 1.0)  # Normalize to 0-1
        
        # 2. Vocabulary Complexity (unique words ratio and average word length)
        unique_words = set(word.lower().strip('.,!?";:()[]{}') for word in words)
        vocabulary_diversity = len(unique_words) / len(words)
        avg_word_length = sum(len(word) for word in words) / len(words)
        vocabulary_complexity = (vocabulary_diversity * 0.6) + (min(avg_word_length / 8, 1.0) * 0.4)
        
        # 3. Readability Score (simplified Flesch-like calculation)
        syllable_count = 0
        for word in words:
            # Simple syllable counting heuristic
            word_clean = re.sub(r'[^a-zA-Z]', '', word.lower())
            if word_clean:
                syllables = max(1, len(re.findall(r'[aeiouAEIOU]', word_clean)))
                if word_clean.endswith('e'):
                    syllables = max(1, syllables - 1)
                syllable_count += syllables
        
        avg_syllables_per_word = syllable_count / len(words) if words else 0
        readability_score = 1.0 - min((avg_syllables_per_word - 1) / 2, 1.0)  # Inverse complexity
        
        # 4. Overall Score (weighted combination)
        overall_score = (
            sentence_complexity * 0.4 +
            vocabulary_complexity * 0.4 +
            readability_score * 0.2
        )
        
        return {
            'overall_score': round(overall_score, 3),
            'sentence_complexity': round(sentence_complexity, 3),
            'vocabulary_complexity': round(vocabulary_complexity, 3),
            'readability_score': round(readability_score, 3),
            'avg_words_per_sentence': round(avg_words_per_sentence, 1),
            'vocabulary_diversity': round(vocabulary_diversity, 3),
            'avg_word_length': round(avg_word_length, 1),
            'total_words': len(words),
            'total_sentences': len(sentences),
            'unique_words': len(unique_words)
        }
        
    except Exception as e:
        return {
            'overall_score': 0,
            'sentence_complexity': 0,
            'vocabulary_complexity': 0,
            'readability_score': 0,
            'error': f'Error calculating complexity: {e}'
        }

def main():
    """Main execution function for enhanced deep analysis."""
    print("Style Transfer AI - Enhanced Deep Stylometry Analyzer")
    print("=" * 70)
    
    while True:  # Main program loop
        try:
            # Get user's model choice
            use_local, selected_model, api_type, processing_mode = get_user_model_choice()
            
            # Initialize API client once if needed
            api_client = None
            
            # Validate setup based on choice
            if use_local:
                model_display = AVAILABLE_MODELS.get(selected_model, selected_model)
                mode_display = f" ({processing_mode.capitalize()} Mode)" if selected_model == "gpt-oss:20b" else ""
                print(f"\nLocal Ollama Model Selected: {model_display}{mode_display}")
                print("=" * 50)
                
                # Check Ollama connection and model availability
                is_connected, message = check_ollama_connection(selected_model)
                if not is_connected:
                    print(f"ERROR: Ollama Setup Error: {message}")
                    print("\nTo fix this:")
                    print("1. Make sure Ollama is installed")
                    print("2. Run 'ollama serve' in a terminal")
                    print(f"3. Ensure you have the model: ollama pull {selected_model}")
                    continue  # Return to main menu instead of exit
                
                print(f"SUCCESS: {message}")
                
            elif api_type == "openai":
                print("\nOpenAI API Model Selected")
                print("=" * 40)
                
                # Get API key and initialize OpenAI client once
                try:
                    api_client, setup_message = setup_openai_client()
                    if not api_client:
                        print(f"ERROR: OpenAI Error: {setup_message}")
                        continue  # Return to main menu instead of exit
                    
                    print(f"SUCCESS: {setup_message}")
                except KeyboardInterrupt:
                    print("\n\nSetup cancelled by user.")
                    continue  # Return to main menu instead of exit
                except Exception as e:
                    print(f"ERROR: Unexpected error: {e}")
                    continue  # Return to main menu instead of exit
                    
            elif api_type == "gemini":
                print("\nGoogle Gemini API Model Selected")
                print("=" * 40)
                
                # Get API key and initialize Gemini client once
                try:
                    api_client, setup_message = setup_gemini_client()
                    if not api_client:
                        print(f"ERROR: Gemini Error: {setup_message}")
                        continue  # Return to main menu instead of exit
                    
                    print(f"SUCCESS: {setup_message}")
                except KeyboardInterrupt:
                    print("\n\nSetup cancelled by user.")
                    continue  # Return to main menu instead of exit
                except Exception as e:
                    print(f"ERROR: Unexpected error: {e}")
                    continue  # Return to main menu instead of exit
            
            # Ask about cleaning up old reports
            cleanup_old_reports()
            
            # Get file paths from user
            file_paths = get_file_paths()
            
            print("\nCreating enhanced deep stylometry profile...")
            print("=" * 70)
            
            # Create enhanced comprehensive style profile
            if use_local:
                style_profile = create_enhanced_style_profile(file_paths, use_local, selected_model, api_type, api_client, processing_mode)
            else:
                style_profile = create_enhanced_style_profile(file_paths, use_local, None, api_type, api_client, processing_mode)
            
            if style_profile['profile_created']:
                # Save the profile locally only
                print(f"\nSaving profile locally...")
                save_result = save_style_profile_dual_format(style_profile, use_cloud_storage=False)
                
                if save_result['success']:
                    print(f"SUCCESS: {save_result['message']}")
                else:
                    print(f"ERROR: {save_result['error']}")
                
                # Display results
                display_enhanced_results(style_profile)
                
                # Ask if user wants to perform another analysis
                print("\n" + "=" * 70)
                while True:
                    another_analysis = input("Would you like to perform another analysis? (y/n): ").strip().lower()
                    if another_analysis in ['y', 'yes']:
                        print("\nReturning to main menu...\n")
                        break  # Continue the main loop
                    elif another_analysis in ['n', 'no']:
                        print("\nThank you for using Style Transfer AI. Goodbye!")
                        return  # Exit the program
                    else:
                        print("Please enter 'y' or 'n'.")
                
            else:
                print("ERROR: Failed to create enhanced style profile")
                print(f"Error: {style_profile.get('error', 'Unknown error')}")
                
                # Ask if user wants to try again
                print("\n" + "=" * 70)
                while True:
                    try_again = input("Would you like to try again? (y/n): ").strip().lower()
                    if try_again in ['y', 'yes']:
                        print("\nReturning to main menu...\n")
                        break  # Continue the main loop
                    elif try_again in ['n', 'no']:
                        print("\nThank you for using Style Transfer AI. Goodbye!")
                        return  # Exit the program
                    else:
                        print("Please enter 'y' or 'n'.")
        
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            # Ask if user wants to return to main menu or exit
            while True:
                try:
                    choice = input("Return to main menu? (y/n): ").strip().lower()
                    if choice in ['y', 'yes']:
                        print("\nReturning to main menu...\n")
                        break  # Continue the main loop
                    elif choice in ['n', 'no']:
                        print("\nGoodbye!")
                        return  # Exit the program
                    else:
                        print("Please enter 'y' or 'n'.")
                except KeyboardInterrupt:
                    print("\n\nGoodbye!")
                    return  # Exit the program


# --- CLI Entrypoint for style-transfer-ai ---
import argparse

def cli_entrypoint():
    parser = argparse.ArgumentParser(
        description="Style Transfer AI - Advanced Stylometry Analysis CLI"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive menu mode (default)"
    )
    parser.add_argument(
        "--analyze", nargs='+', metavar="FILE", help="Analyze one or more text files non-interactively"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Specify model (e.g., gpt-oss:20b, gemma3:1b)"
    )
    parser.add_argument(
        "--local", action="store_true", help="Force use of local model (Ollama)"
    )
    parser.add_argument(
        "--cloud", action="store_true", help="Force use of cloud model (OpenAI/Gemini)"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Base name for output files (no extension)"
    )
    args = parser.parse_args()

    if args.analyze:
        # Non-interactive batch analysis
        use_local = args.local or not args.cloud
        model_name = args.model if args.model else ("gpt-oss:20b" if use_local else "openai")
        api_type = None
        api_client = None
        processing_mode = "normal"
        file_paths = args.analyze
        print(f"Analyzing files: {file_paths}")
        style_profile = create_enhanced_style_profile(file_paths, use_local, model_name, api_type, api_client, processing_mode)
        if style_profile.get('profile_created'):
            print("\nSaving profile locally...")
            save_result = save_style_profile_dual_format(
                style_profile,
                base_filename=args.output or "user_style_profile_enhanced",
                use_cloud_storage=False
            )
            if save_result['success']:
                print(f"SUCCESS: {save_result['message']}")
            else:
                print(f"ERROR: {save_result['error']}")
            display_enhanced_results(style_profile)
        else:
            print("ERROR: Failed to create enhanced style profile")
            print(f"Error: {style_profile.get('error', 'Unknown error')}")
        return

    # Default: interactive menu
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Goodbye!")
    except Exception as e:
        print(f"\nERROR: Unexpected error occurred: {e}")
        print("Please check your setup and try again.")
        print("Returning to main menu...\n")
        try:
            main()
        except:
            print("Unable to recover. Exiting program.")


# Allow both python style_analyzer_enhanced.py and CLI entrypoint
if __name__ == "__main__":
    cli_entrypoint()
