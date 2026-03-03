"""
Text processing utilities for Style Transfer AI.
Handles file reading, text sanitization, and basic text operations.
"""

import re
from ..config.settings import SUPPORTED_ENCODINGS, MAX_FILENAME_LENGTH


def read_text_file(file_path):
    """
    Reads text content from a file with encoding fallback.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        str: File content or error message
    """
    for encoding in SUPPORTED_ENCODINGS:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
                if len(content.strip()) == 0:
                    return f"Error: File '{file_path}' is empty"
                return content
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            return f"Error: File '{file_path}' not found"
        except Exception as e:
            return f"Error: Cannot read file '{file_path}' - {e}"
    
    return f"Error: Cannot read file '{file_path}' with any supported encoding"


def sanitize_filename(name):
    """
    Sanitize a name for use in filenames by removing or replacing invalid characters.
    
    Args:
        name (str): The user's name
        
    Returns:
        str: Sanitized filename-safe string
    """
    if not name or not name.strip():
        return "Anonymous_User"
    
    # Remove or replace invalid filename characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '', name)
    
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    
    # Remove any remaining problematic characters
    sanitized = re.sub(r'[^\w\-_]', '', sanitized)
    
    # Limit length to reasonable filename size
    sanitized = sanitized[:MAX_FILENAME_LENGTH] if len(sanitized) > MAX_FILENAME_LENGTH else sanitized
    
    # Ensure it's not empty after sanitization
    if not sanitized:
        sanitized = "User"
    
    return sanitized


def sanitize_topic_for_filename(topic):
    """
    Sanitize a topic/subject for use in filenames.
    
    Args:
        topic (str): The content topic or subject
        
    Returns:
        str: Sanitized filename-safe topic string
    """
    if not topic or not topic.strip():
        return "general_topic"
    
    # Remove or replace invalid filename characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '', topic)
    
    # Replace spaces and special characters with underscores
    sanitized = re.sub(r'[\s\-\.,;:!?]+', '_', sanitized)
    
    # Remove any remaining problematic characters
    sanitized = re.sub(r'[^\w_]', '', sanitized)
    
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    # Limit length to reasonable filename size (shorter for topics)
    max_topic_length = min(MAX_FILENAME_LENGTH, 50)
    sanitized = sanitized[:max_topic_length] if len(sanitized) > max_topic_length else sanitized
    
    # Ensure it's not empty after sanitization
    if not sanitized:
        sanitized = "general_topic"
    
    # Convert to lowercase for consistency
    return sanitized.lower()


def count_syllables(word):
    """Count syllables in a word using vowel patterns."""
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


def extract_basic_stats(text):
    """Extract basic text statistics."""
    if not text or not text.strip():
        return {
            'word_count': 0,
            'sentence_count': 0,
            'paragraph_count': 0,
            'character_count': 0
        }
    
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    return {
        'word_count': len(words),
        'sentence_count': len(sentences),
        'paragraph_count': len(paragraphs),
        'character_count': len(text)
    }