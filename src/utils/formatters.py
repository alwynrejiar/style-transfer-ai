"""
Output formatting utilities for Style Transfer AI.
Handles JSON and human-readable text report generation.
"""

import json
from datetime import datetime
from ..config.settings import TIMESTAMP_FORMAT


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
    output_lines.append(f"Analysis Method: {style_profile['metadata'].get('analysis_method', 'Enhanced Analysis')}")
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
    if style_profile.get('individual_analyses'):
        output_lines.append("INDIVIDUAL FILE ANALYSES")
        output_lines.append("=" * 50)
        for i, analysis in enumerate(style_profile['individual_analyses'], 1):
            output_lines.append(f"\nFILE {i}: {analysis.get('filename', 'unknown')}")
            output_lines.append("-" * (10 + len(analysis.get('filename', 'unknown'))))
            if 'character_count' in analysis:
                output_lines.append(f"Character Count: {analysis['character_count']}")
            if 'word_count' in analysis:
                output_lines.append(f"Word Count: {analysis['word_count']}")
            output_lines.append("")
            output_lines.append("STYLOMETRIC ANALYSIS:")
            # Handle both string (legacy) and dict (v3.0) analysis formats
            raw = analysis.get('analysis', '')
            if isinstance(raw, dict):
                raw = json.dumps(raw, indent=2, ensure_ascii=False)
            for line in raw.split('\n'):
                if line.strip():
                    output_lines.append(f"  {line}")
            output_lines.append("")
    
    # Consolidated analysis
    if 'consolidated_analysis' in style_profile:
        output_lines.append("CONSOLIDATED STYLOMETRIC PROFILE")
        output_lines.append("=" * 50)
        # Handle both string (legacy) and dict (v3.0) analysis formats
        consolidated = style_profile['consolidated_analysis']
        if isinstance(consolidated, dict):
            consolidated = json.dumps(consolidated, indent=2, ensure_ascii=False)
        for line in consolidated.split('\n'):
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
    output_lines.append("Generated by Style Transfer AI - Enhanced Deep Analysis v4.5")
    output_lines.append("=" * 80)
    
    return '\n'.join(output_lines)


def save_dual_format(style_profile, base_filename, user_name="Anonymous_User"):
    """
    Save style profile in both JSON and TXT formats with user-specific naming.
    
    Args:
        style_profile (dict): The complete style profile data
        base_filename (str): Base filename without extension
        user_name (str): Sanitized user name for filename
        
    Returns:
        tuple: (json_filename, txt_filename)
    """
    import os
    
    # Create stylometry fingerprints directory if it doesn't exist
    fingerprints_dir = "stylometry fingerprints"
    if not os.path.exists(fingerprints_dir):
        os.makedirs(fingerprints_dir)
    
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    json_filename = os.path.join(fingerprints_dir, f"{user_name}_stylometric_profile_{timestamp}.json")
    txt_filename = os.path.join(fingerprints_dir, f"{user_name}_stylometric_profile_{timestamp}.txt")
    
    # Save JSON format
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(style_profile, f, indent=2, ensure_ascii=False)
    
    # Generate and save human-readable content
    human_readable_content = format_human_readable_output(style_profile)
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write(human_readable_content)
    
    return json_filename, txt_filename