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

    output_lines.append("=" * 80)
    output_lines.append("PERSONAL STYLOMETRIC FINGERPRINT ANALYSIS")
    output_lines.append("=" * 80)

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

        output_lines.append("WRITER IDENTITY:")
        output_lines.append(f"  Name: {user_profile.get('name', 'Not provided')}")
        output_lines.append("")

        output_lines.append("LANGUAGE BACKGROUND:")
        output_lines.append(f"  Native Language: {user_profile.get('native_language', 'Not provided')}")
        output_lines.append(f"  English Fluency: {user_profile.get('english_fluency', 'Not provided')}")
        output_lines.append(f"  Other Languages: {user_profile.get('other_languages', 'Not provided')}")
        output_lines.append("")

        output_lines.append("CULTURAL CONTEXT:")
        output_lines.append(f"  Nationality: {user_profile.get('nationality', 'Not provided')}")
        output_lines.append(f"  Cultural Background: {user_profile.get('cultural_background', 'Not provided')}")
        output_lines.append("")

        output_lines.append("EDUCATIONAL BACKGROUND:")
        output_lines.append(f"  Education Level: {user_profile.get('education_level', 'Not provided')}")
        output_lines.append(f"  Field of Study: {user_profile.get('field_of_study', 'Not provided')}")
        output_lines.append("")

        output_lines.append("WRITING EXPERIENCE:")
        output_lines.append(f"  Writing Experience: {user_profile.get('writing_experience', 'Not provided')}")
        output_lines.append(f"  Writing Frequency: {user_profile.get('writing_frequency', 'Not provided')}")
        output_lines.append("")
        output_lines.append("=" * 80)
        output_lines.append("")

    # Metadata
    metadata = style_profile.get('metadata', {})
    output_lines.append("ANALYSIS METADATA")
    output_lines.append("-" * 40)
    output_lines.append(f"Analysis Date: {metadata.get('analysis_date', 'Unknown')}")
    output_lines.append(f"Analysis Method: {metadata.get('analysis_method', 'Enhanced Analysis')}")
    output_lines.append(f"Model Used: {metadata.get('model_used', 'Unknown')}")
    output_lines.append(f"Total Samples Analyzed: {metadata.get('total_samples', 'Unknown')}")
    output_lines.append(f"Combined Text Length: {metadata.get('combined_text_length', 0)} characters")
    output_lines.append("")

    # File information — ALL accesses use .get() with safe defaults
    file_info_list = metadata.get('file_info', [])
    if file_info_list:
        output_lines.append("SOURCE FILES")
        output_lines.append("-" * 40)
        for fi in file_info_list:
            filename  = fi.get('filename', 'unknown')
            word_count = fi.get('word_count', 0)
            # Accept both 'character_count' and legacy 'char_count'
            char_count = fi.get('character_count', fi.get('char_count', 0))
            source    = fi.get('source', '')
            source_tag = f" [{source}]" if source else ""
            output_lines.append(
                f"• {filename}{source_tag}: {word_count} words, {char_count} characters"
            )
        output_lines.append("")

    # Statistical analysis
    stats = style_profile.get('text_statistics', {})
    if stats:
        output_lines.append("STATISTICAL ANALYSIS")
        output_lines.append("-" * 40)
        output_lines.append(f"Total Words: {stats.get('word_count', 'N/A')}")
        output_lines.append(f"Total Sentences: {stats.get('sentence_count', 'N/A')}")
        output_lines.append(f"Total Paragraphs: {stats.get('paragraph_count', 'N/A')}")
        output_lines.append(f"Average Words per Sentence: {stats.get('avg_words_per_sentence', 'N/A')}")
        output_lines.append(f"Lexical Diversity Score: {stats.get('lexical_diversity', 'N/A')}")
        output_lines.append("")

        if 'punctuation_counts' in stats:
            output_lines.append("PUNCTUATION PATTERNS")
            output_lines.append("-" * 25)
            for k, v in stats['punctuation_counts'].items():
                output_lines.append(f"• {k.capitalize()}: {v}")
            output_lines.append("")

        if 'word_frequency' in stats:
            output_lines.append("MOST FREQUENT WORDS")
            output_lines.append("-" * 25)
            for word, freq in list(stats['word_frequency'].items())[:10]:
                output_lines.append(f"• '{word}': {freq} times")
            output_lines.append("")

    # Readability metrics
    rm = style_profile.get('readability_metrics', {})
    if rm:
        output_lines.append("READABILITY ANALYSIS")
        output_lines.append("-" * 40)
        output_lines.append(f"Flesch Reading Ease: {rm.get('flesch_reading_ease', 'N/A')} (0-100, higher = easier)")
        output_lines.append(f"Flesch-Kincaid Grade Level: {rm.get('flesch_kincaid_grade', 'N/A')}")
        output_lines.append(f"Coleman-Liau Index: {rm.get('coleman_liau_index', 'N/A')}")
        output_lines.append(f"Average Syllables per Word: {rm.get('avg_syllables_per_word', 'N/A')}")
        output_lines.append("")

    # Individual file analyses
    individual = style_profile.get('individual_analyses', [])
    if individual:
        output_lines.append("INDIVIDUAL FILE ANALYSES")
        output_lines.append("=" * 50)
        for i, analysis in enumerate(individual, 1):
            fname = analysis.get('filename', f'File {i}')
            output_lines.append(f"\nFILE {i}: {fname}")
            output_lines.append("-" * (10 + len(str(fname))))
            output_lines.append(f"Character Count: {analysis.get('character_count', 'N/A')}")
            output_lines.append(f"Word Count: {analysis.get('word_count', 'N/A')}")
            output_lines.append("")
            output_lines.append("STYLOMETRIC ANALYSIS:")
            raw_analysis = analysis.get('analysis', '')
            if isinstance(raw_analysis, dict):
                raw_analysis = json.dumps(raw_analysis, indent=2)
            for line in str(raw_analysis).split('\n'):
                if line.strip():
                    output_lines.append(f"  {line}")
            output_lines.append("")

    # Consolidated analysis
    ca = style_profile.get('consolidated_analysis', '')
    if ca:
        output_lines.append("CONSOLIDATED STYLOMETRIC PROFILE")
        output_lines.append("=" * 50)
        if isinstance(ca, dict):
            ca = json.dumps(ca, indent=2)
        for line in str(ca).split('\n'):
            if line.strip():
                output_lines.append(line)
        output_lines.append("")

    # Style fingerprint (v3.1+)
    fp = style_profile.get('style_fingerprint_summary', '')
    if fp:
        output_lines.append("STYLE FINGERPRINT")
        output_lines.append("-" * 40)
        output_lines.append(fp)
        output_lines.append("")

    # Most distinctive traits (v3.1+)
    traits = style_profile.get('most_distinctive_traits', [])
    if traits:
        output_lines.append("MOST DISTINCTIVE TRAITS")
        output_lines.append("-" * 40)
        for t in traits:
            output_lines.append(f"• {t}")
        output_lines.append("")

    # Insights footer
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

    output_lines.append("=" * 80)
    output_lines.append("End of Enhanced Deep Stylometry Analysis Report")
    output_lines.append("Generated by Style Transfer AI - Enhanced Deep Analysis v4.5")
    output_lines.append("=" * 80)

    return '\n'.join(output_lines)


def save_dual_format(style_profile, base_filename, user_name="Anonymous_User"):
    """
    Save style profile in both JSON and TXT formats with user-specific naming.
    """
    import os

    fingerprints_dir = "stylometry fingerprints"
    os.makedirs(fingerprints_dir, exist_ok=True)

    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    json_filename = os.path.join(fingerprints_dir, f"{user_name}_stylometric_profile_{timestamp}.json")
    txt_filename  = os.path.join(fingerprints_dir, f"{user_name}_stylometric_profile_{timestamp}.txt")

    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(style_profile, f, indent=2, ensure_ascii=False)

    human_readable = format_human_readable_output(style_profile)
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write(human_readable)

    return json_filename, txt_filename