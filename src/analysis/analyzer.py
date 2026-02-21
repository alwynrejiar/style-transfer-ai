"""
Core analysis engine for Style Transfer AI.
Orchestrates the stylometry analysis workflow.
"""

from datetime import datetime
from .prompts import create_enhanced_deep_prompt
from .metrics import calculate_readability_metrics, analyze_text_statistics, extract_deep_stylometry
from ..utils.text_processing import read_text_file, extract_basic_stats
from ..utils.user_profile import get_user_profile
from ..config.settings import TIMESTAMP_FORMAT


def analyze_style(text_to_analyze, use_local=True, model_name=None, api_type=None, api_client=None, user_profile=None, processing_mode="enhanced"):
    """
    Performs enhanced deep stylometry analysis using specified AI model.
    
    Args:
        text_to_analyze (str): The text to be analyzed
        use_local (bool): Whether to use local Ollama model or cloud APIs
        model_name (str): The specific model to use (for local models)
        api_type (str): 'openai' or 'gemini' for cloud APIs
        api_client: Pre-initialized API client (for OpenAI or Gemini)
        user_profile (dict): User background information for context
        processing_mode (str): 'enhanced' for thorough analysis, 'statistical' for faster processing
        
    Returns:
        str: Structured deep stylometric analysis for style profiling
    """
    # Validate model parameters
    if use_local and not model_name:
        raise ValueError("model_name is required when use_local=True")
    if not use_local and (not api_type or not api_client):
        raise ValueError("api_type and api_client are required when use_local=False")
    
    # Lazy imports to avoid circular dependencies
    from ..models.ollama_client import analyze_with_ollama
    from ..models.openai_client import analyze_with_openai
    from ..models.gemini_client import analyze_with_gemini

    prompt = create_enhanced_deep_prompt(text_to_analyze, user_profile)
    
    if use_local:
        return analyze_with_ollama(prompt, model_name, processing_mode)
    elif api_type == "openai":
        return analyze_with_openai(api_client, prompt)
    elif api_type == "gemini":
        return analyze_with_gemini(api_client, prompt)
    else:
        return "Error: Unknown API type or configuration"


def create_enhanced_style_profile(input_data, use_local=True, model_name=None, api_type=None, api_client=None, processing_mode="enhanced"):
    """
    Creates an enhanced comprehensive style profile from text samples or direct input.
    
    Args:
        input_data (list or dict): Either list of file paths or dict with custom text
        use_local (bool): Whether to use local Ollama model or cloud APIs
        model_name (str): The specific model to use (for local models)
        api_type (str): 'openai' or 'gemini' for cloud APIs
        api_client: Pre-initialized API client (for OpenAI or Gemini)
        processing_mode (str): 'enhanced' for thorough analysis, 'statistical' for faster processing
        
    Returns:
        dict: Enhanced consolidated style profile with deep analysis
    """
    # Validate model parameters
    if use_local and not model_name:
        raise ValueError("model_name is required when use_local=True")
    if not use_local and (not api_type or not api_client):
        raise ValueError("api_type and api_client are required when use_local=False")
    
    # Collect user profile information first
    user_profile = get_user_profile()
    
    all_analyses = []
    combined_text = ""
    file_info = []
    
    # Check if input is custom text or file paths
    if isinstance(input_data, dict) and input_data.get('type') == 'custom_text':
        # Handle custom text input
        print(f"\nAnalyzing custom text input ({input_data['word_count']} words)")
        
        file_content = input_data['text']
        source_name = "custom_text_input"
        
        # Calculate basic statistics for the text
        stats = extract_basic_stats(file_content)
        
        file_info.append({
            'filename': source_name,
            'word_count': stats['word_count'],
            'character_count': stats['character_count'],
            'source': 'direct_input'
        })
        
        # Perform deep analysis on the text
        print(f"  Performing deep analysis...")
        individual_analysis = analyze_style(file_content, use_local, model_name, api_type, api_client, user_profile, processing_mode)
        
        all_analyses.append({
            'filename': source_name,
            'word_count': stats['word_count'],
            'character_count': stats['character_count'],
            'source': 'direct_input',
            'analysis': individual_analysis
        })
        
        combined_text = file_content
        print(f"  Analysis completed for custom text")
        
    else:
        # Handle file paths (original logic)
        file_paths = input_data if isinstance(input_data, list) else []
        
        if not file_paths:
            return {
                'profile_created': False,
                'error': 'No valid input provided'
            }
        
        print(f"\nFound {len(file_paths)} text sample(s) for enhanced deep analysis")
        
        for file_path in file_paths:
            file_content = read_text_file(file_path)
            
            if "Error" not in file_content:
                print(f"  Processing: {file_path}")
                
                # Calculate basic statistics for this file
                stats = extract_basic_stats(file_content)
                
                file_info.append({
                    'filename': file_path,
                    'word_count': stats['word_count'],
                    'character_count': stats['character_count']
                })
                
                # Individual analysis
                print(f"  Performing deep analysis...")
                individual_analysis = analyze_style(file_content, use_local, model_name, api_type, api_client, user_profile, processing_mode)
                
                all_analyses.append({
                    'filename': file_path,
                    'word_count': stats['word_count'],
                    'character_count': stats['character_count'],
                    'analysis': individual_analysis
                })
                
                combined_text += f"\n\n--- From {file_path} ---\n{file_content}"
                print(f"  Analysis completed for {file_path}")
            else:
                print(f"  Error with {file_path}: {file_content}")
    
    if not all_analyses:
        return {
            'profile_created': False,
            'error': 'No valid input could be analyzed'
        }
    
    # Perform enhanced statistical analysis on combined text
    print("Calculating comprehensive statistics...")
    text_statistics = analyze_text_statistics(combined_text)
    
    # Calculate readability metrics
    print("Computing readability metrics...")
    readability_metrics = calculate_readability_metrics(combined_text)

    # Extract deep syntactic stylometry
    print("Extracting deep syntactic stylometry...")
    deep_stylometry = extract_deep_stylometry(combined_text)
    
    # Consolidated analysis of all texts combined
    print("Generating consolidated deep analysis...")
    consolidated_analysis = analyze_style(combined_text, use_local, model_name, api_type, api_client, user_profile, processing_mode)
    
    # Create comprehensive metadata
    analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    analysis_method = "Local Ollama" if use_local else f"Cloud API ({api_type})"
    model_used = model_name if use_local else f"{api_type.title()} API"
    
    # Build the complete style profile
    style_profile = {
        'profile_created': True,
        'user_profile': user_profile,
        'metadata': {
            'analysis_date': analysis_date,
            'analysis_method': analysis_method,
            'model_used': model_used,
            'processing_mode': processing_mode,
            'total_samples': len(all_analyses),
            'combined_text_length': len(combined_text),
            'file_info': file_info
        },
        'text_statistics': text_statistics,
        'readability_metrics': readability_metrics,
        'deep_stylometry': deep_stylometry,
        'individual_analyses': all_analyses,
        'consolidated_analysis': consolidated_analysis
    }
    
    return style_profile


def display_enhanced_results(style_profile):
    """Display a summary of the analysis results including deep stylometry."""
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("="*60)
    
    if 'user_profile' in style_profile and 'name' in style_profile['user_profile']:
        print(f"Profile created for: {style_profile['user_profile']['name']}")
    
    print(f"Files analyzed: {style_profile['metadata']['total_samples']}")
    print(f"Total text length: {style_profile['metadata']['combined_text_length']} characters")
    print(f"Model used: {style_profile['metadata']['model_used']}")
    print(f"Processing mode: {style_profile['metadata']['processing_mode']}")
    
    if 'text_statistics' in style_profile:
        stats = style_profile['text_statistics']
        print(f"\n--- Text Statistics ---")
        print(f"Word count: {stats.get('word_count', 'N/A')}")
        print(f"Sentence count: {stats.get('sentence_count', 'N/A')}")
        print(f"Lexical diversity (TTR): {stats.get('lexical_diversity', 'N/A')}")
    
    if 'readability_metrics' in style_profile:
        metrics = style_profile['readability_metrics']
        print(f"\n--- Readability ---")
        print(f"Flesch Reading Ease: {metrics.get('flesch_reading_ease', 'N/A')}")
        print(f"Grade Level: {metrics.get('flesch_kincaid_grade', 'N/A')}")
        print(f"Coleman-Liau Index: {metrics.get('coleman_liau_index', 'N/A')}")
    
    if 'deep_stylometry' in style_profile:
        ds = style_profile['deep_stylometry']
        print(f"\n--- Deep Stylometry ---")
        
        # Sentence structure
        sl = ds.get('sentence_length_distribution', {})
        if sl.get('mean', 0):
            print(f"Avg sentence length: {sl['mean']} words (std: {sl.get('std_dev', 0)}, range: {sl.get('min', 0)}-{sl.get('max', 0)})")
        
        print(f"Avg dependency depth: {ds.get('avg_dependency_depth', 'N/A')}")
        print(f"Max dependency depth: {ds.get('max_dependency_depth', 'N/A')}")
        print(f"Avg word length: {ds.get('avg_word_length', 'N/A')} chars")
        
        # Voice & style
        pv = ds.get('passive_voice_ratio', 0)
        cr = ds.get('contraction_rate', 0)
        print(f"Passive voice: {round(pv * 100, 1)}%")
        print(f"Contraction rate: {round(cr * 100, 1)}%")
        print(f"Punctuation density: {ds.get('punctuation_density', 'N/A')} per sentence")
        print(f"Question ratio: {round(ds.get('question_ratio', 0) * 100, 1)}%")
        
        # Vocabulary richness
        vr = ds.get('vocabulary_richness', {})
        if vr:
            print(f"\n--- Vocabulary Richness ---")
            print(f"Hapax legomena ratio: {vr.get('hapax_legomena_ratio', 'N/A')}")
            print(f"Yule's K: {vr.get('yules_k', 'N/A')}")
            print(f"Simpson's diversity: {vr.get('simpsons_diversity', 'N/A')}")
            print(f"Brunet's W: {vr.get('brunet_w', 'N/A')}")
            print(f"Honore's R: {vr.get('honore_r', 'N/A')}")
        
        # Top POS
        pos = ds.get('pos_ratios', {})
        if pos:
            top_pos = sorted(pos.items(), key=lambda x: x[1], reverse=True)[:5]
            top_pos_str = ", ".join(f"{k}: {round(v*100, 1)}%" for k, v in top_pos if v > 0)
            print(f"\nTop POS tags: {top_pos_str}")
        
        # Top function words
        fw = ds.get('function_word_freq', {})
        if fw:
            top_fw = sorted(fw.items(), key=lambda x: x[1], reverse=True)[:8]
            top_fw_str = ", ".join(f"'{k}': {round(v*100, 1)}%" for k, v in top_fw if v > 0)
            print(f"Top function words: {top_fw_str}")
    
    print("\n" + "="*60)