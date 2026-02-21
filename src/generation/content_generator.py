"""
Core content generation engine for Style Transfer AI.
Uses analyzed style profiles to generate new content in specific writing styles.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Union

from ..models.ollama_client import analyze_with_ollama
from ..models.openai_client import analyze_with_openai  
from ..models.gemini_client import analyze_with_gemini
from ..utils.text_processing import extract_basic_stats
from ..analysis.metrics import extract_deep_stylometry, calculate_style_similarity
from ..config.settings import TIMESTAMP_FORMAT
from .templates import GenerationTemplates


class ContentGenerator:
    """
    Generates new content based on analyzed writing style profiles.
    
    Supports multiple content types and generation modes:
    - Guided generation (with prompts/topics)
    - Free-form generation (style-only constraints)
    - Template-based generation (emails, articles, etc.)
    """
    
    def __init__(self):
        self.templates = GenerationTemplates()
        self.supported_content_types = [
            'email', 'article', 'story', 'essay', 'letter', 
            'review', 'blog_post', 'social_media', 'academic', 'creative'
        ]
    
    def generate_content(
        self, 
        style_profile: Dict,
        content_type: str,
        topic_or_prompt: str,
        target_length: int = 500,
        tone: str = "neutral",
        additional_context: str = "",
        use_local: bool = True,
        model_name: Optional[str] = None,
        api_type: Optional[str] = None,
        api_client = None
    ) -> Dict:
        """
        Generate new content using a specific writing style profile.
        
        Args:
            style_profile (Dict): Analyzed style profile to emulate
            content_type (str): Type of content to generate
            topic_or_prompt (str): Topic, prompt, or content brief
            target_length (int): Approximate target word count
            tone (str): Desired tone for the content
            additional_context (str): Additional context or requirements
            use_local (bool): Use local Ollama vs API models
            model_name (str): Specific model for generation
            api_type (str): 'openai' or 'gemini' for cloud APIs
            api_client: Pre-initialized API client
            
        Returns:
            Dict: Generated content with metadata and quality metrics
        """
        try:
            # Validate inputs
            if content_type not in self.supported_content_types:
                raise ValueError(f"Unsupported content type: {content_type}")
            
            # Extract style characteristics from profile
            style_essence = self._extract_style_essence(style_profile)
            
            # Build generation prompt
            generation_prompt = self._build_generation_prompt(
                style_essence=style_essence,
                content_type=content_type,
                topic_or_prompt=topic_or_prompt,
                target_length=target_length,
                tone=tone,
                additional_context=additional_context
            )
            
            # Generate content using specified model
            generated_text = self._execute_generation(
                prompt=generation_prompt,
                use_local=use_local,
                model_name=model_name,
                api_type=api_type,
                api_client=api_client
            )
            
            # Analyze and validate generated content
            quality_metrics = self._analyze_generated_content(generated_text, style_profile)
            
            # Package results
            result = {
                'generated_content': generated_text,
                'generation_metadata': {
                    'content_type': content_type,
                    'topic_prompt': topic_or_prompt,
                    'target_length': target_length,
                    'actual_length': len(generated_text.split()),
                    'tone': tone,
                    'additional_context': additional_context,
                    'model_used': model_name or api_type,
                    'timestamp': datetime.now().strftime(TIMESTAMP_FORMAT),
                    'style_profile_source': style_profile.get('metadata', {}).get('source_files', 'Unknown')
                },
                'quality_metrics': quality_metrics,
                'style_adherence_score': self._calculate_style_adherence(generated_text, style_profile)
            }
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'generated_content': None,
                'timestamp': datetime.now().strftime(TIMESTAMP_FORMAT)
            }
    
    def _extract_style_essence(self, style_profile: Dict) -> Dict:
        """
        Extract key stylistic elements from a style profile for generation.
        
        Args:
            style_profile (Dict): Complete style analysis profile
            
        Returns:
            Dict: Essential style characteristics for generation
        """
        essence = {
            'linguistic_patterns': {},
            'vocabulary_preferences': {},
            'structural_tendencies': {},
            'tone_characteristics': {}
        }
        
        try:
            # Extract from statistical analysis
            if 'statistical_analysis' in style_profile:
                stats = style_profile['statistical_analysis']
                essence['linguistic_patterns'] = {
                    'avg_sentence_length': stats.get('average_sentence_length', 15),
                    'lexical_diversity': stats.get('lexical_diversity', 0.5),
                    'readability_level': stats.get('readability_scores', {}).get('flesch_reading_ease', 50)
                }
            
            # Extract from deep analysis (if available)
            if 'deep_analysis' in style_profile:
                deep = style_profile['deep_analysis']
                # Parse deep analysis text for key patterns
                essence['tone_characteristics'] = self._parse_tone_from_analysis(deep)
                essence['vocabulary_preferences'] = self._parse_vocabulary_from_analysis(deep)
                essence['structural_tendencies'] = self._parse_structure_from_analysis(deep)
            
            return essence
            
        except Exception as e:
            # Fallback to basic patterns
            return {
                'linguistic_patterns': {'avg_sentence_length': 15, 'lexical_diversity': 0.5},
                'vocabulary_preferences': {'formality_level': 'moderate'},
                'structural_tendencies': {'paragraph_style': 'standard'},
                'tone_characteristics': {'overall_tone': 'neutral'}
            }
    
    def _build_generation_prompt(
        self, 
        style_essence: Dict, 
        content_type: str, 
        topic_or_prompt: str, 
        target_length: int,
        tone: str = "neutral",
        additional_context: str = ""
    ) -> str:
        """Build the AI prompt for content generation based on style profile."""
        
        # Get content type template
        content_template = self.templates.get_content_template(content_type)
        
        # Build style instructions
        style_instructions = self._build_style_instructions(style_essence)
        
        # Add tone and context sections
        tone_instruction = f"\nDESIRED TONE: {tone}" if tone and tone != "neutral" else ""
        context_instruction = f"\nADDITIONAL CONTEXT/REQUIREMENTS:\n{additional_context}" if additional_context else ""
        
        prompt = f"""
TASK: Generate a {content_type} following the specific writing style profile provided.

TOPIC/PROMPT: {topic_or_prompt}

TARGET LENGTH: Approximately {target_length} words{tone_instruction}{context_instruction}

WRITING STYLE PROFILE TO EMULATE:
{style_instructions}

CONTENT TYPE REQUIREMENTS:
{content_template}

GENERATION GUIDELINES:
1. Strictly adhere to the provided writing style characteristics
2. Maintain the specified tone and formality level
3. Use vocabulary and sentence structures matching the profile
4. Follow the content type conventions while preserving personal style
5. Ensure natural flow and readability
6. Target the specified word count (±20%)

Generate the content now, ensuring it authentically reflects the specified writing style:
"""
        return prompt
    
    def _build_style_instructions(self, style_essence: Dict) -> str:
        """Convert style essence into detailed instructions for AI generation."""
        
        instructions = []
        
        # Linguistic patterns
        if 'linguistic_patterns' in style_essence:
            patterns = style_essence['linguistic_patterns']
            instructions.append(f"- Average sentence length: {patterns.get('avg_sentence_length', 15)} words")
            instructions.append(f"- Lexical diversity: {patterns.get('lexical_diversity', 0.5)} (variety in word choice)")
            instructions.append(f"- Readability level: {patterns.get('readability_level', 50)} (Flesch score)")
        
        # Vocabulary preferences
        if 'vocabulary_preferences' in style_essence:
            vocab = style_essence['vocabulary_preferences']
            instructions.append(f"- Formality level: {vocab.get('formality_level', 'moderate')}")
            if 'preferred_domains' in vocab:
                instructions.append(f"- Preferred vocabulary domains: {', '.join(vocab['preferred_domains'])}")
        
        # Tone characteristics
        if 'tone_characteristics' in style_essence:
            tone = style_essence['tone_characteristics']
            instructions.append(f"- Overall tone: {tone.get('overall_tone', 'neutral')}")
            if 'emotional_range' in tone:
                instructions.append(f"- Emotional expression: {tone['emotional_range']}")
        
        # Structural tendencies
        if 'structural_tendencies' in style_essence:
            structure = style_essence['structural_tendencies']
            instructions.append(f"- Paragraph style: {structure.get('paragraph_style', 'standard')}")
            if 'transition_style' in structure:
                instructions.append(f"- Transition style: {structure['transition_style']}")
        
        return "\n".join(instructions)
    
    def _execute_generation(
        self, 
        prompt: str, 
        use_local: bool, 
        model_name: Optional[str], 
        api_type: Optional[str], 
        api_client
    ) -> str:
        """Execute the content generation using the specified AI model."""
        
        try:
            if use_local and model_name:
                result = analyze_with_ollama(prompt, model_name, processing_mode="enhanced")
                return result
            elif api_type == "openai" and api_client:
                result = analyze_with_openai(api_client, prompt)
                return result
            elif api_type == "gemini" and api_client:
                result = analyze_with_gemini(api_client, prompt)
                return result
            else:
                raise ValueError("Invalid model configuration for generation")
                
        except Exception as e:
            raise RuntimeError(f"Content generation failed: {str(e)}")
    
    def _analyze_generated_content(self, generated_text: str, original_style_profile: Dict) -> Dict:
        """Analyze the quality and characteristics of generated content."""
        
        basic_stats = extract_basic_stats(generated_text)
        
        quality_metrics = {
            'word_count': basic_stats.get('word_count', 0),
            'sentence_count': basic_stats.get('sentence_count', 0),
            'paragraph_count': basic_stats.get('paragraph_count', 0),
            'avg_sentence_length': basic_stats.get('avg_sentence_length', 0),
            'readability_estimate': self._estimate_readability(generated_text),
            'coherence_score': self._estimate_coherence(generated_text),
            'style_consistency': self._estimate_style_consistency(generated_text)
        }
        
        return quality_metrics
    
    def _calculate_style_adherence(self, generated_text: str, style_profile: Dict) -> float:
        """
        Calculate how well the generated text adheres to the target style profile
        using deep stylometry extraction and multi-method similarity comparison.
        
        Returns a score between 0.0 and 1.0 (1.0 = perfect match).
        """
        try:
            # Extract deep stylometry from the generated text
            generated_ds = extract_deep_stylometry(generated_text)
            
            # Get target deep stylometry from the style profile
            target_ds = style_profile.get('deep_stylometry', {})
            
            if generated_ds and target_ds:
                # Full multi-method comparison
                similarity = calculate_style_similarity(generated_ds, target_ds)
                return similarity.get('combined_score', 0.5)
            
            # Fallback to basic stat comparison if deep stylometry unavailable
            generated_stats = extract_basic_stats(generated_text)
            original_stats = style_profile.get('statistical_analysis', style_profile.get('text_statistics', {}))
            
            sentence_length_diff = abs(
                generated_stats.get('avg_sentence_length', 15) - 
                original_stats.get('average_sentence_length', original_stats.get('avg_sentence_length', 15))
            )
            
            score = max(0.0, 1.0 - (sentence_length_diff / 15.0))
            return min(1.0, score)
            
        except Exception:
            return 0.5  # Neutral score if comparison fails
    
    def _parse_tone_from_analysis(self, deep_analysis: str) -> Dict:
        """Extract tone characteristics from deep analysis text."""
        # Placeholder - would use NLP to extract tone markers
        return {'overall_tone': 'neutral', 'emotional_range': 'moderate'}
    
    def _parse_vocabulary_from_analysis(self, deep_analysis: str) -> Dict:
        """Extract vocabulary preferences from deep analysis text."""
        # Placeholder - would analyze vocabulary sophistication
        return {'formality_level': 'moderate', 'complexity': 'medium'}
    
    def _parse_structure_from_analysis(self, deep_analysis: str) -> Dict:
        """Extract structural patterns from deep analysis text."""
        # Placeholder - would identify structural preferences
        return {'paragraph_style': 'standard', 'organization': 'logical'}
    
    def _estimate_readability(self, text: str) -> float:
        """Estimate readability score of generated text."""
        # Simplified readability estimation
        avg_sentence_length = len(text.split()) / max(1, text.count('.'))
        return max(0.0, min(100.0, 100 - avg_sentence_length * 2))
    
    def _estimate_coherence(self, text: str) -> float:
        """Estimate coherence/flow score of generated text."""
        # Placeholder - would analyze logical flow and transitions
        return 0.75  # Default coherence score
    
    def _estimate_style_consistency(self, text: str) -> float:
        """Estimate style consistency throughout the generated text."""
        # Placeholder - would analyze style variation within text
        return 0.80  # Default consistency score