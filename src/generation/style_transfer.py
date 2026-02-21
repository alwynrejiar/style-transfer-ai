"""
Style transfer and content restyling engine for Style Transfer AI.
Transforms existing content to match different writing style profiles.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ..models.ollama_client import analyze_with_ollama
from ..models.openai_client import analyze_with_openai
from ..models.gemini_client import analyze_with_gemini
from ..utils.text_processing import extract_basic_stats
from ..analysis.metrics import extract_deep_stylometry, calculate_style_similarity
from ..config.settings import TIMESTAMP_FORMAT
from .templates import GenerationTemplates


class StyleTransfer:
    """
    Transforms existing content to match different writing style profiles.
    
    Supports:
    - Direct style transfer (content A in style B)
    - Style blending (mix multiple style profiles)
    - Gradual style transformation (subtle to dramatic changes)
    - Bidirectional style comparison and analysis
    """
    
    def __init__(self):
        self.templates = GenerationTemplates()
        self.transfer_types = [
            'direct_transfer', 'style_blend', 'gradual_transform',
            'tone_shift', 'formality_adjust', 'audience_adapt'
        ]
    
    def transfer_style(
        self,
        original_content: str,
        target_style_profile: Dict,
        transfer_type: str = 'direct_transfer',
        intensity: float = 1.0,
        preserve_elements: List[str] = None,
        use_local: bool = True,
        model_name: Optional[str] = None,
        api_type: Optional[str] = None,
        api_client = None
    ) -> Dict:
        """
        Transform content to match a target writing style profile.
        
        Args:
            original_content (str): Content to be restyled
            target_style_profile (Dict): Style profile to emulate
            transfer_type (str): Type of style transfer to perform
            intensity (float): How dramatic the style change should be (0.1-1.0)
            preserve_elements (List[str]): Elements to preserve during transfer
            use_local (bool): Use local Ollama vs API models
            model_name (str): Specific model for transfer
            api_type (str): 'openai' or 'gemini' for cloud APIs
            api_client: Pre-initialized API client
            
        Returns:
            Dict: Transferred content with analysis and metadata
        """
        try:
            # Validate inputs
            if transfer_type not in self.transfer_types:
                raise ValueError(f"Unsupported transfer type: {transfer_type}")
            
            if not 0.1 <= intensity <= 1.0:
                raise ValueError("Intensity must be between 0.1 and 1.0")
            
            # Analyze original content
            original_analysis = self._analyze_original_content(original_content)
            
            # Extract target style characteristics
            target_style = self._extract_style_characteristics(target_style_profile)
            
            # Build transfer prompt
            transfer_prompt = self._build_transfer_prompt(
                original_content=original_content,
                original_analysis=original_analysis,
                target_style=target_style,
                transfer_type=transfer_type,
                intensity=intensity,
                preserve_elements=preserve_elements or []
            )
            
            # Execute style transfer
            transferred_content = self._execute_transfer(
                prompt=transfer_prompt,
                use_local=use_local,
                model_name=model_name,
                api_type=api_type,
                api_client=api_client
            )
            
            # Analyze transfer quality
            transfer_analysis = self._analyze_transfer_quality(
                original_content=original_content,
                transferred_content=transferred_content,
                target_style_profile=target_style_profile
            )
            
            # Package results
            result = {
                'original_content': original_content,
                'transferred_content': transferred_content,
                'transfer_metadata': {
                    'transfer_type': transfer_type,
                    'intensity': intensity,
                    'preserve_elements': preserve_elements or [],
                    'model_used': model_name or api_type,
                    'timestamp': datetime.now().strftime(TIMESTAMP_FORMAT),
                    'target_style_source': target_style_profile.get('metadata', {}).get('source_files', 'Unknown')
                },
                'quality_analysis': transfer_analysis,
                'style_match_score': self._calculate_style_match_score(transferred_content, target_style_profile)
            }
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'original_content': original_content,
                'transferred_content': None,
                'timestamp': datetime.now().strftime(TIMESTAMP_FORMAT)
            }
    
    def blend_styles(
        self,
        original_content: str,
        style_profiles: List[Dict],
        blend_weights: List[float] = None,
        use_local: bool = True,
        model_name: Optional[str] = None,
        api_type: Optional[str] = None,
        api_client = None
    ) -> Dict:
        """
        Blend multiple style profiles and apply to content.
        
        Args:
            original_content (str): Content to be restyled
            style_profiles (List[Dict]): Multiple style profiles to blend
            blend_weights (List[float]): Relative weights for each style (must sum to 1.0)
            use_local (bool): Use local Ollama vs API models
            model_name (str): Specific model for blending
            api_type (str): 'openai' or 'gemini' for cloud APIs
            api_client: Pre-initialized API client
            
        Returns:
            Dict: Content with blended style applied
        """
        try:
            if len(style_profiles) < 2:
                raise ValueError("Style blending requires at least 2 style profiles")
            
            # Normalize blend weights
            if blend_weights is None:
                blend_weights = [1.0 / len(style_profiles)] * len(style_profiles)
            
            if len(blend_weights) != len(style_profiles):
                raise ValueError("Number of blend weights must match number of style profiles")
            
            # Normalize weights to sum to 1.0
            weight_sum = sum(blend_weights)
            blend_weights = [w / weight_sum for w in blend_weights]
            
            # Create blended style characteristics
            blended_style = self._create_blended_style(style_profiles, blend_weights)
            
            # Use direct transfer with the blended style
            return self.transfer_style(
                original_content=original_content,
                target_style_profile={'blended_style': blended_style},
                transfer_type='style_blend',
                use_local=use_local,
                model_name=model_name,
                api_type=api_type,
                api_client=api_client
            )
            
        except Exception as e:
            return {
                'error': str(e),
                'original_content': original_content,
                'transferred_content': None,
                'timestamp': datetime.now().strftime(TIMESTAMP_FORMAT)
            }
    
    def compare_styles(self, content1: str, content2: str) -> Dict:
        """
        Compare the writing styles of two pieces of content.
        
        Args:
            content1 (str): First content sample
            content2 (str): Second content sample
            
        Returns:
            Dict: Detailed style comparison analysis
        """
        try:
            # Analyze both contents
            analysis1 = self._analyze_original_content(content1)
            analysis2 = self._analyze_original_content(content2)
            
            # Calculate differences
            differences = self._calculate_style_differences(analysis1, analysis2)
            
            # Generate comparison summary
            comparison = {
                'content1_analysis': analysis1,
                'content2_analysis': analysis2,
                'style_differences': differences,
                'similarity_score': self._calculate_similarity_score(analysis1, analysis2),
                'recommendations': self._generate_style_recommendations(differences),
                'timestamp': datetime.now().strftime(TIMESTAMP_FORMAT)
            }
            
            return comparison
            
        except Exception as e:
            return {
                'error': str(e),
                'timestamp': datetime.now().strftime(TIMESTAMP_FORMAT)
            }
    
    def _analyze_original_content(self, content: str) -> Dict:
        """Analyze the style characteristics of the original content."""
        
        basic_stats = extract_basic_stats(content)
        
        analysis = {
            'word_count': basic_stats.get('word_count', 0),
            'sentence_count': basic_stats.get('sentence_count', 0),
            'paragraph_count': basic_stats.get('paragraph_count', 0),
            'avg_sentence_length': basic_stats.get('avg_sentence_length', 0),
            'lexical_diversity': self._calculate_lexical_diversity(content),
            'formality_level': self._estimate_formality_level(content),
            'tone_indicators': self._identify_tone_indicators(content),
            'structural_patterns': self._identify_structural_patterns(content)
        }
        
        return analysis
    
    def _extract_style_characteristics(self, style_profile: Dict) -> Dict:
        """Extract key style characteristics from a style profile."""
        
        characteristics = {
            'linguistic_patterns': {},
            'vocabulary_style': {},
            'tone_profile': {},
            'structural_preferences': {}
        }
        
        try:
            # Extract from statistical analysis
            if 'statistical_analysis' in style_profile:
                stats = style_profile['statistical_analysis']
                characteristics['linguistic_patterns'] = {
                    'avg_sentence_length': stats.get('average_sentence_length', 15),
                    'lexical_diversity': stats.get('lexical_diversity', 0.5),
                    'readability_level': stats.get('readability_scores', {}).get('flesch_reading_ease', 50),
                    'punctuation_patterns': stats.get('punctuation_analysis', {})
                }
            
            # Extract from deep analysis if available
            if 'deep_analysis' in style_profile:
                deep = style_profile['deep_analysis']
                characteristics['tone_profile'] = self._extract_tone_from_analysis(deep)
                characteristics['vocabulary_style'] = self._extract_vocabulary_style(deep)
                characteristics['structural_preferences'] = self._extract_structural_preferences(deep)
            
            return characteristics
            
        except Exception:
            # Return default characteristics
            return {
                'linguistic_patterns': {'avg_sentence_length': 15},
                'vocabulary_style': {'formality': 'moderate'},
                'tone_profile': {'primary_tone': 'neutral'},
                'structural_preferences': {'organization': 'standard'}
            }
    
    def _build_transfer_prompt(
        self,
        original_content: str,
        original_analysis: Dict,
        target_style: Dict,
        transfer_type: str,
        intensity: float,
        preserve_elements: List[str]
    ) -> str:
        """Build the AI prompt for style transfer."""
        
        # Get transfer template
        transfer_template = self.templates.get_style_transfer_template(transfer_type)
        
        # Build style instructions
        style_instructions = self._build_style_transfer_instructions(target_style, intensity)
        
        # Build preservation instructions
        preservation_instructions = self._build_preservation_instructions(preserve_elements)
        
        prompt = f"""
TASK: Transform the following content to match the specified writing style profile.

ORIGINAL CONTENT:
{original_content}

TARGET WRITING STYLE PROFILE:
{style_instructions}

TRANSFER TYPE: {transfer_type}
INTENSITY LEVEL: {intensity} (1.0 = complete transformation, 0.1 = subtle changes)

PRESERVATION REQUIREMENTS:
{preservation_instructions}

TRANSFER GUIDELINES:
{transfer_template}

SPECIFIC INSTRUCTIONS:
1. Preserve the core meaning and factual information
2. Transform the writing style to match the target profile
3. Apply changes at the specified intensity level
4. Maintain natural flow and readability
5. Respect any preservation requirements
6. Ensure consistency throughout the transformed content

Transform the content now:
"""
        return prompt
    
    def _build_style_transfer_instructions(self, target_style: Dict, intensity: float) -> str:
        """Build detailed style transfer instructions."""
        
        instructions = []
        
        # Apply intensity modifier
        intensity_text = f"Apply these changes at {intensity*100:.0f}% intensity"
        instructions.append(f"INTENSITY: {intensity_text}")
        
        # Linguistic patterns
        if 'linguistic_patterns' in target_style:
            patterns = target_style['linguistic_patterns']
            if 'avg_sentence_length' in patterns:
                instructions.append(f"- Target sentence length: {patterns['avg_sentence_length']} words")
            if 'lexical_diversity' in patterns:
                instructions.append(f"- Vocabulary variety: {patterns['lexical_diversity']} level")
            if 'readability_level' in patterns:
                instructions.append(f"- Target readability: {patterns['readability_level']} (Flesch score)")
        
        # Vocabulary style
        if 'vocabulary_style' in target_style:
            vocab = target_style['vocabulary_style']
            if 'formality' in vocab:
                instructions.append(f"- Formality level: {vocab['formality']}")
            if 'complexity' in vocab:
                instructions.append(f"- Vocabulary complexity: {vocab['complexity']}")
        
        # Tone profile
        if 'tone_profile' in target_style:
            tone = target_style['tone_profile']
            if 'primary_tone' in tone:
                instructions.append(f"- Primary tone: {tone['primary_tone']}")
            if 'emotional_register' in tone:
                instructions.append(f"- Emotional expression: {tone['emotional_register']}")
        
        # Structural preferences
        if 'structural_preferences' in target_style:
            structure = target_style['structural_preferences']
            if 'organization' in structure:
                instructions.append(f"- Organization style: {structure['organization']}")
            if 'paragraph_style' in structure:
                instructions.append(f"- Paragraph structure: {structure['paragraph_style']}")
        
        return "\n".join(instructions)
    
    def _build_preservation_instructions(self, preserve_elements: List[str]) -> str:
        """Build instructions for elements to preserve during transfer."""
        
        if not preserve_elements:
            return "No specific preservation requirements."
        
        instructions = ["PRESERVE THE FOLLOWING ELEMENTS:"]
        for element in preserve_elements:
            instructions.append(f"- {element}")
        
        return "\n".join(instructions)
    
    def _execute_transfer(
        self,
        prompt: str,
        use_local: bool,
        model_name: Optional[str],
        api_type: Optional[str],
        api_client
    ) -> str:
        """Execute the style transfer using the specified AI model."""
        
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
                raise ValueError("Invalid model configuration for style transfer")
                
        except Exception as e:
            raise RuntimeError(f"Style transfer failed: {str(e)}")
    
    def _analyze_transfer_quality(
        self,
        original_content: str,
        transferred_content: str,
        target_style_profile: Dict
    ) -> Dict:
        """Analyze the quality of the style transfer."""
        
        original_stats = extract_basic_stats(original_content)
        transferred_stats = extract_basic_stats(transferred_content)
        
        quality_analysis = {
            'content_preservation': self._calculate_content_preservation(original_content, transferred_content),
            'style_transformation': self._calculate_style_transformation(original_stats, transferred_stats),
            'readability_improvement': self._calculate_readability_change(original_content, transferred_content),
            'coherence_maintained': self._evaluate_coherence_preservation(original_content, transferred_content),
            'length_change': {
                'original_words': original_stats.get('word_count', 0),
                'transferred_words': transferred_stats.get('word_count', 0),
                'change_ratio': transferred_stats.get('word_count', 0) / max(1, original_stats.get('word_count', 1))
            }
        }
        
        return quality_analysis
    
    def _calculate_style_match_score(self, transferred_content: str, target_style_profile: Dict) -> float:
        """
        Calculate how well the transferred content matches the target style
        using deep stylometry extraction and multi-method similarity.
        
        Returns a score between 0.0 and 1.0 (1.0 = perfect match).
        """
        try:
            # Extract deep stylometry from the transferred content
            transferred_ds = extract_deep_stylometry(transferred_content)
            
            # Get target deep stylometry from the style profile
            target_ds = target_style_profile.get('deep_stylometry', {})
            
            if transferred_ds and target_ds:
                similarity = calculate_style_similarity(transferred_ds, target_ds)
                return similarity.get('combined_score', 0.5)
            
            # Fallback to basic stat comparison
            transferred_stats = extract_basic_stats(transferred_content)
            target_stats = target_style_profile.get('statistical_analysis',
                                                      target_style_profile.get('text_statistics', {}))
            
            sentence_length_score = self._compare_sentence_lengths(
                transferred_stats.get('avg_sentence_length', 15),
                target_stats.get('average_sentence_length',
                                 target_stats.get('avg_sentence_length', 15))
            )
            
            return sentence_length_score
            
        except Exception:
            return 0.5  # Neutral score if comparison fails
    
    def _create_blended_style(self, style_profiles: List[Dict], blend_weights: List[float]) -> Dict:
        """Create a blended style profile from multiple profiles."""
        
        blended_style = {
            'linguistic_patterns': {},
            'vocabulary_style': {},
            'tone_profile': {},
            'structural_preferences': {}
        }
        
        # This would implement sophisticated style blending logic
        # For now, return a simple average-based blend
        
        return blended_style
    
    # Utility methods for analysis (simplified implementations)
    
    def _calculate_lexical_diversity(self, content: str) -> float:
        """Calculate lexical diversity of content."""
        words = content.lower().split()
        unique_words = set(words)
        return len(unique_words) / max(1, len(words))
    
    def _estimate_formality_level(self, content: str) -> str:
        """Estimate the formality level of content."""
        # Simplified formality estimation
        contractions = content.count("'")
        formal_words = sum(1 for word in content.split() if len(word) > 6)
        
        if contractions > formal_words:
            return "casual"
        elif formal_words > contractions * 2:
            return "formal"
        else:
            return "moderate"
    
    def _identify_tone_indicators(self, content: str) -> List[str]:
        """Identify tone indicators in content."""
        # Placeholder implementation
        return ["neutral", "informative"]
    
    def _identify_structural_patterns(self, content: str) -> Dict:
        """Identify structural patterns in content."""
        paragraphs = content.split('\n\n')
        return {
            'paragraph_count': len(paragraphs),
            'avg_paragraph_length': len(content.split()) / max(1, len(paragraphs))
        }
    
    def _extract_tone_from_analysis(self, deep_analysis: str) -> Dict:
        """Extract tone characteristics from deep analysis."""
        return {'primary_tone': 'neutral', 'emotional_register': 'moderate'}
    
    def _extract_vocabulary_style(self, deep_analysis: str) -> Dict:
        """Extract vocabulary style from deep analysis."""
        return {'formality': 'moderate', 'complexity': 'medium'}
    
    def _extract_structural_preferences(self, deep_analysis: str) -> Dict:
        """Extract structural preferences from deep analysis."""
        return {'organization': 'standard', 'paragraph_style': 'balanced'}
    
    def _calculate_content_preservation(self, original: str, transferred: str) -> float:
        """Calculate how well content meaning was preserved."""
        # Simplified content preservation score
        return 0.85  # Placeholder
    
    def _calculate_style_transformation(self, original_stats: Dict, transferred_stats: Dict) -> float:
        """Calculate the degree of style transformation."""
        # Simplified transformation score
        return 0.75  # Placeholder
    
    def _calculate_readability_change(self, original: str, transferred: str) -> float:
        """Calculate change in readability."""
        # Simplified readability comparison
        return 0.1  # Placeholder improvement
    
    def _evaluate_coherence_preservation(self, original: str, transferred: str) -> float:
        """Evaluate how well coherence was preserved."""
        return 0.90  # Placeholder
    
    def _compare_sentence_lengths(self, actual: float, target: float) -> float:
        """Compare sentence lengths for style matching."""
        diff = abs(actual - target)
        return max(0.0, 1.0 - (diff / 15.0))
    
    def _calculate_style_differences(self, analysis1: Dict, analysis2: Dict) -> Dict:
        """Calculate differences between two style analyses."""
        return {
            'sentence_length_diff': abs(
                analysis1.get('avg_sentence_length', 0) - 
                analysis2.get('avg_sentence_length', 0)
            ),
            'formality_diff': analysis1.get('formality_level') != analysis2.get('formality_level')
        }
    
    def _calculate_similarity_score(self, analysis1: Dict, analysis2: Dict) -> float:
        """Calculate similarity score between two analyses using available metrics."""
        try:
            # Use per-metric comparison for basic analyses
            scores = []
            
            # Sentence length similarity
            sl1 = analysis1.get('avg_sentence_length', 15)
            sl2 = analysis2.get('avg_sentence_length', 15)
            scores.append(max(0.0, 1.0 - abs(sl1 - sl2) / 15.0))
            
            # Lexical diversity similarity
            ld1 = analysis1.get('lexical_diversity', 0.5)
            ld2 = analysis2.get('lexical_diversity', 0.5)
            scores.append(max(0.0, 1.0 - abs(ld1 - ld2)))
            
            # Formality match
            scores.append(1.0 if analysis1.get('formality_level') == analysis2.get('formality_level') else 0.5)
            
            return round(sum(scores) / len(scores), 4) if scores else 0.5
        except Exception:
            return 0.5
    
    def _generate_style_recommendations(self, differences: Dict) -> List[str]:
        """Generate recommendations based on style differences."""
        return ["Consider adjusting sentence length for better consistency"]