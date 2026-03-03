"""
Quality control and validation system for generated and transferred content.
Ensures output meets quality standards and user requirements.
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from ..utils.text_processing import extract_basic_stats
from ..config.settings import TIMESTAMP_FORMAT


class QualityController:
    """
    Comprehensive quality control system for content generation and style transfer.
    
    Features:
    - Content quality validation
    - Style consistency checking
    - Readability assessment
    - Grammar and structure validation
    - User requirement compliance
    - Quality scoring and recommendations
    """
    
    def __init__(self):
        self.quality_thresholds = {
            'min_readability_score': 30,
            'max_readability_score': 90,
            'min_lexical_diversity': 0.3,
            'max_lexical_diversity': 0.9,
            'min_sentence_variety': 0.4,
            'max_paragraph_length': 200,
            'min_paragraph_length': 20,
            'coherence_threshold': 0.7,
            'style_consistency_threshold': 0.75
        }
        
        self.quality_categories = [
            'content_accuracy', 'readability', 'style_consistency',
            'grammar_structure', 'coherence', 'requirement_compliance'
        ]
    
    def validate_content_quality(
        self,
        content: str,
        requirements: Dict = None,
        target_style_profile: Dict = None,
        content_type: str = "general"
    ) -> Dict:
        """
        Comprehensive quality validation of generated or transferred content.
        
        Args:
            content (str): Content to validate
            requirements (Dict): User-specified quality requirements
            target_style_profile (Dict): Expected style profile for comparison
            content_type (str): Type of content (email, article, story, etc.)
            
        Returns:
            Dict: Comprehensive quality assessment with scores and recommendations
        """
        try:
            # Basic content validation
            if not content or not content.strip():
                return self._create_failed_validation("Empty or invalid content")
            
            # Initialize quality assessment
            quality_assessment = {
                'overall_score': 0.0,
                'category_scores': {},
                'quality_issues': [],
                'recommendations': [],
                'compliance_status': {},
                'metadata': {
                    'content_type': content_type,
                    'validation_timestamp': datetime.now().strftime(TIMESTAMP_FORMAT),
                    'content_length': len(content),
                    'word_count': len(content.split())
                }
            }
            
            # Perform quality checks
            quality_assessment['category_scores']['content_accuracy'] = self._validate_content_accuracy(content)
            quality_assessment['category_scores']['readability'] = self._validate_readability(content)
            quality_assessment['category_scores']['style_consistency'] = self._validate_style_consistency(
                content, target_style_profile
            )
            quality_assessment['category_scores']['grammar_structure'] = self._validate_grammar_structure(content)
            quality_assessment['category_scores']['coherence'] = self._validate_coherence(content)
            quality_assessment['category_scores']['requirement_compliance'] = self._validate_requirement_compliance(
                content, requirements, content_type
            )
            
            # Calculate overall score
            category_scores = quality_assessment['category_scores']
            quality_assessment['overall_score'] = sum(category_scores.values()) / len(category_scores)
            
            # Generate recommendations
            quality_assessment['recommendations'] = self._generate_quality_recommendations(
                content, category_scores, requirements
            )
            
            # Check compliance with requirements
            quality_assessment['compliance_status'] = self._check_requirement_compliance(
                content, requirements, category_scores
            )
            
            # Identify quality issues
            quality_assessment['quality_issues'] = self._identify_quality_issues(
                content, category_scores
            )
            
            return quality_assessment
            
        except Exception as e:
            return self._create_failed_validation(f"Quality validation error: {str(e)}")
    
    def improve_content_quality(
        self,
        content: str,
        quality_assessment: Dict,
        improvement_focus: List[str] = None
    ) -> Dict:
        """
        Generate specific improvements to enhance content quality.
        
        Args:
            content (str): Original content to improve
            quality_assessment (Dict): Quality assessment from validate_content_quality
            improvement_focus (List[str]): Specific areas to focus on for improvement
            
        Returns:
            Dict: Improvement suggestions and enhanced content recommendations
        """
        try:
            improvements = {
                'improvement_plan': [],
                'specific_fixes': [],
                'rewrite_suggestions': [],
                'enhancement_priorities': [],
                'estimated_impact': {}
            }
            
            # Determine improvement priorities
            category_scores = quality_assessment.get('category_scores', {})
            improvement_priorities = self._determine_improvement_priorities(
                category_scores, improvement_focus
            )
            
            improvements['enhancement_priorities'] = improvement_priorities
            
            # Generate specific improvements for each priority area
            for priority_area in improvement_priorities:
                if priority_area == 'readability':
                    improvements['specific_fixes'].extend(
                        self._generate_readability_improvements(content)
                    )
                elif priority_area == 'style_consistency':
                    improvements['specific_fixes'].extend(
                        self._generate_style_consistency_improvements(content)
                    )
                elif priority_area == 'grammar_structure':
                    improvements['specific_fixes'].extend(
                        self._generate_grammar_improvements(content)
                    )
                elif priority_area == 'coherence':
                    improvements['specific_fixes'].extend(
                        self._generate_coherence_improvements(content)
                    )
            
            # Create improvement plan
            improvements['improvement_plan'] = self._create_improvement_plan(
                improvements['specific_fixes']
            )
            
            # Estimate improvement impact
            improvements['estimated_impact'] = self._estimate_improvement_impact(
                category_scores, improvements['specific_fixes']
            )
            
            return improvements
            
        except Exception as e:
            return {
                'error': f"Quality improvement generation failed: {str(e)}",
                'improvement_plan': [],
                'specific_fixes': [],
                'timestamp': datetime.now().strftime(TIMESTAMP_FORMAT)
            }
    
    def compare_versions(
        self,
        original_content: str,
        improved_content: str,
        requirements: Dict = None
    ) -> Dict:
        """
        Compare quality metrics between original and improved content versions.
        
        Args:
            original_content (str): Original content version
            improved_content (str): Improved content version
            requirements (Dict): Quality requirements for comparison
            
        Returns:
            Dict: Detailed comparison analysis
        """
        try:
            # Validate both versions
            original_quality = self.validate_content_quality(original_content, requirements)
            improved_quality = self.validate_content_quality(improved_content, requirements)
            
            # Calculate improvements
            score_improvements = {}
            for category in self.quality_categories:
                original_score = original_quality['category_scores'].get(category, 0)
                improved_score = improved_quality['category_scores'].get(category, 0)
                score_improvements[category] = improved_score - original_score
            
            comparison = {
                'original_quality': original_quality,
                'improved_quality': improved_quality,
                'score_improvements': score_improvements,
                'overall_improvement': improved_quality['overall_score'] - original_quality['overall_score'],
                'improvement_summary': self._summarize_improvements(score_improvements),
                'recommendation_effectiveness': self._evaluate_recommendation_effectiveness(
                    original_quality, improved_quality
                ),
                'timestamp': datetime.now().strftime(TIMESTAMP_FORMAT)
            }
            
            return comparison
            
        except Exception as e:
            return {
                'error': f"Version comparison failed: {str(e)}",
                'timestamp': datetime.now().strftime(TIMESTAMP_FORMAT)
            }
    
    def _validate_content_accuracy(self, content: str) -> float:
        """Validate content accuracy and factual consistency."""
        
        # Basic content validation checks
        score = 1.0
        
        # Check for obvious inconsistencies
        if self._has_contradictions(content):
            score -= 0.3
        
        # Check for completeness
        if self._is_incomplete_content(content):
            score -= 0.2
        
        # Check for factual plausibility
        if self._has_implausible_claims(content):
            score -= 0.2
        
        return max(0.0, score)
    
    def _validate_readability(self, content: str) -> float:
        """Validate readability metrics."""
        
        stats = extract_basic_stats(content)
        
        # Calculate readability score (simplified Flesch-like calculation)
        avg_sentence_length = stats.get('avg_sentence_length', 15)
        avg_word_length = self._calculate_avg_word_length(content)
        
        # Optimal ranges
        optimal_sentence_length = 15
        optimal_word_length = 5
        
        # Calculate score based on deviation from optimal
        sentence_score = max(0, 1 - abs(avg_sentence_length - optimal_sentence_length) / 20)
        word_score = max(0, 1 - abs(avg_word_length - optimal_word_length) / 5)
        
        return (sentence_score + word_score) / 2
    
    def _validate_style_consistency(self, content: str, target_style_profile: Dict = None) -> float:
        """Validate style consistency throughout the content."""
        
        if not target_style_profile:
            # Basic consistency check without target profile
            return self._check_internal_style_consistency(content)
        
        # Compare against target style profile
        return self._compare_to_target_style(content, target_style_profile)
    
    def _validate_grammar_structure(self, content: str) -> float:
        """Validate grammar and sentence structure."""
        
        # Basic grammar validation
        score = 1.0
        
        # Check for common grammar issues
        if self._has_grammar_errors(content):
            score -= 0.3
        
        # Check sentence structure variety
        structure_variety = self._calculate_structure_variety(content)
        if structure_variety < self.quality_thresholds['min_sentence_variety']:
            score -= 0.2
        
        # Check punctuation usage
        if self._has_punctuation_issues(content):
            score -= 0.1
        
        return max(0.0, score)
    
    def _validate_coherence(self, content: str) -> float:
        """Validate logical flow and coherence."""
        
        # Basic coherence checks
        coherence_score = 0.8  # Start with good assumption
        
        # Check paragraph transitions
        if not self._has_good_transitions(content):
            coherence_score -= 0.2
        
        # Check logical flow
        if not self._has_logical_flow(content):
            coherence_score -= 0.3
        
        # Check topic consistency
        if not self._maintains_topic_focus(content):
            coherence_score -= 0.2
        
        return max(0.0, coherence_score)
    
    def _validate_requirement_compliance(
        self,
        content: str,
        requirements: Dict = None,
        content_type: str = "general"
    ) -> float:
        """Validate compliance with user requirements."""
        
        if not requirements:
            return 0.8  # Neutral score if no requirements specified
        
        compliance_score = 1.0
        
        # Check length requirements
        if 'target_length' in requirements:
            target_length = requirements['target_length']
            actual_length = len(content.split())
            length_diff = abs(actual_length - target_length) / target_length
            if length_diff > 0.2:  # More than 20% difference
                compliance_score -= 0.3
        
        # Check tone requirements
        if 'required_tone' in requirements:
            if not self._matches_required_tone(content, requirements['required_tone']):
                compliance_score -= 0.3
        
        # Check format requirements
        if 'format_requirements' in requirements:
            if not self._meets_format_requirements(content, requirements['format_requirements']):
                compliance_score -= 0.2
        
        # Check content type specific requirements
        if not self._meets_content_type_requirements(content, content_type):
            compliance_score -= 0.2
        
        return max(0.0, compliance_score)
    
    def _generate_quality_recommendations(
        self,
        content: str,
        category_scores: Dict,
        requirements: Dict = None
    ) -> List[str]:
        """Generate specific recommendations for quality improvement."""
        
        recommendations = []
        
        # Readability recommendations
        if category_scores.get('readability', 1.0) < 0.7:
            recommendations.extend([
                "Consider shortening long sentences for better readability",
                "Use simpler vocabulary where appropriate",
                "Break up long paragraphs for easier reading"
            ])
        
        # Style consistency recommendations
        if category_scores.get('style_consistency', 1.0) < 0.7:
            recommendations.extend([
                "Maintain consistent tone throughout the content",
                "Use uniform vocabulary style and formality level",
                "Ensure consistent sentence structure patterns"
            ])
        
        # Grammar recommendations
        if category_scores.get('grammar_structure', 1.0) < 0.7:
            recommendations.extend([
                "Review sentence structure for grammatical correctness",
                "Add more sentence variety to improve flow",
                "Check punctuation usage for clarity"
            ])
        
        # Coherence recommendations
        if category_scores.get('coherence', 1.0) < 0.7:
            recommendations.extend([
                "Improve transitions between paragraphs",
                "Ensure logical progression of ideas",
                "Maintain consistent focus on main topic"
            ])
        
        return recommendations
    
    def _check_requirement_compliance(
        self,
        content: str,
        requirements: Dict = None,
        category_scores: Dict = None
    ) -> Dict:
        """Check compliance with specific requirements."""
        
        compliance_status = {
            'length_compliance': True,
            'tone_compliance': True,
            'format_compliance': True,
            'quality_compliance': True,
            'overall_compliance': True
        }
        
        if requirements:
            # Length compliance
            if 'target_length' in requirements:
                target = requirements['target_length']
                actual = len(content.split())
                compliance_status['length_compliance'] = abs(actual - target) / target <= 0.2
            
            # Tone compliance
            if 'required_tone' in requirements:
                compliance_status['tone_compliance'] = self._matches_required_tone(
                    content, requirements['required_tone']
                )
        
        # Quality threshold compliance
        if category_scores:
            overall_score = sum(category_scores.values()) / len(category_scores)
            compliance_status['quality_compliance'] = overall_score >= 0.7
        
        # Overall compliance
        compliance_status['overall_compliance'] = all(compliance_status.values())
        
        return compliance_status
    
    def _identify_quality_issues(self, content: str, category_scores: Dict) -> List[Dict]:
        """Identify specific quality issues in the content."""
        
        issues = []
        
        # Low readability issues
        if category_scores.get('readability', 1.0) < 0.6:
            issues.append({
                'category': 'readability',
                'severity': 'high',
                'description': 'Content readability is below acceptable standards',
                'suggested_fix': 'Simplify sentence structure and vocabulary'
            })
        
        # Style inconsistency issues
        if category_scores.get('style_consistency', 1.0) < 0.6:
            issues.append({
                'category': 'style_consistency',
                'severity': 'medium',
                'description': 'Inconsistent writing style detected',
                'suggested_fix': 'Maintain consistent tone and vocabulary throughout'
            })
        
        # Grammar issues
        if category_scores.get('grammar_structure', 1.0) < 0.6:
            issues.append({
                'category': 'grammar_structure',
                'severity': 'high',
                'description': 'Grammar and structure issues detected',
                'suggested_fix': 'Review and correct grammatical errors'
            })
        
        return issues
    
    def _create_failed_validation(self, error_message: str) -> Dict:
        """Create a failed validation response."""
        return {
            'overall_score': 0.0,
            'category_scores': {category: 0.0 for category in self.quality_categories},
            'quality_issues': [{'category': 'validation_error', 'description': error_message}],
            'recommendations': ['Fix validation errors before proceeding'],
            'compliance_status': {'overall_compliance': False},
            'metadata': {
                'validation_timestamp': datetime.now().strftime(TIMESTAMP_FORMAT),
                'validation_failed': True,
                'error_message': error_message
            }
        }
    
    # Utility methods for quality assessment (simplified implementations)
    
    def _has_contradictions(self, content: str) -> bool:
        """Check for obvious contradictions in content."""
        # Simplified contradiction detection
        return False
    
    def _is_incomplete_content(self, content: str) -> bool:
        """Check if content appears incomplete."""
        # Check for abrupt endings, missing conclusions
        return content.strip().endswith(('...', 'etc.', 'and so on'))
    
    def _has_implausible_claims(self, content: str) -> bool:
        """Check for implausible factual claims."""
        # Simplified plausibility check
        return False
    
    def _calculate_avg_word_length(self, content: str) -> float:
        """Calculate average word length."""
        words = content.split()
        if not words:
            return 0
        return sum(len(word.strip('.,!?";:')) for word in words) / len(words)
    
    def _check_internal_style_consistency(self, content: str) -> float:
        """Check internal style consistency without target profile."""
        # Simplified consistency check
        return 0.8
    
    def _compare_to_target_style(self, content: str, target_style_profile: Dict) -> float:
        """Compare content style to target profile."""
        # Simplified style comparison
        return 0.75
    
    def _has_grammar_errors(self, content: str) -> bool:
        """Basic grammar error detection."""
        # Check for common patterns that might indicate errors
        double_spaces = '  ' in content
        missing_capitals = re.search(r'\. [a-z]', content) is not None
        return double_spaces or missing_capitals
    
    def _calculate_structure_variety(self, content: str) -> float:
        """Calculate sentence structure variety."""
        sentences = re.split(r'[.!?]+', content)
        if len(sentences) < 2:
            return 0.5
        
        # Simple variety measure based on sentence length variation
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if not lengths:
            return 0.5
        
        avg_length = sum(lengths) / len(lengths)
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        return min(1.0, variance / 50)  # Normalize variance
    
    def _has_punctuation_issues(self, content: str) -> bool:
        """Check for punctuation issues."""
        # Basic punctuation checks
        return content.count('..') > 2 or content.count('!!') > 0
    
    def _has_good_transitions(self, content: str) -> bool:
        """Check for good paragraph transitions."""
        paragraphs = content.split('\n\n')
        if len(paragraphs) < 2:
            return True
        
        # Check for transition words/phrases
        transition_words = ['however', 'therefore', 'furthermore', 'additionally', 'moreover', 'nevertheless']
        transitions_found = sum(1 for p in paragraphs for word in transition_words if word in p.lower())
        return transitions_found >= len(paragraphs) * 0.3
    
    def _has_logical_flow(self, content: str) -> bool:
        """Check for logical flow in content."""
        # Simplified logical flow check
        return True
    
    def _maintains_topic_focus(self, content: str) -> bool:
        """Check if content maintains topic focus."""
        # Simplified topic focus check
        return True
    
    def _matches_required_tone(self, content: str, required_tone: str) -> bool:
        """Check if content matches required tone."""
        # Simplified tone matching
        return True
    
    def _meets_format_requirements(self, content: str, format_requirements: Dict) -> bool:
        """Check if content meets format requirements."""
        # Simplified format checking
        return True
    
    def _meets_content_type_requirements(self, content: str, content_type: str) -> bool:
        """Check if content meets content type requirements."""
        # Simplified content type checking
        return True
    
    def _determine_improvement_priorities(
        self,
        category_scores: Dict,
        improvement_focus: List[str] = None
    ) -> List[str]:
        """Determine priority areas for improvement."""
        
        if improvement_focus:
            return improvement_focus
        
        # Sort categories by lowest scores
        sorted_categories = sorted(
            category_scores.items(),
            key=lambda x: x[1]
        )
        
        # Return categories with scores below 0.7
        return [category for category, score in sorted_categories if score < 0.7]
    
    def _generate_readability_improvements(self, content: str) -> List[Dict]:
        """Generate specific readability improvements."""
        return [
            {
                'type': 'sentence_simplification',
                'description': 'Break down long sentences',
                'impact': 'medium'
            },
            {
                'type': 'vocabulary_simplification',
                'description': 'Replace complex words with simpler alternatives',
                'impact': 'medium'
            }
        ]
    
    def _generate_style_consistency_improvements(self, content: str) -> List[Dict]:
        """Generate style consistency improvements."""
        return [
            {
                'type': 'tone_standardization',
                'description': 'Standardize tone throughout content',
                'impact': 'high'
            }
        ]
    
    def _generate_grammar_improvements(self, content: str) -> List[Dict]:
        """Generate grammar improvements."""
        return [
            {
                'type': 'grammar_correction',
                'description': 'Fix grammatical errors',
                'impact': 'high'
            }
        ]
    
    def _generate_coherence_improvements(self, content: str) -> List[Dict]:
        """Generate coherence improvements."""
        return [
            {
                'type': 'transition_improvement',
                'description': 'Add better paragraph transitions',
                'impact': 'medium'
            }
        ]
    
    def _create_improvement_plan(self, specific_fixes: List[Dict]) -> List[str]:
        """Create a prioritized improvement plan."""
        high_impact = [fix['description'] for fix in specific_fixes if fix.get('impact') == 'high']
        medium_impact = [fix['description'] for fix in specific_fixes if fix.get('impact') == 'medium']
        
        plan = []
        plan.extend(high_impact)
        plan.extend(medium_impact)
        
        return plan[:5]  # Limit to top 5 improvements
    
    def _estimate_improvement_impact(
        self,
        category_scores: Dict,
        specific_fixes: List[Dict]
    ) -> Dict:
        """Estimate the impact of proposed improvements."""
        
        impact_estimates = {}
        
        for category in self.quality_categories:
            current_score = category_scores.get(category, 0.5)
            relevant_fixes = [f for f in specific_fixes if category in f.get('type', '')]
            
            if relevant_fixes:
                estimated_improvement = len(relevant_fixes) * 0.1
                impact_estimates[category] = min(1.0, current_score + estimated_improvement)
            else:
                impact_estimates[category] = current_score
        
        return impact_estimates
    
    def _summarize_improvements(self, score_improvements: Dict) -> str:
        """Summarize the improvements made."""
        
        improved_categories = [cat for cat, improvement in score_improvements.items() if improvement > 0.1]
        
        if not improved_categories:
            return "No significant improvements detected"
        
        return f"Significant improvements in: {', '.join(improved_categories)}"
    
    def _evaluate_recommendation_effectiveness(
        self,
        original_quality: Dict,
        improved_quality: Dict
    ) -> Dict:
        """Evaluate the effectiveness of quality recommendations."""
        
        return {
            'recommendations_followed': 0.8,  # Placeholder
            'improvement_achieved': improved_quality['overall_score'] - original_quality['overall_score'],
            'effectiveness_rating': 'good'  # Placeholder
        }