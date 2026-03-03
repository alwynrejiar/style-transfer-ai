"""
Content templates and prompt management for Style Transfer AI generation.
Provides structured templates for different content types and generation scenarios.
"""

from typing import Dict, List


class GenerationTemplates:
    """
    Manages content type templates and generation prompts.
    Provides structured guidance for different types of content generation.
    """
    
    def __init__(self):
        self.content_templates = self._initialize_content_templates()
        self.style_transfer_templates = self._initialize_style_transfer_templates()
    
    def get_content_template(self, content_type: str) -> str:
        """Get the template/requirements for a specific content type."""
        return self.content_templates.get(content_type, self.content_templates['general'])
    
    def get_style_transfer_template(self, transfer_type: str) -> str:
        """Get the template for style transfer operations."""
        return self.style_transfer_templates.get(transfer_type, self.style_transfer_templates['general'])
    
    def _initialize_content_templates(self) -> Dict[str, str]:
        """Initialize templates for different content types."""
        
        return {
            'email': """
EMAIL STRUCTURE REQUIREMENTS:
- Professional or personal tone as appropriate
- Clear subject matter and purpose
- Proper greeting and closing
- Organized paragraphs with logical flow
- Call to action or next steps (if applicable)
- Appropriate level of formality for the relationship
            """,
            
            'article': """
ARTICLE STRUCTURE REQUIREMENTS:
- Engaging headline/title (if requested)
- Strong opening paragraph that hooks the reader
- Clear thesis or main argument
- Well-organized body paragraphs with supporting evidence
- Smooth transitions between ideas
- Compelling conclusion that reinforces the main points
- Informative and authoritative tone
            """,
            
            'story': """
STORY STRUCTURE REQUIREMENTS:
- Compelling narrative arc with beginning, middle, end
- Well-developed characters (if applicable)
- Vivid descriptions and scene-setting
- Dialogue that feels natural (if applicable)
- Consistent point of view
- Engaging plot progression
- Satisfying resolution
            """,
            
            'essay': """
ESSAY STRUCTURE REQUIREMENTS:
- Clear thesis statement
- Introduction that previews main arguments
- Body paragraphs with topic sentences and supporting details
- Evidence and examples to support claims
- Logical organization and flow
- Conclusion that synthesizes and reinforces the thesis
- Academic or formal tone as appropriate
            """,
            
            'letter': """
LETTER STRUCTURE REQUIREMENTS:
- Appropriate greeting for the relationship level
- Clear purpose stated early
- Personal and engaging tone
- Organized thoughts and smooth flow
- Proper closing that matches the relationship
- Authentic voice that reflects the writer's personality
            """,
            
            'review': """
REVIEW STRUCTURE REQUIREMENTS:
- Clear identification of what's being reviewed
- Balanced evaluation of strengths and weaknesses
- Specific examples and evidence
- Comparison to relevant standards or alternatives
- Overall rating or recommendation
- Helpful and informative tone
- Fair and objective perspective
            """,
            
            'blog_post': """
BLOG POST STRUCTURE REQUIREMENTS:
- Catchy, SEO-friendly title (if requested)
- Engaging opening that hooks readers
- Scannable format with subheadings or bullet points
- Personal insights and opinions
- Conversational and accessible tone
- Call to action or engagement prompt
- Value-driven content for the target audience
            """,
            
            'social_media': """
SOCIAL MEDIA STRUCTURE REQUIREMENTS:
- Concise and impactful messaging
- Platform-appropriate length and format
- Engaging and shareable content
- Clear call to action or engagement hook
- Relevant hashtags or mentions (if applicable)
- Authentic voice that encourages interaction
- Visual or multimedia elements described (if applicable)
            """,
            
            'academic': """
ACADEMIC STRUCTURE REQUIREMENTS:
- Formal academic tone and language
- Clear research question or hypothesis
- Literature review or background context
- Methodology explanation (if applicable)
- Evidence-based arguments with citations
- Objective and analytical perspective
- Conclusions supported by evidence
- Proper academic formatting conventions
            """,
            
            'creative': """
CREATIVE STRUCTURE REQUIREMENTS:
- Innovative and original approach
- Rich descriptive language and imagery
- Emotional resonance and impact
- Unique voice and perspective
- Experimental or unconventional elements (if appropriate)
- Artistic expression that serves the content's purpose
- Engaging and memorable presentation
            """,
            
            'general': """
GENERAL CONTENT REQUIREMENTS:
- Clear purpose and audience awareness
- Logical organization and structure
- Appropriate tone for the context
- Engaging and readable style
- Proper grammar and language usage
- Coherent flow of ideas
- Meaningful and valuable content
            """
        }
    
    def _initialize_style_transfer_templates(self) -> Dict[str, str]:
        """Initialize templates for style transfer operations."""
        
        return {
            'formal_to_casual': """
FORMAL TO CASUAL STYLE TRANSFER:
- Replace formal vocabulary with conversational alternatives
- Shorten and simplify sentence structures
- Add contractions and informal expressions
- Include personal pronouns and direct address
- Use more active voice and dynamic language
- Maintain the core message while making it approachable
            """,
            
            'casual_to_formal': """
CASUAL TO FORMAL STYLE TRANSFER:
- Expand contractions and use complete forms
- Replace colloquialisms with standard language
- Increase sentence complexity and sophistication
- Use third person perspective where appropriate
- Add transitional phrases and formal connectors
- Maintain professional tone throughout
            """,
            
            'technical_to_accessible': """
TECHNICAL TO ACCESSIBLE STYLE TRANSFER:
- Simplify jargon and technical terminology
- Add explanations for complex concepts
- Use analogies and everyday examples
- Break down complex processes into steps
- Maintain accuracy while improving readability
- Include context for specialized knowledge
            """,
            
            'academic_to_popular': """
ACADEMIC TO POPULAR STYLE TRANSFER:
- Simplify academic language and terminology
- Add engaging hooks and interesting examples
- Use shorter paragraphs and sentences
- Include storytelling elements where appropriate
- Maintain scholarly accuracy while improving accessibility
- Add practical applications and relevance
            """,
            
            'neutral_to_persuasive': """
NEUTRAL TO PERSUASIVE STYLE TRANSFER:
- Add compelling arguments and evidence
- Include emotional appeals and storytelling
- Use stronger, more decisive language
- Add calls to action and urgency
- Incorporate social proof and credibility markers
- Maintain ethical persuasion techniques
            """,
            
            'general': """
GENERAL STYLE TRANSFER:
- Preserve the core meaning and information
- Adapt vocabulary and sentence structure to target style
- Maintain logical organization while adjusting presentation
- Ensure consistency in the new style throughout
- Keep the content accurate and truthful
- Adapt tone and formality to match target style
            """
        }
    
    def get_available_content_types(self) -> List[str]:
        """Get list of all available content types."""
        return list(self.content_templates.keys())
    
    def get_available_transfer_types(self) -> List[str]:
        """Get list of all available style transfer types."""
        return list(self.style_transfer_templates.keys())
    
    def create_custom_template(self, template_name: str, template_content: str, template_type: str = 'content'):
        """Add a custom template for content generation or style transfer."""
        if template_type == 'content':
            self.content_templates[template_name] = template_content
        elif template_type == 'transfer':
            self.style_transfer_templates[template_name] = template_content
        else:
            raise ValueError("template_type must be 'content' or 'transfer'")
    
    def get_generation_prompt_template(self) -> str:
        """Get the base template for content generation prompts."""
        return """
You are an expert content writer tasked with generating content that matches a specific writing style profile.

WRITING STYLE PROFILE:
{style_characteristics}

CONTENT REQUIREMENTS:
{content_template}

GENERATION PARAMETERS:
- Content Type: {content_type}
- Topic/Prompt: {topic_prompt}
- Target Length: {target_length} words
- Tone: {target_tone}
- Audience: {target_audience}

INSTRUCTIONS:
1. Study the writing style profile carefully
2. Generate content that authentically reflects this style
3. Follow the content type requirements
4. Maintain consistency throughout the piece
5. Ensure the content is engaging and well-structured

Generate the content now:
"""
    
    def get_style_transfer_prompt_template(self) -> str:
        """Get the base template for style transfer prompts."""
        return """
You are an expert editor tasked with rewriting content to match a specific writing style.

ORIGINAL CONTENT:
{original_content}

TARGET WRITING STYLE:
{target_style_characteristics}

TRANSFER REQUIREMENTS:
{transfer_template}

INSTRUCTIONS:
1. Preserve the core meaning and information from the original
2. Adapt the writing style to match the target profile
3. Ensure smooth flow and natural language
4. Maintain factual accuracy while changing presentation
5. Apply the style consistently throughout

Rewrite the content in the target style:
"""