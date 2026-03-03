"""
Analysis prompt templates for Style Transfer AI.
Contains the enhanced 25-point deep stylometry analysis framework.
"""


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