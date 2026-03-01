"""
Analysis prompt templates for Style Transfer AI — v4.0
Contains the enhanced 25-point deep stylometry analysis framework
PLUS a new Part 8: Human Pattern & Anti-AI Fingerprint section.

New in v4.0:
  - Part 8 added to create_enhanced_deep_prompt() for human pattern extraction
  - create_humanization_injection_prompt() — standalone prompt for rewrite humanization
  - create_anti_ai_audit_prompt() — audits rewritten text for AI detection risk
  - HUMANIZATION_SYSTEM_RULES — constant block injectable into any rewrite system prompt
"""


# ============================================================================
# CONSTANT: Anti-AI Injection Rules Block
# This can be appended verbatim to any rewrite system prompt.
# ============================================================================

HUMANIZATION_SYSTEM_RULES = """
══════════════════════════════════════════════════════
HUMANIZATION RULES — ANTI-AI-DETECTION REQUIREMENTS
══════════════════════════════════════════════════════

SENTENCE LENGTH (most critical):
- You MUST vary sentence lengths radically. Target distribution:
    • 15-25% of sentences should be 1-6 words (micro/short)
    • 30-40% should be 7-14 words (short-medium)
    • 25-35% should be 15-25 words (medium-long)
    • 10-20% should be 26+ words (long)
- Never write more than 2 consecutive sentences of similar length (within 4 words).
- After every 2-3 long sentences, write a short punchy sentence. No exceptions.

SENTENCE STRUCTURE VARIETY:
- Start sentences with: conjunctions (And, But, So, Because), adverbs, gerunds,
  prepositions, fragments. Not just Subject-Verb-Object every time.
- Include at least 1 rhetorical question or self-dialogue pattern per 200 words.
- Include at least 1 sentence fragment (deliberate incomplete sentence) per 300 words.
- Use em-dashes for asides and interruptions — like this — at least once per 150 words.

PARAGRAPH STRUCTURE:
- Vary paragraph length dramatically. At least one single-sentence paragraph.
- At least one paragraph should be noticeably longer than the others.
- Avoid ending paragraphs with summary sentences. End mid-thought occasionally.

FORBIDDEN AI PHRASES (never use these under any circumstances):
- "furthermore", "moreover", "additionally", "consequently"
- "in conclusion", "to summarize", "in summary", "to sum up"
- "it is worth noting", "it is important to note", "it should be noted"
- "delve into", "dive into", "shed light on", "unpack"
- "testament to", "groundbreaking", "revolutionary", "paradigm shift"
- "holistic approach", "leverage", "synergy", "robust"
- "in today's world", "in today's society", "the realm of", "the landscape of"
- "this highlights", "this demonstrates", "this underscores", "as we can see"
- "let's explore", "let's dive", "let me walk you through"

VOCABULARY:
- Mix formal and informal vocabulary unexpectedly.
- Use contractions freely (it's, don't, can't, that's).
- Include occasional colloquialisms and casual phrases.
- Repeat a key word deliberately for emphasis within a paragraph.
- Avoid perfectly parallel list structures — break the pattern occasionally.

PERSONALITY MARKERS:
- Include at least one parenthetical aside per 200 words.
- Use self-correction or qualification at least once (e.g. "or rather", "well,",
  "not exactly — more like").
- Address the reader directly at least once per 400 words.

══════════════════════════════════════════════════════
"""


def create_enhanced_deep_prompt(text_to_analyze, user_profile=None):
    """
    Create the enhanced 25-point deep stylometry analysis prompt
    with user context AND a new Part 8 for human pattern extraction.
    """

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
Perform an ENHANCED DEEP stylometry analysis of the following text for creating a
comprehensive writing style profile AND a humanization blueprint to help rewritten
content avoid AI detection.

Provide specific, quantifiable insights with exact numbers, percentages, and examples.
{user_context}

**PART 1: LINGUISTIC ARCHITECTURE**
1. Sentence Structure Mastery: Calculate exact average sentence length, identify
   complex/compound/simple ratios with percentages, analyze syntactic patterns.
2. Clause Choreography: Measure subordinate clause frequency, coordination vs
   subordination ratios, dependent clause patterns.
3. Punctuation Symphony: Count and categorize ALL punctuation — commas, semicolons,
   dashes, parentheses with specific frequencies per 100 words.
4. Syntactic Sophistication: Identify sentence variety index, grammatical complexity
   scoring, parsing preferences.

**PART 2: LEXICAL INTELLIGENCE**
5. Vocabulary Sophistication: Analyze word complexity levels, formal vs informal
   ratios, academic vocabulary percentage.
6. Semantic Field Preferences: Categorize word choices by domain
   (abstract/concrete, emotional/logical, technical/general).
7. Lexical Diversity Metrics: Calculate type-token ratio, vocabulary richness index,
   word repetition patterns.
8. Register Flexibility: Measure formality spectrum, colloquialisms vs standard
   usage, domain-specific terminology.

**PART 3: STYLISTIC DNA**
9. Tone Architecture: Identify confidence indicators, emotional markers,
   certainty/uncertainty expressions with examples.
10. Voice Consistency: Analyze person preference (1st/2nd/3rd percentages),
    active vs passive voice ratios.
11. Rhetorical Weaponry: Count metaphors, similes, rhetorical questions, parallel
    structures, repetition patterns.
12. Narrative Technique: Point of view consistency, perspective shifts,
    storytelling vs explanatory modes.

**PART 4: COGNITIVE PATTERNS**
13. Logical Flow Design: Analyze argument structure, cause-effect patterns,
    sequential vs thematic organization.
14. Transition Mastery: Count and categorize transition words, coherence
    mechanisms, paragraph linking strategies.
15. Emphasis Engineering: Identify how key points are highlighted — repetition,
    positioning, linguistic intensity.
16. Information Density: Measure concept-to-word ratios, information packaging
    efficiency, elaboration patterns.

**PART 5: PSYCHOLOGICAL MARKERS**
17. Cognitive Processing Style: Analyze linear vs circular thinking, analytical vs
    intuitive patterns, detail vs big-picture focus.
18. Emotional Intelligence: Identify empathy markers, emotional vocabulary
    richness, interpersonal awareness.
19. Authority Positioning: Measure hedging language, assertiveness markers,
    expertise indicators.
20. Risk Tolerance: Analyze certainty language, qualification usage,
    experimental vs conservative expressions.

**PART 6: STRUCTURAL GENIUS**
21. Paragraph Architecture: Calculate paragraph length variance, topic development
    patterns, structural rhythm.
22. Coherence Engineering: Measure text cohesion, referential chains, thematic
    progression strategies.
23. Temporal Dynamics: Analyze tense usage patterns, time reference preferences,
    narrative temporality.
24. Modal Expression: Count modal verbs, probability expressions,
    obligation vs possibility language.

**PART 7: UNIQUE FINGERPRINT**
25. Personal Signature Elements: Identify unique phrases, idiosyncratic expressions,
    personal linguistic habits that would be immediately lost in AI paraphrase.

**PART 8: HUMAN PATTERN & ANTI-AI FINGERPRINT [NEW]**
This section is critical for producing content that passes AI detection.
Analyze the following with precise, actionable detail:

26. Sentence Length Burstiness:
    - Map the EXACT distribution of sentence lengths (count sentences in each range:
      1-5w, 6-10w, 11-18w, 19-28w, 29w+).
    - Identify the specific rhythm pattern (e.g. "three long sentences then a punchy
      short one", "randomly irregular", "gradually building").
    - Calculate the coefficient of variation (std/mean). Humans ≈ 0.45-0.80. AI ≈ 0.20-0.35.
    - Quote at least 2 examples of the author's shortest sentences.
    - Quote at least 2 examples of the author's most complex sentences.

27. Imperfection Profile:
    - Does the author start sentences with conjunctions (And, But, So)? How often?
    - Does the author use comma splices? Give examples.
    - Are there sentence fragments? Give exact examples.
    - Does the author use run-on sentences deliberately? Examples?
    - Any unconventional or inconsistent punctuation patterns?

28. Personality Leak Patterns:
    - Asides: Does the author insert parenthetical thoughts mid-sentence?
      Examples: em-dash asides, bracketed comments, parentheses.
    - Self-correction: Does the author catch themselves and rephrase?
      Examples of mid-thought pivots.
    - Direct reader address: How does the author speak TO the reader?
      Examples of second-person direct address.
    - Emotional leakage: Where do emotions show through unprompted?

29. Vocabulary Mixing & Register Inconsistency:
    - Identify places where the author mixes formal and informal vocabulary
      unexpectedly (e.g. academic vocabulary followed by a casual phrase).
    - List specific idiosyncratic words/phrases this author uses that AI would
      never naturally generate.
    - What casual/colloquial expressions appear despite the overall register?

30. Structural Quirks:
    - One-sentence paragraphs: Does the author use them? How often? For what effect?
    - Abrupt topic pivots: Does the author jump topics without transition?
    - Callback structures: Does the author return to earlier points unexpectedly?
    - Anti-climax or subverted expectations?

31. Anti-AI Pattern Audit:
    - List any AI-like patterns ALREADY PRESENT in this text that a rewriter
      must eliminate (e.g. overused transition words, overly uniform sentences,
      stock phrases, overly balanced structure).
    - Rate the text's current humanness on a scale of 1-10 with justification.
    - Produce a "Humanization Repair List" — 5-7 specific changes a rewriter
      should make to this specific text to reduce AI detection risk.

32. Humanization Blueprint (the most important output of Part 8):
    Produce a precise, numbered list of 8-10 SPECIFIC RULES that a rewriter
    must follow to reproduce this author's voice in a way that will pass
    AI detection. Each rule must be:
    - Concrete (not vague — "vary sentence length" is too vague; 
      "insert a sentence of 3-7 words after every 2-3 sentences of 15+ words" is concrete)
    - Specific to THIS author (derived from the actual analysis above, not generic)
    - Actionable (something a model can directly implement)

PROVIDE YOUR ANALYSIS AS:
1. Quantitative metrics with exact numbers and percentages
2. Specific examples from the text for each point
3. Comparative assessments (high/medium/low with context)
4. Pattern recognition insights
5. Psychological and cognitive style indicators
6. Cultural/linguistic influence markers (based on writer background)
7. Human-pattern analysis with actionable humanization rules

Text to analyze:
{text_to_analyze}
"""


def create_humanization_injection_prompt(
    content_to_rewrite: str,
    style_profile: dict,
    humanization_rules: list[str] | None = None,
    forbidden_phrases: list[str] | None = None,
    target_length_ratio: float = 1.0,
) -> str:
    """
    Creates a complete injection prompt for rewriting content in an author's
    voice with explicit anti-AI-detection rules baked in.

    Args:
        content_to_rewrite: The text to be rewritten.
        style_profile: The full style profile dict from analyze_style().
        humanization_rules: Optional override list of humanization rules.
                            If None, extracted from style_profile.
        forbidden_phrases: Optional override list of forbidden phrases.
                           If None, extracted from style_profile + defaults.
        target_length_ratio: Target output length as ratio of input (default 1.0).

    Returns:
        str: Complete injection prompt ready for an LLM.
    """
    directive = style_profile.get("rewrite_directive", "")
    fingerprint = style_profile.get("style_fingerprint_summary", "")
    traits = style_profile.get("most_distinctive_traits", [])
    do_not_lose = style_profile.get("do_not_lose", [])
    avoid = style_profile.get("avoid_in_rewrite", [])

    # Human-pattern metrics from the profile
    hm = style_profile.get("human_pattern_metrics", {})
    burst_cv = hm.get("burstiness_cv", "unknown")
    burst_label = hm.get("burstiness_label", "unknown")
    ai_risk = hm.get("ai_risk_score", "unknown")
    sent_dist = hm.get("sent_length_distribution", {})

    # Humanization rules: use passed-in or fall back to profile
    rules = humanization_rules or style_profile.get("humanization_rules", [])
    forbidden = forbidden_phrases or style_profile.get("forbidden_ai_phrases", [])

    # Format sections
    traits_block = "\n".join(f"  • {t}" for t in traits) if traits else "  (not available)"
    dnl_block = "\n".join(f"  ✓ {d}" for d in do_not_lose) if do_not_lose else "  (not available)"
    avoid_block = "\n".join(f"  ✗ {a}" for a in avoid) if avoid else "  (not available)"
    rules_block = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(rules)) if rules else ""
    forbidden_block = "\n".join(f"  - \"{p}\"" for p in forbidden) if forbidden else ""

    # Sentence length target
    mean_len = sent_dist.get("mean", 0)
    std_len = sent_dist.get("std_dev", 0)
    length_target = (
        f"Target average sentence length: ~{mean_len} words (std dev ~{std_len}). "
        f"Author burstiness CV is {burst_cv} ({burst_label}) — replicate this variance exactly."
        if mean_len else "Vary sentence length significantly."
    )

    # Source AI risk warning
    risk_warning = ""
    if isinstance(ai_risk, float) and ai_risk > 0.40:
        risk_warning = (
            f"\n⚠ WARNING: The source text itself has AI risk score {ai_risk}. "
            f"The rewrite must be MORE human than the source. Pay extra attention "
            f"to the humanization rules below.\n"
        )

    prompt = f"""You are an expert style transfer writer. Your task is to rewrite the
provided content in the exact voice of a specific author while ensuring the output
will pass AI detection tools.
{risk_warning}
══════════════════════════════════════════════════════
AUTHOR VOICE PROFILE
══════════════════════════════════════════════════════

STYLE FINGERPRINT:
{fingerprint or "(not available)"}

MOST DISTINCTIVE TRAITS:
{traits_block}

REWRITE DIRECTIVE (follow this precisely):
{directive or "(not available — use the traits and rules below)"}

══════════════════════════════════════════════════════
SENTENCE RHYTHM TARGET
══════════════════════════════════════════════════════

{length_target}

Required sentence length distribution:
  - At least 15% of sentences must be 6 words or fewer
  - At least 15% of sentences must be 25 words or more
  - Never write 3+ consecutive sentences within 3 words of the same length

══════════════════════════════════════════════════════
MUST PRESERVE (do not lose these elements)
══════════════════════════════════════════════════════
{dnl_block}

══════════════════════════════════════════════════════
AVOID (these break the voice)
══════════════════════════════════════════════════════
{avoid_block}

══════════════════════════════════════════════════════
HUMANIZATION RULES (anti-AI-detection — NON-NEGOTIABLE)
══════════════════════════════════════════════════════
{rules_block if rules_block else HUMANIZATION_SYSTEM_RULES}

══════════════════════════════════════════════════════
FORBIDDEN PHRASES (never use these)
══════════════════════════════════════════════════════
{forbidden_block if forbidden_block else '''
  - "furthermore", "moreover", "additionally", "consequently"
  - "in conclusion", "to summarize", "it is worth noting"
  - "delve into", "shed light on", "testament to"
  - "in today's world", "the realm of", "this highlights"
  - "let's explore", "groundbreaking", "paradigm shift"
'''}

══════════════════════════════════════════════════════
OUTPUT REQUIREMENTS
══════════════════════════════════════════════════════

- Target length: approximately {target_length_ratio:.0%} of the original
- Preserve all factual content and key arguments
- Do NOT add explanatory headers or section labels unless the original has them
- Do NOT add a summary at the end
- Do NOT start with "Certainly", "Sure", "Of course", or any meta-commentary
- Output only the rewritten content — nothing else

══════════════════════════════════════════════════════
CONTENT TO REWRITE:
══════════════════════════════════════════════════════
{content_to_rewrite}
"""
    return prompt


def create_anti_ai_audit_prompt(rewritten_text: str, original_style_profile: dict | None = None) -> str:
    """
    Creates a prompt to audit rewritten text for AI detection risk.
    Designed to be sent to an LLM for a second-pass review before publishing.

    Returns a prompt that asks the LLM to:
      1. Identify remaining AI-like patterns
      2. Score the text on human-likeness (1-10)
      3. Produce specific sentence-level repair instructions
    """
    fingerprint = ""
    if original_style_profile:
        fingerprint = original_style_profile.get("style_fingerprint_summary", "")

    profile_context = ""
    if fingerprint:
        profile_context = f"""
The text was written to match this author's voice:
{fingerprint}

Compare the rewritten text against this profile — flag any places where the
voice feels generic or AI-generated rather than matching this specific author.
"""

    return f"""You are an AI-detection specialist and forensic writing analyst.
Audit the following text for AI-generation patterns and produce a repair report.
{profile_context}
══════════════════════════════════════════════════════
AUDIT CHECKLIST — evaluate each item:
══════════════════════════════════════════════════════

1. SENTENCE LENGTH VARIETY
   - Are there micro-sentences (1-6 words)? Count them.
   - Are there long complex sentences (25+ words)? Count them.
   - Do 3+ consecutive sentences have similar lengths? Flag which ones.
   - Rate variety: 1 (uniform AI) to 10 (highly varied human)

2. TRANSITION WORDS
   - List any forbidden AI transition phrases found:
     ("furthermore", "moreover", "in conclusion", "it is worth noting",
     "delve into", "this highlights", "let's explore", etc.)
   - Count total AI transitions per 100 words.

3. PARAGRAPH STRUCTURE
   - Are all paragraphs roughly the same length? Flag if yes.
   - Is there at least one single-sentence paragraph?
   - Do paragraphs all end with summary-style sentences?

4. PERSONALITY MARKERS
   - Are there parenthetical asides? Count them.
   - Are there rhetorical questions or self-dialogue?
   - Is there any direct reader address?
   - Does any authentic personality or emotion show through?

5. VOCABULARY
   - Is the vocabulary uniformly formal/academic? Flag if yes.
   - Are there any casual/colloquial words? List them.
   - Any idiosyncratic or unexpected word choices?

6. SENTENCE OPENERS
   - What words/patterns start most sentences? Are they monotonously similar?
   - Are there conjunction-started sentences (And, But, So)?
   - Are there fragment-started sentences?

══════════════════════════════════════════════════════
OUTPUT FORMAT — respond in this exact structure:
══════════════════════════════════════════════════════

HUMAN-LIKENESS SCORE: X/10
ESTIMATED AI DETECTION RISK: [very low / low / moderate / high / very high]

PROBLEMS FOUND:
1. [problem with specific sentence/passage quoted]
2. [problem ...]
...

STRENGTHS:
1. [what's working well]
...

REPAIR INSTRUCTIONS (ordered by priority):
1. [specific change to make — quote the sentence, provide revised version]
2. [specific change ...]
...

TEXT TO AUDIT:
---
{rewritten_text}
---
"""


def create_style_comparison_prompt(text_a: str, text_b: str, label_a: str = "Original", label_b: str = "Rewrite") -> str:
    """
    Creates a prompt to compare two texts stylistically and identify
    where the rewrite diverges from the original voice.
    """
    return f"""You are a forensic writing analyst specializing in style transfer fidelity.
Compare the two texts below and identify:

1. VOICE MATCH: How faithfully does {label_b} capture the voice of {label_a}?
   Score 1-10 with justification.

2. WHAT WAS LOST: List specific voice elements from {label_a} that are absent in {label_b}.
   Be concrete — quote from both texts.

3. WHAT WAS ADDED: List any patterns in {label_b} that are NOT in {label_a}.
   Are these additions AI-typical or human-typical?

4. RHYTHM COMPARISON:
   - {label_a} sentence length pattern vs {label_b} sentence length pattern
   - Which is more varied? Which feels more human?

5. HUMANNESS COMPARISON:
   - Rate {label_a} humanness: X/10
   - Rate {label_b} humanness: X/10
   - What specifically makes one more human than the other?

6. REPAIR PRIORITY LIST: Top 5 changes {label_b} needs to better match {label_a}.

---
{label_a.upper()}:
{text_a}

---
{label_b.upper()}:
{text_b}
---
"""