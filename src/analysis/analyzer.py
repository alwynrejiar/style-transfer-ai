"""
Core analysis engine for Style Transfer AI — v3.0
Self-contained multi-pass structured stylometry pipeline.

7-Pass Pipeline:
  Pass 1 → Lexical Fingerprint       (vocabulary, word choice, signatures, polarity)
  Pass 2 → Syntactic Structure       (sentence architecture, clauses, passives, depth)
  Pass 3 → Voice & Tone              (formality, emotion, hedging, authority, humor)
  Pass 4 → Discourse & Structure     (paragraphs, openings, closings, transitions, argument)
  Pass 5 → Rhythm & Cadence          (punctuation, pacing, lists, self-dialogue, anaphora)
  Pass 6 → Psycholinguistic Layer    (LIWC-style, tense, certainty, sensory, self-reference)
  Pass 7 → Synthesis                 (cross-validates all passes → rewrite directive)

Every pass forces structured JSON output with per-feature confidence scores (0.0–1.0).
Readability metrics (Flesch, Fog, Coleman-Liau, SMOG, ARI) are computed in pure Python
without any external library so this file remains fully self-contained.

The final profile['rewrite_directive'] is a ready-to-inject system prompt block.
No external prompt files, metric modules, or analogy engines are required.
"""

from __future__ import annotations

import json
import math
import re
import textwrap
from datetime import datetime
from typing import Any

from ..config.settings import ANALOGY_AUGMENTATION_ENABLED, DEFAULT_ANALOGY_DOMAIN
from .analogy_engine import AnalogyInjector


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PURE-PYTHON READABILITY METRICS
# These run locally without any model call. Results are embedded into Pass 6.
# ══════════════════════════════════════════════════════════════════════════════

def _count_syllables(word: str) -> int:
    """Rough syllable counter (English heuristic)."""
    word = word.lower().strip(".,!?;:\"'()-")
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    # Silent-e rule
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _tokenize_sentences(text: str) -> list[str]:
    """Split text into sentences on . ! ? boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z]+\b", text)


def compute_readability_metrics(text: str) -> dict:
    """
    Computes 6 standard readability metrics in pure Python.
    Returns a dict ready to embed in the style profile.
    """
    sentences = _tokenize_sentences(text)
    words = _tokenize_words(text)

    num_sentences = max(len(sentences), 1)
    num_words = max(len(words), 1)
    num_chars = sum(len(w) for w in words)

    syllable_counts = [_count_syllables(w) for w in words]
    num_syllables = max(sum(syllable_counts), 1)

    # Words with 3+ syllables (polysyllabic)
    complex_words = sum(1 for s in syllable_counts if s >= 3)

    avg_sent_len = num_words / num_sentences
    avg_syl_per_word = num_syllables / num_words
    avg_word_len = num_chars / num_words

    # Flesch Reading Ease  (higher = easier, 0-100)
    flesch_ease = 206.835 - (1.015 * avg_sent_len) - (84.6 * avg_syl_per_word)
    flesch_ease = round(max(0.0, min(100.0, flesch_ease)), 2)

    # Flesch-Kincaid Grade Level
    fk_grade = (0.39 * avg_sent_len) + (11.8 * avg_syl_per_word) - 15.59
    fk_grade = round(max(0.0, fk_grade), 2)

    # Gunning Fog Index
    fog = 0.4 * (avg_sent_len + 100 * (complex_words / num_words))
    fog = round(max(0.0, fog), 2)

    # Coleman-Liau Index
    L = (num_chars / num_words) * 100   # avg chars per 100 words
    S = (num_sentences / num_words) * 100  # avg sentences per 100 words
    cli = (0.0588 * L) - (0.296 * S) - 15.8
    cli = round(cli, 2)

    # SMOG Index (requires >= 30 sentences for accuracy, approximated otherwise)
    smog = 3.1291 + (1.0430 * math.sqrt(complex_words * (30 / num_sentences)))
    smog = round(max(0.0, smog), 2)

    # Automated Readability Index
    ari = (4.71 * avg_word_len) + (0.5 * avg_sent_len) - 21.43
    ari = round(max(0.0, ari), 2)

    def _ease_label(score: float) -> str:
        if score >= 90: return "Very Easy (5th grade)"
        if score >= 80: return "Easy (6th grade)"
        if score >= 70: return "Fairly Easy (7th grade)"
        if score >= 60: return "Standard (8-9th grade)"
        if score >= 50: return "Fairly Difficult (10-12th grade)"
        if score >= 30: return "Difficult (College)"
        return "Very Confusing (Professional)"

    return {
        "flesch_reading_ease": flesch_ease,
        "flesch_reading_ease_label": _ease_label(flesch_ease),
        "flesch_kincaid_grade": fk_grade,
        "gunning_fog_index": fog,
        "coleman_liau_index": cli,
        "smog_index": smog,
        "automated_readability_index": ari,
        "avg_sentence_length_words": round(avg_sent_len, 2),
        "avg_syllables_per_word": round(avg_syl_per_word, 2),
        "avg_word_length_chars": round(avg_word_len, 2),
        "total_sentences": num_sentences,
        "total_words": num_words,
        "total_syllables": num_syllables,
        "complex_word_count": complex_words,
        "complex_word_ratio": round(complex_words / num_words, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PASS PROMPT TEMPLATES
# 6 focused analysis passes + 1 synthesis pass.
# Each prompt demands strict JSON with per-feature confidence scores.
# ══════════════════════════════════════════════════════════════════════════════

_PASS_PROMPTS: dict[str, str] = {

    # ── PASS 1: Lexical Fingerprint ───────────────────────────────────────────
    "lexical": textwrap.dedent("""
        You are a forensic linguist performing a LEXICAL FINGERPRINT analysis.
        Study only word-level patterns in the writing sample below.

        Return ONLY a valid JSON object. No explanation, no markdown fences.

        {{
          "vocabulary_tier": "<academic|technical|casual|mixed>",
          "vocabulary_tier_confidence": <0.0-1.0>,

          "avg_word_sophistication": "<simple|moderate|advanced>",
          "avg_word_sophistication_confidence": <0.0-1.0>,

          "type_token_ratio": <estimated 0.0-1.0 — ratio of unique words to total words>,
          "ttr_confidence": <0.0-1.0>,

          "hapax_legomena_tendency": "<very low|low|moderate|high — how many words appear only once>",
          "hapax_confidence": <0.0-1.0>,

          "signature_words": ["word1", "word2", "..."],
          "signature_words_notes": "<why these are signature — overused, distinctive, etc.>",
          "signature_words_confidence": <0.0-1.0>,

          "signature_phrases": ["phrase1", "phrase2", "..."],
          "signature_phrases_confidence": <0.0-1.0>,

          "filler_words": ["basically", "actually", "literally", "..."],
          "filler_words_confidence": <0.0-1.0>,

          "hedge_words": ["maybe", "perhaps", "kind of", "..."],
          "hedge_words_confidence": <0.0-1.0>,

          "jargon_domains": ["technology", "finance", "..."],
          "jargon_examples": ["word1", "word2"],
          "jargon_confidence": <0.0-1.0>,

          "rare_word_tendency": "<avoids|occasional|frequent>",
          "rare_word_examples": ["word1", "word2"],
          "rare_word_confidence": <0.0-1.0>,

          "nominalization_tendency": "<low|medium|high>",
          "nominalization_examples": ["example1", "..."],
          "nominalization_confidence": <0.0-1.0>,

          "adjective_density": "<sparse|moderate|heavy>",
          "adverb_density": "<sparse|moderate|heavy>",

          "contraction_rate": "<never|rare|moderate|frequent>",
          "contraction_examples": ["it's", "don't", "..."],

          "concrete_vs_abstract_nouns": "<mostly concrete|balanced|mostly abstract>",
          "concrete_abstract_confidence": <0.0-1.0>,

          "word_polarity": {{
            "positive_ratio": <estimated 0.0-1.0>,
            "negative_ratio": <estimated 0.0-1.0>,
            "neutral_ratio": <estimated 0.0-1.0>,
            "overall_sentiment": "<positive|negative|neutral|mixed>"
          }},
          "polarity_confidence": <0.0-1.0>,

          "intensifier_usage": "<none|rare|moderate|frequent>",
          "intensifier_examples": ["very", "extremely", "..."],

          "foreign_borrowed_words": ["word1", "..."],
          "foreign_word_frequency": "<none|rare|occasional>",

          "qualifier_style": "<brief observation on how author qualifies claims>",
          "qualifier_confidence": <0.0-1.0>,

          "overall_lexical_confidence": <0.0-1.0>,
          "evidence_quotes": ["short quote from text showing lexical style", "..."]
        }}

        WRITING SAMPLE:
        ---
        {text}
        ---
    """),

    # ── PASS 2: Syntactic Structure ───────────────────────────────────────────
    "syntactic": textwrap.dedent("""
        You are a syntactic analyst. Analyze the sentence-level architecture of
        the writing sample below. Focus purely on grammar and structure patterns.

        Return ONLY a valid JSON object. No explanation, no markdown fences.

        {{
          "dominant_sentence_length": "<very short 1-8w|short 9-14w|medium 15-22w|long 23-35w|very long 35w+>",
          "sentence_length_confidence": <0.0-1.0>,

          "sentence_length_variance": "<monotone|moderate|highly varied>",
          "variance_pattern": "<e.g. short-long alternating, gradually building, uniform>",
          "variance_confidence": <0.0-1.0>,

          "dominant_sentence_structure": "<simple|compound|complex|compound-complex|mixed>",
          "structure_confidence": <0.0-1.0>,

          "passive_voice_ratio": <estimated 0.0-1.0>,
          "passive_voice_confidence": <0.0-1.0>,
          "passive_voice_examples": ["example from text", "..."],

          "fragment_usage": <true|false>,
          "fragment_frequency": "<never|rare|occasional|frequent>",
          "fragment_style": "<incomplete thought|deliberate emphasis|conversational>",
          "fragment_confidence": <0.0-1.0>,

          "run_on_tendency": <true|false>,
          "run_on_style": "<never|rare|deliberate rhetorical tool>",

          "clause_depth": "<shallow — 1 level|moderate — 2 levels|deep — 3+ levels>",
          "clause_depth_confidence": <0.0-1.0>,

          "subordinate_vs_coordinate_preference": "<strongly subordinate|balanced|strongly coordinate>",
          "clause_preference_confidence": <0.0-1.0>,

          "appositive_usage": "<never|rare|occasional|frequent>",
          "appositive_examples": ["example", "..."],

          "parenthetical_insertion": "<never|rare|occasional|frequent>",
          "parenthetical_style": "<clarifying|humorous|self-interrupting|none>",

          "parallel_structure_tendency": "<low|medium|high>",
          "parallel_examples": ["example from text", "..."],
          "parallel_confidence": <0.0-1.0>,

          "sentence_opening_patterns": {{
            "starts_with_conjunction": <true|false>,
            "starts_with_adverb": <true|false>,
            "starts_with_pronoun": <true|false>,
            "starts_with_article": <true|false>,
            "starts_with_preposition": <true|false>,
            "starts_with_gerund": <true|false>,
            "common_openers": ["But", "So", "Actually", "When", "..."]
          }},
          "opener_confidence": <0.0-1.0>,

          "question_frequency": "<never|rare|occasional|frequent>",
          "question_type": "<rhetorical|genuine|both|none>",

          "exclamation_frequency": "<never|rare|occasional|frequent>",

          "action_vs_state_verbs": "<mostly action|balanced|mostly state>",
          "verb_preference_confidence": <0.0-1.0>,

          "tense_distribution": {{
            "present_tense_ratio": <0.0-1.0>,
            "past_tense_ratio": <0.0-1.0>,
            "future_tense_ratio": <0.0-1.0>,
            "dominant_tense": "<present|past|future|mixed>"
          }},
          "tense_confidence": <0.0-1.0>,

          "overall_syntactic_confidence": <0.0-1.0>,
          "evidence_quotes": ["quote showing characteristic sentence structure", "..."]
        }}

        WRITING SAMPLE:
        ---
        {text}
        ---
    """),

    # ── PASS 3: Voice & Tone ──────────────────────────────────────────────────
    "voice": textwrap.dedent("""
        You are a voice and tone specialist for a style transfer system. Extract the
        full emotional and interpersonal texture of the author's voice.

        Return ONLY a valid JSON object. No explanation, no markdown fences.

        {{
          "formality_level": "<very informal|informal|neutral|formal|very formal>",
          "formality_confidence": <0.0-1.0>,

          "emotional_register": "<detached|measured|warm|passionate|intense>",
          "emotion_confidence": <0.0-1.0>,

          "directness": "<very direct|direct|hedged|indirect|very indirect>",
          "directness_confidence": <0.0-1.0>,

          "authority_tone": "<tentative|collaborative|authoritative|didactic>",
          "authority_confidence": <0.0-1.0>,

          "hedging_behavior": {{
            "hedges_claims": <true|false>,
            "hedge_frequency": "<never|rare|moderate|frequent>",
            "common_hedges": ["might", "perhaps", "sort of", "..."],
            "hedge_style": "<epistemic — I think|modal — might be|approximative — kind of>"
          }},
          "hedging_confidence": <0.0-1.0>,

          "epistemic_stance": "<how author signals certainty — I think vs it is clear that vs studies show>",
          "epistemic_confidence": <0.0-1.0>,

          "confidence_signaling": "<tentative — I wonder|neutral|assertive — clearly, definitely>",
          "confidence_examples": ["example phrase", "..."],

          "humor_presence": "<none|dry wit|self-deprecating|playful|sarcastic>",
          "humor_examples": ["example from text", "..."],
          "humor_confidence": <0.0-1.0>,

          "uses_first_person": <true|false>,
          "first_person_style": "<avoids|neutral I|opinionated I|inclusive we>",
          "first_person_frequency": "<never|rare|moderate|frequent>",

          "uses_second_person": <true|false>,
          "second_person_style": "<never|occasional you|direct address|instructional>",

          "empathy_signals": "<none|rare|moderate|frequent>",
          "empathy_style": "<acknowledging reader perspective|validating feelings|none>",
          "empathy_examples": ["example from text", "..."],

          "contractions_used": <true|false>,
          "contraction_frequency": "<never|rare|moderate|frequent>",

          "profanity_or_intensifiers": "<none|mild intensifiers|strong language>",

          "exclamation_tone": "<enthusiastic|emphatic|surprised|none>",
          "question_tone": "<curious|rhetorical|Socratic|conversational|none>",

          "voice_description": "<2-3 sentence holistic voice description precise enough for a rewriter>",
          "overall_voice_confidence": <0.0-1.0>,
          "evidence_quotes": ["quote that best captures this author's voice", "..."]
        }}

        WRITING SAMPLE:
        ---
        {text}
        ---
    """),

    # ── PASS 4: Discourse & Structure ─────────────────────────────────────────
    "discourse": textwrap.dedent("""
        You are a discourse analyst. Study how the author organizes ideas at the
        paragraph and document level. Focus purely on structural patterns, not content.

        Return ONLY a valid JSON object. No explanation, no markdown fences.

        {{
          "paragraph_length_tendency": "<very short 1-2 sent|short 3-4|medium 5-7|long 8+|mixed>",
          "paragraph_length_confidence": <0.0-1.0>,

          "topic_sentence_placement": "<always opening|usually opening|buried|absent|varies>",
          "topic_sentence_confidence": <0.0-1.0>,

          "opening_strategy": "<anecdote|bold statement|question|fact|quote|context-setting|direct-claim>",
          "opening_example": "<brief description of how this text actually opens>",
          "opening_confidence": <0.0-1.0>,

          "closing_strategy": "<summary|open-ended|call-to-action|reflective|punchy-statement|question>",
          "closing_example": "<brief description of how this text actually closes>",
          "closing_confidence": <0.0-1.0>,

          "transition_style": "<abrupt|minimal|smooth|heavy-signposting>",
          "common_transitions": ["However,", "So,", "In short,", "The thing is,", "..."],
          "transition_confidence": <0.0-1.0>,

          "argument_structure": "<linear|circular|narrative|compare-contrast|problem-solution|exploratory>",
          "argument_description": "<1 sentence describing how the author moves through ideas>",
          "argument_confidence": <0.0-1.0>,

          "information_density": "<sparse|lean|moderate|dense|very dense>",
          "density_confidence": <0.0-1.0>,

          "digression_tendency": "<stays focused|occasional tangents|frequently digresses>",
          "digression_examples": ["example tangent", "..."],

          "uses_examples": <true|false>,
          "example_style": "<none|brief inline|extended|anecdotal|data-driven>",
          "example_frequency": "<never|rare|moderate|frequent>",

          "uses_analogies": <true|false>,
          "analogy_style": "<simple comparison|extended metaphor|technical analogy>",
          "analogy_frequency": "<never|rare|occasional|frequent>",

          "uses_repetition_for_emphasis": <true|false>,
          "repetition_style": "<none|word repetition|structural repetition|callback>",
          "repetition_examples": ["example", "..."],

          "list_formatting_preference": "<avoids lists|inline natural language lists|bullet points|numbered|mixed>",

          "structural_description": "<2-3 sentence description of how this author builds and moves through ideas>",
          "overall_discourse_confidence": <0.0-1.0>,
          "evidence_quotes": ["quote showing structural tendency", "..."]
        }}

        WRITING SAMPLE:
        ---
        {text}
        ---
    """),

    # ── PASS 5: Rhythm & Cadence ──────────────────────────────────────────────
    "rhythm": textwrap.dedent("""
        You are a prose rhythm and cadence specialist. Analyze the sonic and
        pacing texture of the writing sample below at the punctuation and
        flow level. Focus on how the text *feels* to read aloud.

        Return ONLY a valid JSON object. No explanation, no markdown fences.

        {{
          "overall_cadence_feel": "<staccato — short punchy|flowing — long winding|varied — deliberate contrast|monotone>",
          "cadence_confidence": <0.0-1.0>,
          "cadence_description": "<1-2 sentence description of the prose rhythm feel>",

          "sentence_length_alternation": "<uniform|short-long alternating|gradually building|descending|irregular>",
          "alternation_confidence": <0.0-1.0>,

          "punctuation_density": "<sparse — minimal punctuation|moderate|heavy — lots of commas/dashes>",
          "punctuation_confidence": <0.0-1.0>,

          "comma_usage_style": "<minimal|standard|heavy — serial commas, comma splices>",
          "em_dash_usage": "<never|rare|occasional|frequent>",
          "em_dash_style": "<interruption|elaboration|dramatic pause|none>",
          "em_dash_examples": ["example — like this", "..."],

          "semicolon_vs_period_preference": "<always periods|occasionally semicolons|frequent semicolons>",
          "semicolon_examples": ["example; like this", "..."],

          "colon_usage": "<never|list-introducing|emphatic colon — statement: result>",
          "colon_examples": ["example: result", "..."],

          "ellipsis_usage": "<never|rare|occasional|frequent>",
          "ellipsis_style": "<trailing off|suspense|informality>",

          "parenthesis_usage": "<never|rare|occasional|frequent>",
          "parenthesis_style": "<clarifying aside|humorous aside|data reference>",

          "anaphora_usage": <true|false>,
          "anaphora_examples": ["repeated opener example", "..."],
          "anaphora_frequency": "<never|rare|occasional|frequent>",

          "self_dialogue_pattern": <true|false>,
          "self_dialogue_examples": ["Why does this matter? Because...", "..."],
          "self_dialogue_frequency": "<never|rare|occasional|frequent>",

          "sentence_variety_strategy": "<deliberate short after long for impact|no clear strategy|monotone>",

          "rhetorical_device_usage": {{
            "tricolon": <true|false>,
            "chiasmus": <true|false>,
            "antithesis": <true|false>,
            "hypophora": <true|false>
          }},

          "overall_rhythm_confidence": <0.0-1.0>,
          "evidence_quotes": ["quote that best shows the rhythm pattern", "..."]
        }}

        WRITING SAMPLE:
        ---
        {text}
        ---
    """),

    # ── PASS 6: Psycholinguistic Layer ────────────────────────────────────────
    "psycholinguistic": textwrap.dedent("""
        You are a psycholinguistic analyst. Analyze the deeper cognitive and
        psychological patterns in the writing sample below using LIWC-style
        category analysis and linguistic psychology.

        Return ONLY a valid JSON object. No explanation, no markdown fences.

        {{
          "cognitive_word_usage": {{
            "insight_words": "<never|rare|moderate|frequent — words like think, know, consider>",
            "causation_words": "<never|rare|moderate|frequent — because, hence, therefore>",
            "discrepancy_words": "<never|rare|moderate|frequent — should, would, could>",
            "tentative_words": "<never|rare|moderate|frequent — maybe, perhaps, guess>",
            "certainty_words": "<never|rare|moderate|frequent — always, never, definitely>"
          }},
          "cognitive_confidence": <0.0-1.0>,

          "social_word_usage": {{
            "social_references": "<low|moderate|high — references to people, relationships>",
            "inclusive_language": "<none|occasional we/us|frequent>",
            "exclusive_language": "<none|occasional they/them|frequent>"
          }},
          "social_confidence": <0.0-1.0>,

          "emotional_word_usage": {{
            "positive_emotion_words": "<low|moderate|high>",
            "negative_emotion_words": "<low|moderate|high>",
            "anxiety_words": "<none|rare|present>",
            "anger_words": "<none|rare|present>",
            "sadness_words": "<none|rare|present>"
          }},
          "emotion_word_confidence": <0.0-1.0>,

          "certainty_vs_tentativeness_ratio": "<strongly tentative|slightly tentative|balanced|slightly certain|strongly certain>",
          "certainty_confidence": <0.0-1.0>,

          "self_reference_rate": "<very low — avoids I/me/my|low|moderate|high — frequent self-reference>",
          "self_focus_vs_other_focus": "<self-focused|balanced|other-focused>",
          "self_reference_confidence": <0.0-1.0>,

          "tense_distribution_psychology": {{
            "past_focus": "<low|moderate|high — dwelling on what happened>",
            "present_focus": "<low|moderate|high — describing current state>",
            "future_focus": "<low|moderate|high — projecting forward>",
            "psychological_orientation": "<retrospective|present-grounded|forward-looking|mixed>"
          }},
          "tense_psychology_confidence": <0.0-1.0>,

          "action_vs_state_orientation": "<strongly action-oriented|balanced|strongly state-oriented>",
          "action_examples": ["verb1", "..."],
          "state_examples": ["verb1", "..."],

          "sensory_language": {{
            "visual_words": "<none|rare|moderate|frequent — see, look, bright, dark>",
            "auditory_words": "<none|rare|moderate|frequent — hear, sound, quiet, loud>",
            "tactile_words": "<none|rare|moderate|frequent — feel, touch, rough, smooth>",
            "dominant_sensory_channel": "<visual|auditory|tactile|none|mixed>"
          }},
          "sensory_confidence": <0.0-1.0>,

          "abstract_vs_concrete_thinking": "<highly abstract|balanced|highly concrete>",
          "abstraction_examples": ["abstract phrase", "..."],
          "concrete_examples": ["concrete phrase", "..."],
          "abstraction_confidence": <0.0-1.0>,

          "narrative_vs_analytical_mode": "<strongly narrative|balanced|strongly analytical>",
          "mode_description": "<1 sentence on whether author tells stories or analyses facts>",
          "mode_confidence": <0.0-1.0>,

          "overall_psycholinguistic_confidence": <0.0-1.0>,
          "evidence_quotes": ["quote revealing psychological pattern", "..."]
        }}

        WRITING SAMPLE:
        ---
        {text}
        ---
    """),

    # ── PASS 7: Synthesis → Rewrite Directive ─────────────────────────────────
    "synthesis": textwrap.dedent("""
        You are a master style transfer architect. You have been given the complete
        results of a 6-pass stylometric analysis of an author's writing.

        Your job:
        1. Cross-validate all 6 passes — resolve any contradictions between them.
        2. Identify the author's 3-5 MOST DISTINCTIVE traits (the ones a reader
           would immediately notice if they were missing).
        3. Assign a confidence score to the overall profile.
        4. Produce a REWRITE DIRECTIVE — a complete, precise, self-contained
           system prompt that can be injected verbatim into an LLM to make it
           rewrite ANY content in this exact author's voice. The rewriter must
           need no other context. Be specific about tone, rhythm, word choice,
           sentence structure, transitions, and what to absolutely avoid.

        Return ONLY a valid JSON object. No explanation, no markdown fences.

        {{
          "profile_confidence": <0.0-1.0>,
          "confidence_rationale": "<why this confidence — e.g. short sample, contradictory signals, very consistent>",

          "style_fingerprint_summary": "<3-5 sentence precise fingerprint — the most distinctive, immediately recognizable traits>",

          "cross_validation_notes": "<contradictions found between passes and how they were resolved>",

          "most_distinctive_traits": [
            "<trait 1 — the single most recognizable thing about this voice>",
            "<trait 2>",
            "<trait 3>",
            "<trait 4 — optional>",
            "<trait 5 — optional>"
          ],

          "key_traits": [
            {{"trait": "<trait name>", "strength": "<strong|moderate|weak>", "confidence": <0.0-1.0>}},
            "..."
          ],

          "do_not_lose": [
            "<specific writing habit that must be preserved — be concrete, not vague>",
            "..."
          ],

          "avoid_in_rewrite": [
            "<specific pattern that would break the voice — be concrete>",
            "..."
          ],

          "rewrite_directive": "<A complete self-contained system prompt of 250-400 words. This is the most important output. It must instruct an LLM to rewrite content in this author's voice with enough precision that no other context is needed. Cover: tone and formality, sentence rhythm and length, vocabulary level and signature words, structural habits, opening and closing tendencies, transition style, what to always do, what to never do. Write it as a direct instruction to the rewriter LLM.>"
        }}

        ANALYSIS RESULTS FROM ALL 6 PASSES:
        ---
        {analysis_json}
        ---
    """),
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MODEL CALL WRAPPER
# Calls Ollama or cloud API, strips fences, parses JSON.
# Returns a structured error dict on failure so the pipeline never crashes.
# ══════════════════════════════════════════════════════════════════════════════

def _call_model_for_json(
    prompt: str,
    use_local: bool,
    model_name: str | None,
    api_type: str | None,
    api_client: Any,
    pass_name: str,
) -> dict:
    from ..models.ollama_client import analyze_with_ollama
    from ..models.openai_client import analyze_with_openai
    from ..models.gemini_client import analyze_with_gemini

    try:
        if use_local:
            raw = analyze_with_ollama(prompt, model_name, processing_mode="enhanced")
        elif api_type == "openai":
            raw = analyze_with_openai(api_client, prompt)
        elif api_type == "gemini":
            raw = analyze_with_gemini(api_client, prompt)
        else:
            return {"_pass_error": "Unknown API type", "_pass_name": pass_name}
    except Exception as e:
        return {"_pass_error": f"Model call failed: {e}", "_pass_name": pass_name}

    # Strip markdown fences the model may add despite instructions
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    # Grab the outermost JSON object — models sometimes add preamble
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return {
            "_pass_error": "No JSON object found in model response",
            "_pass_name": pass_name,
            "_raw_response": raw[:600],
        }

    try:
        result = json.loads(match.group())
        result["_pass_name"] = pass_name
        return result
    except json.JSONDecodeError as e:
        return {
            "_pass_error": f"JSON parse failed: {e}",
            "_pass_name": pass_name,
            "_raw_response": raw[:600],
        }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PASS RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def _run_pass(
    pass_name: str,
    text: str,
    use_local: bool,
    model_name: str | None,
    api_type: str | None,
    api_client: Any,
) -> dict:
    """Runs a single named analysis pass and returns the parsed JSON result."""
    print(f"  ▸ Pass [{pass_name}]...", end=" ", flush=True)

    prompt = _PASS_PROMPTS[pass_name].format(text=text)
    result = _call_model_for_json(
        prompt, use_local, model_name, api_type, api_client, pass_name
    )

    if "_pass_error" in result:
        print(f"⚠ FAILED — {result['_pass_error']}")
    else:
        conf_key = f"overall_{pass_name}_confidence"
        conf = result.get(conf_key, "n/a")
        print(f"✓  (confidence: {conf})")

    return result


def _run_synthesis_pass(
    pass_results: dict[str, dict],
    use_local: bool,
    model_name: str | None,
    api_type: str | None,
    api_client: Any,
) -> dict:
    """Runs the synthesis pass using all 6 prior pass results as input."""
    print(f"  ▸ Pass [synthesis]...", end=" ", flush=True)

    # Feed all prior pass results as JSON — exclude internal _keys
    prior_json = json.dumps(
        {k: {ik: iv for ik, iv in v.items() if not ik.startswith("_")}
         for k, v in pass_results.items()},
        indent=2,
        default=str,
    )

    prompt = _PASS_PROMPTS["synthesis"].format(analysis_json=prior_json)
    result = _call_model_for_json(
        prompt, use_local, model_name, api_type, api_client, "synthesis"
    )

    if "_pass_error" in result:
        print(f"⚠ FAILED — {result['_pass_error']}")
    else:
        conf = result.get("profile_confidence", "n/a")
        print(f"✓  (overall profile confidence: {conf})")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CONFIDENCE REPORT BUILDER
# Aggregates per-feature confidence scores across all passes.
# Buckets features into high / medium / low confidence zones.
# ══════════════════════════════════════════════════════════════════════════════

_PASS_OVERALL_CONF_KEYS = {
    "lexical":          "overall_lexical_confidence",
    "syntactic":        "overall_syntactic_confidence",
    "voice":            "overall_voice_confidence",
    "discourse":        "overall_discourse_confidence",
    "rhythm":           "overall_rhythm_confidence",
    "psycholinguistic": "overall_psycholinguistic_confidence",
}


def _build_confidence_report(
    pass_results: dict[str, dict],
    synthesis: dict,
) -> dict:
    report = {
        "overall_profile_confidence": synthesis.get("profile_confidence", 0.0),
        "rationale": synthesis.get("confidence_rationale", ""),
        "per_pass": {},
        "low_confidence_features":    [],   # < 0.6  — uncertain, treat with caution
        "medium_confidence_features": [],   # 0.6-0.84
        "high_confidence_features":   [],   # >= 0.85 — reliable
    }

    for pass_name, conf_key in _PASS_OVERALL_CONF_KEYS.items():
        data = pass_results.get(pass_name, {})
        conf = data.get(conf_key)
        report["per_pass"][pass_name] = (
            round(conf, 3) if isinstance(conf, float) else "parse_error"
        )

        # Walk all keys to surface individual confidence scores
        for key, val in data.items():
            if (
                key.endswith("_confidence")
                and key != conf_key
                and isinstance(val, (int, float))
            ):
                feature_name = f"{pass_name}.{key.replace('_confidence', '')}"
                entry = {"feature": feature_name, "confidence": round(val, 3)}
                if val < 0.6:
                    report["low_confidence_features"].append(entry)
                elif val < 0.85:
                    report["medium_confidence_features"].append(entry)
                else:
                    report["high_confidence_features"].append(entry)

    return report


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PUBLIC API: analyze_style
# Runs the full 7-pass pipeline on a single text sample.
# ══════════════════════════════════════════════════════════════════════════════

def analyze_style(
    text_to_analyze: str,
    use_local: bool = True,
    model_name: str | None = None,
    api_type: str | None = None,
    api_client: Any = None,
    processing_mode: str = "enhanced",
) -> dict:
    """
    Performs a comprehensive 7-pass structured stylometric analysis.

    Passes:
        1. lexical          — vocabulary, word choice, polarity, signatures
        2. syntactic        — sentence structure, passives, fragments, tense
        3. voice            — formality, emotion, hedging, authority, humor
        4. discourse        — paragraph structure, transitions, argument flow
        5. rhythm           — punctuation, cadence, em-dashes, anaphora
        6. psycholinguistic — LIWC-style, certainty, sensory, self-reference
        7. synthesis        — cross-validation + rewrite directive

    Readability metrics (Flesch, Fog, Coleman-Liau, SMOG, ARI) are computed
    locally in pure Python and embedded in the result.

    Args:
        text_to_analyze (str): The writing sample.
        use_local (bool): Use Ollama if True, else cloud API.
        model_name (str): Ollama model name (required if use_local=True).
        api_type (str): 'openai' or 'gemini' (required if use_local=False).
        api_client: Pre-initialized API client.
        processing_mode (str): Reserved for future use.

    Returns:
        dict with keys:
            passes                   → per-pass raw JSON results
            synthesis                → synthesis pass result
            readability_metrics      → computed locally (no model call)
            confidence_report        → per-feature and per-pass breakdown
            rewrite_directive        → ready-to-inject system prompt
            style_fingerprint_summary
            most_distinctive_traits
            key_traits
            do_not_lose
            avoid_in_rewrite
    """
    if use_local and not model_name:
        raise ValueError("model_name is required when use_local=True")
    if not use_local and (not api_type or not api_client):
        raise ValueError("api_type and api_client are required when use_local=False")

    print("\n━━━ 7-Pass Style Analysis Starting ━━━")

    # ── Compute readability locally (no model call needed) ────────────────────
    print("  ▸ Computing readability metrics (local)... ", end="", flush=True)
    readability_metrics = compute_readability_metrics(text_to_analyze)
    print(f"✓  (Flesch Ease: {readability_metrics['flesch_reading_ease']} — "
          f"{readability_metrics['flesch_reading_ease_label']})")

    # ── Run the 6 AI analysis passes ──────────────────────────────────────────
    pass_results: dict[str, dict] = {}
    for pass_name in ("lexical", "syntactic", "voice", "discourse", "rhythm", "psycholinguistic"):
        pass_results[pass_name] = _run_pass(
            pass_name, text_to_analyze, use_local, model_name, api_type, api_client
        )

    # ── Run synthesis over all 6 pass results ─────────────────────────────────
    synthesis = _run_synthesis_pass(
        pass_results, use_local, model_name, api_type, api_client
    )

    # ── Build confidence report ───────────────────────────────────────────────
    confidence_report = _build_confidence_report(pass_results, synthesis)

    print("━━━ 7-Pass Style Analysis Complete ━━━\n")

    return {
        "passes": pass_results,
        "synthesis": synthesis,
        "readability_metrics": readability_metrics,
        "confidence_report": confidence_report,
        # ── Top-level convenience fields for the rewriter ─────────────────────
        "rewrite_directive":          synthesis.get("rewrite_directive", ""),
        "style_fingerprint_summary":  synthesis.get("style_fingerprint_summary", ""),
        "most_distinctive_traits":    synthesis.get("most_distinctive_traits", []),
        "key_traits":                 synthesis.get("key_traits", []),
        "do_not_lose":                synthesis.get("do_not_lose", []),
        "avoid_in_rewrite":           synthesis.get("avoid_in_rewrite", []),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PUBLIC API: create_enhanced_style_profile
# Handles single or multiple inputs, runs analyze_style, assembles profile.
# ══════════════════════════════════════════════════════════════════════════════

def create_enhanced_style_profile(
    input_data,
    use_local: bool = True,
    model_name: str | None = None,
    api_type: str | None = None,
    api_client: Any = None,
    processing_mode: str = "enhanced",
    analogy_augmentation: bool | None = None,
    analogy_domain: str | None = None,
) -> dict:
    """
    Creates a comprehensive style profile from text samples or direct input.

    Args:
        input_data (list | dict): List of file paths OR
                                  {'type': 'custom_text', 'text': '...', 'word_count': N}.
        use_local (bool): Use Ollama if True.
        model_name (str): Ollama model name.
        api_type (str): 'openai' or 'gemini'.
        api_client: Pre-initialized API client.
        processing_mode (str): Passed through to model calls.

    Returns:
        dict: Complete style profile.
    """
    if use_local and not model_name:
        raise ValueError("model_name is required when use_local=True")
    if not use_local and (not api_type or not api_client):
        raise ValueError("api_type and api_client are required when use_local=False")

    all_analyses = []
    combined_text = ""
    file_info = []

    # ── Collect text ──────────────────────────────────────────────────────────
    if isinstance(input_data, dict) and input_data.get("type") == "custom_text":
        print(f"\nCustom text input ({input_data.get('word_count', '?')} words)")
        file_content = input_data["text"]
        file_info.append({
            "filename": "custom_text_input",
            "word_count": input_data.get("word_count", len(file_content.split())),
            "character_count": len(file_content),
            "source": "direct_input",
        })
        analysis = analyze_style(
            file_content, use_local, model_name, api_type, api_client, processing_mode
        )
        all_analyses.append({
            "filename": "custom_text_input",
            "source": "direct_input",
            "analysis": analysis,
        })
        combined_text = file_content

    else:
        from ..utils.text_processing import read_text_file, extract_basic_stats

        file_paths = input_data if isinstance(input_data, list) else []
        if not file_paths:
            return {"profile_created": False, "error": "No valid input provided"}

        print(f"\n{len(file_paths)} text sample(s) found")

        for file_path in file_paths:
            file_content = read_text_file(file_path)
            if "Error" in file_content:
                print(f"  ⚠ Skipping {file_path}: {file_content}")
                continue

            print(f"\n  ── Processing: {file_path}")
            stats = extract_basic_stats(file_content)
            file_info.append({
                "filename": file_path,
                "word_count": stats["word_count"],
                "character_count": stats["character_count"],
            })
            analysis = analyze_style(
                file_content, use_local, model_name, api_type, api_client, processing_mode
            )
            all_analyses.append({
                "filename": file_path,
                "word_count": stats["word_count"],
                "character_count": stats["character_count"],
                "analysis": analysis,
            })
            combined_text += f"\n\n--- From {file_path} ---\n{file_content}"

    if not all_analyses:
        return {"profile_created": False, "error": "No valid input could be analyzed"}

    # ── Consolidated analysis if multiple files ───────────────────────────────
    if len(all_analyses) > 1:
        print("\n── Consolidated analysis across all samples...")
        consolidated = analyze_style(
            combined_text, use_local, model_name, api_type, api_client, processing_mode
        )
    else:
        consolidated = all_analyses[0]["analysis"]

    # ── Assemble final profile ────────────────────────────────────────────────
    analogy_enabled = (
        analogy_augmentation
        if analogy_augmentation is not None
        else ANALOGY_AUGMENTATION_ENABLED
    )
    analogy_result = None
    if analogy_enabled:
        domain = analogy_domain or DEFAULT_ANALOGY_DOMAIN
        injector = AnalogyInjector(domain=domain)
        analogy_result = injector.augment_analysis_result(
            combined_text,
            use_local=use_local,
            model_name=model_name,
            api_type=api_type,
            api_client=api_client,
        )

    return {
        "profile_created": True,
        "schema_version": "3.0",
        "metadata": {
            "analysis_date":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_method":    "Local Ollama" if use_local else f"Cloud API ({api_type})",
            "model_used":         model_name if use_local else f"{api_type.title()} API",
            "processing_mode":    processing_mode,
            "pipeline": [
                "readability (local)",
                "lexical", "syntactic", "voice",
                "discourse", "rhythm", "psycholinguistic",
                "synthesis",
            ],
            "total_samples":         len(all_analyses),
            "combined_text_length":  len(combined_text),
            "file_info":             file_info,
        },
        "individual_analyses":      all_analyses,
        "consolidated_analysis":    consolidated,
        "cognitive_bridging":        analogy_result,
        # ── Top-level fields — no digging into nested dicts ───────────────────
        "rewrite_directive":         consolidated.get("rewrite_directive", ""),
        "style_fingerprint_summary": consolidated.get("style_fingerprint_summary", ""),
        "most_distinctive_traits":   consolidated.get("most_distinctive_traits", []),
        "key_traits":                consolidated.get("key_traits", []),
        "do_not_lose":               consolidated.get("do_not_lose", []),
        "avoid_in_rewrite":          consolidated.get("avoid_in_rewrite", []),
        "confidence_report":         consolidated.get("confidence_report", {}),
        "readability_metrics":       consolidated.get("readability_metrics", {}),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — DISPLAY HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _confidence_bar(score, width: int = 10) -> str:
    """Renders a simple ASCII confidence bar, e.g. [████████░░]"""
    if not isinstance(score, (int, float)):
        return "[??????????]"
    filled = round(max(0.0, min(1.0, score)) * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def display_enhanced_results(style_profile: dict) -> None:
    """Prints a human-readable summary of the v3.0 style profile."""
    W = 68
    DIV = "─" * W

    print("\n" + "=" * W)
    print("  STYLE PROFILE — ANALYSIS COMPLETE  (v3.0)")
    print("=" * W)

    meta = style_profile.get("metadata", {})
    print(f"  Schema   : {style_profile.get('schema_version', '?')}")
    print(f"  Samples  : {meta.get('total_samples', '?')}")
    print(f"  Length   : {meta.get('combined_text_length', '?')} chars")
    print(f"  Model    : {meta.get('model_used', '?')}")
    print(f"  Pipeline : {' → '.join(meta.get('pipeline', []))}")

    # ── Readability ────────────────────────────────────────────────────────
    rm = style_profile.get("readability_metrics", {})
    if rm:
        print(f"\n{DIV}")
        print("  READABILITY METRICS")
        print(DIV)
        print(f"  Flesch Reading Ease     : {rm.get('flesch_reading_ease')}  "
              f"({rm.get('flesch_reading_ease_label', '')})")
        print(f"  Flesch-Kincaid Grade    : {rm.get('flesch_kincaid_grade')}")
        print(f"  Gunning Fog Index       : {rm.get('gunning_fog_index')}")
        print(f"  Coleman-Liau Index      : {rm.get('coleman_liau_index')}")
        print(f"  SMOG Index              : {rm.get('smog_index')}")
        print(f"  Auto Readability Index  : {rm.get('automated_readability_index')}")
        print(f"  Avg sentence length     : {rm.get('avg_sentence_length_words')} words")
        print(f"  Avg syllables/word      : {rm.get('avg_syllables_per_word')}")
        print(f"  Avg word length         : {rm.get('avg_word_length_chars')} chars")
        print(f"  Complex word ratio      : {rm.get('complex_word_ratio')}")
        print(f"  Total words / sentences : {rm.get('total_words')} / {rm.get('total_sentences')}")

    # ── Confidence Report ──────────────────────────────────────────────────
    conf = style_profile.get("confidence_report", {})
    if conf:
        print(f"\n{DIV}")
        print("  CONFIDENCE REPORT")
        print(DIV)
        overall = conf.get("overall_profile_confidence", "n/a")
        bar = _confidence_bar(overall) if isinstance(overall, float) else "          "
        print(f"  Overall  {bar}  {overall}")
        print(f"  Reason : {conf.get('rationale', 'n/a')}")
        print("\n  Per-pass:")
        for pname, score in conf.get("per_pass", {}).items():
            bar = _confidence_bar(score)
            print(f"    {pname:<18} {bar}  {score}")
        low = conf.get("low_confidence_features", [])
        if low:
            print(f"\n  ⚠ Low-confidence features ({len(low)}) — treat as uncertain:")
            for item in low[:10]:
                print(f"      - {item['feature']}  ({item['confidence']})")

    # ── Style Fingerprint ──────────────────────────────────────────────────
    fp = style_profile.get("style_fingerprint_summary", "")
    if fp:
        print(f"\n{DIV}")
        print("  STYLE FINGERPRINT")
        print(DIV)
        print(textwrap.fill(fp, width=W - 2, initial_indent="  ", subsequent_indent="  "))

    # ── Most Distinctive Traits ────────────────────────────────────────────
    mdt = style_profile.get("most_distinctive_traits", [])
    if mdt:
        print(f"\n{DIV}")
        print("  MOST DISTINCTIVE TRAITS")
        print(DIV)
        for i, trait in enumerate(mdt, 1):
            print(f"  {i}. {trait}")

    # ── Key Traits ────────────────────────────────────────────────────────
    traits = style_profile.get("key_traits", [])
    if traits:
        print(f"\n{DIV}")
        print("  KEY TRAITS")
        print(DIV)
        for t in traits:
            bar = _confidence_bar(t.get("confidence", 0))
            print(f"  [{t.get('strength', '?'):<8}] {bar}  {t.get('trait', '?')}")

    # ── Do Not Lose ───────────────────────────────────────────────────────
    dnl = style_profile.get("do_not_lose", [])
    if dnl:
        print(f"\n{DIV}")
        print("  MUST PRESERVE IN REWRITE")
        print(DIV)
        for item in dnl:
            print(f"  ✓ {item}")

    # ── Avoid ─────────────────────────────────────────────────────────────
    avoid = style_profile.get("avoid_in_rewrite", [])
    if avoid:
        print(f"\n{DIV}")
        print("  AVOID IN REWRITE")
        print(DIV)
        for item in avoid:
            print(f"  ✗ {item}")

    # ── Rewrite Directive ─────────────────────────────────────────────────
    directive = style_profile.get("rewrite_directive", "")
    if directive:
        print(f"\n{DIV}")
        print("  REWRITE DIRECTIVE  (inject verbatim as system prompt)")
        print(DIV)
        print(textwrap.fill(
            directive, width=W - 2, initial_indent="  ", subsequent_indent="  "
        ))

    print("\n" + "=" * W + "\n")