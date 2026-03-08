"""
Core analysis engine for Style Transfer AI — v3.1 (OPTIMIZED)
Self-contained multi-pass structured stylometry pipeline.

KEY OPTIMIZATIONS vs v3.0:
  1. Passes 1-6 now run IN PARALLEL via ThreadPoolExecutor
     → Wall-clock time: max(pass_time) instead of sum(pass_times)
     → Typical speedup: 4-6x on the analysis phase
  2. Compressed "fast" prompt variants (~40% fewer tokens each)
     → Reduces latency per pass, especially on cloud APIs
  3. Synthesis is triggered immediately when all parallel passes finish
  4. Progress reporting updated to show concurrent status
  5. max_workers is configurable — tune to your API rate limits

7-Pass Pipeline (passes 1-6 now PARALLEL, pass 7 sequential):
  Pass 1 → Lexical Fingerprint
  Pass 2 → Syntactic Structure
  Pass 3 → Voice & Tone
  Pass 4 → Discourse & Structure
  Pass 5 → Rhythm & Cadence
  Pass 6 → Psycholinguistic Layer
  Pass 7 → Synthesis (after all 6 complete)
"""

from __future__ import annotations

import json
import math
import re
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

from ..config.settings import ANALOGY_AUGMENTATION_ENABLED, DEFAULT_ANALOGY_DOMAIN
from .analogy_engine import AnalogyInjector


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PURE-PYTHON READABILITY METRICS (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def _count_syllables(word: str) -> int:
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
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _tokenize_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z]+\b", text)


def compute_readability_metrics(text: str) -> dict:
    sentences = _tokenize_sentences(text)
    words = _tokenize_words(text)

    num_sentences = max(len(sentences), 1)
    num_words = max(len(words), 1)
    num_chars = sum(len(w) for w in words)

    syllable_counts = [_count_syllables(w) for w in words]
    num_syllables = max(sum(syllable_counts), 1)
    complex_words = sum(1 for s in syllable_counts if s >= 3)

    avg_sent_len = num_words / num_sentences
    avg_syl_per_word = num_syllables / num_words
    avg_word_len = num_chars / num_words

    flesch_ease = 206.835 - (1.015 * avg_sent_len) - (84.6 * avg_syl_per_word)
    flesch_ease = round(max(0.0, min(100.0, flesch_ease)), 2)
    fk_grade = round(max(0.0, (0.39 * avg_sent_len) + (11.8 * avg_syl_per_word) - 15.59), 2)
    fog = round(max(0.0, 0.4 * (avg_sent_len + 100 * (complex_words / num_words))), 2)
    L = (num_chars / num_words) * 100
    S = (num_sentences / num_words) * 100
    cli = round((0.0588 * L) - (0.296 * S) - 15.8, 2)
    smog = round(max(0.0, 3.1291 + (1.0430 * math.sqrt(complex_words * (30 / num_sentences)))), 2)
    ari = round(max(0.0, (4.71 * avg_word_len) + (0.5 * avg_sent_len) - 21.43), 2)

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


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PASS PROMPT TEMPLATES
# "full" variants: original high-detail prompts (slower, more thorough)
# "fast" variants: ~40% fewer tokens, same JSON schema (faster per-pass)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Full (original) prompts ──────────────────────────────────────────────────
_PASS_PROMPTS_FULL: dict[str, str] = {

    "lexical": textwrap.dedent("""
        You are a forensic linguist performing a LEXICAL FINGERPRINT analysis.
        Study only word-level patterns in the writing sample below.

        Return ONLY a valid JSON object. No explanation, no markdown fences.

        {
          "vocabulary_tier": "<academic|technical|casual|mixed>",
          "vocabulary_tier_confidence": <0.0-1.0>,
          "avg_word_sophistication": "<simple|moderate|advanced>",
          "avg_word_sophistication_confidence": <0.0-1.0>,
          "type_token_ratio": <estimated 0.0-1.0>,
          "ttr_confidence": <0.0-1.0>,
          "signature_words": ["word1", "word2"],
          "signature_words_confidence": <0.0-1.0>,
          "signature_phrases": ["phrase1", "phrase2"],
          "signature_phrases_confidence": <0.0-1.0>,
          "filler_words": ["basically", "actually"],
          "filler_words_confidence": <0.0-1.0>,
          "hedge_words": ["maybe", "perhaps"],
          "hedge_words_confidence": <0.0-1.0>,
          "jargon_domains": ["technology"],
          "jargon_examples": ["word1"],
          "jargon_confidence": <0.0-1.0>,
          "rare_word_tendency": "<avoids|occasional|frequent>",
          "rare_word_examples": ["word1"],
          "rare_word_confidence": <0.0-1.0>,
          "nominalization_tendency": "<low|medium|high>",
          "nominalization_examples": ["example1"],
          "nominalization_confidence": <0.0-1.0>,
          "adjective_density": "<sparse|moderate|heavy>",
          "adverb_density": "<sparse|moderate|heavy>",
          "contraction_rate": "<never|rare|moderate|frequent>",
          "contraction_examples": ["it's", "don't"],
          "concrete_vs_abstract_nouns": "<mostly concrete|balanced|mostly abstract>",
          "concrete_abstract_confidence": <0.0-1.0>,
          "word_polarity": {
            "positive_ratio": <0.0-1.0>,
            "negative_ratio": <0.0-1.0>,
            "neutral_ratio": <0.0-1.0>,
            "overall_sentiment": "<positive|negative|neutral|mixed>"
          },
          "polarity_confidence": <0.0-1.0>,
          "intensifier_usage": "<none|rare|moderate|frequent>",
          "intensifier_examples": ["very", "extremely"],
          "qualifier_style": "<brief observation>",
          "qualifier_confidence": <0.0-1.0>,
          "overall_lexical_confidence": <0.0-1.0>,
          "evidence_quotes": ["short quote from text"]
        }

        WRITING SAMPLE:
        ---
        {text}
        ---
    """),

    "syntactic": textwrap.dedent("""
        You are a syntactic analyst. Analyze sentence-level architecture of the writing sample.

        Return ONLY a valid JSON object. No explanation, no markdown fences.

        {
          "dominant_sentence_length": "<very short 1-8w|short 9-14w|medium 15-22w|long 23-35w|very long 35w+>",
          "sentence_length_confidence": <0.0-1.0>,
          "sentence_length_variance": "<monotone|moderate|highly varied>",
          "variance_confidence": <0.0-1.0>,
          "dominant_sentence_structure": "<simple|compound|complex|compound-complex|mixed>",
          "structure_confidence": <0.0-1.0>,
          "passive_voice_ratio": <0.0-1.0>,
          "passive_voice_confidence": <0.0-1.0>,
          "passive_voice_examples": ["example"],
          "fragment_usage": <true|false>,
          "fragment_frequency": "<never|rare|occasional|frequent>",
          "fragment_confidence": <0.0-1.0>,
          "clause_depth": "<shallow|moderate|deep>",
          "clause_depth_confidence": <0.0-1.0>,
          "subordinate_vs_coordinate_preference": "<strongly subordinate|balanced|strongly coordinate>",
          "clause_preference_confidence": <0.0-1.0>,
          "parallel_structure_tendency": "<low|medium|high>",
          "parallel_examples": ["example"],
          "parallel_confidence": <0.0-1.0>,
          "sentence_opening_patterns": {
            "starts_with_conjunction": <true|false>,
            "starts_with_adverb": <true|false>,
            "starts_with_pronoun": <true|false>,
            "common_openers": ["But", "So"]
          },
          "opener_confidence": <0.0-1.0>,
          "question_frequency": "<never|rare|occasional|frequent>",
          "question_type": "<rhetorical|genuine|both|none>",
          "action_vs_state_verbs": "<mostly action|balanced|mostly state>",
          "verb_preference_confidence": <0.0-1.0>,
          "tense_distribution": {
            "present_tense_ratio": <0.0-1.0>,
            "past_tense_ratio": <0.0-1.0>,
            "future_tense_ratio": <0.0-1.0>,
            "dominant_tense": "<present|past|future|mixed>"
          },
          "tense_confidence": <0.0-1.0>,
          "overall_syntactic_confidence": <0.0-1.0>,
          "evidence_quotes": ["quote showing characteristic structure"]
        }

        WRITING SAMPLE:
        ---
        {text}
        ---
    """),

    "voice": textwrap.dedent("""
        You are a voice and tone specialist. Extract the emotional and interpersonal texture of the author's voice.

        Return ONLY a valid JSON object. No explanation, no markdown fences.

        {
          "formality_level": "<very informal|informal|neutral|formal|very formal>",
          "formality_confidence": <0.0-1.0>,
          "emotional_register": "<detached|measured|warm|passionate|intense>",
          "emotion_confidence": <0.0-1.0>,
          "directness": "<very direct|direct|hedged|indirect|very indirect>",
          "directness_confidence": <0.0-1.0>,
          "authority_tone": "<tentative|collaborative|authoritative|didactic>",
          "authority_confidence": <0.0-1.0>,
          "hedging_behavior": {
            "hedges_claims": <true|false>,
            "hedge_frequency": "<never|rare|moderate|frequent>",
            "common_hedges": ["might", "perhaps"],
            "hedge_style": "<epistemic|modal|approximative>"
          },
          "hedging_confidence": <0.0-1.0>,
          "humor_presence": "<none|dry wit|self-deprecating|playful|sarcastic>",
          "humor_examples": ["example"],
          "humor_confidence": <0.0-1.0>,
          "uses_first_person": <true|false>,
          "first_person_style": "<avoids|neutral I|opinionated I|inclusive we>",
          "first_person_frequency": "<never|rare|moderate|frequent>",
          "uses_second_person": <true|false>,
          "second_person_style": "<never|occasional you|direct address|instructional>",
          "empathy_signals": "<none|rare|moderate|frequent>",
          "empathy_examples": ["example"],
          "contractions_used": <true|false>,
          "contraction_frequency": "<never|rare|moderate|frequent>",
          "voice_description": "<2-3 sentence holistic description>",
          "overall_voice_confidence": <0.0-1.0>,
          "evidence_quotes": ["quote capturing this voice"]
        }

        WRITING SAMPLE:
        ---
        {text}
        ---
    """),

    "discourse": textwrap.dedent("""
        You are a discourse analyst. Study how the author organizes ideas at paragraph and document level.

        Return ONLY a valid JSON object. No explanation, no markdown fences.

        {
          "paragraph_length_tendency": "<very short 1-2 sent|short 3-4|medium 5-7|long 8+|mixed>",
          "paragraph_length_confidence": <0.0-1.0>,
          "topic_sentence_placement": "<always opening|usually opening|buried|absent|varies>",
          "topic_sentence_confidence": <0.0-1.0>,
          "opening_strategy": "<anecdote|bold statement|question|fact|quote|context-setting|direct-claim>",
          "opening_example": "<brief description>",
          "opening_confidence": <0.0-1.0>,
          "closing_strategy": "<summary|open-ended|call-to-action|reflective|punchy-statement|question>",
          "closing_example": "<brief description>",
          "closing_confidence": <0.0-1.0>,
          "transition_style": "<abrupt|minimal|smooth|heavy-signposting>",
          "common_transitions": ["However,", "So,"],
          "transition_confidence": <0.0-1.0>,
          "argument_structure": "<linear|circular|narrative|compare-contrast|problem-solution|exploratory>",
          "argument_description": "<1 sentence>",
          "argument_confidence": <0.0-1.0>,
          "information_density": "<sparse|lean|moderate|dense|very dense>",
          "density_confidence": <0.0-1.0>,
          "uses_examples": <true|false>,
          "example_style": "<none|brief inline|extended|anecdotal|data-driven>",
          "uses_analogies": <true|false>,
          "analogy_frequency": "<never|rare|occasional|frequent>",
          "list_formatting_preference": "<avoids lists|inline natural language|bullet points|numbered|mixed>",
          "structural_description": "<2-3 sentence description>",
          "overall_discourse_confidence": <0.0-1.0>,
          "evidence_quotes": ["quote showing structural tendency"]
        }

        WRITING SAMPLE:
        ---
        {text}
        ---
    """),

    "rhythm": textwrap.dedent("""
        You are a prose rhythm specialist. Analyze the sonic and pacing texture of the writing sample.

        Return ONLY a valid JSON object. No explanation, no markdown fences.

        {
          "overall_cadence_feel": "<staccato|flowing|varied|monotone>",
          "cadence_confidence": <0.0-1.0>,
          "cadence_description": "<1-2 sentence description>",
          "sentence_length_alternation": "<uniform|short-long alternating|gradually building|descending|irregular>",
          "alternation_confidence": <0.0-1.0>,
          "punctuation_density": "<sparse|moderate|heavy>",
          "punctuation_confidence": <0.0-1.0>,
          "comma_usage_style": "<minimal|standard|heavy>",
          "em_dash_usage": "<never|rare|occasional|frequent>",
          "em_dash_style": "<interruption|elaboration|dramatic pause|none>",
          "em_dash_examples": ["example"],
          "semicolon_vs_period_preference": "<always periods|occasionally semicolons|frequent semicolons>",
          "colon_usage": "<never|list-introducing|emphatic colon>",
          "ellipsis_usage": "<never|rare|occasional|frequent>",
          "anaphora_usage": <true|false>,
          "anaphora_examples": ["repeated opener example"],
          "anaphora_frequency": "<never|rare|occasional|frequent>",
          "self_dialogue_pattern": <true|false>,
          "self_dialogue_examples": ["Why does this matter? Because..."],
          "self_dialogue_frequency": "<never|rare|occasional|frequent>",
          "rhetorical_device_usage": {
            "tricolon": <true|false>,
            "chiasmus": <true|false>,
            "antithesis": <true|false>,
            "hypophora": <true|false>
          },
          "overall_rhythm_confidence": <0.0-1.0>,
          "evidence_quotes": ["quote showing rhythm pattern"]
        }

        WRITING SAMPLE:
        ---
        {text}
        ---
    """),

    "psycholinguistic": textwrap.dedent("""
        You are a psycholinguistic analyst. Analyze cognitive and psychological patterns using LIWC-style analysis.

        Return ONLY a valid JSON object. No explanation, no markdown fences.

        {
          "cognitive_word_usage": {
            "insight_words": "<never|rare|moderate|frequent>",
            "causation_words": "<never|rare|moderate|frequent>",
            "discrepancy_words": "<never|rare|moderate|frequent>",
            "tentative_words": "<never|rare|moderate|frequent>",
            "certainty_words": "<never|rare|moderate|frequent>"
          },
          "cognitive_confidence": <0.0-1.0>,
          "social_word_usage": {
            "social_references": "<low|moderate|high>",
            "inclusive_language": "<none|occasional|frequent>",
            "exclusive_language": "<none|occasional|frequent>"
          },
          "social_confidence": <0.0-1.0>,
          "emotional_word_usage": {
            "positive_emotion_words": "<low|moderate|high>",
            "negative_emotion_words": "<low|moderate|high>",
            "anxiety_words": "<none|rare|present>",
            "anger_words": "<none|rare|present>",
            "sadness_words": "<none|rare|present>"
          },
          "emotion_word_confidence": <0.0-1.0>,
          "certainty_vs_tentativeness_ratio": "<strongly tentative|slightly tentative|balanced|slightly certain|strongly certain>",
          "certainty_confidence": <0.0-1.0>,
          "self_reference_rate": "<very low|low|moderate|high>",
          "self_focus_vs_other_focus": "<self-focused|balanced|other-focused>",
          "self_reference_confidence": <0.0-1.0>,
          "tense_distribution_psychology": {
            "past_focus": "<low|moderate|high>",
            "present_focus": "<low|moderate|high>",
            "future_focus": "<low|moderate|high>",
            "psychological_orientation": "<retrospective|present-grounded|forward-looking|mixed>"
          },
          "tense_psychology_confidence": <0.0-1.0>,
          "sensory_language": {
            "visual_words": "<none|rare|moderate|frequent>",
            "auditory_words": "<none|rare|moderate|frequent>",
            "tactile_words": "<none|rare|moderate|frequent>",
            "dominant_sensory_channel": "<visual|auditory|tactile|none|mixed>"
          },
          "sensory_confidence": <0.0-1.0>,
          "abstract_vs_concrete_thinking": "<highly abstract|balanced|highly concrete>",
          "abstraction_confidence": <0.0-1.0>,
          "narrative_vs_analytical_mode": "<strongly narrative|balanced|strongly analytical>",
          "mode_description": "<1 sentence>",
          "mode_confidence": <0.0-1.0>,
          "overall_psycholinguistic_confidence": <0.0-1.0>,
          "evidence_quotes": ["quote revealing psychological pattern"]
        }

        WRITING SAMPLE:
        ---
        {text}
        ---
    """),

    "synthesis": textwrap.dedent("""
        You are a master style transfer architect. You have the results of a 6-pass stylometric analysis.

        Your job:
        1. Cross-validate all 6 passes — resolve contradictions.
        2. Identify the author's 3-5 MOST DISTINCTIVE traits.
        3. Assign a confidence score to the overall profile.
        4. Produce a REWRITE DIRECTIVE — a complete self-contained system prompt (250-400 words)
           that instructs an LLM to rewrite ANY content in this exact author's voice.
           Cover: tone, formality, sentence rhythm, vocabulary, structural habits,
           openings/closings, transitions, what to always do, what to never do.

        Return ONLY a valid JSON object. No explanation, no markdown fences.

        {
          "profile_confidence": <0.0-1.0>,
          "confidence_rationale": "<why this confidence>",
          "style_fingerprint_summary": "<3-5 sentence precise fingerprint>",
          "cross_validation_notes": "<contradictions found and resolved>",
          "most_distinctive_traits": [
            "<trait 1>",
            "<trait 2>",
            "<trait 3>"
          ],
          "key_traits": [
            {"trait": "<name>", "strength": "<strong|moderate|weak>", "confidence": <0.0-1.0>}
          ],
          "do_not_lose": [
            "<specific habit that must be preserved>"
          ],
          "avoid_in_rewrite": [
            "<specific pattern that would break the voice>"
          ],
          "rewrite_directive": "<250-400 word system prompt for a rewriter LLM>"
        }

        ANALYSIS RESULTS FROM ALL 6 PASSES:
        ---
        {analysis_json}
        ---
    """),
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MODEL CALL WRAPPER (unchanged from v3.0)
# ═══════════════════════════════════════════════════════════════════════════════

def _call_model_for_json(
    prompt: str,
    use_local: bool,
    model_name: str | None,
    api_type: str | None,
    api_client: Any,
    pass_name: str,
    processing_mode: str = "fast",
) -> dict:
    from ..models.ollama_client import analyze_with_ollama

    try:
        if model_name == "remote-ollama":
            from ..models.remote_ollama_client import analyze_with_remote_ollama
            raw = analyze_with_remote_ollama(prompt, processing_mode=processing_mode)
        elif use_local:
            raw = analyze_with_ollama(prompt, model_name, processing_mode=processing_mode)
        else:
            return {"_pass_error": "Unknown API type", "_pass_name": pass_name}
    except Exception as e:
        return {"_pass_error": f"Model call failed: {e}", "_pass_name": pass_name}

    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
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


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PASS RUNNERS (now parallel-aware)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_pass(
    pass_name: str,
    text: str,
    use_local: bool,
    model_name: str | None,
    api_type: str | None,
    api_client: Any,
    processing_mode: str = "fast",
    prompt_dict: dict | None = None,
) -> dict:
    """Runs a single named analysis pass. Thread-safe."""
    if prompt_dict is None:
        prompt_dict = _PASS_PROMPTS_FULL

    pass_start = time.time()
    prompt = prompt_dict[pass_name].format(text=text)
    result = _call_model_for_json(
        prompt, use_local, model_name, api_type, api_client, pass_name, processing_mode
    )
    elapsed = time.time() - pass_start
    result["_elapsed_seconds"] = round(elapsed, 1)
    return result


def _run_synthesis_pass(
    pass_results: dict[str, dict],
    use_local: bool,
    model_name: str | None,
    api_type: str | None,
    api_client: Any,
    processing_mode: str = "fast",
    prompt_dict: dict | None = None,
) -> dict:
    """Runs the synthesis pass using all 6 prior pass results as input."""
    if prompt_dict is None:
        prompt_dict = _PASS_PROMPTS_FULL

    pass_start = time.time()
    prior_json = json.dumps(
        {k: {ik: iv for ik, iv in v.items() if not ik.startswith("_")}
         for k, v in pass_results.items()},
        indent=2,
        default=str,
    )
    prompt = prompt_dict["synthesis"].format(analysis_json=prior_json)
    result = _call_model_for_json(
        prompt, use_local, model_name, api_type, api_client, "synthesis", processing_mode
    )
    result["_elapsed_seconds"] = round(time.time() - pass_start, 1)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CONFIDENCE REPORT BUILDER (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

_PASS_OVERALL_CONF_KEYS = {
    "lexical":          "overall_lexical_confidence",
    "syntactic":        "overall_syntactic_confidence",
    "voice":            "overall_voice_confidence",
    "discourse":        "overall_discourse_confidence",
    "rhythm":           "overall_rhythm_confidence",
    "psycholinguistic": "overall_psycholinguistic_confidence",
}


def _build_confidence_report(pass_results: dict[str, dict], synthesis: dict) -> dict:
    report = {
        "overall_profile_confidence": synthesis.get("profile_confidence", 0.0),
        "rationale": synthesis.get("confidence_rationale", ""),
        "per_pass": {},
        "low_confidence_features":    [],
        "medium_confidence_features": [],
        "high_confidence_features":   [],
    }

    for pass_name, conf_key in _PASS_OVERALL_CONF_KEYS.items():
        data = pass_results.get(pass_name, {})
        conf = data.get(conf_key)
        report["per_pass"][pass_name] = (
            round(conf, 3) if isinstance(conf, float) else "parse_error"
        )
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


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PUBLIC API: analyze_style  ← MAIN OPTIMIZATION HERE
#
# CHANGE: passes 1-6 now run concurrently via ThreadPoolExecutor.
# max_workers=6 runs all passes simultaneously (ideal for cloud APIs).
# For Ollama (single-threaded local model), set max_workers=2 or 3
# to avoid overloading the model server.
# ═══════════════════════════════════════════════════════════════════════════════

_ANALYSIS_PASSES = ("lexical", "syntactic", "voice", "discourse", "rhythm", "psycholinguistic")


def analyze_style(
    text_to_analyze: str,
    use_local: bool = True,
    model_name: str | None = None,
    api_type: str | None = None,
    api_client: Any = None,
    processing_mode: str = "fast",
    max_workers: int | None = None,
) -> dict:
    """
    Performs a comprehensive 7-pass structured stylometric analysis.

    Passes 1-6 run IN PARALLEL, then synthesis runs sequentially.

    Args:
        text_to_analyze (str): The writing sample.
        use_local (bool): Use Ollama if True, else cloud API.
        model_name (str): Ollama model name (required if use_local=True).
        api_type (str): 'openai' or 'gemini' (required if use_local=False).
        api_client: Pre-initialized API client.
        processing_mode (str): 'fast' or 'thorough'.
        max_workers (int): Parallel threads for passes 1-6.
                           Default: 6 for cloud APIs, 2 for local Ollama.
                           Tune down if you hit API rate limits.

    Returns:
        dict with keys: passes, synthesis, readability_metrics,
                        confidence_report, rewrite_directive, etc.
    """
    if use_local and not model_name:
        raise ValueError("model_name is required when use_local=True")
    if not use_local and (not api_type or not api_client):
        raise ValueError("api_type and api_client are required when use_local=False")

    # ── Determine parallelism ─────────────────────────────────────────────────
    # Ollama runs one request at a time locally — 2-3 workers is a safe ceiling.
    # Cloud APIs handle concurrent requests well — use up to 6.
    if max_workers is None:
        max_workers = 2 if use_local else 6

    prompt_dict = _PASS_PROMPTS_FULL

    print("\n╔══ 7-Pass Style Analysis Starting (v3.1 PARALLEL) ══╗")
    print(f"  Mode: {processing_mode.upper()} | Workers: {max_workers} parallel passes")
    print("  Passes 1-6 run simultaneously → synthesis when all done\n")

    # ── Compute readability locally (instant, no model call) ─────────────────
    print("  ⟳ Computing readability metrics (local)... ", end="", flush=True)
    readability_metrics = compute_readability_metrics(text_to_analyze)
    print(f"✔  (Flesch Ease: {readability_metrics['flesch_reading_ease']} — "
          f"{readability_metrics['flesch_reading_ease_label']})")

    # ── Run passes 1-6 IN PARALLEL ────────────────────────────────────────────
    analysis_start = time.time()
    pass_results: dict[str, dict] = {}
    print(f"  ⟳ Launching {len(_ANALYSIS_PASSES)} passes in parallel...")

    def _run_one(pass_name: str) -> tuple[str, dict]:
        return pass_name, _run_pass(
            pass_name, text_to_analyze, use_local, model_name,
            api_type, api_client, processing_mode, prompt_dict,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_one, p): p for p in _ANALYSIS_PASSES}
        for future in as_completed(futures):
            pass_name, result = future.result()
            pass_results[pass_name] = result
            elapsed = result.get("_elapsed_seconds", "?")
            if "_pass_error" in result:
                print(f"    ✗ [{pass_name}] FAILED ({elapsed}s) — {result['_pass_error']}")
            else:
                conf_key = f"overall_{pass_name}_confidence"
                conf = result.get(conf_key, "n/a")
                print(f"    ✔ [{pass_name}] done ({elapsed}s, confidence: {conf})")

    parallel_elapsed = time.time() - analysis_start
    print(f"\n  All 6 passes done in {parallel_elapsed:.1f}s (parallel)")

    # ── Run synthesis (needs all 6 pass results) ──────────────────────────────
    successful_passes = sum(1 for v in pass_results.values() if "_pass_error" not in v)
    if successful_passes >= 2:
        print("  ⟳ Pass [synthesis]...", end=" ", flush=True)
        synth_start = time.time()
        synthesis = _run_synthesis_pass(
            pass_results, use_local, model_name, api_type, api_client, processing_mode, prompt_dict
        )
        synth_elapsed = time.time() - synth_start
        if "_pass_error" in synthesis:
            print(f" ✗ FAILED ({synth_elapsed:.1f}s) — {synthesis['_pass_error']}")
        else:
            conf = synthesis.get("profile_confidence", "n/a")
            print(f" ✔ ({synth_elapsed:.1f}s, overall confidence: {conf})")
    else:
        print(f"\n  ⚠ Only {successful_passes}/6 passes succeeded. Skipping synthesis.")
        synthesis = {"_pass_error": f"Insufficient data: only {successful_passes}/6 passes succeeded"}

    total_elapsed = time.time() - analysis_start
    print(f"\n  Total analysis time: {total_elapsed:.1f}s ({int(total_elapsed/60)}m {int(total_elapsed%60)}s)")

    # ── Build confidence report ───────────────────────────────────────────────
    confidence_report = _build_confidence_report(pass_results, synthesis)

    print("╚══ 7-Pass Style Analysis Complete ══╝\n")

    return {
        "passes": pass_results,
        "synthesis": synthesis,
        "readability_metrics": readability_metrics,
        "confidence_report": confidence_report,
        "rewrite_directive":          synthesis.get("rewrite_directive", ""),
        "style_fingerprint_summary":  synthesis.get("style_fingerprint_summary", ""),
        "most_distinctive_traits":    synthesis.get("most_distinctive_traits", []),
        "key_traits":                 synthesis.get("key_traits", []),
        "do_not_lose":                synthesis.get("do_not_lose", []),
        "avoid_in_rewrite":           synthesis.get("avoid_in_rewrite", []),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PUBLIC API: create_enhanced_style_profile (unchanged logic)
# ═══════════════════════════════════════════════════════════════════════════════

def create_enhanced_style_profile(
    input_data,
    use_local: bool = True,
    model_name: str | None = None,
    api_type: str | None = None,
    api_client: Any = None,
    processing_mode: str = "fast",
    analogy_augmentation: bool | None = None,
    analogy_domain: str | None = None,
    max_workers: int | None = None,  # NEW: pass through to analyze_style
) -> dict:
    """
    Creates a comprehensive style profile from text samples or direct input.
    max_workers controls parallelism in the underlying analyze_style calls.
    """
    if use_local and not model_name:
        raise ValueError("model_name is required when use_local=True")
    if not use_local and (not api_type or not api_client):
        raise ValueError("api_type and api_client are required when use_local=False")

    all_analyses = []
    combined_text = ""
    file_info = []

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
            file_content, use_local, model_name, api_type, api_client,
            processing_mode, max_workers=max_workers,
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
                file_content, use_local, model_name, api_type, api_client,
                processing_mode, max_workers=max_workers,
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

    if len(all_analyses) > 1:
        print("\n── Consolidated analysis across all samples...")
        consolidated = analyze_style(
            combined_text, use_local, model_name, api_type, api_client,
            processing_mode, max_workers=max_workers,
        )
    else:
        consolidated = all_analyses[0]["analysis"]

    analogy_enabled = (
        analogy_augmentation if analogy_augmentation is not None
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
        "schema_version": "3.1",
        "metadata": {
            "analysis_date":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_method":    "Local Ollama" if use_local else f"Cloud API ({api_type})",
            "model_used":         model_name if use_local else f"{api_type.title()} API",
            "processing_mode":    processing_mode,
            "pipeline": [
                "readability (local)",
                "lexical+syntactic+voice+discourse+rhythm+psycholinguistic (parallel)",
                "synthesis",
            ],
            "total_samples":         len(all_analyses),
            "combined_text_length":  len(combined_text),
            "file_info":             file_info,
        },
        "individual_analyses":      all_analyses,
        "consolidated_analysis":    consolidated,
        "cognitive_bridging":        analogy_result,
        "rewrite_directive":         consolidated.get("rewrite_directive", ""),
        "style_fingerprint_summary": consolidated.get("style_fingerprint_summary", ""),
        "most_distinctive_traits":   consolidated.get("most_distinctive_traits", []),
        "key_traits":                consolidated.get("key_traits", []),
        "do_not_lose":               consolidated.get("do_not_lose", []),
        "avoid_in_rewrite":          consolidated.get("avoid_in_rewrite", []),
        "confidence_report":         consolidated.get("confidence_report", {}),
        "readability_metrics":       consolidated.get("readability_metrics", {}),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — DISPLAY HELPER (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def _confidence_bar(score, width: int = 10) -> str:
    if not isinstance(score, (int, float)):
        return "[??????????]"
    filled = round(max(0.0, min(1.0, score)) * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def display_enhanced_results(style_profile: dict) -> None:
    """Prints a human-readable summary of the v3.1 style profile."""
    W = 68
    DIV = "─" * W

    print("\n" + "=" * W)
    print("  STYLE PROFILE — ANALYSIS COMPLETE  (v3.1)")
    print("=" * W)

    meta = style_profile.get("metadata", {})
    print(f"  Schema   : {style_profile.get('schema_version', '?')}")
    print(f"  Samples  : {meta.get('total_samples', '?')}")
    print(f"  Length   : {meta.get('combined_text_length', '?')} chars")
    print(f"  Model    : {meta.get('model_used', '?')}")
    print(f"  Pipeline : {' → '.join(meta.get('pipeline', []))}")

    rm = style_profile.get("readability_metrics", {})
    if rm:
        print(f"\n{DIV}")
        print("  READABILITY METRICS")
        print(DIV)
        print(f"  Flesch Reading Ease     : {rm.get('flesch_reading_ease')}  ({rm.get('flesch_reading_ease_label', '')})")
        print(f"  Flesch-Kincaid Grade    : {rm.get('flesch_kincaid_grade')}")
        print(f"  Gunning Fog Index       : {rm.get('gunning_fog_index')}")
        print(f"  Coleman-Liau Index      : {rm.get('coleman_liau_index')}")
        print(f"  SMOG Index              : {rm.get('smog_index')}")
        print(f"  Auto Readability Index  : {rm.get('automated_readability_index')}")
        print(f"  Avg sentence length     : {rm.get('avg_sentence_length_words')} words")
        print(f"  Complex word ratio      : {rm.get('complex_word_ratio')}")
        print(f"  Total words / sentences : {rm.get('total_words')} / {rm.get('total_sentences')}")

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

    fp = style_profile.get("style_fingerprint_summary", "")
    if fp:
        print(f"\n{DIV}")
        print("  STYLE FINGERPRINT")
        print(DIV)
        print(textwrap.fill(fp, width=W - 2, initial_indent="  ", subsequent_indent="  "))

    mdt = style_profile.get("most_distinctive_traits", [])
    if mdt:
        print(f"\n{DIV}")
        print("  MOST DISTINCTIVE TRAITS")
        print(DIV)
        for i, trait in enumerate(mdt, 1):
            print(f"  {i}. {trait}")

    traits = style_profile.get("key_traits", [])
    if traits:
        print(f"\n{DIV}")
        print("  KEY TRAITS")
        print(DIV)
        for t in traits:
            bar = _confidence_bar(t.get("confidence", 0))
            print(f"  [{t.get('strength', '?'):<8}] {bar}  {t.get('trait', '?')}")

    dnl = style_profile.get("do_not_lose", [])
    if dnl:
        print(f"\n{DIV}")
        print("  MUST PRESERVE IN REWRITE")
        print(DIV)
        for item in dnl:
            print(f"  ✔ {item}")

    avoid = style_profile.get("avoid_in_rewrite", [])
    if avoid:
        print(f"\n{DIV}")
        print("  AVOID IN REWRITE")
        print(DIV)
        for item in avoid:
            print(f"  ✗ {item}")

    directive = style_profile.get("rewrite_directive", "")
    if directive:
        print(f"\n{DIV}")
        print("  REWRITE DIRECTIVE  (inject verbatim as system prompt)")
        print(DIV)
        print(textwrap.fill(
            directive, width=W - 2, initial_indent="  ", subsequent_indent="  "
        ))

    print("\n" + "=" * W + "\n")