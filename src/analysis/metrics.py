"""
Metrics calculation module for Style Transfer AI — v4.0
Handles readability metrics, statistical text analysis, advanced
computational stylometry (Burrows' Delta, n-grams, vocabulary richness,
syntactic complexity) AND human-vs-AI pattern metrics.

New in v4.0:
  - compute_humanization_metrics()  → burstiness, entropy, AI-risk composite
  - compute_ai_risk_score()         → single 0.0-1.0 AI-detection risk estimate
  - compute_sentence_burstiness()   → coefficient of variation of sentence lengths
  - detect_ai_clichés()             → counts known AI-generation marker phrases
  - compute_vocabulary_burstiness() → local word clustering vs global
  - post_process_for_humanness()    → rewrite post-processor that enforces human patterns
"""

import re
import math
import sys
import subprocess
import statistics
from collections import Counter
from ..utils.text_processing import count_syllables

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FUNCTION_WORDS = [
    "the", "and", "to", "of", "in", "a", "that", "it", "is", "was",
    "for", "on", "with", "as", "at", "by", "this", "an", "be", "not",
    "but", "from", "or", "have", "had", "has", "its", "are", "were", "been",
    "which", "their", "if", "will", "each", "about", "how", "up", "out", "them",
    "then", "she", "he", "his", "her", "would", "there", "what", "so", "can",
]

POS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
    "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
    "SCONJ", "SYM", "VERB", "X",
]

CONTRACTIONS = {
    "n't", "'re", "'ve", "'ll", "'d", "'m", "'s",
    "can't", "won't", "don't", "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't", "doesn't", "didn't", "couldn't",
    "shouldn't", "wouldn't", "mustn't", "let's", "that's", "who's",
    "what's", "here's", "there's", "it's", "i'm", "you're", "they're",
    "we're", "i've", "you've", "we've", "they've", "i'll", "you'll",
    "he'll", "she'll", "we'll", "they'll", "i'd", "you'd", "he'd",
    "she'd", "we'd", "they'd",
}

MODAL_VERBS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}

# ── AI cliché phrases — statistically overrepresented in AI-generated text ──
# Sourced from: GPTZero, Originality.ai, academic studies on AI text patterns.
AI_CLICHE_PHRASES = [
    # Stock transitions AI overuses
    "furthermore", "moreover", "additionally", "consequently", "therefore",
    "in conclusion", "to summarize", "in summary", "to sum up",
    "in essence", "in other words", "that being said", "with that said",
    "as mentioned", "as noted", "as discussed", "as stated",
    # AI filler openers
    "it is worth noting", "it is important to note", "it is important to",
    "it should be noted", "it's worth noting", "needless to say",
    "it goes without saying", "one must acknowledge", "one cannot deny",
    # AI enthusiasm words (hallmarks of ChatGPT/Claude defaults)
    "delve into", "dive into", "explore", "shed light on", "unpack",
    "testament to", "a testament", "groundbreaking", "revolutionary",
    "game-changer", "paradigm shift", "at the forefront", "cutting-edge",
    "holistic approach", "leverage", "synergy", "robust",
    # AI hedging boilerplate
    "in today's world", "in today's society", "in today's fast-paced",
    "in the realm of", "the realm of", "the world of",
    "the landscape of", "the fabric of",
    # AI summary clichés
    "this highlights", "this demonstrates", "this underscores",
    "this illustrates", "this shows that", "this reveals",
    "this suggests that", "as we can see", "we can see that",
    # AI instructional boilerplate
    "let's explore", "let us explore", "let's dive", "let's delve",
    "let's take a look", "let's examine", "let me walk you through",
    # Hollow emphasis
    "truly", "certainly", "undoubtedly", "unquestionably",
    "absolutely", "without a doubt", "indeed",
]

# ── Human writing markers — patterns humans use more than AI ─────────────────
HUMAN_MARKERS = [
    # Casual connectors as sentence starters
    r'\bso\b', r'\bbut\b', r'\band\b', r'\bbecause\b', r'\byet\b',
    # Self-dialogue
    r'\?.*?[Bb]ecause', r'\bwhy\?', r'\bwhat\?', r'\bhow\?',
    # Hesitation / informality
    r'\bactually\b', r'\bhonestly\b', r'\blook\b', r'\bhere\'s the thing\b',
    r'\bthe thing is\b', r'\bfunny thing\b', r'\bweird(ly)?\b',
    # Mild profanity / intensifiers
    r'\bdamn\b', r'\bhell\b', r'\bpretty\b', r'\bkind of\b', r'\bsort of\b',
    # Self-correction
    r'\bor rather\b', r'\bor maybe\b', r'\bwell,\b', r'\bi mean\b',
    r'\bthat is\b',
]


# ---------------------------------------------------------------------------
# spaCy bootstrap
# ---------------------------------------------------------------------------

def _ensure_spacy_model():
    try:
        import spacy
    except Exception:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "spacy"], check=True)
            import spacy
        except Exception:
            print("spaCy install failed. Please run: python -m pip install spacy")
            return None
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        try:
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            return spacy.load("en_core_web_sm")
        except Exception:
            print("spaCy model download failed. Please run: python -m spacy download en_core_web_sm")
            return None


# ---------------------------------------------------------------------------
# Vocabulary richness helpers (unchanged from v3.0)
# ---------------------------------------------------------------------------

def _hapax_legomena_ratio(word_freq: Counter, total_words: int) -> float:
    if total_words == 0:
        return 0.0
    hapax = sum(1 for count in word_freq.values() if count == 1)
    return round(hapax / total_words, 4)


def _dis_legomena_ratio(word_freq: Counter, total_words: int) -> float:
    if total_words == 0:
        return 0.0
    dis = sum(1 for count in word_freq.values() if count == 2)
    return round(dis / total_words, 4)


def _yules_k(word_freq: Counter, total_words: int) -> float:
    if total_words == 0:
        return 0.0
    freq_spectrum = Counter(word_freq.values())
    m2 = sum(i * i * v_i for i, v_i in freq_spectrum.items())
    k = 10_000 * (m2 - total_words) / (total_words * total_words) if total_words > 1 else 0.0
    return round(k, 4)


def _simpsons_diversity(word_freq: Counter, total_words: int) -> float:
    if total_words <= 1:
        return 0.0
    s = sum(n * (n - 1) for n in word_freq.values())
    d = 1 - s / (total_words * (total_words - 1))
    return round(d, 4)


def _brunet_w(total_words: int, vocab_size: int) -> float:
    if total_words <= 0 or vocab_size <= 0:
        return 0.0
    return round(total_words ** (vocab_size ** -0.172), 4)


def _honore_r(total_words: int, vocab_size: int, hapax_count: int) -> float:
    if total_words == 0 or vocab_size == 0:
        return 0.0
    if hapax_count == vocab_size:
        hapax_count = vocab_size - 1
    if vocab_size - hapax_count == 0:
        return 0.0
    r = 100 * math.log(total_words) / (1 - hapax_count / vocab_size)
    return round(r, 4)


# ---------------------------------------------------------------------------
# N-gram helpers (unchanged)
# ---------------------------------------------------------------------------

def _character_ngrams(text: str, n: int = 3, top_k: int = 25) -> dict:
    cleaned = text.lower()
    ngrams = Counter()
    for i in range(len(cleaned) - n + 1):
        ngrams[cleaned[i:i + n]] += 1
    total = sum(ngrams.values())
    if total == 0:
        return {}
    return {ng: round(cnt / total, 6) for ng, cnt in ngrams.most_common(top_k)}


def _word_bigrams(tokens_lower: list, top_k: int = 25) -> dict:
    bigrams = Counter()
    for i in range(len(tokens_lower) - 1):
        bigrams[(tokens_lower[i], tokens_lower[i + 1])] += 1
    total = sum(bigrams.values())
    if total == 0:
        return {}
    return {
        f"{a} {b}": round(cnt / total, 6)
        for (a, b), cnt in bigrams.most_common(top_k)
    }


# ============================================================================
# NEW IN v4.0 — HUMAN-VS-AI PATTERN METRICS
# ============================================================================

def compute_sentence_burstiness(text: str) -> dict:
    """
    Compute sentence length burstiness — the primary human-vs-AI signal.

    AI text has burstiness_cv around 0.2-0.35.
    Human text typically ranges 0.45-0.80+.

    Returns a detailed burstiness profile including:
      - coefficient of variation (CV = std/mean)
      - consecutive-run penalty (AI tends to run same-length sentences)
      - length bin distribution and entropy
      - specific short/long sentence ratios
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s.strip()]

    if len(sentences) < 3:
        return {
            "burstiness_cv": 0.0,
            "burstiness_label": "insufficient data",
            "consecutive_uniformity": 0.0,
            "short_sentence_ratio": 0.0,
            "long_sentence_ratio": 0.0,
            "sentence_entropy": 0.0,
            "length_bins": {},
        }

    sent_lengths = [len(re.findall(r"\b[a-zA-Z]+\b", s)) for s in sentences]
    sent_lengths = [l for l in sent_lengths if l > 0]

    if not sent_lengths:
        return {"burstiness_cv": 0.0, "burstiness_label": "no data"}

    mean_len = statistics.mean(sent_lengths)
    std_len = statistics.pstdev(sent_lengths)
    cv = round(std_len / mean_len, 4) if mean_len > 0 else 0.0

    # Consecutive uniformity: fraction of adjacent pairs within 3 words of each other
    runs = sum(1 for i in range(1, len(sent_lengths))
               if abs(sent_lengths[i] - sent_lengths[i - 1]) <= 3)
    uniformity = round(runs / max(len(sent_lengths) - 1, 1), 4)

    # Length bins
    bins = {"micro(1-4)": 0, "short(5-10)": 0, "medium(11-20)": 0,
            "long(21-35)": 0, "very_long(36+)": 0}
    for l in sent_lengths:
        if l <= 4:      bins["micro(1-4)"] += 1
        elif l <= 10:   bins["short(5-10)"] += 1
        elif l <= 20:   bins["medium(11-20)"] += 1
        elif l <= 35:   bins["long(21-35)"] += 1
        else:           bins["very_long(36+)"] += 1

    total = len(sent_lengths)
    # Entropy of the bin distribution
    entropy = 0.0
    for count in bins.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    short_ratio = round((bins["micro(1-4)"] + bins["short(5-10)"]) / total, 4)
    long_ratio = round((bins["long(21-35)"] + bins["very_long(36+)"]) / total, 4)

    label = (
        "very human" if cv >= 0.55
        else "human" if cv >= 0.40
        else "borderline" if cv >= 0.28
        else "AI-like (low variety)"
    )

    return {
        "burstiness_cv": cv,
        "burstiness_label": label,
        "mean_sentence_length": round(mean_len, 2),
        "std_sentence_length": round(std_len, 2),
        "min_length": min(sent_lengths),
        "max_length": max(sent_lengths),
        "consecutive_uniformity": uniformity,
        "uniformity_label": (
            "uniform (AI-risk)" if uniformity > 0.65
            else "moderate" if uniformity > 0.45
            else "varied (human-like)"
        ),
        "short_sentence_ratio": short_ratio,
        "long_sentence_ratio": long_ratio,
        "sentence_entropy": round(entropy, 4),
        "entropy_label": (
            "high variety" if entropy >= 2.0
            else "moderate variety" if entropy >= 1.4
            else "low variety (AI-risk)"
        ),
        "length_bins": bins,
        "bin_percentages": {k: round(v / total, 3) for k, v in bins.items()},
    }


def compute_vocabulary_burstiness(text: str, window_size: int = 60) -> dict:
    """
    Measure local vocabulary clustering — humans tend to use words in bursts
    (topic-driven local repetition) rather than evenly distributed across text.

    High window_std_dev of local TTR = bursty = human.
    Low window_std_dev of local TTR = uniform = AI.
    """
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if len(words) < window_size * 2:
        return {"vocabulary_burstiness": 0.0, "label": "insufficient data"}

    window_ttrs = []
    for i in range(0, len(words) - window_size, window_size // 2):
        chunk = words[i:i + window_size]
        ttr = len(set(chunk)) / len(chunk)
        window_ttrs.append(ttr)

    if len(window_ttrs) < 2:
        return {"vocabulary_burstiness": 0.0, "label": "insufficient data"}

    mean_ttr = statistics.mean(window_ttrs)
    std_ttr = statistics.pstdev(window_ttrs)
    cv = round(std_ttr / mean_ttr, 4) if mean_ttr > 0 else 0.0

    return {
        "vocabulary_burstiness_cv": cv,
        "mean_window_ttr": round(mean_ttr, 4),
        "std_window_ttr": round(std_ttr, 4),
        "num_windows": len(window_ttrs),
        "label": (
            "bursty (human-like)" if cv >= 0.08
            else "moderately bursty" if cv >= 0.04
            else "uniform (AI-like)"
        ),
    }


def detect_ai_clichés(text: str) -> dict:
    """
    Detect AI-generation marker phrases in text.

    Returns counts, density per 100 words, and a flagged list of
    phrases found. High density → high AI-detection risk.
    """
    text_lower = text.lower()
    words = re.findall(r"\b[a-zA-Z]+\b", text_lower)
    total_words = max(len(words), 1)

    found_phrases = []
    total_count = 0

    for phrase in AI_CLICHE_PHRASES:
        count = text_lower.count(phrase)
        if count > 0:
            found_phrases.append({"phrase": phrase, "count": count})
            total_count += count

    density = round(total_count / (total_words / 100), 4)

    return {
        "ai_cliche_count": total_count,
        "ai_cliche_density_per_100w": density,
        "ai_cliche_risk_label": (
            "high risk" if density > 3.0
            else "moderate risk" if density > 1.5
            else "low risk" if density > 0.5
            else "clean"
        ),
        "flagged_phrases": sorted(found_phrases, key=lambda x: -x["count"])[:20],
        "unique_clichés_found": len(found_phrases),
    }


def count_human_markers(text: str) -> dict:
    """
    Count the presence of patterns that are statistically more common
    in human writing than AI writing.
    """
    text_lower = text.lower()
    total_words = max(len(re.findall(r"\b[a-zA-Z]+\b", text)), 1)

    found = {}
    total = 0
    for pattern in HUMAN_MARKERS:
        matches = len(re.findall(pattern, text_lower))
        if matches > 0:
            found[pattern] = matches
            total += matches

    density = round(total / (total_words / 100), 4)

    return {
        "human_marker_count": total,
        "human_marker_density_per_100w": density,
        "human_marker_label": (
            "strongly human" if density > 4.0
            else "moderately human" if density > 2.0
            else "weakly human" if density > 0.5
            else "neutral"
        ),
        "patterns_found": found,
    }


def compute_ai_risk_score(text: str) -> dict:
    """
    Composite AI-detection risk score (0.0 = definitely human, 1.0 = definitely AI).

    Combines 6 independently computed signals:
      1. Sentence burstiness (CV)
      2. Sentence entropy
      3. AI cliché density
      4. Consecutive uniformity
      5. Vocabulary burstiness
      6. Human marker presence

    Each signal is normalized and weighted based on published research
    on AI text detection feature importance.
    """
    burstiness = compute_sentence_burstiness(text)
    vocab_burst = compute_vocabulary_burstiness(text)
    ai_clichés = detect_ai_clichés(text)
    human_marks = count_human_markers(text)

    # ── Signal 1: Sentence CV (weight 0.28) ──────────────────────────────────
    cv = burstiness.get("burstiness_cv", 0.0)
    # CV < 0.25 = AI. CV > 0.55 = human. Normalize to 0-1 risk.
    s1 = max(0.0, min(1.0, (0.55 - cv) / 0.55))

    # ── Signal 2: Sentence entropy (weight 0.18) ─────────────────────────────
    entropy = burstiness.get("sentence_entropy", 0.0)
    # entropy < 1.2 = AI. entropy > 2.2 = human.
    s2 = max(0.0, min(1.0, (2.2 - entropy) / 2.2))

    # ── Signal 3: AI cliché density (weight 0.22) ────────────────────────────
    cliché_density = ai_clichés.get("ai_cliche_density_per_100w", 0.0)
    # density > 4.0 = AI. density < 0.5 = human.
    s3 = max(0.0, min(1.0, cliché_density / 4.0))

    # ── Signal 4: Consecutive uniformity (weight 0.18) ───────────────────────
    uniformity = burstiness.get("consecutive_uniformity", 0.5)
    # uniformity > 0.70 = AI. uniformity < 0.35 = human.
    s4 = max(0.0, min(1.0, (uniformity - 0.35) / 0.35))

    # ── Signal 5: Vocab burstiness CV (weight 0.08) ──────────────────────────
    vb_cv = vocab_burst.get("vocabulary_burstiness_cv", 0.05)
    # vb_cv < 0.03 = AI. vb_cv > 0.10 = human.
    s5 = max(0.0, min(1.0, (0.10 - vb_cv) / 0.10))

    # ── Signal 6: Human markers (weight 0.06) ────────────────────────────────
    hm_density = human_marks.get("human_marker_density_per_100w", 0.0)
    # hm_density < 0.5 = AI. hm_density > 4.0 = human.
    s6 = max(0.0, min(1.0, (4.0 - hm_density) / 4.0))

    # Weighted composite
    ai_risk = (
        0.28 * s1 +
        0.18 * s2 +
        0.22 * s3 +
        0.18 * s4 +
        0.08 * s5 +
        0.06 * s6
    )
    ai_risk = round(min(1.0, max(0.0, ai_risk)), 4)
    human_likeness = round(1.0 - ai_risk, 4)

    return {
        "ai_risk_score": ai_risk,
        "human_likeness_score": human_likeness,
        "risk_label": (
            "very high risk (AI-like)" if ai_risk >= 0.75
            else "high risk" if ai_risk >= 0.60
            else "moderate risk" if ai_risk >= 0.40
            else "low risk (human-like)" if ai_risk >= 0.20
            else "very human-like"
        ),
        "signal_breakdown": {
            "sentence_cv_risk":        round(s1, 4),
            "sentence_entropy_risk":   round(s2, 4),
            "ai_cliche_risk":          round(s3, 4),
            "uniformity_risk":         round(s4, 4),
            "vocab_burstiness_risk":   round(s5, 4),
            "human_marker_risk":       round(s6, 4),
        },
        "burstiness_detail": burstiness,
        "vocab_burstiness_detail": vocab_burst,
        "ai_clichés_detail": ai_clichés,
        "human_markers_detail": human_marks,
    }


def compute_humanization_metrics(text: str) -> dict:
    """
    Master function returning all human-vs-AI metrics in one dict.
    Designed to be called once and passed to both the analysis pipeline
    and the post-processor.
    """
    ai_risk = compute_ai_risk_score(text)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    words = re.findall(r"\b[a-zA-Z]+\b", text)

    # ── Paragraph length variance ────────────────────────────────────────────
    para_lengths = [len(re.findall(r"\b[a-zA-Z]+\b", p)) for p in paragraphs if p]
    para_cv = 0.0
    if len(para_lengths) >= 2:
        pm = statistics.mean(para_lengths)
        ps = statistics.pstdev(para_lengths)
        para_cv = round(ps / pm, 4) if pm > 0 else 0.0

    # ── Punctuation irregularity across paragraphs ───────────────────────────
    punct_densities = []
    for p in paragraphs:
        pw = len(re.findall(r"\b[a-zA-Z]+\b", p))
        pc = sum(1 for ch in p if ch in ".,;:!?—-")
        if pw > 0:
            punct_densities.append(pc / pw)
    punct_irregularity = round(statistics.pstdev(punct_densities), 4) if len(punct_densities) > 1 else 0.0

    return {
        **ai_risk,
        "paragraph_count": len(paragraphs),
        "paragraph_length_cv": para_cv,
        "para_cv_label": (
            "varied (human-like)" if para_cv >= 0.45
            else "moderate" if para_cv >= 0.25
            else "uniform (AI-like)"
        ),
        "punct_irregularity_across_paragraphs": punct_irregularity,
        "total_words": len(words),
        "total_sentences": len([s for s in sentences if s.strip()]),
    }


def post_process_for_humanness(text: str, style_profile: dict | None = None) -> dict:
    """
    Analyzes a piece of text (typically rewriter output) and returns:
      1. A humanness audit with specific problems identified
      2. Concrete repair instructions for a second-pass rewrite
      3. The AI risk score before and after estimated repairs

    This is meant to be called on the OUTPUT of a style injection rewrite
    to check whether it has become more human-like or still reads as AI.

    Args:
        text: The rewritten text to audit
        style_profile: Optional — the original style profile to compare against

    Returns:
        dict with 'audit', 'repair_instructions', 'ai_risk_score'
    """
    metrics = compute_humanization_metrics(text)
    burstiness = metrics.get("burstiness_detail", {})
    clichés = metrics.get("ai_clichés_detail", {})

    audit = {
        "ai_risk_score": metrics.get("ai_risk_score"),
        "risk_label": metrics.get("risk_label"),
        "human_likeness_score": metrics.get("human_likeness_score"),
        "problems": [],
        "strengths": [],
    }

    repair_instructions = []

    # ── Check burstiness ─────────────────────────────────────────────────────
    cv = burstiness.get("burstiness_cv", 0.0)
    if cv < 0.28:
        audit["problems"].append(
            f"Very low sentence length variety (CV={cv}). All sentences are similar length — "
            f"major AI detection trigger."
        )
        repair_instructions.append(
            "URGENT: Break up consecutive same-length sentences. After every block of 2-3 "
            "sentences of 15+ words, insert a standalone sentence of 3-7 words maximum."
        )
    elif cv < 0.40:
        audit["problems"].append(
            f"Moderate sentence length variety (CV={cv}). Needs more extreme short/long contrast."
        )
        repair_instructions.append(
            "Add at least 3 micro-sentences (1-5 words) spread across the text. "
            "These can be rhetorical fragments or standalone punchy points."
        )
    else:
        audit["strengths"].append(f"Good sentence length burstiness (CV={cv}).")

    # ── Check uniformity ─────────────────────────────────────────────────────
    uniformity = burstiness.get("consecutive_uniformity", 0.0)
    if uniformity > 0.65:
        audit["problems"].append(
            f"High consecutive uniformity ({uniformity}). Too many adjacent sentences of "
            f"similar length — pattern AI detectors flag heavily."
        )
        repair_instructions.append(
            "Find any sequence of 3+ sentences all between 12-20 words. Split the "
            "longest into two shorter ones, or merge the two shortest."
        )

    # ── Check short sentence ratio ────────────────────────────────────────────
    short_ratio = burstiness.get("short_sentence_ratio", 0.0)
    if short_ratio < 0.10:
        audit["problems"].append(
            f"Almost no short sentences ({short_ratio:.0%}). Human writers naturally "
            f"produce short punchy sentences. AI avoids them."
        )
        repair_instructions.append(
            "Add at least 2-4 sentences of 4 words or fewer. These can be rhetorical "
            "questions, emphatic declarations, or abrupt pivots."
        )
    else:
        audit["strengths"].append(f"Healthy short sentence presence ({short_ratio:.0%}).")

    # ── Check AI clichés ─────────────────────────────────────────────────────
    cliché_density = clichés.get("ai_cliche_density_per_100w", 0.0)
    if cliché_density > 1.5:
        flagged = [p["phrase"] for p in clichés.get("flagged_phrases", [])[:5]]
        audit["problems"].append(
            f"High AI cliché density ({cliché_density}/100w). Found: {flagged}"
        )
        repair_instructions.append(
            f"Remove or rephrase these AI-marker phrases: {flagged}. "
            f"Replace with direct, specific, concrete language."
        )
    elif cliché_density == 0.0:
        audit["strengths"].append("No AI cliché phrases detected.")

    # ── Check paragraph variety ──────────────────────────────────────────────
    para_cv = metrics.get("paragraph_length_cv", 0.0)
    if para_cv < 0.20 and metrics.get("paragraph_count", 0) >= 3:
        audit["problems"].append(
            f"Uniform paragraph lengths (CV={para_cv}). All paragraphs are roughly the "
            f"same size — a hallmark of AI-generated structure."
        )
        repair_instructions.append(
            "Vary paragraph lengths: make one paragraph a single sentence, make another "
            "paragraph at least twice the average length."
        )
    elif para_cv >= 0.40:
        audit["strengths"].append(f"Good paragraph length variety (CV={para_cv}).")

    # ── Compare against source profile if available ──────────────────────────
    if style_profile:
        src_metrics = style_profile.get("human_pattern_metrics", {})
        src_burst = src_metrics.get("burstiness_cv", 0.0)
        if src_burst and cv < src_burst * 0.70:
            repair_instructions.append(
                f"Source author had burstiness CV={src_burst} but rewrite is only {cv}. "
                f"The rewrite is MORE uniform than the original. Increase sentence variety."
            )

    # ── Overall verdict ──────────────────────────────────────────────────────
    risk = metrics.get("ai_risk_score", 0.5)
    if risk >= 0.60:
        audit["verdict"] = (
            f"⚠ HIGH AI DETECTION RISK ({risk}). This text will likely be flagged. "
            f"Apply all repair instructions before publishing."
        )
    elif risk >= 0.40:
        audit["verdict"] = (
            f"⚡ MODERATE RISK ({risk}). Apply the repair instructions for best results."
        )
    else:
        audit["verdict"] = (
            f"✓ LOW RISK ({risk}). Text reads as human-like. "
            f"Minor improvements possible."
        )

    return {
        "audit": audit,
        "repair_instructions": repair_instructions,
        "full_metrics": metrics,
        "estimated_post_repair_risk": max(0.0, round(risk - 0.15 * len(repair_instructions), 4)),
    }


# ============================================================================
# DEEP STYLOMETRY (unchanged from v3.0, kept for compatibility)
# ============================================================================

_EMPTY_PROFILE = {
    "pos_ratios": {tag: 0.0 for tag in POS_TAGS},
    "function_word_freq": {w: 0.0 for w in FUNCTION_WORDS},
    "avg_dependency_depth": 0.0,
    "max_dependency_depth": 0,
    "sentence_length_distribution": {
        "mean": 0.0, "median": 0.0, "std_dev": 0.0, "min": 0, "max": 0,
    },
    "vocabulary_richness": {
        "hapax_legomena_ratio": 0.0,
        "dis_legomena_ratio": 0.0,
        "yules_k": 0.0,
        "simpsons_diversity": 0.0,
        "brunet_w": 0.0,
        "honore_r": 0.0,
    },
    "avg_word_length": 0.0,
    "contraction_rate": 0.0,
    "passive_voice_ratio": 0.0,
    "modal_verb_freq": {v: 0.0 for v in MODAL_VERBS},
    "sentence_starter_pos": {},
    "named_entity_distribution": {},
    "char_trigram_profile": {},
    "word_bigram_profile": {},
    "punctuation_density": 0.0,
    "quotation_density": 0.0,
    "question_ratio": 0.0,
    "exclamation_ratio": 0.0,
}


def extract_deep_stylometry(text):
    """
    Extract a comprehensive deep stylometry feature set using spaCy.
    Now also appends humanization metrics to the returned dict.
    """
    if not text or not text.strip():
        return dict(_EMPTY_PROFILE)

    nlp = _ensure_spacy_model()
    if nlp is None:
        return dict(_EMPTY_PROFILE)

    max_len = nlp.max_length
    if len(text) > max_len:
        nlp.max_length = len(text) + 1000

    doc = nlp(text)
    tokens = [t for t in doc if not t.is_space]
    total_tokens = len(tokens)
    if total_tokens == 0:
        return dict(_EMPTY_PROFILE)

    # POS ratios
    pos_counter = Counter(t.pos_ for t in tokens)
    pos_ratios = {
        tag: round(pos_counter.get(tag, 0) / total_tokens, 4)
        for tag in POS_TAGS
    }

    # Function word frequencies
    token_texts_lower = [t.text.lower() for t in tokens]
    fw_counter = Counter(w for w in token_texts_lower if w in set(FUNCTION_WORDS))
    function_word_freq = {
        w: round(fw_counter.get(w, 0) / total_tokens, 4)
        for w in FUNCTION_WORDS
    }

    # Dependency tree depth
    def _node_depth(node, _seen=None):
        if _seen is None:
            _seen = set()
        if node.i in _seen:
            return 1
        _seen.add(node.i)
        children = list(node.children)
        if not children:
            return 1
        return 1 + max(_node_depth(c, _seen) for c in children)

    sentence_depths = []
    sentence_lengths = []
    sentence_starter_pos_counter = Counter()
    question_count = 0
    exclamation_count = 0

    for sent in doc.sents:
        sent_tokens = [t for t in sent if not t.is_space]
        if not sent_tokens:
            continue
        sentence_lengths.append(len(sent_tokens))
        sentence_starter_pos_counter[sent_tokens[0].pos_] += 1
        sent_text = sent.text.strip()
        if sent_text.endswith("?"):
            question_count += 1
        elif sent_text.endswith("!"):
            exclamation_count += 1
        roots = [t for t in sent if t.head == t]
        if roots:
            sentence_depths.append(max(_node_depth(r) for r in roots))

    total_sentences = len(sentence_lengths) if sentence_lengths else 1

    avg_dep_depth = sum(sentence_depths) / len(sentence_depths) if sentence_depths else 0.0
    max_dep_depth = max(sentence_depths) if sentence_depths else 0

    if sentence_lengths:
        sent_len_dist = {
            "mean": round(statistics.mean(sentence_lengths), 2),
            "median": round(statistics.median(sentence_lengths), 2),
            "std_dev": round(statistics.pstdev(sentence_lengths), 2),
            "min": min(sentence_lengths),
            "max": max(sentence_lengths),
        }
    else:
        sent_len_dist = {"mean": 0.0, "median": 0.0, "std_dev": 0.0, "min": 0, "max": 0}

    words_alpha = [w for w in token_texts_lower if w.isalpha()]
    word_freq = Counter(words_alpha)
    total_alpha = len(words_alpha)
    vocab_size = len(word_freq)
    hapax_count = sum(1 for c in word_freq.values() if c == 1)

    vocabulary_richness = {
        "hapax_legomena_ratio": _hapax_legomena_ratio(word_freq, total_alpha),
        "dis_legomena_ratio": _dis_legomena_ratio(word_freq, total_alpha),
        "yules_k": _yules_k(word_freq, total_alpha),
        "simpsons_diversity": _simpsons_diversity(word_freq, total_alpha),
        "brunet_w": _brunet_w(total_alpha, vocab_size),
        "honore_r": _honore_r(total_alpha, vocab_size, hapax_count),
    }

    avg_word_length = (
        round(sum(len(w) for w in words_alpha) / total_alpha, 4)
        if total_alpha else 0.0
    )

    contraction_count = sum(
        1 for t in tokens if t.text.lower() in CONTRACTIONS or "'" in t.text
    )
    contraction_rate = round(contraction_count / total_tokens, 4)

    nsubj_count = sum(1 for t in doc if t.dep_ == "nsubj")
    nsubjpass_count = sum(1 for t in doc if t.dep_ in ("nsubjpass", "nsubj:pass"))
    passive_total = nsubj_count + nsubjpass_count
    passive_voice_ratio = (
        round(nsubjpass_count / passive_total, 4) if passive_total > 0 else 0.0
    )

    modal_counter = Counter(
        t.text.lower() for t in tokens if t.text.lower() in MODAL_VERBS
    )
    modal_verb_freq = {
        v: round(modal_counter.get(v, 0) / total_tokens, 4) for v in MODAL_VERBS
    }

    starter_total = sum(sentence_starter_pos_counter.values())
    sentence_starter_pos = {
        pos: round(cnt / starter_total, 4)
        for pos, cnt in sentence_starter_pos_counter.most_common()
    } if starter_total else {}

    ent_counter = Counter(ent.label_ for ent in doc.ents)
    ent_total = sum(ent_counter.values())
    named_entity_distribution = {
        label: round(cnt / ent_total, 4) for label, cnt in ent_counter.most_common()
    } if ent_total else {}

    char_trigram_profile = _character_ngrams(text, n=3, top_k=25)
    word_bigram_profile = _word_bigrams(words_alpha, top_k=25)

    punct_count = sum(1 for t in tokens if t.is_punct)
    punctuation_density = round(punct_count / total_sentences, 4)

    quote_count = text.count('"') + text.count('\u201c') + text.count('\u201d') + text.count("'")
    quotation_density = round(quote_count / total_tokens, 4)

    question_ratio = round(question_count / total_sentences, 4)
    exclamation_ratio = round(exclamation_count / total_sentences, 4)

    # ── Append humanization metrics (NEW in v4.0) ────────────────────────────
    humanization = compute_humanization_metrics(text)

    return {
        "pos_ratios": pos_ratios,
        "function_word_freq": function_word_freq,
        "avg_dependency_depth": round(avg_dep_depth, 4),
        "max_dependency_depth": max_dep_depth,
        "sentence_length_distribution": sent_len_dist,
        "vocabulary_richness": vocabulary_richness,
        "avg_word_length": avg_word_length,
        "contraction_rate": contraction_rate,
        "passive_voice_ratio": passive_voice_ratio,
        "modal_verb_freq": modal_verb_freq,
        "sentence_starter_pos": sentence_starter_pos,
        "named_entity_distribution": named_entity_distribution,
        "char_trigram_profile": char_trigram_profile,
        "word_bigram_profile": word_bigram_profile,
        "punctuation_density": punctuation_density,
        "quotation_density": quotation_density,
        "question_ratio": question_ratio,
        "exclamation_ratio": exclamation_ratio,
        # ── New in v4.0 ───────────────────────────────────────────────────────
        "humanization_metrics": humanization,
    }


# ---------------------------------------------------------------------------
# Style similarity / Burrows' Delta (unchanged from v3.0)
# ---------------------------------------------------------------------------

def calculate_style_similarity(profile_a, profile_b):
    """
    Calculate style similarity using cosine similarity, Burrows' Delta,
    and N-gram overlap.
    """
    if not profile_a or not profile_b:
        return {
            "cosine_similarity": 0.0,
            "burrows_delta": 0.0,
            "ngram_overlap": 0.0,
            "combined_score": 0.0,
        }

    def _vectorize(profile):
        vec = []
        pos = profile.get("pos_ratios", {})
        vec.extend(float(pos.get(tag, 0.0)) for tag in POS_TAGS)
        fw = profile.get("function_word_freq", {})
        vec.extend(float(fw.get(w, 0.0)) for w in FUNCTION_WORDS)
        vec.append(float(profile.get("avg_dependency_depth", 0.0)))
        vec.append(float(profile.get("avg_word_length", 0.0)))
        vec.append(float(profile.get("contraction_rate", 0.0)))
        vec.append(float(profile.get("passive_voice_ratio", 0.0)))
        vec.append(float(profile.get("punctuation_density", 0.0)))
        vec.append(float(profile.get("question_ratio", 0.0)))
        vec.append(float(profile.get("exclamation_ratio", 0.0)))
        vec.append(float(profile.get("quotation_density", 0.0)))
        mv = profile.get("modal_verb_freq", {})
        vec.extend(float(mv.get(v, 0.0)) for v in MODAL_VERBS)
        vr = profile.get("vocabulary_richness", {})
        vec.append(float(vr.get("hapax_legomena_ratio", 0.0)))
        vec.append(float(vr.get("yules_k", 0.0)))
        vec.append(float(vr.get("simpsons_diversity", 0.0)))
        sl = profile.get("sentence_length_distribution", {})
        vec.append(float(sl.get("mean", 0.0)))
        vec.append(float(sl.get("std_dev", 0.0)))
        return vec

    vec_a = _vectorize(profile_a)
    vec_b = _vectorize(profile_b)

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    cosine_sim = (dot / (norm_a * norm_b)) if (norm_a > 0 and norm_b > 0) else 0.0
    cosine_sim = max(0.0, min(1.0, cosine_sim))

    fw_a = profile_a.get("function_word_freq", {})
    fw_b = profile_b.get("function_word_freq", {})
    deltas = []
    for w in FUNCTION_WORDS:
        fa = float(fw_a.get(w, 0.0))
        fb = float(fw_b.get(w, 0.0))
        mean_val = (fa + fb) / 2
        std_val = math.sqrt(((fa - mean_val) ** 2 + (fb - mean_val) ** 2) / 2)
        if std_val > 0:
            deltas.append(abs((fa - fb) / std_val))
        else:
            deltas.append(0.0)
    burrows_delta_raw = sum(deltas) / len(deltas) if deltas else 0.0
    burrows_similarity = max(0.0, 1.0 - burrows_delta_raw)

    ngram_a = set(profile_a.get("char_trigram_profile", {}).keys())
    ngram_b = set(profile_b.get("char_trigram_profile", {}).keys())
    if ngram_a or ngram_b:
        ngram_overlap = len(ngram_a & ngram_b) / len(ngram_a | ngram_b)
    else:
        ngram_overlap = 0.0

    combined = (
        0.30 * cosine_sim +
        0.45 * burrows_similarity +
        0.25 * ngram_overlap
    )
    combined = round(max(0.0, min(1.0, combined)), 4)

    return {
        "cosine_similarity": round(cosine_sim, 4),
        "burrows_delta": round(burrows_similarity, 4),
        "ngram_overlap": round(ngram_overlap, 4),
        "combined_score": combined,
    }


# ---------------------------------------------------------------------------
# Readability & text statistics (unchanged from v3.0)
# ---------------------------------------------------------------------------

def calculate_readability_metrics(text):
    if not text or not text.strip():
        return {}
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = text.split()
    if not words or not sentences:
        return {}

    syllables = sum([count_syllables(word) for word in words])
    avg_sentence_length = len(words) / len(sentences)
    avg_syllables_per_word = syllables / len(words)
    flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    fk_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
    avg_letters_per_100_words = (sum(len(word) for word in words) / len(words)) * 100
    avg_sentences_per_100_words = (len(sentences) / len(words)) * 100
    coleman_liau = (0.0588 * avg_letters_per_100_words) - (0.296 * avg_sentences_per_100_words) - 15.8

    return {
        "flesch_reading_ease": round(flesch_score, 2),
        "flesch_kincaid_grade": round(fk_grade, 2),
        "coleman_liau_index": round(coleman_liau, 2),
        "avg_sentence_length": round(avg_sentence_length, 2),
        "avg_syllables_per_word": round(avg_syllables_per_word, 2),
    }


def analyze_text_statistics(text):
    empty_result = {
        'word_count': 0, 'sentence_count': 0, 'paragraph_count': 0,
        'character_count': 0, 'avg_words_per_sentence': 0,
        'avg_sentences_per_paragraph': 0, 'avg_word_length': 0,
        'word_frequency': {}, 'punctuation_counts': {}, 'sentence_types': {},
        'unique_words': 0, 'lexical_diversity': 0,
        'vocabulary_richness': {
            'hapax_legomena_ratio': 0.0, 'dis_legomena_ratio': 0.0,
            'yules_k': 0.0, 'simpsons_diversity': 0.0,
        },
        'humanization_metrics': {},
    }
    if not text or not text.strip():
        return dict(empty_result)

    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    if not words:
        empty_result.update({
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'character_count': len(text),
        })
        return empty_result

    word_freq = Counter(word.lower().strip('.,!?";:()[]{}') for word in words)
    punctuation_counts = {
        'commas': text.count(','), 'periods': text.count('.'),
        'semicolons': text.count(';'), 'colons': text.count(':'),
        'exclamations': text.count('!'), 'questions': text.count('?'),
        'dashes': text.count('—') + text.count('--'),
        'parentheses': text.count('('),
    }
    sentence_types = {
        'declarative': len([s for s in sentences if s.strip().endswith('.')]),
        'interrogative': len([s for s in sentences if s.strip().endswith('?')]),
        'exclamatory': len([s for s in sentences if s.strip().endswith('!')]),
        'imperative': 0,
    }

    unique_words_count = len(set(word.lower() for word in words))
    alpha_words = [w.lower().strip('.,!?";:()[]{}') for w in words if w.strip('.,!?";:()[]{}').isalpha()]
    alpha_freq = Counter(alpha_words)
    total_alpha = len(alpha_words)
    avg_word_length = round(sum(len(w) for w in alpha_words) / total_alpha, 2) if total_alpha else 0

    # ── NEW: humanization metrics appended ───────────────────────────────────
    humanization = compute_humanization_metrics(text)

    return {
        'word_count': len(words),
        'sentence_count': len(sentences),
        'paragraph_count': len(paragraphs),
        'character_count': len(text),
        'avg_words_per_sentence': round(len(words) / len(sentences), 2) if sentences else 0,
        'avg_sentences_per_paragraph': round(len(sentences) / len(paragraphs), 2) if paragraphs else 0,
        'avg_word_length': avg_word_length,
        'word_frequency': dict(word_freq.most_common(20)),
        'punctuation_counts': punctuation_counts,
        'sentence_types': sentence_types,
        'unique_words': unique_words_count,
        'lexical_diversity': round(unique_words_count / len(words), 3) if words else 0,
        'vocabulary_richness': {
            'hapax_legomena_ratio': _hapax_legomena_ratio(alpha_freq, total_alpha),
            'dis_legomena_ratio': _dis_legomena_ratio(alpha_freq, total_alpha),
            'yules_k': _yules_k(alpha_freq, total_alpha),
            'simpsons_diversity': _simpsons_diversity(alpha_freq, total_alpha),
        },
        'humanization_metrics': humanization,
    }