"""
Metrics calculation module for Style Transfer AI.
Handles readability metrics, statistical text analysis, and advanced
computational stylometry features (Burrows' Delta, n-grams, vocabulary
richness, syntactic complexity, etc.).
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

# Expanded function word list (top 50 most common English function words).
# Function-word frequencies are among the strongest authorship signals.
FUNCTION_WORDS = [
    "the", "and", "to", "of", "in", "a", "that", "it", "is", "was",
    "for", "on", "with", "as", "at", "by", "this", "an", "be", "not",
    "but", "from", "or", "have", "had", "has", "its", "are", "were", "been",
    "which", "their", "if", "will", "each", "about", "how", "up", "out", "them",
    "then", "she", "he", "his", "her", "would", "there", "what", "so", "can",
]

# All major Universal POS tags tracked
POS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
    "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
    "SCONJ", "SYM", "VERB", "X",
]

# Common contractions for contraction-rate calculation
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

# Modal verbs (epistemic / deontic markers)
MODAL_VERBS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}

# ---------------------------------------------------------------------------
# spaCy bootstrap
# ---------------------------------------------------------------------------

def _ensure_spacy_model():
    """Ensure spaCy and the English model are available, returning a loaded nlp or None."""
    try:
        import spacy
    except Exception:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "spacy"],
                check=True
            )
            import spacy
        except Exception:
            print("spaCy install failed. Please run: python -m pip install spacy")
            return None

    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        try:
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                check=True
            )
            return spacy.load("en_core_web_sm")
        except Exception:
            print("spaCy model download failed. Please run: python -m spacy download en_core_web_sm")
            return None


# ---------------------------------------------------------------------------
# Vocabulary richness helpers
# ---------------------------------------------------------------------------

def _hapax_legomena_ratio(word_freq: Counter, total_words: int) -> float:
    """Words appearing exactly once / total words (strong authorship marker)."""
    if total_words == 0:
        return 0.0
    hapax = sum(1 for count in word_freq.values() if count == 1)
    return round(hapax / total_words, 4)


def _dis_legomena_ratio(word_freq: Counter, total_words: int) -> float:
    """Words appearing exactly twice / total words."""
    if total_words == 0:
        return 0.0
    dis = sum(1 for count in word_freq.values() if count == 2)
    return round(dis / total_words, 4)


def _yules_k(word_freq: Counter, total_words: int) -> float:
    """Yule's K – vocabulary richness independent of text length.
    Lower K → richer vocabulary.  Typical prose: 80-200.
    """
    if total_words == 0:
        return 0.0
    freq_spectrum = Counter(word_freq.values())
    m2 = sum(i * i * v_i for i, v_i in freq_spectrum.items())
    if total_words == 0:
        return 0.0
    k = 10_000 * (m2 - total_words) / (total_words * total_words) if total_words > 1 else 0.0
    return round(k, 4)


def _simpsons_diversity(word_freq: Counter, total_words: int) -> float:
    """Simpson's Diversity Index – probability two random words differ."""
    if total_words <= 1:
        return 0.0
    s = sum(n * (n - 1) for n in word_freq.values())
    d = 1 - s / (total_words * (total_words - 1))
    return round(d, 4)


def _brunet_w(total_words: int, vocab_size: int) -> float:
    """Brunet's W – another length-independent vocabulary measure.
    Typical values around 20; lower = richer vocabulary.
    """
    if total_words <= 0 or vocab_size <= 0:
        return 0.0
    return round(total_words ** (vocab_size ** -0.172), 4)


def _honore_r(total_words: int, vocab_size: int, hapax_count: int) -> float:
    """Honoré's R statistic – rewards large vocabularies with many hapax legomena."""
    if total_words == 0 or vocab_size == 0:
        return 0.0
    if hapax_count == vocab_size:
        hapax_count = vocab_size - 1  # prevent division by zero
    if vocab_size - hapax_count == 0:
        return 0.0
    r = 100 * math.log(total_words) / (1 - hapax_count / vocab_size)
    return round(r, 4)


# ---------------------------------------------------------------------------
# N-gram helpers
# ---------------------------------------------------------------------------

def _character_ngrams(text: str, n: int = 3, top_k: int = 25) -> dict:
    """Extract character n-gram frequency profile (case-insensitive).
    Character n-grams are among the *best* features for authorship attribution.
    """
    cleaned = text.lower()
    ngrams = Counter()
    for i in range(len(cleaned) - n + 1):
        ngrams[cleaned[i:i + n]] += 1
    total = sum(ngrams.values())
    if total == 0:
        return {}
    return {ng: round(cnt / total, 6) for ng, cnt in ngrams.most_common(top_k)}


def _word_bigrams(tokens_lower: list, top_k: int = 25) -> dict:
    """Extract word bigram frequency profile."""
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


# ---------------------------------------------------------------------------
# Main deep stylometry extraction
# ---------------------------------------------------------------------------

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

    Features extracted:
    ─ Full POS tag distribution (17 Universal POS tags)
    ─ 50 function-word frequencies
    ─ Dependency tree depth (avg & max)
    ─ Sentence length distribution (mean / median / std / min / max)
    ─ Vocabulary richness (hapax, Yule's K, Simpson's D, Brunet's W, Honoré's R)
    ─ Average word length
    ─ Contraction rate
    ─ Passive voice ratio
    ─ Modal verb frequency
    ─ Sentence-starter POS distribution
    ─ Named-entity type distribution
    ─ Character trigram profile (top 25)
    ─ Word bigram profile (top 25)
    ─ Punctuation / quotation density
    ─ Question & exclamation ratios
    """
    if not text or not text.strip():
        return dict(_EMPTY_PROFILE)

    nlp = _ensure_spacy_model()
    if nlp is None:
        return dict(_EMPTY_PROFILE)

    # spaCy has a default max_length; process in chunks if needed
    max_len = nlp.max_length
    if len(text) > max_len:
        nlp.max_length = len(text) + 1000

    doc = nlp(text)
    tokens = [t for t in doc if not t.is_space]
    total_tokens = len(tokens)
    if total_tokens == 0:
        return dict(_EMPTY_PROFILE)

    # --- POS ratios (all 17 tags) ---
    pos_counter = Counter(t.pos_ for t in tokens)
    pos_ratios = {
        tag: round(pos_counter.get(tag, 0) / total_tokens, 4)
        for tag in POS_TAGS
    }

    # --- Function word frequencies ---
    token_texts_lower = [t.text.lower() for t in tokens]
    fw_counter = Counter(w for w in token_texts_lower if w in set(FUNCTION_WORDS))
    function_word_freq = {
        w: round(fw_counter.get(w, 0) / total_tokens, 4)
        for w in FUNCTION_WORDS
    }

    # --- Dependency tree depth ---
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

        # Sentence starter POS
        sentence_starter_pos_counter[sent_tokens[0].pos_] += 1

        # Question / exclamation detection
        sent_text = sent.text.strip()
        if sent_text.endswith("?"):
            question_count += 1
        elif sent_text.endswith("!"):
            exclamation_count += 1

        # Depth
        roots = [t for t in sent if t.head == t]
        if roots:
            sentence_depths.append(max(_node_depth(r) for r in roots))

    total_sentences = len(sentence_lengths) if sentence_lengths else 1

    avg_dep_depth = (
        sum(sentence_depths) / len(sentence_depths) if sentence_depths else 0.0
    )
    max_dep_depth = max(sentence_depths) if sentence_depths else 0

    # --- Sentence length distribution ---
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

    # --- Vocabulary richness ---
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

    # --- Average word length ---
    avg_word_length = (
        round(sum(len(w) for w in words_alpha) / total_alpha, 4)
        if total_alpha else 0.0
    )

    # --- Contraction rate ---
    contraction_count = sum(
        1 for t in tokens if t.text.lower() in CONTRACTIONS or "'" in t.text
    )
    contraction_rate = round(contraction_count / total_tokens, 4)

    # --- Passive voice ratio (nsubjpass / nsubj+nsubjpass) ---
    nsubj_count = sum(1 for t in doc if t.dep_ == "nsubj")
    nsubjpass_count = sum(1 for t in doc if t.dep_ in ("nsubjpass", "nsubj:pass"))
    passive_total = nsubj_count + nsubjpass_count
    passive_voice_ratio = (
        round(nsubjpass_count / passive_total, 4) if passive_total > 0 else 0.0
    )

    # --- Modal verb frequency ---
    modal_counter = Counter(
        t.text.lower() for t in tokens if t.text.lower() in MODAL_VERBS
    )
    modal_verb_freq = {
        v: round(modal_counter.get(v, 0) / total_tokens, 4) for v in MODAL_VERBS
    }

    # --- Sentence-starter POS distribution ---
    starter_total = sum(sentence_starter_pos_counter.values())
    sentence_starter_pos = {
        pos: round(cnt / starter_total, 4)
        for pos, cnt in sentence_starter_pos_counter.most_common()
    } if starter_total else {}

    # --- Named entity distribution ---
    ent_counter = Counter(ent.label_ for ent in doc.ents)
    ent_total = sum(ent_counter.values())
    named_entity_distribution = {
        label: round(cnt / ent_total, 4) for label, cnt in ent_counter.most_common()
    } if ent_total else {}

    # --- Character trigram profile ---
    char_trigram_profile = _character_ngrams(text, n=3, top_k=25)

    # --- Word bigram profile ---
    word_bigram_profile = _word_bigrams(words_alpha, top_k=25)

    # --- Punctuation density (punctuation tokens per sentence) ---
    punct_count = sum(1 for t in tokens if t.is_punct)
    punctuation_density = round(punct_count / total_sentences, 4)

    # --- Quotation density ---
    quote_count = text.count('"') + text.count('"') + text.count('"') + text.count("'")
    quotation_density = round(quote_count / total_tokens, 4)

    # --- Question / exclamation ratios ---
    question_ratio = round(question_count / total_sentences, 4)
    exclamation_ratio = round(exclamation_count / total_sentences, 4)

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
    }


# ---------------------------------------------------------------------------
# Similarity / Burrows' Delta
# ---------------------------------------------------------------------------

def calculate_style_similarity(profile_a, profile_b):
    """
    Calculate style similarity using multiple complementary methods:

    1. **Cosine similarity** on the full feature vector (POS + function words +
       scalar features).
    2. **Burrows' Delta** – the gold-standard distance in computational
       stylometry, using z-scores of the most-frequent-word frequencies.
    3. **N-gram overlap** (Jaccard) on character trigram keys.

    Returns a dict with individual scores and a combined score (0.0–1.0,
    where 1.0 = identical style).
    """
    if not profile_a or not profile_b:
        return {
            "cosine_similarity": 0.0,
            "burrows_delta": 0.0,
            "ngram_overlap": 0.0,
            "combined_score": 0.0,
        }

    # ------ Feature vector for cosine similarity ------
    def _vectorize(profile):
        vec = []
        # POS ratios
        pos = profile.get("pos_ratios", {})
        vec.extend(float(pos.get(tag, 0.0)) for tag in POS_TAGS)
        # Function word frequencies
        fw = profile.get("function_word_freq", {})
        vec.extend(float(fw.get(w, 0.0)) for w in FUNCTION_WORDS)
        # Scalar features
        vec.append(float(profile.get("avg_dependency_depth", 0.0)))
        vec.append(float(profile.get("avg_word_length", 0.0)))
        vec.append(float(profile.get("contraction_rate", 0.0)))
        vec.append(float(profile.get("passive_voice_ratio", 0.0)))
        vec.append(float(profile.get("punctuation_density", 0.0)))
        vec.append(float(profile.get("question_ratio", 0.0)))
        vec.append(float(profile.get("exclamation_ratio", 0.0)))
        vec.append(float(profile.get("quotation_density", 0.0)))
        # Modal verb frequencies
        mv = profile.get("modal_verb_freq", {})
        vec.extend(float(mv.get(v, 0.0)) for v in MODAL_VERBS)
        # Vocabulary richness scalars
        vr = profile.get("vocabulary_richness", {})
        vec.append(float(vr.get("hapax_legomena_ratio", 0.0)))
        vec.append(float(vr.get("yules_k", 0.0)))
        vec.append(float(vr.get("simpsons_diversity", 0.0)))
        # Sentence length distribution
        sl = profile.get("sentence_length_distribution", {})
        vec.append(float(sl.get("mean", 0.0)))
        vec.append(float(sl.get("std_dev", 0.0)))
        return vec

    vec_a = _vectorize(profile_a)
    vec_b = _vectorize(profile_b)

    # Cosine similarity
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    cosine_sim = (dot / (norm_a * norm_b)) if (norm_a > 0 and norm_b > 0) else 0.0
    cosine_sim = max(0.0, min(1.0, cosine_sim))

    # ------ Burrows' Delta ------
    # Uses function word frequencies; delta = mean |z_a - z_b| across features.
    fw_a = profile_a.get("function_word_freq", {})
    fw_b = profile_b.get("function_word_freq", {})
    # Compute z-scores relative to the two-text "corpus"
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
    # Convert to similarity (0-1); delta of 0 → similarity 1
    burrows_similarity = max(0.0, 1.0 - burrows_delta_raw)

    # ------ N-gram (character trigram) overlap (Jaccard) ------
    ngram_a = set(profile_a.get("char_trigram_profile", {}).keys())
    ngram_b = set(profile_b.get("char_trigram_profile", {}).keys())
    if ngram_a or ngram_b:
        ngram_overlap = len(ngram_a & ngram_b) / len(ngram_a | ngram_b)
    else:
        ngram_overlap = 0.0
    ngram_overlap = round(ngram_overlap, 4)

    # ------ Combined score (weighted average) ------
    # Weights chosen to emphasise Burrows' Delta (most validated in literature)
    combined = (
        0.30 * cosine_sim
        + 0.45 * burrows_similarity
        + 0.25 * ngram_overlap
    )
    combined = round(max(0.0, min(1.0, combined)), 4)

    return {
        "cosine_similarity": round(cosine_sim, 4),
        "burrows_delta": round(burrows_similarity, 4),
        "ngram_overlap": ngram_overlap,
        "combined_score": combined,
    }


def calculate_readability_metrics(text):
    """Calculate various readability and complexity metrics."""
    # Input validation
    if not text or not text.strip():
        return {}
    
    # Basic text statistics
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = text.split()
    
    # Additional validation
    if not words or not sentences:
        return {}
    
    syllables = sum([count_syllables(word) for word in words])
    
    # Readability scores
    avg_sentence_length = len(words) / len(sentences)
    avg_syllables_per_word = syllables / len(words)
    
    # Flesch Reading Ease
    flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    
    # Flesch-Kincaid Grade Level
    fk_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
    
    # Coleman-Liau Index
    avg_letters_per_100_words = (sum(len(word) for word in words) / len(words)) * 100
    avg_sentences_per_100_words = (len(sentences) / len(words)) * 100
    coleman_liau = (0.0588 * avg_letters_per_100_words) - (0.296 * avg_sentences_per_100_words) - 15.8
    
    return {
        "flesch_reading_ease": round(flesch_score, 2),
        "flesch_kincaid_grade": round(fk_grade, 2),
        "coleman_liau_index": round(coleman_liau, 2),
        "avg_sentence_length": round(avg_sentence_length, 2),
        "avg_syllables_per_word": round(avg_syllables_per_word, 2)
    }


def analyze_text_statistics(text):
    """Perform detailed statistical analysis of text with vocabulary richness."""
    # Input validation
    empty_result = {
        'word_count': 0,
        'sentence_count': 0,
        'paragraph_count': 0,
        'character_count': 0,
        'avg_words_per_sentence': 0,
        'avg_sentences_per_paragraph': 0,
        'avg_word_length': 0,
        'word_frequency': {},
        'punctuation_counts': {},
        'sentence_types': {},
        'unique_words': 0,
        'lexical_diversity': 0,
        'vocabulary_richness': {
            'hapax_legomena_ratio': 0.0,
            'dis_legomena_ratio': 0.0,
            'yules_k': 0.0,
            'simpsons_diversity': 0.0,
        },
    }
    if not text or not text.strip():
        return dict(empty_result)
    
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    if not words:
        empty_result['sentence_count'] = len(sentences)
        empty_result['paragraph_count'] = len(paragraphs)
        empty_result['character_count'] = len(text)
        return empty_result
    
    # Word frequency analysis
    word_freq = Counter(word.lower().strip('.,!?";:()[]{}') for word in words)
    
    # Punctuation analysis
    punctuation_counts = {
        'commas': text.count(','),
        'periods': text.count('.'),
        'semicolons': text.count(';'),
        'colons': text.count(':'),
        'exclamations': text.count('!'),
        'questions': text.count('?'),
        'dashes': text.count('—') + text.count('--'),
        'parentheses': text.count('(')
    }
    
    # Sentence type analysis
    sentence_types = {
        'declarative': len([s for s in sentences if s.strip().endswith('.')]),
        'interrogative': len([s for s in sentences if s.strip().endswith('?')]),
        'exclamatory': len([s for s in sentences if s.strip().endswith('!')]),
        'imperative': 0
    }
    
    # Safe calculations with validation
    unique_words_count = len(set(word.lower() for word in words))
    alpha_words = [w.lower().strip('.,!?";:()[]{}') for w in words if w.strip('.,!?";:()[]{}').isalpha()]
    alpha_freq = Counter(alpha_words)
    total_alpha = len(alpha_words)
    
    avg_word_length = round(
        sum(len(w) for w in alpha_words) / total_alpha, 2
    ) if total_alpha else 0
    
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
    }