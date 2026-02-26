"""
Analogy Engine for Style Transfer AI.

Provides the ``AnalogyInjector`` service that detects conceptually dense
passages and generates contextual analogies to aid comprehension.

The engine can run **standalone** or as a post-processing layer on top of
any style-transfer / analysis result (hybrid mode).

Key design rule
---------------
When both a primary style transfer AND analogy augmentation are active,
the analogies are injected in clearly delimited ``[Cognitive Note: …]``
blocks so they *never* corrupt the primary style output.
"""

from __future__ import annotations

import re
import math
from typing import Dict, List, Optional, Tuple

from ..config.settings import (
    ANALOGY_DOMAINS,
    CONCEPTUAL_DENSITY_THRESHOLD,
    DEFAULT_ANALOGY_DOMAIN,
)
from ..utils.text_processing import count_syllables


# ---------------------------------------------------------------------------
# Conceptual density detection (pure computation — no LLM needed)
# ---------------------------------------------------------------------------

def detect_conceptual_density(text: str) -> Dict:
    """Scan *text* and return per-sentence density scores plus an overall score.

    Density is a heuristic composite of:
    * **Lexical density** — ratio of content words (nouns, verbs, adjectives,
      adverbs approximation) to total words.
    * **Syllabic complexity** — average syllables per word.
    * **Information density** — unique-word ratio within the sentence.
    * **Sentence length factor** — longer sentences carry more cognitive load.

    Returns
    -------
    dict
        ``overall_density``  : float 0-1 (mean across sentences)
        ``sentence_scores``  : list[dict] with ``text``, ``density``, ``factors``
        ``high_density_count``: int — sentences above the configured threshold
    """
    if not text or not text.strip():
        return {
            "overall_density": 0.0,
            "sentence_scores": [],
            "high_density_count": 0,
        }

    sentences = _split_sentences(text)
    if not sentences:
        return {
            "overall_density": 0.0,
            "sentence_scores": [],
            "high_density_count": 0,
        }

    scored: List[Dict] = []
    for sent in sentences:
        score, factors = _score_sentence(sent)
        scored.append({
            "text": sent,
            "density": round(score, 4),
            "factors": factors,
        })

    overall = sum(s["density"] for s in scored) / len(scored) if scored else 0.0
    high_count = sum(
        1 for s in scored if s["density"] >= CONCEPTUAL_DENSITY_THRESHOLD
    )

    return {
        "overall_density": round(overall, 4),
        "sentence_scores": scored,
        "high_density_count": high_count,
    }


# ---------------------------------------------------------------------------
# Analogy injection (LLM-powered)
# ---------------------------------------------------------------------------

class AnalogyInjector:
    """Generate and inject contextual analogies for dense text passages.

    Parameters
    ----------
    domain : str
        One of the keys in ``ANALOGY_DOMAINS`` (e.g. ``"sports"``).
    threshold : float
        Override for ``CONCEPTUAL_DENSITY_THRESHOLD``.
    """

    def __init__(
        self,
        domain: str = DEFAULT_ANALOGY_DOMAIN,
        threshold: float = CONCEPTUAL_DENSITY_THRESHOLD,
    ):
        if domain not in ANALOGY_DOMAINS:
            raise ValueError(
                f"Unknown analogy domain '{domain}'. "
                f"Choose from: {list(ANALOGY_DOMAINS.keys())}"
            )
        self.domain = domain
        self.domain_info = ANALOGY_DOMAINS[domain]
        self.threshold = threshold

    # ----- public API -----

    def augment_text(
        self,
        text: str,
        *,
        use_local: bool = True,
        model_name: Optional[str] = None,
        api_type: Optional[str] = None,
        api_client=None,
    ) -> Dict:
        """Analyse *text* for density and inject analogies where needed.

        Returns a dict with:
        * ``augmented_text``   — the original text with ``[Cognitive Note: …]``
          blocks inserted after each high-density sentence.
        * ``density_report``   — full output of ``detect_conceptual_density``.
        * ``analogies``        — list of generated analogies with metadata.
        * ``analogy_count``    — how many analogies were injected.
        """
        density_report = detect_conceptual_density(text)
        dense_sentences = [
            s for s in density_report["sentence_scores"]
            if s["density"] >= self.threshold
        ]

        if not dense_sentences:
            return {
                "augmented_text": text,
                "density_report": density_report,
                "analogies": [],
                "analogy_count": 0,
            }

        # Build a single batched prompt for all dense sentences so we only
        # make one LLM call.
        prompt = self._build_analogy_prompt(dense_sentences)

        raw_response = self._call_model(
            prompt,
            use_local=use_local,
            model_name=model_name,
            api_type=api_type,
            api_client=api_client,
        )

        analogies = self._parse_analogy_response(raw_response, dense_sentences)

        augmented = self._inject_analogies(text, analogies)

        return {
            "augmented_text": augmented,
            "density_report": density_report,
            "analogies": analogies,
            "analogy_count": len(analogies),
        }

    def augment_analysis_result(
        self,
        analysis_text: str,
        *,
        use_local: bool = True,
        model_name: Optional[str] = None,
        api_type: Optional[str] = None,
        api_client=None,
    ) -> Dict:
        """Augment an *existing* analysis / style-transfer result.

        This is the **hybrid-mode** entry point.  The primary style output is
        preserved verbatim; analogies are appended in a clearly separated
        ``COGNITIVE BRIDGING NOTES`` section at the end.
        """
        density_report = detect_conceptual_density(analysis_text)
        dense_sentences = [
            s for s in density_report["sentence_scores"]
            if s["density"] >= self.threshold
        ]

        if not dense_sentences:
            return {
                "augmented_text": analysis_text,
                "density_report": density_report,
                "analogies": [],
                "analogy_count": 0,
            }

        prompt = self._build_analogy_prompt(dense_sentences)
        raw_response = self._call_model(
            prompt,
            use_local=use_local,
            model_name=model_name,
            api_type=api_type,
            api_client=api_client,
        )
        analogies = self._parse_analogy_response(raw_response, dense_sentences)

        # Append a separated section so the primary output stays intact.
        notes_section = self._format_cognitive_notes(analogies)
        augmented = analysis_text.rstrip() + "\n\n" + notes_section

        return {
            "augmented_text": augmented,
            "density_report": density_report,
            "analogies": analogies,
            "analogy_count": len(analogies),
        }

    # ----- prompt building -----

    def _build_analogy_prompt(self, dense_sentences: List[Dict]) -> str:
        """Create the LLM prompt for a batch of dense sentences."""
        domain_desc = self.domain_info["description"]
        numbered = "\n".join(
            f"{i+1}. \"{s['text']}\""
            for i, s in enumerate(dense_sentences)
        )
        return (
            "You are an educational analogy generation agent.\n"
            "Your task is to analyze the following academic content and identify "
            "concepts that may be abstract, technical, or difficult to understand.\n\n"
            "Rules:\n"
            "- Generate a simple, accurate real-world analogy for each concept.\n"
            "- Preserve the original meaning — do not distort technical correctness.\n"
            "- Keep explanations concise and structured.\n"
            "- Add a practical example when it helps understanding.\n"
            "- Do NOT replace the original explanation; supplement it.\n"
            "- Your goal is conceptual clarity, not literary storytelling.\n\n"
            f"Domain preference: {domain_desc}\n\n"
            "PASSAGES TO EXPLAIN:\n"
            f"{numbered}\n\n"
            "FORMAT your response as a numbered list matching the input numbers.\n"
            "For each entry provide:\n"
            "  - Concept: the key idea identified\n"
            "  - Analogy: a simple real-world analogy (1-2 sentences)\n"
            "  - Example: a brief practical example (optional, only if helpful)\n\n"
            "Example format:\n"
            "1. Concept: [key idea]\n"
            "   Analogy: [real-world analogy]\n"
            "   Example: [practical example]\n"
            "2. Concept: [key idea]\n"
            "   Analogy: [real-world analogy]\n"
            "...\n"
        )

    # ----- model dispatch -----

    def _call_model(
        self,
        prompt: str,
        *,
        use_local: bool,
        model_name: Optional[str],
        api_type: Optional[str],
        api_client,
    ) -> str:
        """Route the prompt to the correct model backend (lazy imports)."""
        from ..models.ollama_client import analyze_with_ollama
        from ..models.openai_client import analyze_with_openai
        from ..models.gemini_client import analyze_with_gemini

        if use_local:
            if not model_name:
                raise ValueError("model_name required for local Ollama inference")
            return analyze_with_ollama(prompt, model_name, "enhanced")
        elif api_type == "openai":
            return analyze_with_openai(api_client, prompt)
        elif api_type == "gemini":
            return analyze_with_gemini(api_client, prompt)
        else:
            return "Error: Unknown API type for analogy generation"

    # ----- response parsing -----

    @staticmethod
    def _parse_analogy_response(
        raw: str, dense_sentences: List[Dict]
    ) -> List[Dict]:
        """Parse the numbered-list LLM response into structured analogies.

        Supports two formats:
        - Simple:     ``1. <text>``
        - Structured: ``1. Concept: … / Analogy: … / Example: …``
        """
        analogies: List[Dict] = []

        # Split into numbered blocks ("1. …", "2. …", etc.)
        block_pattern = re.compile(r"(?:^|\n)\s*(\d+)\.\s*", re.MULTILINE)
        splits = block_pattern.split(raw)
        # splits looks like ["", "1", "block1 text", "2", "block2 text", …]
        block_map: Dict[int, str] = {}
        idx = 1
        while idx < len(splits) - 1:
            try:
                num = int(splits[idx])
                block_map[num] = splits[idx + 1].strip()
            except (ValueError, IndexError):
                pass
            idx += 2

        # Sub-field extraction helpers
        def _extract_field(block: str, label: str) -> str:
            m = re.search(
                rf"(?:^|\n)\s*{label}\s*:\s*(.+?)(?=\n\s*(?:Concept|Analogy|Example)\s*:|$)",
                block,
                re.IGNORECASE | re.DOTALL,
            )
            return m.group(1).strip() if m else ""

        for i, sent_info in enumerate(dense_sentences):
            num = i + 1
            block = block_map.get(num, "")
            if not block:
                continue

            concept = _extract_field(block, "Concept")
            analogy_text = _extract_field(block, "Analogy")
            example = _extract_field(block, "Example")

            # Fallback: treat whole block as analogy if no sub-fields found
            if not analogy_text:
                analogy_text = block

            entry: Dict = {
                "source_sentence": sent_info["text"],
                "density_score": sent_info["density"],
                "analogy": analogy_text,
            }
            if concept:
                entry["concept"] = concept
            if example:
                entry["example"] = example

            analogies.append(entry)

        return analogies

    # ----- injection helpers -----

    @staticmethod
    def _inject_analogies(text: str, analogies: List[Dict]) -> str:
        """Insert ``[Cognitive Note: …]`` blocks after each source sentence."""
        result = text
        for item in reversed(analogies):  # reverse to preserve offsets
            src = item["source_sentence"]
            parts = []
            if item.get("concept"):
                parts.append(f"Concept: {item['concept']}")
            parts.append(f"Analogy: {item['analogy']}")
            if item.get("example"):
                parts.append(f"Example: {item['example']}")
            note = "\n[Cognitive Note: " + " | ".join(parts) + "]\n"
            # Insert right after the source sentence
            idx = result.find(src)
            if idx != -1:
                end = idx + len(src)
                result = result[:end] + note + result[end:]
        return result

    @staticmethod
    def _format_cognitive_notes(analogies: List[Dict]) -> str:
        """Format a standalone Cognitive Bridging Notes section."""
        lines = [
            "=" * 60,
            "COGNITIVE BRIDGING NOTES",
            "=" * 60,
            "",
        ]
        for i, item in enumerate(analogies, 1):
            preview = (
                item["source_sentence"][:80] + "..."
                if len(item["source_sentence"]) > 80
                else item["source_sentence"]
            )
            lines.append(f"{i}. Dense passage (score {item['density_score']:.2f}):")
            lines.append(f"   \"{preview}\"")
            if item.get("concept"):
                lines.append(f"   Concept:  {item['concept']}")
            lines.append(f"   Analogy:  {item['analogy']}")
            if item.get("example"):
                lines.append(f"   Example:  {item['example']}")
            lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

# Approximate "content word" detection without spaCy — we exclude the most
# common English function words.
_FUNCTION_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "is", "are", "was", "were",
    "be", "been", "being", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "as", "into", "through", "during", "before", "after",
    "above", "below", "between", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "can", "will", "just", "should", "now", "do", "did",
    "does", "has", "have", "had", "it", "its", "this", "that", "these",
    "those", "i", "me", "my", "we", "our", "you", "your", "he", "him",
    "his", "she", "her", "they", "them", "their", "what", "which", "who",
    "whom", "whose", "would", "could", "shall", "may", "might", "must",
})


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences (simple regex-based)."""
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


def _score_sentence(sentence: str) -> Tuple[float, Dict]:
    """Return a 0-1 density score for a single sentence plus factor breakdown."""
    words = sentence.split()
    if not words:
        return 0.0, {}

    word_count = len(words)
    lower_words = [w.lower().strip(".,!?;:'\"()[]{}") for w in words]
    alpha_words = [w for w in lower_words if w.isalpha()]

    # --- Factor 1: Lexical density (content-word ratio) ---
    content_words = [w for w in alpha_words if w not in _FUNCTION_WORDS]
    lexical_density = len(content_words) / len(alpha_words) if alpha_words else 0.0

    # --- Factor 2: Syllabic complexity ---
    total_syllables = sum(count_syllables(w) for w in alpha_words) if alpha_words else 0
    avg_syllables = total_syllables / len(alpha_words) if alpha_words else 1.0
    # Normalise: 1 syll → 0, 4+ syll → 1
    syllable_factor = min(1.0, max(0.0, (avg_syllables - 1.0) / 3.0))

    # --- Factor 3: Information density (unique ratio within sentence) ---
    unique_ratio = len(set(alpha_words)) / len(alpha_words) if alpha_words else 0.0

    # --- Factor 4: Length factor ---
    # Short sentences are rarely "dense"; long ones carry more load.
    # Normalise: <=5 words → 0, >=30 words → 1
    length_factor = min(1.0, max(0.0, (word_count - 5) / 25.0))

    # Weighted composite
    density = (
        0.35 * lexical_density
        + 0.25 * syllable_factor
        + 0.20 * unique_ratio
        + 0.20 * length_factor
    )

    factors = {
        "lexical_density": round(lexical_density, 4),
        "syllable_complexity": round(syllable_factor, 4),
        "information_density": round(unique_ratio, 4),
        "length_factor": round(length_factor, 4),
    }

    return density, factors
