"""
Analogy Engine for Style Transfer AI  —  v2 (two-stage pipeline)
=================================================================

Stage 1 — Content Expansion
    The raw user input (topic / short phrase / paragraph) is first expanded
    into a thorough, factual explanation by the LLM.  This ensures the
    analogy stage always has rich material to work with, even when the user
    types only "solar flare".

Stage 2 — Domain Analogy Transformation
    The expanded content is re-written as a comprehensive, domain-specific
    analogy.  This is a *full transformation*, not just a one-line cognitive
    note appended to the original text.

The old per-sentence injection mode (``[Cognitive Note: …]`` blocks) is
preserved as an optional ``inject_mode`` for hybrid / post-processing use,
but it is no longer the default output.

Key design rules
----------------
* One LLM call per stage — no silent retries that confuse the user.
* Domain analogies must be narrative and thorough (≥ 3 paragraphs).
* The original expanded explanation is always returned alongside the analogy
  so callers can display / log both.
"""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, Tuple

from ..config.settings import (
    ANALOGY_DOMAINS,
    CONCEPTUAL_DENSITY_THRESHOLD,
    DEFAULT_ANALOGY_DOMAIN,
)
from ..utils.text_processing import count_syllables


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

DOMAIN_INSTRUCTIONS: Dict[str, str] = {
    "sports": (
        "Map every concept to elements of a sport or athletic competition "
        "(teams, players, rules, training, scoring, strategy, tournaments). "
        "Use vivid play-by-play language."
    ),
    "gaming": (
        "Map every concept to video-game mechanics "
        "(quests, levels, stats, power-ups, bosses, respawns, inventory, "
        "multiplayer, game engines). Write as if guiding a player through "
        "a game world."
    ),
    "cooking": (
        "Map every concept to cooking or baking "
        "(ingredients, recipes, heat, timing, mixing, flavours, kitchen tools). "
        "Walk the reader through the process like a chef's tutorial."
    ),
    "nature": (
        "Map every concept to natural phenomena "
        "(ecosystems, weather, animal behaviour, geology, seasons, rivers). "
        "Use sensory, immersive nature-writing language."
    ),
    "daily_life": (
        "Map every concept to everyday household or social situations "
        "(morning routines, commuting, shopping, conversations, home repairs). "
        "Keep it grounded and immediately relatable."
    ),
    "tech": (
        "Map every concept to familiar technology "
        "(computers, networks, apps, APIs, databases, version control, "
        "UI/UX design). Use developer-friendly language without jargon."
    ),
    "general_simplification": (
        "Simplify the content as clearly as possible for a curious 14-year-old "
        "with no prior background. Use short sentences, concrete comparisons, "
        "and everyday vocabulary."
    ),
}


# ---------------------------------------------------------------------------
# Conceptual density detection  (unchanged from v1, pure computation)
# ---------------------------------------------------------------------------

def detect_conceptual_density(text: str) -> Dict:
    """Return per-sentence density scores plus an overall score.

    Returns
    -------
    dict with keys:
        overall_density    : float 0-1
        sentence_scores    : list[dict]  — text, density, factors
        high_density_count : int
    """
    if not text or not text.strip():
        return {"overall_density": 0.0, "sentence_scores": [], "high_density_count": 0}

    sentences = _split_sentences(text)
    if not sentences:
        return {"overall_density": 0.0, "sentence_scores": [], "high_density_count": 0}

    scored: List[Dict] = []
    for sent in sentences:
        score, factors = _score_sentence(sent)
        scored.append({"text": sent, "density": round(score, 4), "factors": factors})

    overall = sum(s["density"] for s in scored) / len(scored) if scored else 0.0
    high_count = sum(1 for s in scored if s["density"] >= CONCEPTUAL_DENSITY_THRESHOLD)

    return {
        "overall_density": round(overall, 4),
        "sentence_scores": scored,
        "high_density_count": high_count,
    }


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class AnalogyInjector:
    """Two-stage analogy pipeline.

    Parameters
    ----------
    domain : str
        One of the keys in ``ANALOGY_DOMAINS`` / ``DOMAIN_INSTRUCTIONS``.
    threshold : float
        Density threshold used only in inject-mode (legacy).
    """

    # Minimum word count that counts as "already detailed content".
    # Inputs below this are sent through Stage 1 (expansion) first.
    EXPANSION_THRESHOLD = 40

    def __init__(
        self,
        domain: str = DEFAULT_ANALOGY_DOMAIN,
        threshold: float = CONCEPTUAL_DENSITY_THRESHOLD,
    ):
        # Accept both old ANALOGY_DOMAINS keys and new DOMAIN_INSTRUCTIONS keys
        all_valid = set(ANALOGY_DOMAINS.keys()) | set(DOMAIN_INSTRUCTIONS.keys())
        if domain not in all_valid:
            raise ValueError(
                f"Unknown analogy domain '{domain}'. "
                f"Choose from: {sorted(all_valid)}"
            )
        self.domain = domain
        self.domain_info = ANALOGY_DOMAINS.get(domain, {"description": domain})
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Primary public API  — two-stage pipeline
    # ------------------------------------------------------------------

    def augment_text(
        self,
        text: str,
        *,
        use_local: bool = True,
        model_name: Optional[str] = None,
        api_type: Optional[str] = None,
        api_client=None,
        inject_mode: bool = False,       # True → legacy per-sentence notes
        verbose: bool = False,
    ) -> Dict:
        """Run the full two-stage pipeline on *text*.

        Stage 1  ── If the input is too short / sparse, expand it into a full
                    factual explanation first.
        Stage 2  ── Transform the (now detailed) text into a comprehensive
                    domain-specific analogy.

        Parameters
        ----------
        inject_mode : bool
            When True, falls back to the old per-sentence ``[Cognitive Note]``
            injection instead of a full transformation.  Use this only when
            *text* is already a long analysis you want to annotate.

        Returns
        -------
        dict
            augmented_text   : str  — the final analogy / annotated output
            expanded_text    : str  — Stage-1 expansion (same as input if
                                      input was already detailed)
            density_report   : dict — density analysis of the expanded text
            analogies        : list — structured analogy objects (inject_mode)
            analogy_count    : int
            stages_run       : list[str] — which stages were executed
        """
        _call = lambda prompt: self._call_model(
            prompt,
            use_local=use_local,
            model_name=model_name,
            api_type=api_type,
            api_client=api_client,
        )

        stages_run: List[str] = []
        word_count = len(text.split())

        # ── Stage 1: Content Expansion ────────────────────────────────────
        if word_count < self.EXPANSION_THRESHOLD:
            if verbose:
                print(f"  [Stage 1] Input has only {word_count} words — expanding…")
            expansion_prompt = self._build_expansion_prompt(text)
            expanded_text = _call(expansion_prompt)
            stages_run.append("content_expansion")
            if verbose:
                print(f"  [Stage 1] Expansion complete ({len(expanded_text.split())} words).")
        else:
            expanded_text = text
            if verbose:
                print(f"  [Stage 1] Input already detailed ({word_count} words) — skipping expansion.")

        # Density analysis is always on the *expanded* text
        density_report = detect_conceptual_density(expanded_text)

        # ── Stage 2a: Legacy inject-mode (per-sentence notes) ─────────────
        if inject_mode:
            dense_sentences = [
                s for s in density_report["sentence_scores"]
                if s["density"] >= self.threshold
            ]
            if not dense_sentences:
                return {
                    "augmented_text": expanded_text,
                    "expanded_text": expanded_text,
                    "density_report": density_report,
                    "analogies": [],
                    "analogy_count": 0,
                    "stages_run": stages_run,
                }
            prompt = self._build_analogy_prompt_legacy(dense_sentences)
            raw = _call(prompt)
            analogies = self._parse_analogy_response(raw, dense_sentences)
            augmented = self._inject_analogies(expanded_text, analogies)
            stages_run.append("inject_mode_annotation")
            return {
                "augmented_text": augmented,
                "expanded_text": expanded_text,
                "density_report": density_report,
                "analogies": analogies,
                "analogy_count": len(analogies),
                "stages_run": stages_run,
            }

        # ── Stage 2b: Full domain analogy transformation (default) ────────
        if verbose:
            print(f"  [Stage 2] Generating '{self.domain}' domain analogy…")
        transform_prompt = self._build_transform_prompt(expanded_text)
        analogy_text = _call(transform_prompt)
        stages_run.append("domain_analogy_transformation")
        if verbose:
            print(f"  [Stage 2] Transformation complete.")

        return {
            "augmented_text": analogy_text,
            "expanded_text": expanded_text,
            "density_report": density_report,
            "analogies": [],           # structured notes not produced in transform mode
            "analogy_count": 0,
            "stages_run": stages_run,
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
        """Augment an existing style-transfer / analysis result (hybrid mode).

        The primary output is preserved verbatim; a ``COGNITIVE BRIDGING
        NOTES`` section is appended.  The input is assumed to be already
        detailed so Stage 1 is skipped.
        """
        return self.augment_text(
            analysis_text,
            use_local=use_local,
            model_name=model_name,
            api_type=api_type,
            api_client=api_client,
            inject_mode=True,
        )

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_expansion_prompt(topic: str) -> str:
        """Stage 1 — ask the model to write a detailed explanation of the topic."""
        return (
            "You are an expert educational writer.\n"
            "Your task: write a thorough, accurate, and detailed explanation of the "
            "following topic or passage.  The explanation should:\n"
            "  - Be at least 4 paragraphs long.\n"
            "  - Cover what it is, how it works, why it matters, and any key "
            "    sub-concepts or mechanisms involved.\n"
            "  - Be factually correct and written in clear, neutral prose.\n"
            "  - NOT use bullet points — write in full paragraphs.\n\n"
            f"TOPIC: {topic}\n\n"
            "Write the explanation now:"
        )

    def _build_transform_prompt(self, detailed_text: str) -> str:
        """Stage 2 — transform detailed content into a domain analogy."""
        domain_instruction = DOMAIN_INSTRUCTIONS.get(
            self.domain,
            self.domain_info.get("description", "Simplify the content clearly."),
        )
        return (
            "You are an expert educational analogy writer specialising in "
            "cognitive load reduction.\n\n"
            "TASK\n"
            "----\n"
            "Transform the CONTENT below into a comprehensive, engaging analogy "
            "explanation using the DOMAIN INSTRUCTION provided.\n\n"
            "REQUIREMENTS\n"
            "------------\n"
            "1. Map EVERY major concept from the content to a concrete element in "
            "   the chosen domain — do not leave any key idea unexplained.\n"
            "2. Write at least 4 substantial paragraphs.\n"
            "3. Follow the narrative flow of the original content (same order of "
            "   ideas) so the reader can match the analogy back to the source.\n"
            "4. Use vivid, specific details from the domain — avoid vague phrases "
            "   like 'just like in real life'.\n"
            "5. After the analogy, add a short 'CONCEPT MAP' section (a simple "
            "   two-column table or bullet list) that explicitly maps each original "
            "   term to its analogy counterpart.\n"
            "6. Preserve factual accuracy — the analogy must not distort the "
            "   underlying science / meaning.\n"
            "7. Write in second person ('you') to keep it engaging.\n\n"
            f"DOMAIN INSTRUCTION: {domain_instruction}\n\n"
            "CONTENT TO TRANSFORM:\n"
            "---------------------\n"
            f"{detailed_text}\n\n"
            "Write the full analogy explanation now:"
        )

    def _build_analogy_prompt_legacy(self, dense_sentences: List[Dict]) -> str:
        """Legacy Stage 2 — per-sentence cognitive notes (inject_mode)."""
        domain_desc = self.domain_info.get("description", self.domain)
        numbered = "\n".join(
            f"{i+1}. \"{s['text']}\""
            for i, s in enumerate(dense_sentences)
        )
        return (
            "You are an educational analogy generation agent.\n"
            "Generate a simple, accurate real-world analogy for each passage below.\n\n"
            "Rules:\n"
            "- Preserve the original meaning — do not distort technical correctness.\n"
            "- Keep analogies concise (1-3 sentences) but genuinely illuminating.\n"
            "- Use the domain preference for all analogies.\n"
            "- Add a practical example where it helps.\n\n"
            f"Domain preference: {domain_desc}\n\n"
            "PASSAGES:\n"
            f"{numbered}\n\n"
            "FORMAT (match the input numbers exactly):\n"
            "1. Concept: [key idea]\n"
            "   Analogy: [domain-specific analogy]\n"
            "   Example: [practical example]\n"
            "2. Concept: [key idea]\n"
            "   Analogy: [domain-specific analogy]\n"
            "...\n"
        )

    # ------------------------------------------------------------------
    # Model dispatch
    # ------------------------------------------------------------------

    def _call_model(
        self,
        prompt: str,
        *,
        use_local: bool,
        model_name: Optional[str],
        api_type: Optional[str],
        api_client,
    ) -> str:
        from ..models.ollama_client import analyze_with_ollama

        if use_local:
            if not model_name:
                raise ValueError("model_name is required for local Ollama inference.")
            return analyze_with_ollama(prompt, model_name, "enhanced")

        return "Error: No remote API configured for analogy generation."

    # ------------------------------------------------------------------
    # Legacy response parsing & injection helpers (inject_mode)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_analogy_response(
        raw: str, dense_sentences: List[Dict]
    ) -> List[Dict]:
        analogies: List[Dict] = []
        block_pattern = re.compile(r"(?:^|\n)\s*(\d+)\.\s*", re.MULTILINE)
        splits = block_pattern.split(raw)
        block_map: Dict[int, str] = {}
        idx = 1
        while idx < len(splits) - 1:
            try:
                num = int(splits[idx])
                block_map[num] = splits[idx + 1].strip()
            except (ValueError, IndexError):
                pass
            idx += 2

        def _field(block: str, label: str) -> str:
            m = re.search(
                rf"(?:^|\n)\s*{label}\s*:\s*(.+?)(?=\n\s*(?:Concept|Analogy|Example)\s*:|$)",
                block, re.IGNORECASE | re.DOTALL,
            )
            return m.group(1).strip() if m else ""

        for i, sent_info in enumerate(dense_sentences):
            block = block_map.get(i + 1, "")
            if not block:
                continue
            concept = _field(block, "Concept")
            analogy_text = _field(block, "Analogy") or block
            example = _field(block, "Example")
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

    @staticmethod
    def _inject_analogies(text: str, analogies: List[Dict]) -> str:
        result = text
        for item in reversed(analogies):
            src = item["source_sentence"]
            parts = []
            if item.get("concept"):
                parts.append(f"Concept: {item['concept']}")
            parts.append(f"Analogy: {item['analogy']}")
            if item.get("example"):
                parts.append(f"Example: {item['example']}")
            note = "\n[Cognitive Note: " + " | ".join(parts) + "]\n"
            idx = result.find(src)
            if idx != -1:
                end = idx + len(src)
                result = result[:end] + note + result[end:]
        return result

    @staticmethod
    def _format_cognitive_notes(analogies: List[Dict]) -> str:
        lines = ["=" * 60, "COGNITIVE BRIDGING NOTES", "=" * 60, ""]
        for i, item in enumerate(analogies, 1):
            preview = (
                item["source_sentence"][:80] + "…"
                if len(item["source_sentence"]) > 80
                else item["source_sentence"]
            )
            lines.append(f"{i}. Dense passage (score {item['density_score']:.2f}):")
            lines.append(f'   "{preview}"')
            if item.get("concept"):
                lines.append(f"   Concept:  {item['concept']}")
            lines.append(f"   Analogy:  {item['analogy']}")
            if item.get("example"):
                lines.append(f"   Example:  {item['example']}")
            lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point  (called from the main menu — option 7)
# ---------------------------------------------------------------------------

def run_analogy_cli(model_name: str, use_local: bool = True) -> None:
    """Interactive CLI for the two-stage analogy pipeline."""

    DOMAIN_KEYS = list(DOMAIN_INSTRUCTIONS.keys())
    DOMAIN_LABELS = {
        "sports": "Sports",
        "gaming": "Gaming",
        "cooking": "Cooking",
        "nature": "Nature",
        "daily_life": "Daily Life",
        "tech": "Tech",
        "general_simplification": "General Simplification",
    }

    print("\n" + "=" * 60)
    print("COGNITIVE BRIDGING / ANALOGY ENGINE  (v2 — Two-Stage)")
    print("=" * 60)
    print(
        "\nThis engine first expands your input into a detailed explanation,\n"
        "then transforms it into a rich, domain-specific analogy.\n"
    )

    # --- Input ---
    print("How would you like to provide text?")
    print("  1. Type / paste content")
    print("  2. Load from file path")
    choice = input("Choice (1/2): ").strip()

    if choice == "2":
        path = input("File path: ").strip()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw_input = fh.read()
        except OSError as exc:
            print(f"  Error reading file: {exc}")
            return
    else:
        print("Paste or type your text below.")
        print("When finished, press Enter on an empty line:\n")
        lines: List[str] = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        raw_input = "\n".join(lines).strip()

    if not raw_input:
        print("  No input provided — returning to menu.")
        return

    word_count = len(raw_input.split())
    print(f"\nInput received: {word_count} word(s).")

    if word_count < AnalogyInjector.EXPANSION_THRESHOLD:
        print(
            f"  ℹ  Input is short ({word_count} words). "
            "Stage 1 will expand it into a full explanation before applying the analogy."
        )

    # --- Domain selection ---
    print("\nSelect analogy domain:")
    for i, key in enumerate(DOMAIN_KEYS, 1):
        print(f"  {i}. {DOMAIN_LABELS.get(key, key.title())}")

    default_idx = DOMAIN_KEYS.index(DEFAULT_ANALOGY_DOMAIN) + 1 \
        if DEFAULT_ANALOGY_DOMAIN in DOMAIN_KEYS else len(DOMAIN_KEYS)
    dom_choice = input(f"Choice (1-{len(DOMAIN_KEYS)}) [{default_idx}]: ").strip()

    try:
        domain = DOMAIN_KEYS[int(dom_choice) - 1] if dom_choice else DEFAULT_ANALOGY_DOMAIN
    except (ValueError, IndexError):
        domain = DEFAULT_ANALOGY_DOMAIN

    print(f"\nUsing domain: {DOMAIN_LABELS.get(domain, domain.title())}")

    # --- Run pipeline ---
    injector = AnalogyInjector(domain=domain)

    # Stage 1
    if word_count < AnalogyInjector.EXPANSION_THRESHOLD:
        print("\n[Stage 1] Expanding topic into detailed content…")
        _print_spinner(lambda: None, label="", duration=0)  # just cosmetic
        from ..models.ollama_client import analyze_with_ollama
        expansion_prompt = injector._build_expansion_prompt(raw_input)
        print("  Calling model", end="", flush=True)
        t0 = time.time()
        expanded = analyze_with_ollama(expansion_prompt, model_name, "enhanced")
        elapsed = time.time() - t0
        print(f"\r  ✔ Expansion complete ({len(expanded.split())} words, {elapsed:.1f}s)")
    else:
        expanded = raw_input
        print("\n[Stage 1] Input already detailed — skipping expansion.")

    # Density of expanded text
    density = detect_conceptual_density(expanded)
    print(f"\nExpanded content density: {density['overall_density']:.3f}")

    # Stage 2
    print(f"\n[Stage 2] Generating '{DOMAIN_LABELS.get(domain, domain)}' analogy…")
    transform_prompt = injector._build_transform_prompt(expanded)
    t0 = time.time()
    print("  Calling model", end="", flush=True)
    from ..models.ollama_client import analyze_with_ollama
    analogy_output = analyze_with_ollama(transform_prompt, model_name, "enhanced")
    elapsed = time.time() - t0
    print(f"\r  ✔ Analogy transformation complete ({elapsed:.1f}s)")

    # --- Display ---
    print("\n" + "=" * 60)
    print("STAGE 1 — EXPANDED EXPLANATION")
    print("=" * 60)
    print(expanded)

    print("\n" + "=" * 60)
    print(f"STAGE 2 — {DOMAIN_LABELS.get(domain, domain).upper()} ANALOGY")
    print("=" * 60)
    print(analogy_output)
    print("=" * 60)


def _print_spinner(fn, label: str = "Working", duration: float = 0) -> None:
    """Tiny cosmetic spinner — not blocking."""
    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

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
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


def _score_sentence(sentence: str) -> Tuple[float, Dict]:
    words = sentence.split()
    if not words:
        return 0.0, {}

    word_count = len(words)
    lower_words = [w.lower().strip(".,!?;:'\"()[]{}") for w in words]
    alpha_words = [w for w in lower_words if w.isalpha()]

    content_words = [w for w in alpha_words if w not in _FUNCTION_WORDS]
    lexical_density = len(content_words) / len(alpha_words) if alpha_words else 0.0

    total_syllables = sum(count_syllables(w) for w in alpha_words) if alpha_words else 0
    avg_syllables = total_syllables / len(alpha_words) if alpha_words else 1.0
    syllable_factor = min(1.0, max(0.0, (avg_syllables - 1.0) / 3.0))

    unique_ratio = len(set(alpha_words)) / len(alpha_words) if alpha_words else 0.0
    length_factor = min(1.0, max(0.0, (word_count - 5) / 25.0))

    density = (
        0.35 * lexical_density
        + 0.25 * syllable_factor
        + 0.20 * unique_ratio
        + 0.20 * length_factor
    )

    return density, {
        "lexical_density": round(lexical_density, 4),
        "syllable_complexity": round(syllable_factor, 4),
        "information_density": round(unique_ratio, 4),
        "length_factor": round(length_factor, 4),
    }