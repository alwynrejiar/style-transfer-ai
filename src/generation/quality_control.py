"""
Quality control and humanization validation system — v4.0
Replaces placeholder checks with real AI-detection signal measurement.

Core capabilities:
  - Full AI-detection risk audit (burstiness, entropy, clichés, uniformity)
  - Sentence-level repair suggestions with before/after diffs
  - Style fidelity validation against the source profile
  - Multi-category scoring with weighted composite
  - Automated rule-based text fixer (no LLM needed for quick repairs)
  - Publish readiness gate: PASS / WARN / FAIL
"""

from __future__ import annotations

import re
import math
import statistics
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..utils.text_processing import extract_basic_stats
from ..analysis.metrics import (
    compute_humanization_metrics,
    compute_sentence_burstiness,
    compute_vocabulary_burstiness,
    detect_ai_clichés,
    count_human_markers,
    post_process_for_humanness,
    calculate_style_similarity,
    extract_deep_stylometry,
)
from ..config.settings import TIMESTAMP_FORMAT

# ── Thresholds ────────────────────────────────────────────────────────────────
PASS_RISK_THRESHOLD = 0.40   # ai_risk_score below this = PASS
WARN_RISK_THRESHOLD = 0.58   # between PASS and WARN = WARN, above = FAIL

TARGET_BURSTINESS_CV      = 0.45
MIN_SHORT_SENTENCE_RATIO  = 0.12
MIN_LONG_SENTENCE_RATIO   = 0.10
MAX_AI_CLICHE_DENSITY     = 1.0   # per 100 words
MIN_HUMAN_MARKER_DENSITY  = 0.8   # per 100 words
MAX_UNIFORMITY_RATIO      = 0.60
MIN_PARAGRAPH_CV          = 0.25
# ─────────────────────────────────────────────────────────────────────────────

# ── AI cliché replacements: phrase → better alternative ──────────────────────
CLICHE_REPLACEMENTS: Dict[str, str] = {
    "furthermore":              "also",
    "moreover":                 "and",
    "additionally":             "plus",
    "consequently":             "so",
    "therefore":                "so",
    "in conclusion":            "",           # just delete it — end is the conclusion
    "to summarize":             "",
    "in summary":               "",
    "to sum up":                "",
    "in essence":               "basically",
    "it is worth noting":       "note that",
    "it is important to note":  "note that",
    "it should be noted":       "note that",
    "it's worth noting":        "worth noting:",
    "needless to say":          "",
    "as mentioned":             "",
    "as noted":                 "",
    "delve into":               "look at",
    "dive into":                "get into",
    "shed light on":            "explain",
    "unpack":                   "break down",
    "testament to":             "proof of",
    "groundbreaking":           "new",
    "revolutionary":            "major",
    "game-changer":             "big shift",
    "paradigm shift":           "major change",
    "holistic approach":        "full approach",
    "leverage":                 "use",
    "synergy":                  "teamwork",
    "robust":                   "strong",
    "in today's world":         "today",
    "in today's society":       "today",
    "the realm of":             "",
    "the landscape of":         "the world of",
    "the fabric of":            "",
    "this highlights":          "this shows",
    "this demonstrates":        "this shows",
    "this underscores":         "this confirms",
    "this illustrates":         "this shows",
    "as we can see":            "",
    "let's explore":            "let's look at",
    "let's dive":               "let's get into",
    "let me walk you through":  "here's how",
    "truly":                    "",
    "certainly":                "",
    "undoubtedly":              "",
    "unquestionably":           "",
}


class QualityController:
    """
    Comprehensive quality control and humanization validation system.

    Primary methods:
      validate_content_quality()    → full audit + scores + recommendations
      quick_ai_risk_check()         → fast risk score only (no LLM)
      apply_rule_based_fixes()      → automated text fixes (no LLM needed)
      improve_content_quality()     → structured improvement plan
      compare_versions()            → before/after comparison
      get_publish_readiness()       → PASS / WARN / FAIL gate
    """

    def __init__(self) -> None:
        self.quality_categories = [
            "humanness",          # AI-detection risk metrics
            "style_fidelity",     # Match to target style profile
            "readability",        # Readability and clarity
            "content_integrity",  # Factual completeness and coherence
            "structure",          # Paragraph and sentence structure variety
        ]

        # Category weights for overall_score
        self._weights = {
            "humanness":         0.35,   # highest weight — this is the core problem
            "style_fidelity":    0.25,
            "readability":       0.18,
            "content_integrity": 0.12,
            "structure":         0.10,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # PRIMARY API — validate_content_quality
    # ══════════════════════════════════════════════════════════════════════════

    def validate_content_quality(
        self,
        content: str,
        requirements: Optional[Dict] = None,
        target_style_profile: Optional[Dict] = None,
        content_type: str = "general",
    ) -> Dict:
        """
        Full quality audit combining AI-risk, style fidelity, and readability.

        Args:
            content:              Text to evaluate
            requirements:         Dict with optional keys: target_length, required_tone,
                                  max_ai_risk (float), min_burstiness_cv (float)
            target_style_profile: Profile from create_enhanced_style_profile()
            content_type:         Informational — used in metadata

        Returns:
            Dict: scores, issues, recommendations, publish_readiness
        """
        if not content or not content.strip():
            return self._failed_validation("Empty content")

        req = requirements or {}

        # ── Category scores ───────────────────────────────────────────────────
        humanness_score,   humanness_detail   = self._score_humanness(content, req)
        fidelity_score,    fidelity_detail    = self._score_style_fidelity(content, target_style_profile)
        readability_score, readability_detail = self._score_readability(content, req)
        integrity_score,   integrity_detail   = self._score_content_integrity(content, req)
        structure_score,   structure_detail   = self._score_structure(content)

        category_scores = {
            "humanness":         humanness_score,
            "style_fidelity":    fidelity_score,
            "readability":       readability_score,
            "content_integrity": integrity_score,
            "structure":         structure_score,
        }

        # Weighted overall score
        overall = sum(
            category_scores[cat] * self._weights[cat]
            for cat in self.quality_categories
        )
        overall = round(overall, 4)

        # Issues and recommendations
        issues = self._identify_issues(content, category_scores, humanness_detail)
        recs   = self._generate_recommendations(content, category_scores, humanness_detail, req)

        # Publish readiness
        readiness = self._get_publish_readiness_internal(
            humanness_detail.get("ai_risk_score", 0.5),
            overall,
            issues,
        )

        return {
            "overall_score":     overall,
            "category_scores":   category_scores,
            "publish_readiness": readiness,
            "quality_issues":    issues,
            "recommendations":   recs,
            "humanness_detail":  humanness_detail,
            "structure_detail":  structure_detail,
            "readability_detail":readability_detail,
            "compliance_status": self._check_compliance(content, req, category_scores),
            "metadata": {
                "content_type":          content_type,
                "validation_timestamp":  datetime.now().strftime(TIMESTAMP_FORMAT),
                "content_length":        len(content),
                "word_count":            len(content.split()),
                "schema_version":        "4.0",
            },
        }

    # ══════════════════════════════════════════════════════════════════════════
    # QUICK RISK CHECK (no LLM, fast)
    # ══════════════════════════════════════════════════════════════════════════

    def quick_ai_risk_check(self, content: str) -> Dict:
        """
        Fast AI-risk check using only local computation (no model call).
        Suitable for real-time feedback during editing.

        Returns:
            Dict with ai_risk_score, risk_label, top 3 problems, quick_fixes
        """
        hm    = compute_humanization_metrics(content)
        burst = compute_sentence_burstiness(content)
        cliché= detect_ai_clichés(content)
        human = count_human_markers(content)

        risk       = hm.get("ai_risk_score", 0.5)
        risk_label = hm.get("risk_label", "unknown")

        problems = []
        quick_fixes = []

        cv = burst.get("burstiness_cv", 0.0)
        if cv < 0.30:
            problems.append(f"Very uniform sentence lengths (CV={cv:.2f}) — major AI signal")
            quick_fixes.append("Insert a 2-4 word sentence somewhere in each paragraph")
        elif cv < TARGET_BURSTINESS_CV:
            problems.append(f"Low sentence variety (CV={cv:.2f})")
            quick_fixes.append("Add one very short and one very long sentence")

        cd = cliché.get("ai_cliche_density_per_100w", 0.0)
        if cd > 1.0:
            top_phrases = [p["phrase"] for p in cliché.get("flagged_phrases", [])[:4]]
            problems.append(f"AI clichés detected ({cd:.1f}/100w): {top_phrases}")
            quick_fixes.append(f"Replace: {top_phrases}")

        ur = burst.get("consecutive_uniformity", 0.0)
        if ur > 0.65:
            problems.append(f"Too many consecutive same-length sentences ({ur:.0%})")
            quick_fixes.append("After every 2-3 long sentences, write one short one")

        sr = burst.get("short_sentence_ratio", 0.0)
        if sr < MIN_SHORT_SENTENCE_RATIO:
            problems.append(f"Almost no short sentences ({sr:.0%})")
            quick_fixes.append("Add standalone 3-5 word sentences (rhetorical or emphatic)")

        hmd = human.get("human_marker_density_per_100w", 0.0)
        if hmd < MIN_HUMAN_MARKER_DENSITY:
            problems.append("Very few human writing markers detected")
            quick_fixes.append("Add: conversational aside, self-correction, or conjunction starter")

        return {
            "ai_risk_score":       risk,
            "risk_label":          risk_label,
            "human_likeness":      hm.get("human_likeness_score"),
            "burstiness_cv":       cv,
            "burstiness_label":    burst.get("burstiness_label"),
            "ai_cliche_density":   cd,
            "short_sentence_ratio":sr,
            "human_marker_density":hmd,
            "problems_found":      problems,
            "quick_fixes":         quick_fixes,
            "pass_fail":           "PASS" if risk <= PASS_RISK_THRESHOLD else
                                   "WARN" if risk <= WARN_RISK_THRESHOLD else "FAIL",
        }

    # ══════════════════════════════════════════════════════════════════════════
    # AUTOMATED RULE-BASED TEXT FIXER (no LLM)
    # ══════════════════════════════════════════════════════════════════════════

    def apply_rule_based_fixes(self, content: str) -> Dict:
        """
        Apply deterministic text fixes that reduce AI-detection risk
        without requiring a model call. Fast and predictable.

        Fixes applied:
          1. Strip AI cliché phrases using CLICHE_REPLACEMENTS dict
          2. Break up runs of same-length sentences by splitting the longest
          3. Lowercase sentence-starting transition openers
             (e.g. 'Furthermore, ...' → 'Also, ...')

        Returns:
            Dict with fixed_content, changes_made, risk_before, risk_after
        """
        before_risk = compute_humanization_metrics(content).get("ai_risk_score", 0.5)

        fixed    = content
        changes  = []

        # Fix 1 — replace AI cliché phrases
        fixed, cliché_changes = self._strip_ai_clichés(fixed)
        changes.extend(cliché_changes)

        # Fix 2 — break consecutive same-length sentence runs
        fixed, run_changes = self._break_sentence_runs(fixed)
        changes.extend(run_changes)

        # Fix 3 — lowercase stock sentence openers
        fixed, opener_changes = self._fix_ai_openers(fixed)
        changes.extend(opener_changes)

        after_risk = compute_humanization_metrics(fixed).get("ai_risk_score", before_risk)

        return {
            "fixed_content":  fixed,
            "changes_made":   changes,
            "change_count":   len(changes),
            "risk_before":    before_risk,
            "risk_after":     after_risk,
            "risk_reduction": round(before_risk - after_risk, 4),
            "note": (
                "Rule-based fixes applied. For larger improvements, use StyleTransfer.humanize_only()."
            ),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # IMPROVEMENT PLAN
    # ══════════════════════════════════════════════════════════════════════════

    def improve_content_quality(
        self,
        content: str,
        quality_assessment: Dict,
        improvement_focus: Optional[List[str]] = None,
    ) -> Dict:
        """
        Generate a structured improvement plan based on quality_assessment.
        """
        cat_scores   = quality_assessment.get("category_scores", {})
        hum_detail   = quality_assessment.get("humanness_detail", {})
        priorities   = self._determine_priorities(cat_scores, improvement_focus)

        specific_fixes   = []
        improvement_plan = []

        for area in priorities:
            if area == "humanness":
                specific_fixes.extend(self._humanness_fixes(content, hum_detail))
            elif area == "structure":
                specific_fixes.extend(self._structure_fixes(content))
            elif area == "readability":
                specific_fixes.extend(self._readability_fixes(content))
            elif area == "style_fidelity":
                specific_fixes.append({
                    "type": "style_alignment",
                    "description": "Re-run style transfer with a higher intensity setting",
                    "impact": "high",
                })
            elif area == "content_integrity":
                specific_fixes.extend(self._integrity_fixes(content))

        improvement_plan = [f["description"] for f in specific_fixes
                            if f.get("impact") in ("high", "medium")][:7]

        estimated_impact = {}
        for cat in self.quality_categories:
            current = cat_scores.get(cat, 0.5)
            relevant = [f for f in specific_fixes if cat in f.get("type", "")]
            estimated_impact[cat] = min(1.0, round(current + len(relevant) * 0.08, 3))

        return {
            "improvement_plan":      improvement_plan,
            "specific_fixes":        specific_fixes,
            "enhancement_priorities":priorities,
            "estimated_impact":      estimated_impact,
            "rule_based_fixes_available": True,
            "note": "Call apply_rule_based_fixes() for instant no-LLM improvements.",
        }

    # ══════════════════════════════════════════════════════════════════════════
    # VERSION COMPARISON
    # ══════════════════════════════════════════════════════════════════════════

    def compare_versions(
        self,
        original_content: str,
        improved_content: str,
        requirements: Optional[Dict] = None,
        target_style_profile: Optional[Dict] = None,
    ) -> Dict:
        """Compare quality metrics between two content versions."""
        orig_qa = self.validate_content_quality(original_content, requirements, target_style_profile)
        impr_qa = self.validate_content_quality(improved_content, requirements, target_style_profile)

        improvements = {
            cat: round(
                impr_qa["category_scores"].get(cat, 0) -
                orig_qa["category_scores"].get(cat, 0),
                4
            )
            for cat in self.quality_categories
        }

        orig_risk = orig_qa["humanness_detail"].get("ai_risk_score", 0.5)
        impr_risk = impr_qa["humanness_detail"].get("ai_risk_score", 0.5)

        return {
            "original_quality":      orig_qa,
            "improved_quality":      impr_qa,
            "score_improvements":    improvements,
            "overall_improvement":   round(
                impr_qa["overall_score"] - orig_qa["overall_score"], 4
            ),
            "ai_risk_improvement":   round(orig_risk - impr_risk, 4),
            "original_ai_risk":      orig_risk,
            "improved_ai_risk":      impr_risk,
            "original_readiness":    orig_qa["publish_readiness"]["status"],
            "improved_readiness":    impr_qa["publish_readiness"]["status"],
            "improvement_summary":   self._summarize(improvements),
            "timestamp":             datetime.now().strftime(TIMESTAMP_FORMAT),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLISH READINESS GATE
    # ══════════════════════════════════════════════════════════════════════════

    def get_publish_readiness(self, content: str, style_profile: Optional[Dict] = None) -> Dict:
        """
        Quick publish gate — returns PASS / WARN / FAIL with a clear reason.
        Use this as the final checkpoint before sending content to production.
        """
        qa = self.validate_content_quality(content, target_style_profile=style_profile)
        return qa["publish_readiness"]

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY SCORING — HUMANNESS
    # ══════════════════════════════════════════════════════════════════════════

    def _score_humanness(self, content: str, req: Dict) -> Tuple[float, Dict]:
        """Score AI-detection risk. Higher = more human (less AI risk)."""
        hm    = compute_humanization_metrics(content)
        burst = compute_sentence_burstiness(content)
        cliché= detect_ai_clichés(content)
        human = count_human_markers(content)
        vburst= compute_vocabulary_burstiness(content)

        ai_risk  = hm.get("ai_risk_score", 0.5)
        # Score is inverse of risk: risk 0.0 → score 1.0, risk 1.0 → score 0.0
        base_score = 1.0 - ai_risk

        # Apply penalty for exceeding max_ai_risk requirement
        if "max_ai_risk" in req:
            if ai_risk > req["max_ai_risk"]:
                base_score *= 0.6

        detail = {
            "ai_risk_score":          ai_risk,
            "human_likeness_score":   hm.get("human_likeness_score"),
            "burstiness_cv":          burst.get("burstiness_cv"),
            "burstiness_label":       burst.get("burstiness_label"),
            "sentence_entropy":       burst.get("sentence_entropy"),
            "short_sentence_ratio":   burst.get("short_sentence_ratio"),
            "long_sentence_ratio":    burst.get("long_sentence_ratio"),
            "consecutive_uniformity": burst.get("consecutive_uniformity"),
            "ai_cliche_density":      cliché.get("ai_cliche_density_per_100w"),
            "ai_cliche_count":        cliché.get("ai_cliche_count"),
            "flagged_clichés":        [p["phrase"] for p in cliché.get("flagged_phrases", [])[:8]],
            "human_marker_density":   human.get("human_marker_density_per_100w"),
            "vocab_burstiness_cv":    vburst.get("vocabulary_burstiness_cv"),
            "signal_breakdown":       hm.get("signal_breakdown", {}),
        }

        return round(max(0.0, min(1.0, base_score)), 4), detail

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY SCORING — STYLE FIDELITY
    # ══════════════════════════════════════════════════════════════════════════

    def _score_style_fidelity(
        self, content: str, profile: Optional[Dict]
    ) -> Tuple[float, Dict]:
        if not profile:
            return 0.75, {"note": "No target profile provided — neutral score"}

        detail: Dict = {}
        scores = []

        # Deep stylometry comparison
        try:
            ds_content = extract_deep_stylometry(content)
            ds_target  = profile.get("deep_stylometry", {})
            if ds_content and ds_target:
                sim = calculate_style_similarity(ds_content, ds_target)
                scores.append(sim.get("combined_score", 0.5))
                detail["stylometry_similarity"] = sim
        except Exception:
            pass

        # Burstiness CV closeness
        target_cv  = profile.get("human_pattern_metrics", {}).get("burstiness_cv", 0.0)
        content_cv = compute_sentence_burstiness(content).get("burstiness_cv", 0.0)
        if target_cv > 0:
            cv_closeness = max(0.0, 1.0 - abs(target_cv - content_cv) / target_cv)
            scores.append(cv_closeness)
            detail["burstiness_cv_target"]  = target_cv
            detail["burstiness_cv_actual"]  = content_cv
            detail["burstiness_cv_closeness"] = round(cv_closeness, 4)

        # Average sentence length closeness
        rm = profile.get("readability_metrics", {})
        target_asl = rm.get("avg_sentence_length_words", 0)
        stats      = extract_basic_stats(content)
        actual_asl = stats.get("avg_sentence_length", 15)
        if target_asl > 0:
            asl_closeness = max(0.0, 1.0 - abs(target_asl - actual_asl) / target_asl)
            scores.append(asl_closeness)
            detail["avg_sent_len_target"]   = target_asl
            detail["avg_sent_len_actual"]   = actual_asl
            detail["avg_sent_len_closeness"]= round(asl_closeness, 4)

        score = round(sum(scores) / len(scores), 4) if scores else 0.5
        return score, detail

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY SCORING — READABILITY
    # ══════════════════════════════════════════════════════════════════════════

    def _score_readability(self, content: str, req: Dict) -> Tuple[float, Dict]:
        stats = extract_basic_stats(content)
        words = content.split()
        if not words:
            return 0.0, {}

        asl = stats.get("avg_sentence_length", 15)

        # Target: 12-20 words is generally readable
        if 12 <= asl <= 20:
            asl_score = 1.0
        elif asl < 6:
            asl_score = 0.7  # too choppy
        elif asl > 30:
            asl_score = max(0.4, 1.0 - (asl - 30) / 30)
        else:
            asl_score = max(0.5, 1.0 - abs(asl - 16) / 16)

        # Word length score (avg 4-6 chars = readable)
        avg_wl = sum(len(w.strip(".,!?;:\"'()")) for w in words) / len(words)
        if 4 <= avg_wl <= 6:
            wl_score = 1.0
        else:
            wl_score = max(0.5, 1.0 - abs(avg_wl - 5) / 5)

        score = round((asl_score * 0.6 + wl_score * 0.4), 4)

        detail = {
            "avg_sentence_length": asl,
            "avg_word_length":     round(avg_wl, 2),
            "asl_score":           round(asl_score, 4),
            "word_length_score":   round(wl_score, 4),
        }
        return score, detail

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY SCORING — CONTENT INTEGRITY
    # ══════════════════════════════════════════════════════════════════════════

    def _score_content_integrity(self, content: str, req: Dict) -> Tuple[float, Dict]:
        score = 1.0
        detail: Dict = {}

        # Incomplete endings
        stripped = content.strip()
        if stripped.endswith(("...", "etc.", "and so on", "and more")):
            score -= 0.25
            detail["incomplete_ending"] = True

        # Very short content
        wc = len(content.split())
        if wc < 30:
            score -= 0.30
            detail["too_short"] = True

        # Length compliance
        if "target_length" in req:
            target = req["target_length"]
            diff   = abs(wc - target) / target
            if diff > 0.25:
                score -= 0.20
                detail["length_off_target"] = f"target={target}, actual={wc}, diff={diff:.0%}"

        detail["word_count"] = wc
        return round(max(0.0, score), 4), detail

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY SCORING — STRUCTURE
    # ══════════════════════════════════════════════════════════════════════════

    def _score_structure(self, content: str) -> Tuple[float, Dict]:
        sentences  = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content) if s.strip()]
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        if not sentences:
            return 0.0, {}

        # Sentence length variance
        lengths = [len(re.findall(r"\b[a-zA-Z]+\b", s)) for s in sentences]
        if len(lengths) >= 2:
            mean_l = statistics.mean(lengths)
            std_l  = statistics.pstdev(lengths)
            cv     = std_l / mean_l if mean_l > 0 else 0.0
        else:
            cv = 0.0

        # Paragraph count and variety
        para_score = 1.0
        if len(paragraphs) < 2:
            para_score = 0.6
        else:
            para_lengths = [len(re.findall(r"\b[a-zA-Z]+\b", p)) for p in paragraphs]
            if len(para_lengths) >= 2:
                pm = statistics.mean(para_lengths)
                ps = statistics.pstdev(para_lengths)
                para_cv = ps / pm if pm > 0 else 0.0
            else:
                para_cv = 0.0
            if para_cv < MIN_PARAGRAPH_CV:
                para_score = max(0.5, para_cv / MIN_PARAGRAPH_CV)

        # Sentence variety score from CV
        variety_score = min(1.0, cv / TARGET_BURSTINESS_CV)

        score = round((variety_score * 0.6 + para_score * 0.4), 4)

        detail = {
            "sentence_count":        len(sentences),
            "paragraph_count":       len(paragraphs),
            "sentence_length_cv":    round(cv, 4),
            "paragraph_variety_score": round(para_score, 4),
            "variety_score":         round(variety_score, 4),
        }
        return score, detail

    # ══════════════════════════════════════════════════════════════════════════
    # ISSUES AND RECOMMENDATIONS
    # ══════════════════════════════════════════════════════════════════════════

    def _identify_issues(
        self,
        content: str,
        scores: Dict,
        hum_detail: Dict,
    ) -> List[Dict]:
        issues = []

        # Humanness issues (most critical)
        risk = hum_detail.get("ai_risk_score", 0.5)
        if risk >= 0.60:
            issues.append({
                "category": "humanness",
                "severity": "critical",
                "description": f"High AI detection risk ({risk:.2f}). Content will likely be flagged.",
                "fix": "Run StyleTransfer.humanize_only() or QualityController.apply_rule_based_fixes()",
            })
        elif risk >= REPAIR_THRESHOLD:
            issues.append({
                "category": "humanness",
                "severity": "high",
                "description": f"Moderate AI detection risk ({risk:.2f}).",
                "fix": "Apply humanization rules and check burstiness",
            })

        cv = hum_detail.get("burstiness_cv", 0.0)
        if cv < 0.30:
            issues.append({
                "category": "humanness",
                "severity": "critical",
                "description": f"Very uniform sentence lengths (CV={cv:.2f}) — primary AI detector signal.",
                "fix": "Add micro-sentences (≤5 words) and at least one 30+ word sentence",
            })

        cd = hum_detail.get("ai_cliche_density", 0.0)
        if cd > MAX_AI_CLICHE_DENSITY:
            phrases = hum_detail.get("flagged_clichés", [])[:5]
            issues.append({
                "category": "humanness",
                "severity": "high",
                "description": f"AI clichés: {cd:.1f}/100 words. Found: {phrases}",
                "fix": "Call apply_rule_based_fixes() to strip these automatically",
            })

        ur = hum_detail.get("consecutive_uniformity", 0.0)
        if ur > MAX_UNIFORMITY_RATIO:
            issues.append({
                "category": "structure",
                "severity": "medium",
                "description": f"Consecutive sentence uniformity too high ({ur:.0%})",
                "fix": "Break up runs of similar-length sentences",
            })

        # Style fidelity
        if scores.get("style_fidelity", 1.0) < 0.55:
            issues.append({
                "category": "style_fidelity",
                "severity": "medium",
                "description": "Poor match to target style profile",
                "fix": "Re-run transfer with higher intensity or check profile quality",
            })

        # Readability
        if scores.get("readability", 1.0) < 0.55:
            issues.append({
                "category": "readability",
                "severity": "medium",
                "description": "Readability below acceptable threshold",
                "fix": "Check average sentence length — target 12–20 words",
            })

        return issues

    def _generate_recommendations(
        self,
        content: str,
        scores: Dict,
        hum_detail: Dict,
        req: Dict,
    ) -> List[str]:
        recs = []
        risk = hum_detail.get("ai_risk_score", 0.5)
        cv   = hum_detail.get("burstiness_cv", 0.0)

        if risk > PASS_RISK_THRESHOLD:
            recs.append(
                f"Priority 1: Reduce AI risk from {risk:.2f} to below {PASS_RISK_THRESHOLD}. "
                "Call apply_rule_based_fixes() first, then StyleTransfer.humanize_only() if needed."
            )

        if cv < TARGET_BURSTINESS_CV:
            recs.append(
                f"Sentence variety is low (CV={cv:.2f}, target ≥{TARGET_BURSTINESS_CV}). "
                "Insert short punchy sentences (3-5 words) after long ones."
            )

        cd = hum_detail.get("ai_cliche_density", 0.0)
        if cd > 0.5:
            phrases = hum_detail.get("flagged_clichés", [])[:3]
            recs.append(f"Remove AI-marker phrases: {phrases}. Use direct language instead.")

        short_r = hum_detail.get("short_sentence_ratio", 0.0)
        if short_r < MIN_SHORT_SENTENCE_RATIO:
            recs.append(
                f"Only {short_r:.0%} of sentences are short (≤6 words). "
                "Add at least 2-3 micro-sentences for human rhythm."
            )

        if scores.get("style_fidelity", 1.0) < 0.65:
            recs.append(
                "Style fidelity is weak. Ensure the full profile (rewrite_directive + "
                "humanization_rules) is being passed to the generation prompt."
            )

        hmd = hum_detail.get("human_marker_density", 0.0)
        if hmd < MIN_HUMAN_MARKER_DENSITY:
            recs.append(
                "Human writing markers are sparse. Add: conjunction sentence-starters "
                "(But/And/So), em-dash asides, or self-correction phrases."
            )

        return recs

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLISH READINESS
    # ══════════════════════════════════════════════════════════════════════════

    def _get_publish_readiness_internal(
        self,
        ai_risk: float,
        overall_score: float,
        issues: List[Dict],
    ) -> Dict:
        critical_issues = [i for i in issues if i.get("severity") == "critical"]

        if ai_risk <= PASS_RISK_THRESHOLD and overall_score >= 0.65 and not critical_issues:
            status  = "PASS"
            message = f"Content is ready to publish. AI risk={ai_risk:.2f}, score={overall_score:.2f}."
        elif ai_risk <= WARN_RISK_THRESHOLD and not critical_issues:
            status  = "WARN"
            message = (
                f"Publishable but improveable. AI risk={ai_risk:.2f}. "
                "Apply recommendations before publishing."
            )
        else:
            status  = "FAIL"
            message = (
                f"Not ready. AI risk={ai_risk:.2f} (threshold={PASS_RISK_THRESHOLD}). "
                f"{len(critical_issues)} critical issue(s) must be resolved."
            )

        return {
            "status":           status,
            "message":          message,
            "ai_risk_score":    ai_risk,
            "overall_score":    overall_score,
            "critical_issues":  len(critical_issues),
            "pass_threshold":   PASS_RISK_THRESHOLD,
            "warn_threshold":   WARN_RISK_THRESHOLD,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # COMPLIANCE CHECK
    # ══════════════════════════════════════════════════════════════════════════

    def _check_compliance(
        self,
        content: str,
        req: Dict,
        scores: Dict,
    ) -> Dict:
        status: Dict = {
            "overall_compliance": True,
            "length_compliance":  True,
            "quality_compliance": scores.get("humanness", 0.0) >= 0.58,
            "risk_compliance":    True,
        }

        if "target_length" in req:
            wc     = len(content.split())
            target = req["target_length"]
            status["length_compliance"] = abs(wc - target) / target <= 0.20
            status["actual_length"]     = wc
            status["target_length"]     = target

        if "max_ai_risk" in req:
            hm    = compute_humanization_metrics(content)
            risk  = hm.get("ai_risk_score", 0.5)
            status["risk_compliance"] = risk <= req["max_ai_risk"]
            status["actual_risk"]     = risk
            status["required_max_risk"] = req["max_ai_risk"]

        status["overall_compliance"] = all(
            v for k, v in status.items()
            if k.endswith("_compliance")
        )
        return status

    # ══════════════════════════════════════════════════════════════════════════
    # AUTOMATED RULE-BASED FIXERS
    # ══════════════════════════════════════════════════════════════════════════

    def _strip_ai_clichés(self, content: str) -> Tuple[str, List[str]]:
        """Replace AI cliché phrases using CLICHE_REPLACEMENTS."""
        fixed   = content
        changes = []

        for phrase, replacement in CLICHE_REPLACEMENTS.items():
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            if pattern.search(fixed):
                fixed = pattern.sub(replacement, fixed)
                changes.append(f'Replaced "{phrase}" → "{replacement}"')

        # Clean up double spaces and leading commas left by deletions
        fixed = re.sub(r" {2,}", " ", fixed)
        fixed = re.sub(r",\s*,", ",", fixed)
        fixed = re.sub(r"\.\s*\.", ".", fixed)
        fixed = re.sub(r"^\s*,\s*", "", fixed, flags=re.MULTILINE)

        return fixed, changes

    def _break_sentence_runs(self, content: str, run_length: int = 3) -> Tuple[str, List[str]]:
        """
        Detect runs of ≥run_length consecutive sentences with similar lengths
        (within 4 words) and split the longest sentence in the run at a comma.
        Returns modified content and list of changes.
        """
        sentences = re.split(r"(?<=[.!?])\s+", content.strip())
        if len(sentences) < run_length + 1:
            return content, []

        lengths  = [len(re.findall(r"\b[a-zA-Z]+\b", s)) for s in sentences]
        changes  = []
        modified = sentences[:]

        i = 0
        while i <= len(modified) - run_length:
            window = lengths[i:i + run_length]
            if max(window) - min(window) <= 4 and min(window) >= 10:
                # Find the longest in the run and split at a comma
                longest_idx = i + window.index(max(window))
                sent = modified[longest_idx]
                comma_pos = sent.find(",", len(sent) // 3)
                if comma_pos > 0:
                    part1 = sent[:comma_pos].strip()
                    part2 = sent[comma_pos + 1:].strip().capitalize()
                    modified[longest_idx] = part1 + "."
                    modified.insert(longest_idx + 1, part2)
                    lengths.insert(longest_idx + 1, len(re.findall(r"\b[a-zA-Z]+\b", part2)))
                    lengths[longest_idx] = len(re.findall(r"\b[a-zA-Z]+\b", part1))
                    changes.append(f"Split long sentence at position {longest_idx} to break uniform run")
            i += 1

        return " ".join(modified), changes

    def _fix_ai_openers(self, content: str) -> Tuple[str, List[str]]:
        """
        Lowercase and replace stock sentence-opening AI transition words.
        E.g. 'Furthermore, ...' → 'Also, ...'
        """
        changes = []
        opener_fixes = {
            r"^Furthermore,":   "Also,",
            r"^Moreover,":      "And",
            r"^Additionally,":  "Plus,",
            r"^Consequently,":  "So",
            r"^Therefore,":     "So",
            r"^In conclusion,": "",
            r"^To summarize,":  "",
            r"^In summary,":    "",
        }
        lines = content.split("\n")
        fixed_lines = []
        for line in lines:
            fixed_line = line
            for pattern, replacement in opener_fixes.items():
                new_line = re.sub(pattern, replacement, fixed_line, flags=re.MULTILINE)
                if new_line != fixed_line:
                    changes.append(f"Fixed opener: '{pattern}' → '{replacement}'")
                    fixed_line = new_line
            fixed_lines.append(fixed_line.lstrip(". ,"))

        return "\n".join(fixed_lines), changes

    # ══════════════════════════════════════════════════════════════════════════
    # IMPROVEMENT FIX GENERATORS
    # ══════════════════════════════════════════════════════════════════════════

    def _humanness_fixes(self, content: str, hm_detail: Dict) -> List[Dict]:
        fixes = []
        cv = hm_detail.get("burstiness_cv", 0.5)
        if cv < TARGET_BURSTINESS_CV:
            fixes.append({
                "type": "humanness_burstiness",
                "description": (
                    f"Add micro-sentences (≤5 words) — current burstiness CV={cv:.2f}, "
                    f"need ≥{TARGET_BURSTINESS_CV}"
                ),
                "impact": "high",
            })
        cd = hm_detail.get("ai_cliche_density", 0.0)
        if cd > 0.5:
            fixes.append({
                "type": "humanness_clichés",
                "description": f"Strip AI clichés ({cd:.1f}/100w) — call apply_rule_based_fixes()",
                "impact": "high",
            })
        hmd = hm_detail.get("human_marker_density", 0.0)
        if hmd < MIN_HUMAN_MARKER_DENSITY:
            fixes.append({
                "type": "humanness_markers",
                "description": "Add conjunction sentence-starters and em-dash asides",
                "impact": "medium",
            })
        return fixes

    def _structure_fixes(self, content: str) -> List[Dict]:
        return [
            {
                "type": "structure_paragraph",
                "description": "Vary paragraph lengths — add one single-sentence paragraph",
                "impact": "medium",
            },
            {
                "type": "structure_sentences",
                "description": "Break consecutive same-length sentence runs",
                "impact": "medium",
            },
        ]

    def _readability_fixes(self, content: str) -> List[Dict]:
        stats = extract_basic_stats(content)
        asl   = stats.get("avg_sentence_length", 15)
        fixes = []
        if asl > 25:
            fixes.append({
                "type": "readability_length",
                "description": f"Sentences too long (avg {asl:.0f}w). Split some at conjunctions.",
                "impact": "medium",
            })
        elif asl < 8:
            fixes.append({
                "type": "readability_length",
                "description": f"Sentences too short (avg {asl:.0f}w). Merge some short ones.",
                "impact": "low",
            })
        return fixes

    def _integrity_fixes(self, content: str) -> List[Dict]:
        fixes = []
        if content.strip().endswith(("...", "etc.", "and so on")):
            fixes.append({
                "type": "content_integrity_ending",
                "description": "Content appears incomplete — add a proper ending",
                "impact": "high",
            })
        return fixes

    # ══════════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ══════════════════════════════════════════════════════════════════════════

    def _determine_priorities(
        self,
        scores: Dict,
        focus: Optional[List[str]],
    ) -> List[str]:
        if focus:
            return focus
        return [
            cat for cat, score in sorted(scores.items(), key=lambda x: x[1])
            if score < 0.72
        ]

    def _summarize(self, improvements: Dict) -> str:
        improved = [c for c, d in improvements.items() if d > 0.05]
        declined = [c for c, d in improvements.items() if d < -0.05]
        parts = []
        if improved:
            parts.append(f"Improved: {', '.join(improved)}")
        if declined:
            parts.append(f"Declined: {', '.join(declined)}")
        return ". ".join(parts) if parts else "No significant change detected."

    def _failed_validation(self, message: str) -> Dict:
        return {
            "overall_score":     0.0,
            "category_scores":   {c: 0.0 for c in self.quality_categories},
            "publish_readiness": {"status": "FAIL", "message": message},
            "quality_issues":    [{"category": "validation", "severity": "critical",
                                   "description": message}],
            "recommendations":   ["Fix the validation error before proceeding"],
            "compliance_status": {"overall_compliance": False},
            "metadata": {
                "validation_timestamp": datetime.now().strftime(TIMESTAMP_FORMAT),
                "validation_failed":    True,
                "error_message":        message,
            },
        }