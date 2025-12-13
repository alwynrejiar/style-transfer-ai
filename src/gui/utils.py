"""
Utility helpers for the GUI (threading, file IO, metric normalization).
"""
import threading
import os
from typing import Any, Callable, Dict, List, Tuple

from src.analysis.metrics import analyze_text_statistics, calculate_readability_metrics


class ThreadedTask(threading.Thread):
    """Background thread that returns result via a callback."""

    def __init__(self, target_func: Callable[..., Any], callback_func: Callable[[Any], None], *args, **kwargs):
        super().__init__(daemon=True)
        self.target_func = target_func
        self.callback_func = callback_func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.target_func(*self.args, **self.kwargs)
            self.callback_func(result)
        except Exception as exc:  # pylint: disable=broad-except
            self.callback_func({"error": str(exc)})


def safe_read_text(file_path: str) -> str:
    """Read a text file with UTF-8 then latin-1 fallback."""
    if not file_path:
        return ""
    if not os.path.exists(file_path):
        return ""
    for enc in ("utf-8", "latin-1"):
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except OSError:
            return ""
    return ""


def compute_metrics(text: str) -> Tuple[Dict, Dict]:
    """Convenience wrapper to compute statistics and readability."""
    stats = analyze_text_statistics(text)
    readability = calculate_readability_metrics(text)
    return stats, readability


def radar_from_metrics(stats: Dict, readability: Dict) -> Tuple[List[str], List[float]]:
    """
    Map text metrics to 6 radar dimensions (0-1 scale) for visualization.
    Dimensions: Lexical Diversity, Sentence Length, Readability, Complexity, Punctuation, Information Density.
    """
    labels = [
        "Lexical Diversity",
        "Sentence Length",
        "Readability",
        "Structural Complexity",
        "Punctuation",
        "Information Density",
    ]

    if not stats:
        stats = {}
    if not readability:
        readability = {}

    lexical_div = float(stats.get("lexical_diversity", 0))
    avg_words = float(stats.get("avg_words_per_sentence", 0))
    flesch = float(readability.get("flesch_reading_ease", 0))
    unique_words = float(stats.get("unique_words", 0))
    word_count = float(stats.get("word_count", 1))
    punctuation_total = sum(stats.get("punctuation_counts", {}).values()) if stats.get("punctuation_counts") else 0

    def clamp01(x):
        return max(0.0, min(1.0, x))

    values = [
        clamp01(lexical_div),
        clamp01(avg_words / 30.0),  # assume 30 words as upper comfortable
        clamp01(flesch / 100.0),
        clamp01((avg_words / 25.0 + lexical_div) / 2.0),
        clamp01(punctuation_total / max(50.0, word_count)),
        clamp01((unique_words / max(word_count, 1.0))),
    ]
    return labels, values
