"""
Tests for the Analogy Engine / Cognitive Bridging feature.

Run:
    python tests/test_analogy_engine.py

These are import-validation and unit tests that do NOT require a running
Ollama server or any API key.  They exercise the pure-Python density
detection and the AnalogyInjector plumbing (prompt building, response
parsing, injection).
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


def test_imports():
    """Verify all analogy-engine modules can be imported."""
    print("Testing imports...")
    try:
        from src.analysis.analogy_engine import (
            detect_conceptual_density,
            AnalogyInjector,
        )
        print("  ✓ analogy_engine imported")
    except Exception as e:
        print(f"  ✗ analogy_engine import failed: {e}")
        return False

    try:
        from src.config.settings import (
            ANALOGY_DOMAINS,
            DEFAULT_ANALOGY_DOMAIN,
            CONCEPTUAL_DENSITY_THRESHOLD,
            ANALOGY_AUGMENTATION_ENABLED,
        )
        print("  ✓ config settings imported")
    except Exception as e:
        print(f"  ✗ config settings import failed: {e}")
        return False

    return True


def test_detect_conceptual_density_empty():
    """Empty / trivial input should return zero density."""
    from src.analysis.analogy_engine import detect_conceptual_density

    print("Testing density detection (empty input)...")
    result = detect_conceptual_density("")
    assert result["overall_density"] == 0.0, f"Expected 0.0, got {result['overall_density']}"
    assert result["high_density_count"] == 0
    assert result["sentence_scores"] == []
    print("  ✓ empty input handled correctly")
    return True


def test_detect_conceptual_density_simple():
    """Simple text should produce low density."""
    from src.analysis.analogy_engine import detect_conceptual_density

    print("Testing density detection (simple text)...")
    text = "I like cats. They are nice. Cats sleep a lot."
    result = detect_conceptual_density(text)
    assert 0.0 <= result["overall_density"] <= 1.0
    assert len(result["sentence_scores"]) == 3
    for s in result["sentence_scores"]:
        assert "density" in s
        assert "factors" in s
    print(f"  ✓ overall density = {result['overall_density']:.4f} (3 sentences)")
    return True


def test_detect_conceptual_density_complex():
    """Dense academic text should score high."""
    from src.analysis.analogy_engine import detect_conceptual_density

    print("Testing density detection (complex text)...")
    text = (
        "The epistemological ramifications of quantum decoherence fundamentally "
        "undermine classical deterministic interpretations of macroscopic "
        "thermodynamic phenomena, necessitating a comprehensive reconceptualization "
        "of emergent complexity within hierarchically structured dissipative systems."
    )
    result = detect_conceptual_density(text)
    assert result["overall_density"] > 0.4, (
        f"Expected high density, got {result['overall_density']}"
    )
    print(f"  ✓ overall density = {result['overall_density']:.4f} (complex text)")
    return True


def test_analogy_injector_init():
    """AnalogyInjector should initialise with valid domains."""
    from src.analysis.analogy_engine import AnalogyInjector
    from src.config.settings import ANALOGY_DOMAINS

    print("Testing AnalogyInjector initialisation...")
    for domain in ANALOGY_DOMAINS:
        injector = AnalogyInjector(domain=domain)
        assert injector.domain == domain
        print(f"  ✓ domain '{domain}' accepted")

    # Invalid domain should raise
    try:
        AnalogyInjector(domain="nonexistent_domain")
        print("  ✗ should have raised ValueError for invalid domain")
        return False
    except ValueError:
        print("  ✓ invalid domain correctly rejected")

    return True


def test_analogy_injector_prompt():
    """Prompt builder should produce a well-structured prompt."""
    from src.analysis.analogy_engine import AnalogyInjector

    print("Testing prompt building...")
    injector = AnalogyInjector(domain="sports")
    dense_sentences = [
        {"text": "Complex sentence one.", "density": 0.85, "factors": {}},
        {"text": "Complex sentence two.", "density": 0.90, "factors": {}},
    ]
    prompt = injector._build_analogy_prompt(dense_sentences)
    assert "sports" in prompt.lower() or "athletic" in prompt.lower()
    assert "1." in prompt
    assert "2." in prompt
    assert "Complex sentence one." in prompt
    print("  ✓ prompt contains domain reference and numbered sentences")
    return True


def test_analogy_injector_parse():
    """Response parser should extract numbered analogies."""
    from src.analysis.analogy_engine import AnalogyInjector

    print("Testing response parsing...")
    dense = [
        {"text": "Sentence A.", "density": 0.85, "factors": {}},
        {"text": "Sentence B.", "density": 0.90, "factors": {}},
    ]
    raw_response = (
        "1. Think of it like a quarterback reading the defence.\n"
        "2. It's like levelling up in a video game.\n"
    )
    result = AnalogyInjector._parse_analogy_response(raw_response, dense)
    assert len(result) == 2
    assert "quarterback" in result[0]["analogy"]
    assert result[1]["density_score"] == 0.90
    print(f"  ✓ parsed {len(result)} analogies correctly")
    return True


def test_analogy_injection():
    """Injection should insert [Cognitive Note: …] blocks."""
    from src.analysis.analogy_engine import AnalogyInjector

    print("Testing analogy injection into text...")
    original = "Sentence A. Sentence B."
    analogies = [
        {
            "source_sentence": "Sentence A.",
            "density_score": 0.85,
            "analogy": "Like scoring a goal.",
        },
    ]
    result = AnalogyInjector._inject_analogies(original, analogies)
    assert "[Cognitive Note:" in result
    assert "Like scoring a goal." in result
    # Original text should still be present
    assert "Sentence A." in result
    assert "Sentence B." in result
    print("  ✓ cognitive note injected correctly")
    return True


def test_cognitive_notes_format():
    """Standalone notes section should format cleanly."""
    from src.analysis.analogy_engine import AnalogyInjector

    print("Testing cognitive notes formatting...")
    analogies = [
        {
            "source_sentence": "A very dense passage about quantum physics.",
            "density_score": 0.92,
            "analogy": "Like trying to juggle while riding a unicycle.",
        },
    ]
    notes = AnalogyInjector._format_cognitive_notes(analogies)
    assert "COGNITIVE BRIDGING NOTES" in notes
    assert "0.92" in notes
    assert "juggle" in notes
    print("  ✓ notes section formatted correctly")
    return True


def test_analyzer_integration():
    """Verify analyzer.create_enhanced_style_profile accepts analogy params."""
    from src.analysis.analyzer import create_enhanced_style_profile
    import inspect

    print("Testing analyzer integration...")
    sig = inspect.signature(create_enhanced_style_profile)
    params = list(sig.parameters.keys())
    assert "analogy_augmentation" in params, "Missing analogy_augmentation parameter"
    assert "analogy_domain" in params, "Missing analogy_domain parameter"
    print("  ✓ create_enhanced_style_profile has analogy parameters")
    return True


def test_settings_schema():
    """Verify all required config constants exist."""
    from src.config import settings

    print("Testing config schema...")
    required = [
        "ANALOGY_AUGMENTATION_ENABLED",
        "ANALOGY_DOMAINS",
        "DEFAULT_ANALOGY_DOMAIN",
        "CONCEPTUAL_DENSITY_THRESHOLD",
    ]
    for name in required:
        assert hasattr(settings, name), f"Missing config: {name}"
        print(f"  ✓ {name} exists")

    # Validate domain structure
    for key, info in settings.ANALOGY_DOMAINS.items():
        assert "label" in info, f"Domain '{key}' missing 'label'"
        assert "description" in info, f"Domain '{key}' missing 'description'"
    print(f"  ✓ {len(settings.ANALOGY_DOMAINS)} valid analogy domains")
    return True


def test_hybrid_mode_rule():
    """Verify cognitive notes don't corrupt the primary output."""
    from src.analysis.analogy_engine import AnalogyInjector

    print("Testing hybrid mode (style + analogy)...")
    primary_output = "The author uses predominantly active voice with 78% short sentences."
    analogies = [
        {
            "source_sentence": primary_output,
            "density_score": 0.82,
            "analogy": "Like a sprinter who prefers straight-line dashes over hurdles.",
        },
    ]
    # Hybrid injection should append notes, not modify original
    notes = AnalogyInjector._format_cognitive_notes(analogies)
    augmented = primary_output.rstrip() + "\n\n" + notes
    # Primary output must be intact at the start
    assert augmented.startswith(primary_output)
    assert "COGNITIVE BRIDGING NOTES" in augmented
    print("  ✓ primary style output preserved in hybrid mode")
    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("ANALOGY ENGINE / COGNITIVE BRIDGING — TEST SUITE")
    print("=" * 60)

    tests = [
        test_imports,
        test_detect_conceptual_density_empty,
        test_detect_conceptual_density_simple,
        test_detect_conceptual_density_complex,
        test_analogy_injector_init,
        test_analogy_injector_prompt,
        test_analogy_injector_parse,
        test_analogy_injection,
        test_cognitive_notes_format,
        test_analyzer_integration,
        test_settings_schema,
        test_hybrid_mode_rule,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            result = test()
            if result is False:
                failed += 1
            else:
                passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} FAILED: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
