"""Tests for the overclaim lens (Step 11 — reviewer 2 bait detection).

Two surfaces:

1. The lexicon of overclaim words is exhaustive enough to catch the
   common patterns. Tested via ``find_overclaim_candidates``.
2. The lens is registered in LENS_PROMPTS as a per-section lens (not
   multi-section), with a prompt that mentions all four overclaim
   categories.
"""

from __future__ import annotations

from andamentum.whetstone.v2.agents.lens_prompts import (
    LENS_MULTI_SECTION,
    LENS_PROMPTS,
)
from andamentum.whetstone.v2.structural.overclaim_lexicon import (
    find_overclaim_candidates,
)


# ── Lexicon ────────────────────────────────────────────────────────────


def test_finds_novelty_words():
    candidates = find_overclaim_candidates(
        "We present the first comprehensive analysis of this novel approach."
    )
    cats = [c[0] for c in candidates]
    assert "novelty" in cats
    # Should match both "first" and "novel"
    phrases = [c[1].lower() for c in candidates]
    assert "first" in phrases
    assert "novel" in phrases


def test_finds_strength_words():
    candidates = find_overclaim_candidates(
        "We observed dramatic improvements with robust statistical support."
    )
    cats = [c[0] for c in candidates]
    assert "strength" in cats


def test_finds_causal_words():
    candidates = find_overclaim_candidates(
        "X causes Y, leading to a marked decrease in Z."
    )
    cats = [c[0] for c in candidates]
    assert "causal" in cats


def test_finds_generalisation_words():
    candidates = find_overclaim_candidates(
        "These findings apply broadly in humans and across the population."
    )
    cats = [c[0] for c in candidates]
    assert "generalisation" in cats


def test_case_insensitive():
    candidates = find_overclaim_candidates("FIRST AND NOVEL APPROACH")
    assert len(candidates) >= 2


def test_word_boundary_respected():
    # "robustness" should not match the "robust" word
    candidates = find_overclaim_candidates(
        "We tested the robustness of the algorithm."
    )
    phrases = [c[1].lower() for c in candidates]
    assert "robust" not in phrases


def test_multi_word_phrase_matches():
    candidates = find_overclaim_candidates(
        "This is a paradigm-shifting result."
    )
    phrases = [c[1].lower() for c in candidates]
    assert "paradigm-shifting" in phrases


def test_returns_empty_on_clean_prose():
    candidates = find_overclaim_candidates(
        "We measured the temperature of three samples and recorded the results."
    )
    assert candidates == []


# ── Lens registration ──────────────────────────────────────────────────


def test_overclaim_registered_as_lens():
    assert "overclaim" in LENS_PROMPTS


def test_overclaim_is_per_section_not_multi():
    # Overclaim is detected at the section level — each section's
    # claims need to match THAT section's evidence. Multi-section
    # reading would dilute focus.
    assert LENS_MULTI_SECTION.get("overclaim", False) is False


def test_overclaim_prompt_mentions_all_four_categories():
    prompt = LENS_PROMPTS["overclaim"]
    # The persona prompt should explicitly call out each overclaim
    # category so the LLM has clear coverage.
    for keyword in ("novelty", "strength", "mechanistic", "generalisation"):
        assert keyword.lower() in prompt.lower(), (
            f"overclaim prompt missing {keyword!r} category"
        )


def test_overclaim_prompt_lists_specific_words():
    """Prompt should enumerate the actual words to flag, not just describe
    them abstractly. This is what step 2 (prompt enrichment) demanded
    for every lens."""
    prompt = LENS_PROMPTS["overclaim"]
    # Sample a few words from each category — prompt should mention
    # them by name so the LLM has concrete recall targets.
    for word in ("first", "novel", "unprecedented", "dramatic", "robust"):
        assert word in prompt.lower(), (
            f"overclaim prompt missing concrete word {word!r}"
        )


def test_overclaim_prompt_warns_against_legitimate_uses():
    """Prompt should explicitly carve out cases where strong language IS
    appropriate so the lens doesn't generate 50 false positives per
    paragraph."""
    prompt = LENS_PROMPTS["overclaim"].lower()
    # Should mention either "supported by", "evidence", or "limitations"
    assert "supported" in prompt or "limitations" in prompt
