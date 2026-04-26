"""Tests for prompt template + domain hints."""

from andamentum.chunker.prompts import (
    DOMAIN_HINTS,
    SYSTEM_PROMPT,
    build_user_prompt,
)


def test_domain_hints_cover_expected_domains():
    expected = {"academic", "web", "code", "transcript", "general"}
    assert set(DOMAIN_HINTS) == expected


def test_system_prompt_forbids_rewriting():
    assert (
        "do NOT rewrite" in SYSTEM_PROMPT.lower()
        or "not rewrite" in SYSTEM_PROMPT.lower()
        or "verbatim" in SYSTEM_PROMPT.lower()
    )


def test_build_user_prompt_includes_window_text():
    prompt = build_user_prompt(
        window_text="Some text here.",
        domain="academic",
        window_size=10_000,
        prior_unit_titles=[],
    )
    assert "Some text here." in prompt
    assert "10000" in prompt or "10,000" in prompt or "first" in prompt.lower()


def test_build_user_prompt_includes_domain_hint():
    prompt = build_user_prompt(
        window_text="x",
        domain="web",
        window_size=1000,
        prior_unit_titles=[],
    )
    # Web hint mentions navigation
    assert "navigation" in prompt.lower() or "ads" in prompt.lower()


def test_build_user_prompt_includes_prior_titles_when_present():
    prompt = build_user_prompt(
        window_text="x",
        domain="general",
        window_size=1000,
        prior_unit_titles=["Introduction", "Methods"],
    )
    assert "Introduction" in prompt
    assert "Methods" in prompt


def test_build_user_prompt_omits_prior_section_when_empty():
    prompt = build_user_prompt(
        window_text="x",
        domain="general",
        window_size=1000,
        prior_unit_titles=[],
    )
    # No "previous units" section header when no priors
    assert "previous unit" not in prompt.lower() or "no previous" in prompt.lower()
