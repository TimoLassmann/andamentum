"""The AI-use statement is built from logged events only."""

from __future__ import annotations

from andamentum.compose.harness.draft.disclosure import HEADING, ai_use_statement


def step(step: str, model: str, detail: str = "") -> dict[str, object]:
    return {
        "type": "draft_step",
        "step": step,
        "role": "drafter",
        "model": model,
        "detail": detail,
    }


def test_heading() -> None:
    assert HEADING == "Statement on the use of AI"


def test_an_empty_log_says_only_what_is_certain() -> None:
    text = ai_use_statement([])
    assert text.startswith("This document was drafted with the help of an AI system")
    assert "drafted by" not in text and "Citations" not in text and "checks" not in text
    assert text.endswith(
        "The authors are responsible for the content of this document."
    )


def test_models_sections_and_revisions() -> None:
    events = [
        step("brief", "m:a"),
        step("evidence", "m:a", "src1"),
        step("outline", "m:a"),
        step("draft", "m:a", "Methods"),
        step("draft", "m:b", "Results"),
        step("revise", "m:b", "Results"),
        step("claims", "m:j", "Results"),
    ]
    text = ai_use_statement(events)
    assert "were produced by m:a." in text  # the same model three times is named once
    assert 'Section "Methods" was drafted by m:a.' in text
    assert 'Section "Results" was drafted by m:b.' in text
    assert (
        'Section "Results" was revised by m:b after automated checks reported errors.'
        in text
    )
    assert "m:j" not in text


def test_checks_citations_and_numbers_are_conditional() -> None:
    events = [
        {"type": "check_run", "checks": ["crossrefs", "statistics"]},
        {"type": "check_run", "checks": ["crossrefs", "citations"]},
        {"type": "draft_run", "citations": 2, "numbers_bound": 0},
        {"type": "draft_run", "citations": 1, "numbers_bound": 3},
    ]
    text = ai_use_statement(events)
    assert (
        "Automated checks run on the draft: crossrefs, statistics and citations."
        in text
    )
    assert (
        "Citations were inserted only for works whose DOI and metadata were verified"
        in text
    )
    assert "Numbers taken from result files are bound" in text
    only_checks = ai_use_statement([events[0]])
    assert "Citations" not in only_checks and "Numbers" not in only_checks


def test_it_is_one_paragraph() -> None:
    text = ai_use_statement([step("draft", "m:a", "A"), step("draft", "m:a", "B")])
    assert "\n" not in text
    assert 'Sections "A" and "B" were drafted by m:a.' in text
