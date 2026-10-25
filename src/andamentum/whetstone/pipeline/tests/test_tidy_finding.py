"""Holding a model's finding to its fields: issue, headline, section names."""

from __future__ import annotations

from andamentum.whetstone.pipeline.digest import build_document_model
from andamentum.whetstone.pipeline.model import DocumentModel, Section, Span
from andamentum.whetstone.pipeline.prompt_blocks import (
    name_sections,
    sections_block,
    tidy_finding,
    title_from,
)
from andamentum.whetstone.pipeline.review import Finding
from andamentum.whetstone.pipeline.synth import _to_wfinding


def test_headline_and_issue_labels_are_split() -> None:
    issue, headline = tidy_finding(
        "Headline: Scope claim exceeds the data\nIssue: You claim X but show Y.",
        "",
        criterion="Story",
    )
    assert headline == "Scope claim exceeds the data"
    assert issue == "You claim X but show Y."


def test_an_unlabelled_first_line_before_issue_is_the_headline() -> None:
    text = "Over-general claim — Claim lacks evidence\nIssue: You state Z."
    issue, headline = tidy_finding(text, "", criterion="Evidence")
    assert headline == "Over-general claim — Claim lacks evidence"
    assert issue == "You state Z."


def test_a_given_headline_wins_and_the_first_line_is_dropped() -> None:
    text = "Over-general claim — Claim lacks evidence\nIssue: You state Z."
    issue, headline = tidy_finding(text, "My own headline", criterion="Evidence")
    assert headline == "My own headline"
    assert issue == "You state Z."


def test_a_criterion_prefix_is_removed() -> None:
    for prefix in ("Significance — ", "Significance: ", "significance - "):
        issue, _ = tidy_finding(
            f"{prefix}Scalability is asserted without evidence.",
            "",
            criterion="Significance",
        )
        assert issue == "Scalability is asserted without evidence."
    issue, _ = tidy_finding(
        "Significance: Scalability is asserted.", "", criterion="significance"
    )
    assert issue == "Scalability is asserted."


def test_a_headline_equal_to_the_criterion_is_empty() -> None:
    for given in ("Evaluations", "evaluations", " EVALUATIONS "):
        _, headline = tidy_finding(
            "You skip the baseline.", given, criterion="Evaluations"
        )
        assert headline == ""


def test_an_ordinary_issue_is_unchanged() -> None:
    text = "You report one run.\nThe second paragraph says more."
    assert tidy_finding(text, "Single run", criterion="Evidence") == (
        text,
        "Single run",
    )


def test_issue_mid_line_is_not_cut() -> None:
    text = "The real issue: the sample is small, and the issue: it recurs."
    issue, headline = tidy_finding(text, "", criterion="Evidence")
    assert issue == text
    assert headline == ""


def test_title_from() -> None:
    twelve = " ".join(f"w{i}" for i in range(12))
    assert title_from(twelve) == " ".join(f"w{i}" for i in range(8)) + " …"
    assert title_from("one two three") == "one two three"
    assert title_from("") == ""
    assert title_from("   ") == ""


TITLES = {"s19": "Methods", "s10": "Results"}


def test_section_ids_become_titles() -> None:
    out = name_sections("the methods section (s19) and section s10", TITLES)
    assert out == "the methods section (“Methods”) and the section “Results”"


def test_unknown_ids_and_table_labels_are_left_alone() -> None:
    assert name_sections("see s99 and Table S1", TITLES) == "see s99 and Table S1"


def _model() -> DocumentModel:
    text = "Intro text here. Methods text here."
    return build_document_model(
        text,
        [
            Section(
                id="s1", title="Introduction", text="Intro text here. ", start=0, end=16
            ),
            Section(
                id="s2",
                title="Methods",
                text="Methods text here.",
                start=16,
                end=len(text),
            ),
        ],
        [],
    )


def test_to_wfinding_titles_from_the_issue_not_the_criterion() -> None:
    f = Finding(
        criterion="Story",
        issue="Headline:\nIssue: You rely on s2 alone but never show the design.",
        quote="Methods",
        fix="Describe the design in section s2.",
        span=Span(section_id="s2", start=16, end=23),
    )
    public = _to_wfinding(f, _model())
    assert public.title != "Story"
    assert public.title.startswith("You rely on “Methods” alone")
    assert "s2" not in public.rationale
    assert "“Methods”" in public.rationale
    assert "“Methods”" in public.fix and "s2" not in public.fix
    assert "Headline:" not in public.rationale


def test_to_wfinding_keeps_a_headline_written_into_the_issue() -> None:
    f = Finding(
        criterion="Story",
        issue="Headline: Scope outruns the data\nIssue: You claim X but show Y.",
        quote="q",
    )
    public = _to_wfinding(f, _model())
    assert public.title == "Scope outruns the data"
    assert public.rationale == "You claim X but show Y."


def test_sections_block_names_parents_without_an_arrow() -> None:
    model = DocumentModel(
        source="x",
        sections=[
            Section(
                id="s2",
                title="Benchmark tasks",
                text="body",
                start=0,
                end=4,
                parents=["Results"],
            )
        ],
    )
    block = sections_block(model)
    assert "[s2] Benchmark tasks (in: Results)" in block
    assert "›" not in block
