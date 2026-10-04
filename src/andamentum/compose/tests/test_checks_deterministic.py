from __future__ import annotations

from typing import Any

from andamentum.compose.checks.deterministic import (
    errors,
    run_deterministic_checks,
    to_fill_items,
)
from andamentum.compose.genres import genre_for
from andamentum.compose.schemas import Brief, Fact, Outline, SectionPlan
from andamentum.scribe import Section


def brief(**kw) -> Brief:
    base: dict[str, Any] = dict(
        task="t",
        genre="memo",
        title="Apollo plan",
        audience="the Apollo Labs board",
        purpose="decide",
        key_messages=["k"],
        must_include=[],
        constraints=[],
        word_range=(10, 200),
        reader_questions=[],
    )
    base.update(kw)
    return Brief(**base)


def section(i: int, title: str, body: str) -> Section:
    return Section(
        id=f"s{i:02d}",
        title=title,
        order=i * 10,
        revision=1,
        status="drafted",
        body=body,
    )


def fact(fid: str) -> Fact:
    return Fact(id=fid, source_id="src1", statement="s", quote="q", start=0, end=1)


def names(results, *, failed_only=True):
    return {r.name for r in results if not (failed_only and r.passed)}


def test_clean_document_passes():
    genre = genre_for("email")
    sections = [
        section(1, "Body", "Jane Doe, please send the Apollo budget by Friday. " * 3)
    ]
    results = run_deterministic_checks(
        sections, brief=brief(word_range=(20, 100)), genre=genre, outline=None, facts=[]
    )
    assert errors(results) == []


def test_total_words_out_of_range_is_error():
    sections = [section(1, "Body", "Short text here.")]
    results = run_deterministic_checks(
        sections,
        brief=brief(word_range=(50, 100)),
        genre=genre_for("email"),
        outline=None,
        facts=[],
    )
    bad = [r for r in errors(results) if r.name == "total_words"]
    assert len(bad) == 1 and "too short" in bad[0].details[0]


def test_required_sections_missing():
    genre = genre_for("memo")
    sections = [section(1, "Summary", "word " * 30)]
    results = run_deterministic_checks(
        sections, brief=brief(), genre=genre, outline=None, facts=[]
    )
    bad = next(r for r in errors(results) if r.name == "required_sections")
    assert any("Recommendation" in d for d in bad.details)


def test_required_sections_skipped_for_flexible_with_guidelines():
    genre = genre_for("grant")
    sections = [section(1, "Aims", "word " * 30)]
    results = run_deterministic_checks(
        sections,
        brief=brief(guidelines="Fictional Foundation call text."),
        genre=genre,
        outline=None,
        facts=[],
    )
    assert "required_sections" not in names(errors(results), failed_only=False)


def test_flexible_genre_uses_outline_titles():
    genre = genre_for("report")
    outline = Outline(
        sections=[
            SectionPlan(
                title="Overview",
                purpose="p",
                key_points=[],
                fact_ids=[],
                target_words=30,
            )
        ]
    )
    sections = [section(1, "Overview", "word " * 30)]
    results = run_deterministic_checks(
        sections, brief=brief(), genre=genre, outline=outline, facts=[]
    )
    assert "required_sections" not in names(errors(results))


def test_must_include_case_insensitive_and_names_missing():
    sections = [section(1, "Body", "The launch is on friday. " * 6)]
    results = run_deterministic_checks(
        sections,
        brief=brief(must_include=["Friday", "Saltlog pilot"], word_range=(10, 200)),
        genre=genre_for("email"),
        outline=None,
        facts=[],
    )
    bad = next(r for r in errors(results) if r.name == "must_include")
    assert bad.details == ["missing: Saltlog pilot"]


def test_heading_in_body_is_section_error():
    sections = [section(1, "Body", "Intro text.\n## A heading\nMore text here now.")]
    results = run_deterministic_checks(
        sections,
        brief=brief(word_range=(1, 200)),
        genre=genre_for("email"),
        outline=None,
        facts=[],
    )
    bad = next(r for r in errors(results) if r.name == "headings_in_bodies")
    assert bad.section_id == "s01"


def test_unknown_fact_marker_is_error():
    sections = [
        section(1, "Body", "Sales rose [[f1]] and fell [[f9]]. " + "word " * 20)
    ]
    results = run_deterministic_checks(
        sections,
        brief=brief(word_range=(1, 200)),
        genre=genre_for("email"),
        outline=None,
        facts=[fact("f1")],
    )
    bad = next(r for r in errors(results) if r.name == "fact_markers")
    assert "f9" in bad.details[0] and bad.section_id == "s01"


def test_to_fill_is_warning_and_listed():
    sections = [section(1, "Body", "Budget is [TO FILL: total cost]. " + "word " * 20)]
    results = run_deterministic_checks(
        sections,
        brief=brief(word_range=(1, 200)),
        genre=genre_for("email"),
        outline=None,
        facts=[],
    )
    assert "to_fill" not in names(errors(results))
    assert any("total cost" in d for d in to_fill_items(results))


def test_section_words_warning_only():
    outline = Outline(
        sections=[
            SectionPlan(
                title="Body", purpose="p", key_points=[], fact_ids=[], target_words=500
            )
        ]
    )
    sections = [section(1, "Body", "word " * 60)]
    results = run_deterministic_checks(
        sections,
        brief=brief(word_range=(1, 200)),
        genre=genre_for("email"),
        outline=outline,
        facts=[],
    )
    assert "section_words" in names(results)
    assert "section_words" not in names(errors(results))


def test_grade_level_ceiling():
    hard = (
        "Notwithstanding considerable organisational heterogeneity, the interdisciplinary "
        "committee's deliberations predominantly necessitated comprehensive "
        "reconsideration of institutional methodological assumptions. "
    ) * 3
    sections = [section(1, "Body", hard)]
    results = run_deterministic_checks(
        sections,
        brief=brief(word_range=(1, 500)),
        genre=genre_for("email"),
        outline=None,
        facts=[],
    )
    bad = next(r for r in errors(results) if r.name == "grade_level")
    assert bad.section_id == "s01"


def test_no_grade_check_without_ceiling():
    results = run_deterministic_checks(
        [section(1, "Summary", "word " * 40)],
        brief=brief(),
        genre=genre_for("memo"),
        outline=None,
        facts=[],
    )
    assert "grade_level" not in {r.name for r in results}
