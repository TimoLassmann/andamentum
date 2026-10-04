from __future__ import annotations

from andamentum.compose.route import route
from andamentum.compose.schemas import (
    CheckResult,
    Outline,
    ReaderQuestion,
    SectionPlan,
    WorkItem,
)
from andamentum.compose.bundle import Section

from ._stub import make_brief

BRIEF = make_brief(word_range=(100, 200))


def sec(i: int, title: str) -> Section:
    return Section(
        id=f"s{i:02d}",
        title=title,
        order=i * 10,
        revision=1,
        status="drafted",
        body="x",
    )


SECTIONS = [sec(1, "Summary"), sec(2, "Findings")]
OUTLINE = Outline(
    sections=[
        SectionPlan(
            title="Summary",
            purpose="p",
            key_points=["Apollo launches in May"],
            fact_ids=[],
            target_words=50,
        ),
        SectionPlan(
            title="Findings",
            purpose="p",
            key_points=["Saltlog cut costs by 12 percent"],
            fact_ids=[],
            target_words=80,
        ),
    ]
)


def test_groups_items_per_section():
    checks = [
        CheckResult(
            name="grounding",
            passed=False,
            details=["f1: bad", "f2: worse"],
            section_id="s02",
        ),
        CheckResult(
            name="headings_in_bodies", passed=False, details=["h"], section_id="s02"
        ),
        CheckResult(name="grounding", passed=True, details=[], section_id="s01"),
    ]
    routed = route(checks, sections=SECTIONS, outline=OUTLINE, brief=BRIEF)
    assert list(routed.by_section) == ["s02"]
    assert len(routed.by_section["s02"]) == 3
    assert routed.outline == []


def test_warnings_and_passes_are_not_routed():
    checks = [
        CheckResult(
            name="section_words", passed=False, details=["x"], section_id="s01"
        ),
        CheckResult(name="to_fill", passed=False, details=["y"]),
        CheckResult(name="total_words", passed=True, details=[]),
    ]
    assert route(checks, sections=SECTIONS, outline=OUTLINE, brief=BRIEF).items == []


def test_must_include_maps_to_planned_section():
    checks = [
        CheckResult(
            name="must_include", passed=False, details=["missing: Apollo launches"]
        )
    ]
    routed = route(checks, sections=SECTIONS, outline=OUTLINE, brief=BRIEF)
    assert list(routed.by_section) == ["s01"]


def test_must_include_unknown_goes_to_outline():
    checks = [
        CheckResult(name="must_include", passed=False, details=["missing: Zeta budget"])
    ]
    routed = route(checks, sections=SECTIONS, outline=OUTLINE, brief=BRIEF)
    assert routed.by_section == {} and len(routed.outline) == 1


def test_total_words_routes_to_sections_that_can_absorb_it():
    sections = [
        sec(1, "Summary").model_copy(update={"body": "word " * 20}),
        sec(2, "Findings").model_copy(update={"body": "word " * 40}),
    ]
    checks = [CheckResult(name="total_words", passed=False, details=["too short"])]
    routed = route(checks, sections=sections, outline=OUTLINE, brief=BRIEF)
    assert routed.outline == []
    assert sorted(routed.by_section) == ["s01", "s02"]
    assert (
        "lengthen this section to about 50 words" in routed.by_section["s01"][0].reason
    )
    long = [s.model_copy(update={"body": "word " * 200}) for s in sections]
    cut = route(checks, sections=long, outline=OUTLINE, brief=BRIEF)
    assert "cut this section to about 75 words" in cut.by_section["s01"][0].reason


def test_total_words_and_required_sections_are_outline_level():
    sections = [s.model_copy(update={"body": ""}) for s in SECTIONS]
    checks = [
        CheckResult(name="total_words", passed=False, details=["too short"]),
        CheckResult(
            name="required_sections", passed=False, details=["missing section: Risks"]
        ),
    ]
    routed = route(checks, sections=sections, outline=OUTLINE, brief=BRIEF)
    assert len(routed.outline) == 2


def test_reader_miss_maps_by_key_point_overlap():
    miss = ReaderQuestion(
        question="What did Saltlog achieve?",
        expected_answer="Saltlog cut costs by 12 percent",
    )
    routed = route(
        [], sections=SECTIONS, outline=OUTLINE, brief=BRIEF, reader_misses=[miss]
    )
    assert list(routed.by_section) == ["s02"]
    assert routed.by_section["s02"][0].source == "reader"


def test_reader_miss_unidentifiable_goes_to_outline():
    miss = ReaderQuestion(question="Q?", expected_answer="entirely unrelated matter")
    routed = route(
        [], sections=SECTIONS, outline=OUTLINE, brief=BRIEF, reader_misses=[miss]
    )
    assert len(routed.outline) == 1


def test_critic_items_pass_through_and_duplicates_collapse():
    item = WorkItem(
        level="section", section_id="s01", reason="[major] r", source="critic"
    )
    out = WorkItem(
        level="outline", section_id=None, reason="[moderate] o", source="critic"
    )
    routed = route(
        [],
        sections=SECTIONS,
        outline=OUTLINE,
        brief=BRIEF,
        critic_items=[item, item, out],
    )
    assert len(routed.by_section["s01"]) == 1
    assert routed.outline == [out]
    assert len(routed.items) == 2


def test_no_outline_still_routes():
    checks = [CheckResult(name="must_include", passed=False, details=["missing: X"])]
    routed = route(checks, sections=SECTIONS, outline=None, brief=BRIEF)
    assert len(routed.outline) == 1
