"""Unit tests for the structural extractors (citations, terms, numerics, crossrefs)."""

from andamentum.whetstone.structural.citations import extract_citations
from andamentum.whetstone.structural.crossrefs import (
    extract_cross_references,
    find_figure_anchors,
    find_section_anchors,
    find_table_anchors,
)
from andamentum.whetstone.structural.numerics import extract_numeric_claims
from andamentum.whetstone.structural.terms import extract_term_glossary
from andamentum.whetstone.structural.types import SectionRef


def _section(id_: str, text: str, title: str = "") -> SectionRef:
    return SectionRef(id=id_, title=title, text=text, char_start=0, char_end=len(text))


# ── Citations ──────────────────────────────────────────────────────────


def test_numeric_citations_are_extracted():
    sec = _section("sec_001", "As shown in [1] and [3], plus [5-7] also support this.")
    graph = extract_citations([sec])
    keys = sorted(o.key.key for o in graph.occurrences)
    assert keys == ["1", "3", "5", "6", "7"]
    assert all(o.key.style == "numeric" for o in graph.occurrences)


def test_pandoc_citations_are_extracted():
    sec = _section("sec_001", "See [@smith2020] and [@jones2021; @lee2022].")
    graph = extract_citations([sec])
    keys = sorted(o.key.key for o in graph.occurrences)
    assert keys == ["jones2021", "lee2022", "smith2020"]


def test_references_section_is_recognised_and_entries_extracted():
    body = _section("sec_001", "Cited [1] and [2].")
    refs = _section(
        "sec_002",
        "[1] Author A. (2020). Title one.\n[2] Author B. (2021). Title two.",
        title="References",
    )
    graph = extract_citations([body, refs])
    assert graph.references_section_ids == ["sec_002"]
    assert "1" in graph.references_defined
    assert "Title one" in graph.references_defined["1"]
    # In-text citations from the references section itself are NOT counted.
    assert {o.key.key for o in graph.occurrences} == {"1", "2"}


def test_references_section_split_across_multiple_sections_is_handled():
    """When the chunker splits a long bibliography into multiple sections,
    we must recognise ALL of them as references, not just the titled one.

    This was a real bug: on the POET PDF (~80 references), the chunker
    produced sec_022 (just the heading), sec_023, sec_024, sec_025, sec_026
    (each containing ~20 reference entries). Only sec_022 was being
    recognised, so all real reference entries looked "missing" and the
    entry headers in sec_023+ were being counted as in-text citations.
    """
    body = _section("sec_001", "We cite [1], [2], [3], [4], and [5].")
    # First references section: just the heading, body is empty / minimal
    refs1 = _section("sec_002", "", title="References")
    # Continuation section: pure reference entries
    refs2 = _section(
        "sec_003",
        "- [1] Author A. (2020). Title one.\n"
        "- [2] Author B. (2021). Title two.\n"
        "- [3] Author C. (2022). Title three.",
    )
    # Another continuation
    refs3 = _section(
        "sec_004",
        "- [4] Author D. (2023). Title four.\n"
        "- [5] Author E. (2024). Title five.",
    )
    # A real prose section AFTER references — must NOT be counted as references
    appendix = _section(
        "sec_005",
        "## Appendix\n\nSome more text without [N]-style references.",
        title="Appendix",
    )
    graph = extract_citations([body, refs1, refs2, refs3, appendix])
    # All three reference sections detected
    assert graph.references_section_ids == ["sec_002", "sec_003", "sec_004"]
    # All five reference entries recognised
    assert sorted(graph.references_defined.keys()) == ["1", "2", "3", "4", "5"]
    # In-text citations from sec_001 only (5 unique keys), NOT from the
    # reference-list bodies in sec_003/sec_004.
    assert sorted({o.key.key for o in graph.occurrences}) == ["1", "2", "3", "4", "5"]


def test_references_continuation_does_not_consume_appendix():
    """An appendix or other section after the references list must not be
    misclassified as reference continuation."""
    refs = _section("sec_001", "", title="References")
    refs_body = _section(
        "sec_002",
        "[1] First. (2020).\n[2] Second. (2021).\n[3] Third. (2022).",
    )
    appendix = _section(
        "sec_003",
        "Long prose appendix. We discuss prior work [12] briefly here. "
        "More prose follows. Even more prose. Conclusion.",
        title="Appendix A",
    )
    graph = extract_citations([refs, refs_body, appendix])
    # sec_003 (appendix) should NOT be recognised as references
    assert graph.references_section_ids == ["sec_001", "sec_002"]
    # The [12] in the appendix IS counted as an in-text citation
    assert any(o.key.key == "12" for o in graph.occurrences)


# ── Terms / acronyms ───────────────────────────────────────────────────


def test_acronym_definition_is_extracted():
    sec = _section(
        "sec_001",
        "We use Reinforcement Learning (RL) to optimise the policy.",
    )
    glossary = extract_term_glossary([sec])
    assert len(glossary.definitions) == 1
    d = glossary.definitions[0]
    assert d.term == "RL"
    assert "Reinforcement Learning" in d.expansion


def test_acronym_usages_after_definition_are_collected():
    sec = _section(
        "sec_001",
        "We use Reinforcement Learning (RL) to optimise the policy. "
        "RL is a powerful framework. We applied RL to the maze task.",
    )
    glossary = extract_term_glossary([sec])
    # 1 definition + 3 usages (the parenthesised one + two more in prose)
    assert len(glossary.definitions) == 1
    rl_usages = [u for u in glossary.usages if u.term == "RL"]
    assert len(rl_usages) >= 3


def test_acronym_is_skipped_when_initials_dont_match():
    """`(XYZ)` after `Reinforcement Learning` should NOT be paired up."""
    sec = _section("sec_001", "Reinforcement Learning (XYZ) is great.")
    glossary = extract_term_glossary([sec])
    assert glossary.definitions == []


# ── Numerics ───────────────────────────────────────────────────────────


def test_sample_size_is_extracted():
    sec = _section("sec_001", "We recruited N = 50 participants. n=10 per group.")
    claims = extract_numeric_claims([sec])
    sample_size_values = [c.value for c in claims if c.kind == "sample_size"]
    assert "N=50" in sample_size_values
    assert "n=10" in sample_size_values


def test_p_value_is_extracted():
    sec = _section("sec_001", "Significant difference (p < 0.05) and another at p=.001.")
    claims = extract_numeric_claims([sec])
    p_values = [c.value for c in claims if c.kind == "p_value"]
    assert "<0.05" in p_values
    # p=.001 → normalised to "=0.001"
    assert any(v.startswith("=") and "0.001" in v for v in p_values)


def test_percentage_is_extracted():
    sec = _section("sec_001", "Accuracy improved by 12.5% over baseline.")
    claims = extract_numeric_claims([sec])
    pcts = [c.value for c in claims if c.kind == "percentage"]
    assert "12.5" in pcts


# ── Cross-references ───────────────────────────────────────────────────


def test_figure_reference_is_extracted():
    sec = _section("sec_001", "Figure 3 shows the layout. See also Fig. 4 below.")
    refs = extract_cross_references([sec])
    fig_targets = sorted(r.target for r in refs if r.kind == "figure")
    assert fig_targets == ["3", "4"]


def test_section_reference_is_extracted():
    sec = _section("sec_001", "As described in Section 2.1, the method works.")
    refs = extract_cross_references([sec])
    sec_targets = [r.target for r in refs if r.kind == "section"]
    assert "2.1" in sec_targets


def test_figure_anchor_detection():
    sec = _section(
        "sec_002",
        "## Results\n\nSome text.\n\nFigure 3: A plot of accuracy vs. epochs.",
    )
    assert "3" in find_figure_anchors([sec])


def test_table_anchor_detection():
    sec = _section("sec_002", "Table 5: Comparison of model sizes.")
    assert "5" in find_table_anchors([sec])


def test_section_anchor_detection_from_numbered_heading():
    sec = _section(
        "sec_003",
        "## 4.1 Experimental Setup\n\nWe ran 100 trials.",
        title="4.1 Experimental Setup",
    )
    assert "4.1" in find_section_anchors([sec])
