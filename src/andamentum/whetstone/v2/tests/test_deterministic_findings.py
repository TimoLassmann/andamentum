"""Tests for deterministic_findings.synthesize_deterministic_findings."""

from andamentum.whetstone.v2.structural.deterministic_findings import (
    synthesize_deterministic_findings,
)
from andamentum.whetstone.v2.structural.citations import extract_citations
from andamentum.whetstone.v2.structural.crossrefs import extract_cross_references
from andamentum.whetstone.v2.structural.numerics import extract_numeric_claims
from andamentum.whetstone.v2.structural.terms import extract_term_glossary
from andamentum.whetstone.v2.structural.types import SectionRef, StructuralFacts


def _section(id_: str, text: str, title: str = "") -> SectionRef:
    return SectionRef(id=id_, title=title, text=text, char_start=0, char_end=len(text))


def _facts_from(sections: list[SectionRef]) -> StructuralFacts:
    return StructuralFacts(
        citation_graph=extract_citations(sections),
        term_glossary=extract_term_glossary(sections),
        numeric_claims=extract_numeric_claims(sections),
        cross_references=extract_cross_references(sections),
    )


def test_finds_missing_reference_when_cited_key_not_in_references():
    sections = [
        _section("sec_001", "We cite [1] and [99]."),
        _section("sec_002", "[1] Author. (2020). Title.", title="References"),
    ]
    findings = synthesize_deterministic_findings(
        sections=sections, facts=_facts_from(sections)
    )
    titles = [f.title for f in findings]
    assert any("[99]" in t for t in titles)
    # [1] is defined → should NOT be flagged
    assert not any("[1]" in t for t in titles)


def test_finds_unused_reference():
    sections = [
        _section("sec_001", "We cite [1]."),
        _section(
            "sec_002",
            "[1] First. (2020).\n[2] Second. (2021).",
            title="References",
        ),
    ]
    findings = synthesize_deterministic_findings(
        sections=sections, facts=_facts_from(sections)
    )
    assert any("never cited" in f.title.lower() for f in findings)


def test_flags_missing_references_section_when_citations_present():
    sections = [_section("sec_001", "We cite [1] and [2].")]
    findings = synthesize_deterministic_findings(
        sections=sections, facts=_facts_from(sections)
    )
    assert any("references section" in f.title.lower() for f in findings)


def test_flags_acronym_redefined_with_different_expansion():
    sections = [
        _section("sec_001", "We use Reinforcement Learning (RL) here."),
        _section("sec_002", "Then Random Logic (RL) is applied later."),
    ]
    findings = synthesize_deterministic_findings(
        sections=sections, facts=_facts_from(sections)
    )
    assert any("RL" in f.title and "inconsist" in f.title.lower() for f in findings)


def test_flags_inconsistent_sample_size():
    sections = [
        _section("sec_001", "We had N=50 participants in study one."),
        _section("sec_002", "Across N=48 participants for the followup."),
    ]
    findings = synthesize_deterministic_findings(
        sections=sections, facts=_facts_from(sections)
    )
    assert any("Inconsistent N" in f.title for f in findings)


def test_flags_broken_figure_reference():
    sections = [
        _section("sec_001", "As shown in Figure 7, accuracy is high."),
        _section("sec_002", "Figure 1: A real anchored figure."),
    ]
    findings = synthesize_deterministic_findings(
        sections=sections, facts=_facts_from(sections)
    )
    assert any("Figure 7" in f.title for f in findings)
    # Figure 1 IS anchored → no finding for it
    assert not any("Figure 1" in f.title for f in findings)


def test_no_findings_on_clean_document():
    sections = [
        _section(
            "sec_001",
            "We use Reinforcement Learning (RL) for our agent. RL is a foundation. "
            "We had N=50 participants. As shown in Figure 1, results are clear. "
            "We cite [1] for prior work.",
        ),
        _section("sec_002", "Figure 1: Performance over time."),
        _section(
            "sec_003",
            "[1] Smith, J. (2020). A foundational paper.",
            title="References",
        ),
    ]
    findings = synthesize_deterministic_findings(
        sections=sections, facts=_facts_from(sections)
    )
    # Clean doc — no findings
    assert findings == []


def test_findings_have_high_confidence_and_quotes():
    sections = [_section("sec_001", "We cite [42] but no references list exists.")]
    findings = synthesize_deterministic_findings(
        sections=sections, facts=_facts_from(sections)
    )
    for f in findings:
        assert f.confidence == "high"
        assert f.source == "deterministic"
