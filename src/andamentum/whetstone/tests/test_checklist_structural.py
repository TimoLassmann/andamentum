"""Tests for the deterministic checklist checks (Step 1 — feature parity).

Each test is a tight focused case: one input, one expected outcome.
The point is to lock in the regex behaviour so future edits to the
patterns don't silently regress detection.
"""

from __future__ import annotations

from andamentum.whetstone.structural.checklist import run_checklist_checks
from andamentum.whetstone.structural.types import SectionRef


def _section(id_: str, title: str, text: str) -> SectionRef:
    return SectionRef(id=id_, title=title, text=text, char_start=0, char_end=len(text))


# ── Required statements ────────────────────────────────────────────────


def test_no_findings_when_all_statements_present():
    md = """\
# Some Paper

By Jane Doe, Department of Test Science, University of Test.

Keywords: alpha, beta, gamma

## Abstract

Background: We studied X. Methods: A randomised trial of N=50 patients
with ethics approval from the IRB. Results: Significant findings observed.
Conclusion: This matters because Y. The study contains enough words to
reach the wordcount minimum without padding by repeating itself a lot,
and so the abstract checker accepts it as well-sized for purposes of
this test which simulates a real-world abstract length adequately for
the keyword counter and the imrad cue detector to all be satisfied
together with the wordcount checker happy.

## Methods

We enrolled patients with informed consent.

## Conflicts of interest

The authors declare no conflicts of interest.

## Data availability

Data are available on request.

## Funding

This work was supported by grant number ABC-123.
"""
    sections = [
        _section("sec_001", "Some Paper", "# Some Paper\n"),
        _section(
            "sec_002",
            "Abstract",
            "## Abstract\n\nBackground: We studied X. Methods: A randomised trial "
            "of N=50 patients with ethics approval from the IRB. Results: "
            "Significant findings observed. Conclusion: This matters because Y. "
            "The study contains enough words to reach the wordcount minimum "
            "without padding by repeating itself a lot, and so the abstract "
            "checker accepts it as well-sized for purposes of this test which "
            "simulates a real-world abstract length adequately for the keyword "
            "counter and the imrad cue detector to all be satisfied together "
            "with the wordcount checker happy.",
        ),
    ]
    findings = run_checklist_checks(markdown=md, sections=sections)
    titles = [f.title for f in findings]
    # All four required-statement checks should pass:
    assert not any("Conflict-of-interest" in t for t in titles)
    assert not any("Data availability" in t for t in titles)
    assert not any("Ethics statement" in t for t in titles)
    assert not any("Funding / acknowledgements" in t for t in titles)


def test_missing_coi_flagged():
    md = "# Title\n\nBy Jane, Department of Test\n\nKeywords: a, b, c\n\nData availability: data on request. Ethics approval from IRB. Funding: grant ABC."
    sections = [_section("sec_001", "Title", md)]
    findings = run_checklist_checks(markdown=md, sections=sections)
    assert any("Conflict-of-interest" in f.title for f in findings)


def test_missing_data_availability_flagged():
    md = "# Title\n\nDepartment of Test\n\nKeywords: a, b, c\n\nCompeting interests: none. Ethics: IRB approved. Funding: none."
    sections = [_section("sec_001", "Title", md)]
    findings = run_checklist_checks(markdown=md, sections=sections)
    assert any("Data availability" in f.title for f in findings)


def test_ethics_only_required_when_subjects_mentioned():
    md = "# Title\n\nDepartment of Test\n\nKeywords: a, b, c\n\nCOI: none. Data sharing: yes. Funding: none. We computed prime numbers up to 1000."
    sections = [_section("sec_001", "Title", md)]
    findings = run_checklist_checks(markdown=md, sections=sections)
    # No subjects mentioned → no ethics finding
    assert not any("Ethics statement" in f.title for f in findings)


def test_ethics_required_when_animals_mentioned():
    md = "# Title\n\nDepartment of Test\n\nKeywords: a, b, c\n\nCOI: none. Data sharing: yes. Funding: none. We injected mice with reagent X."
    sections = [_section("sec_001", "Title", md)]
    findings = run_checklist_checks(markdown=md, sections=sections)
    assert any("Ethics statement" in f.title for f in findings)


def test_ethics_satisfied_by_iacuc_keyword():
    md = "# Title\n\nDepartment of Test\n\nKeywords: a, b, c\n\nCOI: none. Data sharing: yes. Funding: none. Mice were treated under IACUC protocol 42."
    sections = [_section("sec_001", "Title", md)]
    findings = run_checklist_checks(markdown=md, sections=sections)
    assert not any("Ethics statement" in f.title for f in findings)


def test_funding_picked_up_by_grant_number():
    md = "# Title\n\nDepartment of Test\n\nKeywords: a, b, c\n\nCOI: none. Data sharing: yes. Supported by grant number ABC-123 from NIH."
    sections = [_section("sec_001", "Title", md)]
    findings = run_checklist_checks(markdown=md, sections=sections)
    assert not any("Funding" in f.title for f in findings)


# ── Author affiliations ────────────────────────────────────────────────


def test_authors_block_detected_in_head():
    md = "# Title\n\nJane Doe, Department of Test, University of Test\n\nrest of doc..."
    sections = [_section("sec_001", "Title", md)]
    findings = run_checklist_checks(markdown=md, sections=sections)
    assert not any("Author affiliations" in f.title for f in findings)


def test_authors_block_missing_flagged_as_moderate():
    md = "Just plain prose with no institutional words at all in the head 2000 chars. " * 30
    sections = [_section("sec_001", "Title", md)]
    findings = run_checklist_checks(markdown=md, sections=sections)
    matching = [f for f in findings if "Author affiliations" in f.title]
    assert len(matching) == 1
    assert matching[0].severity == "moderate"


# ── Keywords ───────────────────────────────────────────────────────────


def test_keywords_count_in_range():
    md = "# Title\n\n## Keywords: alpha, beta, gamma, delta\n\nrest..."
    sections = [_section("sec_001", "Title", md)]
    findings = run_checklist_checks(markdown=md, sections=sections)
    assert not any("Keywords" in f.title for f in findings)


def test_keywords_too_few():
    md = "# Title\n\nKeywords: alpha, beta\n\nrest..."
    sections = [_section("sec_001", "Title", md)]
    findings = run_checklist_checks(markdown=md, sections=sections)
    matching = [f for f in findings if "at least" in f.title.lower()]
    assert len(matching) == 1


def test_keywords_too_many():
    md = "# Title\n\nKeywords: a, b, c, d, e, f, g, h, i, j, k\n\nrest..."
    sections = [_section("sec_001", "Title", md)]
    findings = run_checklist_checks(markdown=md, sections=sections)
    matching = [f for f in findings if "at most" in f.title.lower()]
    assert len(matching) == 1


def test_keywords_section_missing():
    md = "# Title\n\nNo keywords anywhere\n\nrest..."
    sections = [_section("sec_001", "Title", md)]
    findings = run_checklist_checks(markdown=md, sections=sections)
    matching = [f for f in findings if f.title == "Keywords section missing"]
    assert len(matching) == 1


# ── Title ──────────────────────────────────────────────────────────────


def test_title_in_range():
    md = "# Spaced repetition improves retention in undergraduate students\n\nrest..."
    sections = [_section("sec_001", "Title", md)]
    findings = run_checklist_checks(markdown=md, sections=sections)
    assert not any("Title" in f.title and "word" in f.title for f in findings)


def test_title_missing_h1():
    md = "Plain text with no heading anywhere.\n\nrest..."
    sections = [_section("sec_001", "Plain", md)]
    findings = run_checklist_checks(markdown=md, sections=sections)
    assert any(f.title == "Document title not identified" for f in findings)


def test_title_too_short():
    md = "# A study\n\nrest..."
    sections = [_section("sec_001", "Title", md)]
    findings = run_checklist_checks(markdown=md, sections=sections)
    assert any("only" in f.title.lower() and "word" in f.title.lower() for f in findings)


def test_title_too_long():
    md = "# " + " ".join(["word"] * 30) + "\n\nrest..."
    sections = [_section("sec_001", "Title", md)]
    findings = run_checklist_checks(markdown=md, sections=sections)
    assert any("trimming" in f.title.lower() for f in findings)


# ── Abstract ───────────────────────────────────────────────────────────


def test_abstract_section_missing():
    md = "# Title\n\nNo abstract section anywhere."
    sections = [_section("sec_001", "Title", md)]
    findings = run_checklist_checks(markdown=md, sections=sections)
    assert any(f.title == "Abstract section not identified" for f in findings)


def test_abstract_too_short():
    abstract_text = "## Abstract\n\nThis is too short."
    sections = [
        _section("sec_001", "Title", "# Title\n"),
        _section("sec_002", "Abstract", abstract_text),
    ]
    findings = run_checklist_checks(markdown="# Title\n\n" + abstract_text, sections=sections)
    assert any("only" in f.title.lower() and "word" in f.title.lower() for f in findings)


def test_abstract_too_long():
    long_body = " ".join(["word"] * 400)
    abstract_text = f"## Abstract\n\n{long_body} background methods results conclusion."
    sections = [
        _section("sec_001", "Title", "# Title\n"),
        _section("sec_002", "Abstract", abstract_text),
    ]
    findings = run_checklist_checks(markdown="# Title\n\n" + abstract_text, sections=sections)
    assert any("trimming" in f.title.lower() for f in findings)


def test_abstract_imrad_cues_present():
    body = (
        "Background: studied X. Methods: trial design. Results: significant. "
        "Conclusion: matters."
    ) * 10  # repeated to push above 150-word minimum
    abstract_text = f"## Abstract\n\n{body}"
    sections = [
        _section("sec_001", "Title", "# Title\n"),
        _section("sec_002", "Abstract", abstract_text),
    ]
    findings = run_checklist_checks(markdown="# Title\n\n" + abstract_text, sections=sections)
    assert not any("IMRAD" in f.title for f in findings)


def test_abstract_imrad_cues_missing():
    body = "We thought about it for a while and then decided to publish a paper. " * 30
    abstract_text = f"## Abstract\n\n{body}"
    sections = [
        _section("sec_001", "Title", "# Title\n"),
        _section("sec_002", "Abstract", abstract_text),
    ]
    findings = run_checklist_checks(markdown="# Title\n\n" + abstract_text, sections=sections)
    assert any("IMRAD" in f.title for f in findings)


# ── Determinism / source / category ────────────────────────────────────


def test_findings_have_compliance_or_metadata_categories():
    md = "Plain text, missing everything."
    sections = [_section("sec_001", "Plain", md)]
    findings = run_checklist_checks(markdown=md, sections=sections)
    # Every finding should have a category set (no empties)
    assert all(f.category in {"compliance", "metadata", "abstract"} for f in findings)


def test_findings_default_to_deterministic_source():
    md = "Plain text, missing everything."
    sections = [_section("sec_001", "Plain", md)]
    findings = run_checklist_checks(markdown=md, sections=sections)
    assert all(f.source == "deterministic" for f in findings)
