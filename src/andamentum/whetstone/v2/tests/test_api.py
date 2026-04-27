"""End-to-end test for review_document on a synthetic-fault paper.

No LLM calls in Phase 1 — tests the full pipeline (harvest → chunk →
scan → deterministic findings) against a paper engineered to trip every
deterministic check.
"""


from andamentum.whetstone.v2 import ReviewResult, review_document


# A small "paper" with planted issues:
#   • Cites [42] which has no entry in references
#   • References [1] and [2] but [2] is never cited
#   • Defines "RL" twice with different expansions
#   • Says N=50 in one section and N=48 in another
#   • References "Figure 7" but only Figure 1 is anchored
PAPER_WITH_FAULTS = """\
## 1 Introduction

This paper studies Reinforcement Learning (RL) applied to bipedal walking.
We had N = 50 participants in our user study, and cite prior work [1, 42].
As shown in Figure 7, the results are striking.

## 2 Methods

We compare two variants of Random Logic (RL) on the same benchmark.
Across N=48 trials, the new method outperforms baselines significantly.
Figure 1: Comparison of accuracy across methods.

## References

[1] First Author. (2020). Title one.
[2] Second Author. (2021). Title two.
"""


async def test_review_document_returns_result_with_deterministic_findings():
    result = await review_document(PAPER_WITH_FAULTS)
    assert isinstance(result, ReviewResult)
    # Every deterministic finding has high confidence and source="deterministic"
    for f in result.deterministic_findings:
        assert f.confidence == "high"
        assert f.source == "deterministic"


async def test_review_document_finds_missing_citation_42():
    result = await review_document(PAPER_WITH_FAULTS)
    titles = [f.title for f in result.deterministic_findings]
    assert any("[42]" in t for t in titles), f"Got titles: {titles}"


async def test_review_document_finds_unused_reference_2():
    result = await review_document(PAPER_WITH_FAULTS)
    titles = [f.title for f in result.deterministic_findings]
    assert any("never cited" in t.lower() for t in titles), f"Got titles: {titles}"


async def test_review_document_finds_redefined_acronym_RL():
    result = await review_document(PAPER_WITH_FAULTS)
    titles = [f.title for f in result.deterministic_findings]
    assert any("RL" in t and "inconsist" in t.lower() for t in titles), (
        f"Got titles: {titles}"
    )


async def test_review_document_finds_inconsistent_sample_size():
    result = await review_document(PAPER_WITH_FAULTS)
    titles = [f.title for f in result.deterministic_findings]
    assert any("Inconsistent N" in t for t in titles), f"Got titles: {titles}"


async def test_review_document_finds_broken_figure_reference():
    result = await review_document(PAPER_WITH_FAULTS)
    titles = [f.title for f in result.deterministic_findings]
    assert any("Figure 7" in t for t in titles), f"Got titles: {titles}"


async def test_review_document_populates_document_map():
    result = await review_document(PAPER_WITH_FAULTS)
    assert len(result.document_map) >= 2  # at least Intro + Methods
    # Each card has section_id + title; one_line_gist is a deterministic best-effort
    for card in result.document_map:
        assert card.section_id.startswith("sec_")
        assert card.title  # non-empty


async def test_review_document_reports_metrics():
    result = await review_document(PAPER_WITH_FAULTS)
    m = result.metrics
    assert m.llm_calls == 0  # Phase 1: no LLM
    assert m.deterministic_findings_count == len(result.deterministic_findings)
    assert m.sections_processed >= 2
    assert m.wall_seconds >= 0


async def test_review_document_no_findings_on_clean_paper():
    """A trivially clean paper produces an empty findings list."""
    clean = """## 1 Introduction

We study a simple problem and present results.

## 2 Conclusion

Our findings are encouraging.
"""
    result = await review_document(clean)
    # No citations, no acronyms, no figure refs → no deterministic findings
    assert result.deterministic_findings == []


async def test_review_document_handles_path_input(tmp_path):
    """Passing a file path works the same as raw markdown."""
    p = tmp_path / "paper.md"
    p.write_text(PAPER_WITH_FAULTS)
    result = await review_document(p)
    assert len(result.deterministic_findings) > 0


async def test_phase_1_summary_is_empty():
    """Phase 4's Synthesise hasn't shipped yet — summary is empty."""
    result = await review_document(PAPER_WITH_FAULTS)
    assert result.summary == ""
    assert result.findings == []  # no LLM-driven findings yet
    assert result.author_questions == []
