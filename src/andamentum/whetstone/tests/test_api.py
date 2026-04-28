"""End-to-end test for review_document on a synthetic-fault paper.

No LLM calls in Phase 1 — tests the full pipeline (harvest → chunk →
scan → deterministic findings) against a paper engineered to trip every
deterministic check.
"""


from andamentum.whetstone import ReviewResult, review_document


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
    # Every deterministic finding has source="deterministic"; structural
    # checks are high-confidence, checklist body-shape checks (e.g.
    # "Abstract not identified", "Author block not detected") are medium.
    for f in result.deterministic_findings:
        assert f.confidence in ("high", "medium")
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
    """A trivially clean paper produces an empty findings list.

    "Clean" here means: no structural faults (citations, acronyms,
    cross-refs, sample sizes) AND no missing pre-submission checklist
    items (title, keywords, abstract, required statements). The
    deterministic substrate covers both, so this test asserts the
    union is empty.
    """
    clean = """\
# A study of how spaced repetition affects retention in students

By Jane Doe, Department of Educational Psychology, University of Test.

Keywords: spaced repetition, retention, learning, undergraduates

## Abstract

Background: Spaced repetition is a learning technique whose effect on long-term
retention has been studied across multiple disciplines and contexts, but the
evidence in undergraduate cohorts is fragmented and the effect size estimates
vary widely. The present study seeks to replicate and extend prior work in a
new task domain. Methods: We conducted a randomised crossover trial with N=50
undergraduate participants who were assigned to spaced or massed practice
schedules over four weeks. Each participant completed standardised recall
assessments at one-week and four-week follow-up timepoints, and analyses
were preregistered. Results: Participants in the spaced condition showed
significantly improved recall at the four-week follow-up compared with the
massed condition, with a moderate effect size that was consistent across
subgroups defined by prior coursework, gender, and weekly study time.
Conclusion: Spaced repetition meaningfully improves retention in this
undergraduate population, replicating prior findings and extending them to a
new task domain. We discuss implications for course-level study-skill
guidance and outline pre-registered follow-up work.

## 1 Introduction

We study a simple problem and present results.

## 2 Conclusion

Our findings are encouraging.

## Conflicts of interest

The author declares no conflicts of interest.

## Data availability

Data are available on request from the corresponding author.

## Funding

This work was supported by a Test University internal grant.

## Ethics

This study was approved by the Test University IRB (protocol #42).
"""
    result = await review_document(clean)
    # No structural faults AND no missing checklist items → no findings.
    titles = [f.title for f in result.deterministic_findings]
    assert result.deterministic_findings == [], f"Got findings: {titles}"


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
