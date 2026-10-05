"""The v2 records: validation, content addressing, and the result's exports."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from andamentum.deep_research.v2.types import (
    Budget,
    Candidate,
    CandidateScore,
    Conflict,
    Criterion,
    Finding,
    Passage,
    ResearchResult,
    passage_id_for,
)
from andamentum.references import LICENCE_NO_REDISTRIBUTE, Work


def test_criterion_weight_is_bounded_and_the_model_is_frozen() -> None:
    assert Criterion(id="c1", text="Is it a trial?", kind="must").weight == 1
    for weight in (0, 4):
        with pytest.raises(ValidationError):
            Criterion(id="c1", text="Is it a trial?", kind="must", weight=weight)
    with pytest.raises(ValidationError):
        Criterion(id="c1", text="Is it a trial?", kind="maybe")  # type: ignore[arg-type]
    criterion = Criterion(id="c1", text="Is it a trial?", kind="must")
    with pytest.raises(ValidationError):
        criterion.weight = 2  # type: ignore[misc]


def test_candidate_needs_a_source() -> None:
    with pytest.raises(ValidationError):
        Candidate(work=Work(), sources=[])


def test_candidate_score_probabilities_are_checked() -> None:
    CandidateScore(
        work_id="w", subquestion_id="s", criterion_p={"c1": 0.4}, score=0.4, kept=True
    )
    with pytest.raises(ValidationError, match="not in"):
        CandidateScore(
            work_id="w",
            subquestion_id="s",
            criterion_p={"c1": 1.4},
            score=0.4,
            kept=True,
        )
    with pytest.raises(ValidationError):
        CandidateScore(work_id="w", subquestion_id="s", score=1.2, kept=True)


def test_passage_is_content_addressed_and_an_exact_slice() -> None:
    passage = Passage.make(work_id="w_1", start=10, text="Some verbatim text.")
    assert passage.end == 10 + len("Some verbatim text.")
    assert passage.passage_id == passage_id_for("w_1", "Some verbatim text.")
    other_work = Passage.make(work_id="w_2", start=10, text="Some verbatim text.")
    assert other_work.passage_id != passage.passage_id
    with pytest.raises(ValidationError, match="do not span"):
        Passage(
            passage_id=passage.passage_id,
            work_id="w_1",
            start=0,
            end=5,
            text="Some verbatim text.",
        )
    with pytest.raises(ValidationError, match="sha256"):
        Passage(
            passage_id="x", work_id="w_1", start=0, end=19, text="Some verbatim text."
        )


def _finding(finding_id: str, work_id: str) -> Finding:
    return Finding(
        finding_id=finding_id,
        subquestion_id="s1",
        claim="claim",
        quote="quote",
        passage_id="p",
        work_id=work_id,
    )


def test_finding_id_has_the_citation_form() -> None:
    assert _finding("F12", "w").finding_id == "F12"
    with pytest.raises(ValidationError):
        _finding("12", "w")


def test_a_conflict_is_a_pair() -> None:
    Conflict(subquestion_id="s1", finding_ids=["F1", "F2"])
    with pytest.raises(ValidationError):
        Conflict(subquestion_id="s1", finding_ids=["F1"])


def test_budget_defaults_are_named_and_bounds_are_positive() -> None:
    budget = Budget()
    assert (
        budget.max_rounds,
        budget.queries_per_subq,
        budget.candidates_per_round,
    ) == (3, 3, 60)
    assert (budget.expand_per_round, budget.passages_per_work) == (40, 6)
    with pytest.raises(ValidationError):
        Budget(max_rounds=0)
    with pytest.raises(ValidationError):
        Budget(max_wall_s=0)


def _result(report: str) -> ResearchResult:
    return ResearchResult(
        run_id="r_1",
        question="q",
        mode="scholarly",
        works=[
            Work(
                work_id="w_a",
                doi="10.1001/a",
                title="Alpha",
                authors=["Doe, Jane"],
                year=2020,
            ),
            Work(
                work_id="w_b",
                doi="10.1001/b",
                title="Beta",
                abstract="hidden",
                licence=LICENCE_NO_REDISTRIBUTE,
            ),
            Work(work_id="w_c", doi="10.1001/c", title="Gamma"),
        ],
        findings=[_finding("F1", "w_a"), _finding("F2", "w_b"), _finding("F3", "w_c")],
        report_markdown=report,
    )


def test_cited_works_follow_the_report_in_first_citation_order() -> None:
    result = _result("Beta says so [F2]. Alpha too [F1, F2]; and again [F1].")
    assert result.cited_work_ids() == ["w_b", "w_a"]


def test_text_that_only_looks_like_a_citation_is_not_one() -> None:
    result = _result("See F3 and [note F3] and [F9] and [F1].")
    assert result.cited_work_ids() == ["w_a"]


def test_without_a_report_every_finding_counts() -> None:
    assert _result("").cited_work_ids() == ["w_a", "w_b", "w_c"]


def test_to_csl_exports_the_cited_works_without_restricted_text() -> None:
    items = _result("Claim [F2][F1].").to_csl()
    assert [item["id"] for item in items] == ["w_b", "w_a"]
    assert "abstract" not in items[0]
    assert items[1]["author"] == [{"family": "Doe", "given": "Jane"}]


def test_the_result_round_trips_through_json() -> None:
    result = _result("Claim [F1].")
    again = ResearchResult.model_validate_json(result.model_dump_json())
    # The one difference is the licence: a restricted abstract is not exported.
    assert again.works[1].abstract == ""
    again.works[1].abstract = result.works[1].abstract
    assert again == result


def test_no_dump_of_a_result_carries_a_restricted_abstract() -> None:
    result = _result("Claim [F2].")
    assert "hidden" not in result.model_dump_json()
    assert "hidden" not in str(result.model_dump(mode="json"))
    assert "hidden" not in str(result.model_dump())
    # The record in memory keeps it; only an export drops it.
    assert result.works[1].abstract == "hidden"
