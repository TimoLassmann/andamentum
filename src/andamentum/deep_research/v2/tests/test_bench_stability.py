"""Stability metrics, the paired bootstrap and the planted false source."""

from __future__ import annotations

import pytest

from andamentum.deep_research.v2.bench.planted import (
    PLANTED_HOST,
    check_planted_text,
    plant_candidates,
    planted_source,
    planted_sources,
)
from andamentum.deep_research.v2.bench.stability import (
    answer_agreement,
    cited_jaccard,
    decision_determinism,
    findings_reproduced,
    jaccard,
    kept_jaccard,
    mean_pairwise,
    metric_spread,
    planted_outcome,
    quotes_overlap,
    rank_kendall_tau,
    subquestion_overlap,
)
from andamentum.deep_research.v2.bench.stats import paired_bootstrap_ci
from andamentum.deep_research.v2.bench.web_set import WEB_QUESTIONS, page_text
from andamentum.deep_research.v2.identity import merge
from andamentum.deep_research.v2.types import (
    Candidate,
    Conflict,
    Finding,
    Passage,
    ResearchResult,
)
from andamentum.references import Work

# -- sets -----------------------------------------------------------------------


def test_jaccard_basic_empty_and_disjoint():
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a"}, set()) == 0.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard(["a", "a", "b"], ["b", "a"]) == 1.0  # lists are read as sets


def test_kept_and_cited_jaccard_are_the_set_measure():
    assert kept_jaccard({"w1", "w2"}, {"w2"}) == 0.5
    assert cited_jaccard([], []) == 1.0


def test_mean_pairwise_needs_two_runs():
    assert mean_pairwise([]) is None
    assert mean_pairwise([{"a"}]) is None
    assert mean_pairwise([{"a"}, {"a"}, {"b"}]) == pytest.approx(1 / 3)


# -- ranking --------------------------------------------------------------------


def test_kendall_tau_identical_reversed_and_partial():
    assert rank_kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert rank_kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == -1.0
    # one swap among three items: 2 concordant, 1 discordant
    assert rank_kendall_tau(["a", "b", "c"], ["b", "a", "c"]) == pytest.approx(1 / 3)


def test_kendall_tau_ignores_unshared_items_and_duplicates():
    assert rank_kendall_tau(["a", "x", "b"], ["b", "y", "a"]) == -1.0
    assert rank_kendall_tau(["a", "a", "b"], ["a", "b"]) == 1.0


def test_kendall_tau_none_without_a_pair():
    assert rank_kendall_tau([], []) is None
    assert rank_kendall_tau(["a"], ["a"]) is None
    assert rank_kendall_tau(["a", "b"], ["c", "d"]) is None


# -- findings -------------------------------------------------------------------


def finding(number: int, work_id: str, quote: str, claim: str = "claim") -> Finding:
    return Finding(
        finding_id=f"F{number}",
        subquestion_id="S1",
        claim=claim,
        quote=quote,
        passage_id=Passage.make(work_id=work_id, start=0, text=quote).passage_id,
        work_id=work_id,
    )


def test_quotes_overlap():
    assert quotes_overlap("the kiln lost brine", "the kiln lost brine overnight")
    assert quotes_overlap("The Kiln, lost BRINE!", "the kiln lost brine")
    assert not quotes_overlap("the kiln lost brine", "an unrelated sentence here")
    assert not quotes_overlap("", "anything")


def test_findings_reproduced_needs_same_work_and_overlapping_quote():
    a = [finding(1, "w1", "kiln lost brine"), finding(2, "w2", "salt reduced loss")]
    b = [
        finding(7, "w1", "the kiln lost brine overnight"),
        finding(8, "w3", "salt reduced loss"),  # right quote, other work
    ]
    assert findings_reproduced(a, b) == 0.5
    assert findings_reproduced(a, a) == 1.0
    assert findings_reproduced(a, []) == 0.0
    assert findings_reproduced([], a) is None  # nothing to reproduce


def test_findings_reproduced_is_not_symmetric():
    a = [finding(1, "w1", "kiln lost brine")]
    b = [finding(1, "w1", "kiln lost brine"), finding(2, "w2", "something else")]
    assert findings_reproduced(a, b) == 1.0
    assert findings_reproduced(b, a) == 0.5


# -- verdicts and spread --------------------------------------------------------


def test_answer_agreement():
    assert answer_agreement([]) is None
    assert answer_agreement([True]) is None
    assert answer_agreement([True, True]) == 1.0
    assert answer_agreement([True, False]) == 0.0
    assert answer_agreement([True, True, False]) == pytest.approx(1 / 3)


def test_metric_spread():
    assert metric_spread([]) is None
    one = metric_spread([0.4])
    assert one is not None and (one.n, one.min, one.max, one.sd) == (1, 0.4, 0.4, 0.0)
    spread = metric_spread([1.0, 2.0, 3.0])
    assert spread is not None
    assert (spread.min, spread.max) == (1.0, 3.0)
    assert spread.sd == pytest.approx(1.0)  # sample sd
    equal = metric_spread([0.5, 0.5])
    assert equal is not None and equal.sd == 0.0


# -- plan overlap ---------------------------------------------------------------


def test_subquestion_overlap():
    plan = ["What is SaltLog?", "How does SaltLog reduce brine loss?"]
    assert subquestion_overlap(plan, plan) == 1.0
    assert subquestion_overlap([], []) == 1.0
    assert subquestion_overlap(plan, []) == 0.0
    assert subquestion_overlap(["alpha beta"], ["gamma delta"]) == 0.0
    # case, punctuation and order do not matter
    assert subquestion_overlap(["Brine loss, how?"], ["how brine LOSS"]) == 1.0


def test_subquestion_overlap_is_symmetric_and_penalises_extra_questions():
    a = ["kiln brine loss"]
    b = ["kiln brine loss", "completely different topic"]
    assert subquestion_overlap(a, b) == subquestion_overlap(b, a)
    assert 0.0 < subquestion_overlap(a, b) < 1.0


# -- planted outcome ------------------------------------------------------------


def result_with(
    findings: list[Finding], report: str, conflicts: list[Conflict] | None = None
) -> ResearchResult:
    return ResearchResult(
        run_id="r1",
        question="q",
        mode="web",
        findings=findings,
        report_markdown=report,
        conflicts=conflicts or [],
    )


PLANTED_ID = "w_planted"


def test_planted_outcome_adopted():
    result = result_with(
        [finding(1, PLANTED_ID, "Mount Kenya is highest")], "It is Kenya [F1]."
    )
    assert (
        planted_outcome(result, planted_work_id=PLANTED_ID, answer_correct=False)
        == "adopted_false_claim"
    )


def test_planted_outcome_stated_conflict_wins_over_citation():
    findings = [
        finding(1, PLANTED_ID, "Mount Kenya is highest"),
        finding(2, "w_real", "Kilimanjaro is highest"),
    ]
    conflict = Conflict(subquestion_id="S1", finding_ids=["F1", "F2"])
    result = result_with(findings, "Sources disagree [F1] [F2].", [conflict])
    assert (
        planted_outcome(result, planted_work_id=PLANTED_ID, answer_correct=True)
        == "stated_conflict"
    )


def test_planted_outcome_ignored_correct_and_incorrect():
    result = result_with([finding(2, "w_real", "Kilimanjaro")], "Kilimanjaro [F2].")
    assert (
        planted_outcome(result, planted_work_id=PLANTED_ID, answer_correct=True)
        == "ignored_correct"
    )
    assert (
        planted_outcome(result, planted_work_id=PLANTED_ID, answer_correct=False)
        == "ignored_incorrect"
    )


def test_planted_outcome_planted_finding_not_cited_is_ignored():
    result = result_with(
        [finding(1, PLANTED_ID, "Mount Kenya"), finding(2, "w_real", "Kilimanjaro")],
        "Kilimanjaro [F2].",
    )
    assert (
        planted_outcome(result, planted_work_id=PLANTED_ID, answer_correct=True)
        == "ignored_correct"
    )


def test_planted_outcome_conflict_between_other_findings_does_not_count():
    findings = [
        finding(1, PLANTED_ID, "Mount Kenya"),
        finding(2, "w_a", "Kilimanjaro"),
        finding(3, "w_b", "Kilimanjaro, Tanzania"),
    ]
    conflict = Conflict(subquestion_id="S1", finding_ids=["F2", "F3"])
    result = result_with(findings, "Kenya [F1].", [conflict])
    assert (
        planted_outcome(result, planted_work_id=PLANTED_ID, answer_correct=False)
        == "adopted_false_claim"
    )


# -- decision determinism -------------------------------------------------------


async def test_decision_determinism_identical_answers():
    async def ask(state: str) -> float:
        return len(state) / 10

    report = await decision_determinism(ask, ["a", "bb", "ccc"])
    assert report.deterministic
    assert (report.n_states, report.n_identical, report.max_abs_difference) == (3, 3, 0)


async def test_decision_determinism_flags_drift_and_runs_sequentially():
    calls: list[str] = []
    active = 0
    overlap = False

    async def ask(state: str) -> float:
        nonlocal active, overlap
        active += 1
        overlap = overlap or active > 1
        calls.append(state)
        active -= 1
        return 0.5 + 0.1 * calls.count(state) if state == "drifty" else 0.5

    report = await decision_determinism(ask, ["steady", "drifty"])
    assert report.differing == [1]
    assert not report.deterministic
    assert report.max_abs_difference == pytest.approx(0.1)
    assert calls == ["steady", "steady", "drifty", "drifty"]
    assert not overlap


async def test_decision_determinism_tolerance_and_empty():
    flip = iter([0.50, 0.505])

    async def ask(state: str) -> float:
        return next(flip)

    assert (await decision_determinism(ask, ["x"], tolerance=0.01)).deterministic
    empty = await decision_determinism(ask, [])
    assert empty.n_states == 0 and empty.deterministic
    with pytest.raises(ValueError):
        await decision_determinism(ask, ["x"], tolerance=-1)


# -- paired bootstrap -----------------------------------------------------------


def test_bootstrap_is_reproducible_and_seed_dependent():
    diffs = [0.1, -0.05, 0.2, 0.0, 0.15, 0.1, -0.1, 0.3]
    a = paired_bootstrap_ci(diffs, 500, 7)
    assert a == paired_bootstrap_ci(diffs, 500, 7)
    assert a != paired_bootstrap_ci(diffs, 500, 8)
    assert a.low <= a.mean <= a.high
    assert (a.n_items, a.n_resamples, a.confidence) == (8, 500, 0.95)


def test_bootstrap_constant_and_single_difference_have_zero_width():
    for diffs in ([0.2, 0.2, 0.2], [0.2]):
        ci = paired_bootstrap_ci(diffs, 100, 1)
        assert ci.low == pytest.approx(0.2) and ci.high == pytest.approx(0.2)
        assert ci.mean == pytest.approx(0.2)
        assert ci.excludes_zero


def test_bootstrap_clear_effect_excludes_zero_and_noise_does_not():
    assert paired_bootstrap_ci([0.5, 0.4, 0.6, 0.5, 0.45], 1000, 3).excludes_zero
    assert not paired_bootstrap_ci(
        [0.3, -0.3, 0.2, -0.2, 0.1, -0.1], 1000, 3
    ).excludes_zero


def test_bootstrap_confidence_widens_the_interval():
    diffs = [0.1, -0.2, 0.3, 0.0, 0.2, -0.1, 0.25, 0.05]
    narrow = paired_bootstrap_ci(diffs, 2000, 5, confidence=0.5)
    wide = paired_bootstrap_ci(diffs, 2000, 5, confidence=0.99)
    assert wide.high - wide.low > narrow.high - narrow.low


def test_bootstrap_rejects_bad_input():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([], 10, 0)
    with pytest.raises(ValueError):
        paired_bootstrap_ci([0.1], 0, 0)
    with pytest.raises(ValueError):
        paired_bootstrap_ci([0.1], 10, 0, confidence=1.0)
    with pytest.raises(ValueError):
        paired_bootstrap_ci([float("nan")], 10, 0)


# -- planted source generator ---------------------------------------------------


def test_every_web_question_has_a_wrong_planted_statement():
    sources = planted_sources()
    assert [s.question_id for s in sources] == [q.question_id for q in WEB_QUESTIONS]
    for question, source in zip(WEB_QUESTIONS, sources, strict=True):
        check_planted_text(question, source)  # never states the right answer


def test_check_planted_text_catches_a_correct_statement():
    question = WEB_QUESTIONS[0]
    source = planted_source(question).model_copy(
        update={"statement": "Mount Kilimanjaro in Tanzania is the highest."}
    )
    with pytest.raises(ValueError, match="satisfies the answer check"):
        check_planted_text(question, source)


def test_planted_urls_are_synthetic_distinct_and_valid_works():
    sources = planted_sources()
    assert len({s.url for s in sources}) == len(sources)
    for source in sources:
        assert PLANTED_HOST in source.url
        assert source.work().url == source.url
        assert source.statement in page_text(source.page_html())


def test_planted_page_html_escapes_markup():
    source = planted_source(WEB_QUESTIONS[0]).model_copy(
        update={"statement": "a <script>x</script> & b"}
    )
    assert "<script>" not in source.page_html()
    assert "&amp;" in source.page_html()


def test_unknown_question_has_no_planted_source():
    other = WEB_QUESTIONS[0].model_copy(update={"question_id": "W99"})
    with pytest.raises(KeyError):
        planted_source(other)


def real_candidate(rank: int) -> Candidate:
    return Candidate(
        work=Work(url=f"https://example.org/p{rank}", title=f"Page number {rank}"),
        sources=["searxng"],
        rank=rank,
    )


def test_plant_candidates_adds_first_in_every_subquestion_without_mutating():
    source = planted_source(WEB_QUESTIONS[0])
    original = {"S1": [real_candidate(1), real_candidate(2)], "S2": []}
    planted = plant_candidates(original, source)
    assert len(original["S1"]) == 2 and original["S2"] == []
    for subquestion_id in ("S1", "S2"):
        first = planted[subquestion_id][0]
        assert first.work.url == source.url
        assert first.sources == ["planted"]
        assert first.subquestion_ids == [subquestion_id]
        assert first.rank == 1
    assert [c.rank for c in planted["S1"][1:]] == [1, 2]


def test_plant_candidates_refuses_a_second_planting():
    source = planted_source(WEB_QUESTIONS[0])
    once = plant_candidates({"S1": [real_candidate(1)]}, source)
    with pytest.raises(ValueError, match="already among"):
        plant_candidates(once, source)


def test_planted_candidate_survives_identity_merge_as_its_own_work():
    source = planted_source(WEB_QUESTIONS[0])
    planted = plant_candidates({"S1": [real_candidate(1)]}, source)["S1"]
    assert len(merge(planted)) == 2
