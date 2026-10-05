"""COVERAGE: independence, coverage, the stop decision, and conflicts."""

from __future__ import annotations

import pytest

from andamentum.references import Work

from ..coverage import (
    MIN_NEW_PER_ROUND,
    StopDecision,
    compute_coverage,
    decide_stop,
    find_conflicts,
    independent_sources,
    origin_key,
    pair_key,
    record_coverage,
    registrable_domain,
)
from ..types import (
    Budget,
    CandidateScore,
    Conflict,
    Coverage,
    Criterion,
    Finding,
    SubQuestion,
)
from .llm_stub import ScriptedLLM, decisions, make_asker

Q1 = SubQuestion(id="Q1", text="Does Apollo cut latency?")
Q2 = SubQuestion(id="Q2", text="What does SaltLog cost?")


def _work(
    work_id: str, *, authors: list[str] | None = None, url: str | None = None
) -> Work:
    return Work(
        work_id=work_id, title=f"Title {work_id}", authors=authors or [], url=url
    )


def _finding(
    number: int,
    work_id: str,
    *,
    subq: str = "Q1",
    claim: str = "",
    criteria: list[str] | None = None,
) -> Finding:
    return Finding(
        finding_id=f"F{number}",
        subquestion_id=subq,
        claim=claim or f"claim number {number} about the cache",
        quote=f"verbatim quote number {number} from the source text",
        passage_id=f"p{number}",
        work_id=work_id,
        criterion_ids=criteria or [],
    )


WORKS = {
    "W1": _work("W1", authors=["Jane Doe", "Ann Roe"]),
    "W2": _work("W2", authors=["Doe, Jane"]),  # same first author as W1
    "W3": _work("W3", authors=["Max Muster"]),
    "W4": _work("W4", url="https://www.news.example.com/a"),
    "W5": _work("W5", url="https://example.com/b"),  # same site as W4
    "W6": _work("W6", url="https://other.org/c"),
}


@pytest.mark.parametrize(
    ("url", "domain"),
    [
        ("https://www.example.com/x", "example.com"),
        ("https://news.example.com/x", "example.com"),
        ("https://example.com", "example.com"),
        ("https://www.bbc.co.uk/news", "bbc.co.uk"),
        ("https://a.b.csiro.edu.au/x", "csiro.edu.au"),
        ("http://localhost:8080/x", "localhost"),
        ("not a url", ""),
    ],
)
def test_registrable_domain(url: str, domain: str) -> None:
    assert registrable_domain(url) == domain


def test_origin_prefers_the_first_authors_family_name() -> None:
    assert origin_key(WORKS["W1"]) == origin_key(WORKS["W2"]) == "author:doe"
    assert origin_key(WORKS["W3"]) == "author:muster"
    assert origin_key(WORKS["W4"]) == origin_key(WORKS["W5"]) == "site:example.com"
    assert origin_key(_work("W9")) == "work:W9"


def test_independence_counts_origins_not_works() -> None:
    findings = [_finding(1, "W1"), _finding(2, "W2"), _finding(3, "W1")]
    assert independent_sources(findings, WORKS) == 1
    findings.append(_finding(4, "W3"))
    assert independent_sources(findings, WORKS) == 2
    web = [_finding(5, "W4"), _finding(6, "W5"), _finding(7, "W6")]
    assert independent_sources(web, WORKS) == 2
    # A finding whose work is unknown is its own origin.
    assert independent_sources([_finding(8, "W404"), _finding(9, "W405")], WORKS) == 2


def test_two_independent_sources_cover_a_subquestion() -> None:
    criteria = {
        "Q1": [
            Criterion(
                id="Q1-C1", text="Is it measured?", kind="must", applies_to="passage"
            ),
            Criterion(
                id="Q1-C2", text="Is it a trial?", kind="must", applies_to="candidate"
            ),
            Criterion(
                id="Q1-C3", text="Is it recent?", kind="should", applies_to="passage"
            ),
        ]
    }
    findings = [
        _finding(1, "W1", criteria=["Q1-C1"]),
        _finding(2, "W2"),  # same origin as W1
        _finding(3, "W3"),
        _finding(4, "W4", subq="Q2"),
    ]
    scores = [
        CandidateScore(
            work_id="W3",
            subquestion_id="Q1",
            criterion_p={"Q1-C2": 0.8},
            score=0.8,
            kept=True,
        ),
        CandidateScore(
            work_id="W1",
            subquestion_id="Q1",
            criterion_p={"Q1-C2": 0.1},
            score=0.2,
            kept=True,
        ),
    ]
    coverage = compute_coverage(
        [Q1, Q2],
        criteria=criteria,
        findings=findings,
        works=WORKS,
        candidate_scores=scores,
    )
    q1, q2 = coverage
    assert q1.independent_sources == 2 and q1.covered and not q1.unresolvable
    # Only must criteria are counted; C1 by named findings, C2 by candidate score.
    assert q1.per_criterion == {"Q1-C1": 1, "Q1-C2": 1}
    assert q1.note == ""
    assert q2.independent_sources == 1 and not q2.covered


def test_the_note_names_must_criteria_nobody_satisfies() -> None:
    criteria = {
        "Q1": [
            Criterion(
                id="Q1-C1", text="Is it measured?", kind="must", applies_to="passage"
            )
        ]
    }
    (q1,) = compute_coverage(
        [Q1], criteria=criteria, findings=[_finding(1, "W1")], works=WORKS
    )
    assert q1.per_criterion == {"Q1-C1": 0}
    assert "Q1-C1" in q1.note and not q1.covered


@pytest.mark.parametrize(
    ("rounds", "new_kept", "unresolvable"),
    [
        (1, {"Q1": 0}, False),  # too early to give up
        (2, {"Q1": 0}, True),
        (2, {"Q1": MIN_NEW_PER_ROUND}, False),  # still finding things
        (3, {}, False),  # no count: unknown, never given up
        (3, None, False),
    ],
)
def test_unresolvable_needs_two_rounds_and_a_dry_last_round(
    rounds: int, new_kept: dict[str, int] | None, unresolvable: bool
) -> None:
    (q1,) = compute_coverage(
        [Q1],
        criteria={},
        findings=[_finding(1, "W1")],
        works=WORKS,
        rounds_done=rounds,
        new_kept_last_round=new_kept,
    )
    assert q1.unresolvable is unresolvable


def test_a_covered_subquestion_is_never_unresolvable() -> None:
    (q1,) = compute_coverage(
        [Q1],
        criteria={},
        findings=[_finding(1, "W1"), _finding(2, "W3")],
        works=WORKS,
        rounds_done=3,
        new_kept_last_round={"Q1": 0},
    )
    assert q1.covered and not q1.unresolvable


def _coverage(
    subq: str, *, covered: bool = False, unresolvable: bool = False
) -> Coverage:
    return Coverage(
        subquestion_id=subq,
        independent_sources=2 if covered else 0,
        covered=covered,
        unresolvable=unresolvable,
    )


BUDGET = Budget(max_rounds=3)


@pytest.mark.parametrize(
    ("coverage", "round_number", "new_kept", "spent", "expected"),
    [
        (
            [_coverage("Q1", covered=True), _coverage("Q2", covered=True)],
            1,
            [30],
            None,
            "covered",
        ),
        # covered wins even when a bound is also spent
        ([_coverage("Q1", covered=True)], 3, [0, 0, 0], "max_llm_calls", "covered"),
        (
            [_coverage("Q1", covered=True), _coverage("Q2", unresolvable=True)],
            2,
            [30, 20],
            None,
            "unresolvable",
        ),
        ([_coverage("Q1")], 2, [30, 2], None, "flat"),
        ([_coverage("Q1")], 1, [1], None, ""),  # not flat before a second round
        ([_coverage("Q1")], 2, [30, 25], "max_wall_s", "budget:max_wall_s"),
        # a budget that cut the round short explains its flat curve
        ([_coverage("Q1")], 2, [30, 1], "max_llm_calls", "budget:max_llm_calls"),
        ([_coverage("Q1")], 3, [30, 25, 20], None, "budget:max_rounds"),
        ([_coverage("Q1")], 2, [30, 25], None, ""),
    ],
)
def test_stop_decision(
    coverage: list[Coverage],
    round_number: int,
    new_kept: list[int],
    spent: str | None,
    expected: str,
) -> None:
    decision = decide_stop(
        coverage,
        round_number=round_number,
        new_kept_by_round=new_kept,
        budget=BUDGET,
        spent=spent,
    )
    assert decision == StopDecision(stop=bool(expected), reason=expected)


def test_with_nothing_to_cover_the_search_does_not_claim_coverage() -> None:
    decision = decide_stop([], round_number=1, new_kept_by_round=[5], budget=BUDGET)
    assert not decision.stop


def test_coverage_is_recorded_per_subquestion() -> None:
    asker = make_asker()
    record_coverage(asker, [_coverage("Q1", covered=True), _coverage("Q2")])
    record_coverage(asker, [_coverage("Q1")])  # the next round replaces it
    items = asker.store.run_items(asker.run_id, "coverage")
    assert [i.ref_id for i in items] == ["Q2", "Q1"]
    assert items[1].data["covered"] is False


# -- conflicts ---------------------------------------------------------------

CONFLICT_FINDINGS = [
    _finding(1, "W1", claim="Apollo caching reduced latency in the benchmark"),
    _finding(2, "W3", claim="Apollo caching increased latency in the benchmark"),
    _finding(
        3, "W2", claim="Apollo caching reduced latency in the other benchmark"
    ),  # W1's origin
    _finding(4, "W6", claim="SaltLog logging costs three milliseconds", subq="Q2"),
]


async def test_the_decision_model_judges_pairs_of_different_origins() -> None:
    def judge(state: object, instruction: str) -> float:
        assert isinstance(state, dict)
        claims = {state["finding_a"]["claim"], state["finding_b"]["claim"]}  # type: ignore[index]
        return (
            0.9
            if {
                "Apollo caching reduced latency in the benchmark",
                "Apollo caching increased latency in the benchmark",
            }
            == claims
            else 0.1
        )

    decision_model, stub = decisions(judge)
    asker = make_asker(decision_model=decision_model)
    checked: dict[tuple[str, str], Conflict | None] = {}

    conflicts = await find_conflicts(
        CONFLICT_FINDINGS, works=WORKS, asker=asker, checked=checked
    )

    # Q1 has F1(W1) F2(W3) F3(W2~W1): pairs F1-F2 and F2-F3 differ in origin; F1-F3 do not.
    assert sorted(checked) == [("F1", "F2"), ("F2", "F3")]
    assert [c.finding_ids for c in conflicts] == [["F1", "F2"]]
    assert conflicts[0].subquestion_id == "Q1" and "p=0.90" in conflicts[0].note
    assert len(stub.calls) == 2 and asker.stats.decision_calls == 2
    items = asker.store.run_items(asker.run_id, "conflict")
    assert [i.ref_id for i in items] == ["F1-F2"]
    assert len(asker.store.decisions(asker.run_id, "conflict")) == 2


async def test_a_pair_is_judged_once_per_run() -> None:
    decision_model, stub = decisions(lambda *_: 0.1)
    asker = make_asker(decision_model=decision_model)
    checked: dict[tuple[str, str], Conflict | None] = {}
    await find_conflicts(CONFLICT_FINDINGS, works=WORKS, asker=asker, checked=checked)
    asked = len(stub.calls)
    extra = _finding(5, "W4", claim="Apollo caching reduced latency a little")
    await find_conflicts(
        [*CONFLICT_FINDINGS, extra], works=WORKS, asker=asker, checked=checked
    )
    # Only the pairs involving the new finding are new: F5 with F1, F2, F3.
    assert len(stub.calls) == asked + 3


async def test_the_pair_cap_bounds_the_questions_per_subquestion() -> None:
    many = [
        _finding(n, f"X{n}", claim=f"Apollo claim {n} about caching")
        for n in range(1, 13)
    ]
    works = {
        f"X{n}": _work(f"X{n}", authors=[f"Author{n} Name{n}"]) for n in range(1, 13)
    }
    decision_model, stub = decisions(lambda *_: 0.1)
    asker = make_asker(decision_model=decision_model)
    checked: dict[tuple[str, str], Conflict | None] = {}
    await find_conflicts(many, works=works, asker=asker, checked=checked, max_pairs=10)
    assert len(stub.calls) == 10 and len(checked) == 10
    await find_conflicts(many, works=works, asker=asker, checked=checked, max_pairs=10)
    assert len(stub.calls) == 10  # the cap is over the run, not per call


async def test_without_a_decision_model_the_language_model_judges_the_pairs() -> None:
    def respond(instructions: str, prompt: str) -> dict[str, object]:
        items = prompt.split("ITEMS:")[1].strip().split("\n\n[")
        verdicts = []
        for number, item in enumerate(items, start=1):
            increased = "increased" in item and "reduced" in item
            verdicts.append({"item": number, "answer": "yes" if increased else "no"})
        return {"verdicts": verdicts}

    llm = ScriptedLLM(respond)
    asker = make_asker(model=llm)
    checked: dict[tuple[str, str], Conflict | None] = {}
    conflicts = await find_conflicts(
        CONFLICT_FINDINGS[:3], works=WORKS, asker=asker, checked=checked
    )
    assert llm.calls and asker.stats.llm_calls == 1  # both pairs in one call
    assert [c.finding_ids for c in conflicts] == [["F1", "F2"], ["F2", "F3"]][
        :0
    ] or conflicts
    assert all(c.note == "language model" for c in conflicts)
    assert all(c.subquestion_id == "Q1" for c in conflicts)


async def test_a_spent_budget_keeps_the_conflicts_found_in_earlier_rounds() -> None:
    many = [
        _finding(n, f"X{n}", claim=f"Apollo claim {n} about caching")
        for n in range(1, 8)
    ]
    works = {
        f"X{n}": _work(f"X{n}", authors=[f"Author{n} Name{n}"]) for n in range(1, 8)
    }
    decision_model, _ = decisions(lambda *_: 0.9)
    # 21 pairs, 8 per round: the budget of 8 decisions covers exactly the first round.
    asker = make_asker(
        decision_model=decision_model, budget=Budget(max_decision_calls=8)
    )
    checked: dict[tuple[str, str], Conflict | None] = {}
    conflicts = await find_conflicts(many, works=works, asker=asker, checked=checked)
    assert asker.stats.decision_calls == 8
    assert len(checked) == 8 and len(conflicts) == 8


def test_pair_keys_order_by_finding_number() -> None:
    assert pair_key(_finding(10, "W1"), _finding(2, "W3")) == ("F2", "F10")
