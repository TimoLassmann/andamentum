"""Query-dependent criteria: compile, score, combine, refine, log.

Offline: the LLM is a scripted stand-in for ``run_agent_with_usage`` and the
decision model is ``core.decisions.testing.StubDecisions``. Every rule of
``combine`` and the refinement bound has a test here. All data is synthetic.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai import ModelRetry
from pydantic_ai.usage import RunUsage

from andamentum.core.decisions import Answer
from andamentum.core.decisions.testing import StubDecisions, choice_answer, noul_answer
from andamentum.deep_research.v2.ask import Asker, BudgetExhausted
from andamentum.deep_research.v2.criteria import (
    EXCLUDE_DROP_AT,
    MUST_DROP_BELOW,
    REFINE_SAMPLE,
    SOURCE_TYPE_KEY,
    CandidateJudgement,
    CriteriaDraft,
    CriteriaError,
    CriterionDraft,
    LabelAnswer,
    PassageJudgement,
    PassageScore,
    ReviewDraft,
    Screener,
    ScreeningError,
    boundary_margin,
    combine,
    compile_criteria,
    criterion_problems,
    rank,
    score_candidates,
    select_passages,
    validate_criteria,
)
from andamentum.deep_research.v2.store import ResearchStore
from andamentum.deep_research.v2.types import (
    Budget,
    Candidate,
    CandidateScore,
    Criterion,
    Passage,
    RunStats,
    SubQuestion,
)
from andamentum.references import Work

SUBQ = SubQuestion(id="S1", text="Does SaltLog reduce brine loss in Apollo kilns?")
LLM = "ollama:stub-llm"

# -- fixtures ---------------------------------------------------------------------


def draft(
    text: str, kind: str = "must", weight: int = 2, applies_to: str = "candidate"
) -> CriterionDraft:
    return CriterionDraft.model_validate(
        {"text": text, "kind": kind, "weight": weight, "applies_to": applies_to}
    )


GOOD = [
    draft("Is the study about SaltLog?", "must", 3),
    draft("Does it measure brine loss?", "must", 2),
    draft("Is it a controlled experiment?", "should", 1),
    draft("Is it an advertisement?", "exclude", 1),
    draft("Does the passage report a brine loss figure?", "should", 2, "passage"),
]


def criteria_set(**weights: int) -> list[Criterion]:
    """S1-C1 must (w3), S1-C2 must (w2), S1-C3 should (w1), S1-C4 exclude, S1-C5 passage."""
    return [
        Criterion(id="S1-C1", text=GOOD[0].text, kind="must", weight=3),
        Criterion(id="S1-C2", text=GOOD[1].text, kind="must", weight=2),
        Criterion(id="S1-C3", text=GOOD[2].text, kind="should", weight=1),
        Criterion(id="S1-C4", text=GOOD[3].text, kind="exclude", weight=1),
        Criterion(
            id="S1-C5",
            text=GOOD[4].text,
            kind="should",
            weight=2,
            applies_to="passage",
        ),
    ]


def candidate(n: int, *, title: str | None = None) -> Candidate:
    return Candidate(
        work=Work(
            work_id=f"w{n}",
            title=title or f"SaltLog trial number {n}",
            abstract=f"Synthetic abstract {n} about brine loss in Apollo kilns.",
            year=2020 + n % 5,
            venue="Journal of Invented Results",
        ),
        sources=["openalex"],
    )


class FakeLLM:
    """Stands in for ``run_agent_with_usage``: per output type, a handler
    turns the user message into the output. Records every call."""

    def __init__(self) -> None:
        self.handlers: dict[type[BaseModel], Callable[[str], BaseModel]] = {}
        self.calls: list[dict[str, Any]] = []

    def on(
        self, output_type: type[BaseModel], handler: Callable[[str], BaseModel]
    ) -> None:
        self.handlers[output_type] = handler

    def queue(self, output_type: type[BaseModel], outputs: list[BaseModel]) -> None:
        pending = list(outputs)
        self.on(output_type, lambda _message: pending.pop(0))

    async def __call__(self, model: Any, **kwargs: Any) -> tuple[BaseModel, RunUsage]:
        self.calls.append({"model": model, **kwargs})
        output = self.handlers[kwargs["output_type"]](kwargs["user_message"])
        assert isinstance(output, kwargs["output_type"])
        return output, RunUsage(input_tokens=5, output_tokens=3)

    def of(self, output_type: type[BaseModel]) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["output_type"] is output_type]


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    fake = FakeLLM()
    monkeypatch.setattr("andamentum.deep_research.v2.ask.run_agent_with_usage", fake)
    return fake


@pytest.fixture
def store() -> ResearchStore:
    return ResearchStore.in_memory()


@pytest.fixture
def run_id(store: ResearchStore) -> str:
    return store.start_run(question="q", mode="scholarly", config={}, models={})


def make_asker(
    store: ResearchStore,
    run_id: str,
    *,
    decision: StubDecisions | None = None,
    budget: Budget | None = None,
) -> Asker:
    return Asker(
        model=LLM,
        store=store,
        run_id=run_id,
        budget=budget or Budget(),
        stats=RunStats(),
        decision_model=decision.model() if decision else None,
    )


def responder(
    p: Callable[[str, str], float], source_type: str = "primary_study"
) -> Callable[[object, dict[str, dict[str, object]]], dict[str, Answer]]:
    """A StubDecisions responder: ``p(title_or_passage, instructions)`` for
    each noul question, ``source_type`` for the choice."""

    def respond(
        state: object, questions: dict[str, dict[str, object]]
    ) -> dict[str, Answer]:
        assert isinstance(state, dict)
        subject = str(state.get("title") or state.get("passage"))
        answers: dict[str, Answer] = {}
        for key, q in questions.items():
            if q["type"] == "choice":
                options = list(q["criteria"])  # type: ignore[arg-type]
                answers[key] = choice_answer(options, source_type, 0.8)
            else:
                answers[key] = noul_answer(p(subject, str(q["instructions"])))
        return answers

    return respond


def p_table(
    table: dict[tuple[str, str], float], default: float = 0.9
) -> Callable[[str, str], float]:
    """p per (subject, question); otherwise ``default``, except that the
    exclude criterion ("an advertisement?") does not hold."""
    return lambda subject, text: table.get(
        (subject, text), 0.1 if text == GOOD[3].text else default
    )


def labelled(message: str, answer: Callable[[str], str]) -> list[LabelAnswer]:
    """LLM scorer answers: one per 'Cn: question' line, by the question text."""
    out = []
    for line in message.splitlines():
        if line[:1] == "C" and ": " in line and line.split(":")[0][1:].isdigit():
            label, question = line.split(": ", 1)
            out.append(LabelAnswer(label=label, answer=answer(question)))  # type: ignore[arg-type]
    return out


# -- validation ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Is the study about SaltLog?",
        "Does it compare SaltLog and a placebo kiln?",
        "Was it published after 2020?",
    ],
)
def test_atomic_questions_pass(text: str) -> None:
    assert criterion_problems(text) == []


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("Is the study about SaltLog", "ending in '?'"),
        ("Is it about SaltLog? Is it recent?", "ending in '?'"),
        ("Is it about SaltLog and does it measure brine loss?", "joins two"),
        ("Is it a trial or is it a cohort?", "joins two"),
        ("Does it study kilns and/or furnaces?", "joins two"),
        ("Does it report cost as well as yield?", "joins two"),
        ("", "empty"),
        ("Is it " + "very " * 60 + "long?", "longer than"),
    ],
)
def test_compound_or_malformed_questions_are_rejected(text: str, fragment: str) -> None:
    problems = criterion_problems(text)
    assert problems and any(fragment in p for p in problems)


def test_a_good_set_validates() -> None:
    assert validate_criteria(GOOD) == []


def test_too_few_and_too_many_criteria() -> None:
    assert any("2 criteria" in p for p in validate_criteria(GOOD[:2]))
    many = [draft(f"Does it mention kiln number {i}?", "should") for i in range(8)]
    assert any("9 criteria" in p for p in validate_criteria([GOOD[0], *many]))


def test_duplicates_are_rejected_even_reordered() -> None:
    dup = [*GOOD[:3], draft("Is SaltLog the study about?", "should")]
    assert any("duplicates" in p for p in validate_criteria(dup))


def test_a_candidate_must_is_required() -> None:
    only_passage_must = [
        draft("Does the passage name SaltLog?", "must", 2, "passage"),
        draft("Is it recent?", "should"),
        draft("Is it a controlled experiment?", "should"),
    ]
    assert any("no 'must'" in p for p in validate_criteria(only_passage_must))


# -- compiling ----------------------------------------------------------------------------


async def test_compile_once_assigns_ids_and_logs_version_1(
    llm: FakeLLM, store: ResearchStore, run_id: str
) -> None:
    llm.queue(CriteriaDraft, [CriteriaDraft(criteria=GOOD)])
    screener = Screener(asker=make_asker(store, run_id))
    criteria = await screener.compile(SUBQ, question="How do Apollo kilns lose brine?")
    assert [c.id for c in criteria] == [f"S1-C{i}" for i in range(1, 6)]
    assert criteria[4].applies_to == "passage"
    assert len(llm.calls) == 1
    assert "How do Apollo kilns lose brine?" in llm.calls[0]["user_message"]
    [row] = store.run_items(run_id, "criterion")
    assert row.ref_id == "S1/v1"
    assert row.data["origin"] == "compiled"
    assert row.data["rejected_attempts"] == []
    assert screener.asker.stats.llm_calls == 1


async def test_compile_re_asks_once_with_the_problems(
    llm: FakeLLM, store: ResearchStore, run_id: str
) -> None:
    bad = CriteriaDraft(
        criteria=[draft("Is it about SaltLog and does it measure loss?"), *GOOD[1:3]]
    )
    llm.queue(CriteriaDraft, [bad, CriteriaDraft(criteria=GOOD)])
    screener = Screener(asker=make_asker(store, run_id))
    criteria = await screener.compile(SUBQ)
    assert len(criteria) == len(GOOD)
    assert len(llm.calls) == 2
    assert "joins two properties" in llm.calls[1]["user_message"]
    [row] = store.run_items(run_id, "criterion")
    rejected = row.data["rejected_attempts"]
    assert isinstance(rejected, list) and len(rejected) == 1


async def test_compile_fails_loud_after_the_re_ask(
    llm: FakeLLM, store: ResearchStore, run_id: str
) -> None:
    bad = CriteriaDraft(criteria=GOOD[:2])
    llm.queue(CriteriaDraft, [bad, bad, CriteriaDraft(criteria=GOOD)])
    screener = Screener(asker=make_asker(store, run_id))
    with pytest.raises(CriteriaError) as raised:
        await screener.compile(SUBQ)
    assert len(llm.calls) == 2
    assert any("2 criteria" in p for p in raised.value.problems)


async def test_one_off_compile_criteria(llm: FakeLLM) -> None:
    llm.queue(CriteriaDraft, [CriteriaDraft(criteria=GOOD)])
    criteria = await compile_criteria(SUBQ, model=LLM)
    assert [c.kind for c in criteria] == ["must", "must", "should", "exclude", "should"]
    assert llm.calls[0]["model"] == LLM


# -- combine: every rule ------------------------------------------------------------------


def judge(**p: float) -> CandidateScore:
    defaults = {"S1-C1": 0.9, "S1-C2": 0.9, "S1-C3": 0.9, "S1-C4": 0.1}
    probabilities = {**defaults, **{k.replace("_", "-"): v for k, v in p.items()}}
    return combine(
        work_id="w1",
        subquestion_id="S1",
        criteria=criteria_set(),
        criterion_p=probabilities,
        source_type="news",
    )


def test_exclude_at_threshold_drops_with_its_text() -> None:
    score = judge(S1_C4=EXCLUDE_DROP_AT)
    assert not score.kept
    assert "exclude S1-C4" in score.reason and "advertisement" in score.reason


def test_exclude_below_threshold_keeps() -> None:
    assert judge(S1_C4=EXCLUDE_DROP_AT - 0.01).kept


def test_confident_must_miss_drops() -> None:
    score = judge(S1_C2=MUST_DROP_BELOW - 0.01)
    assert not score.kept
    assert "must S1-C2 fails" in score.reason and "brine loss" in score.reason


def test_must_at_threshold_keeps_recall_first() -> None:
    assert judge(S1_C2=MUST_DROP_BELOW).kept


def test_both_rules_are_named_when_both_fire() -> None:
    score = judge(S1_C1=0.05, S1_C4=0.9)
    assert "exclude S1-C4" in score.reason and "must S1-C1" in score.reason


def test_score_is_the_weighted_mean_over_must_and_should() -> None:
    score = judge(S1_C1=1.0, S1_C2=0.5, S1_C3=0.0, S1_C4=0.4)
    assert score.kept
    assert score.score == pytest.approx((3 * 1.0 + 2 * 0.5 + 1 * 0.0) / 6)
    assert score.reason.startswith("kept: score 0.67")


def test_passage_criteria_take_no_part_in_candidate_decisions() -> None:
    score = judge()
    assert "S1-C5" not in score.criterion_p


def test_source_type_is_recorded_never_used_to_drop() -> None:
    score = judge()
    assert score.kept and score.source_type == "news"


def test_a_missing_answer_fails_loud() -> None:
    with pytest.raises(ScreeningError, match="S1-C3"):
        combine(
            work_id="w1",
            subquestion_id="S1",
            criteria=criteria_set(),
            criterion_p={"S1-C1": 0.9, "S1-C2": 0.9, "S1-C4": 0.1},
        )


def test_rank_puts_kept_first_by_score() -> None:
    low = judge(S1_C3=0.0)
    high = judge(S1_C3=1.0)
    dropped = judge(S1_C4=0.9).model_copy(update={"work_id": "w0"})
    assert rank([dropped, low, high]) == [high, low, dropped]


def test_boundary_margin() -> None:
    criteria = criteria_set()
    assert boundary_margin(judge(S1_C2=0.3, S1_C4=0.1), criteria) == pytest.approx(0.1)
    assert boundary_margin(judge(S1_C4=0.45), criteria) == pytest.approx(0.05)
    # Dropped: every violation must flip, so the largest distance counts.
    both = judge(S1_C1=0.15, S1_C2=0.0)
    assert boundary_margin(both, criteria) == pytest.approx(0.2)


# -- scoring with a decision model ---------------------------------------------------------


async def test_one_request_per_candidate_with_every_criterion_and_the_source_type(
    store: ResearchStore, run_id: str
) -> None:
    table = {("SaltLog trial number 2", GOOD[1].text): 0.05}
    stub = StubDecisions(responder(p_table(table)))
    screener = Screener(asker=make_asker(store, run_id, decision=stub))
    scores = await screener.score_candidates(
        SUBQ, criteria_set(), [candidate(1), candidate(2)]
    )
    assert len(stub.calls) == 2
    state, questions = stub.calls[0]
    assert isinstance(state, dict) and state["title"] == "SaltLog trial number 1"
    assert state["question"] == SUBQ.text
    assert set(questions) == {"S1-C1", "S1-C2", "S1-C3", "S1-C4", SOURCE_TYPE_KEY}
    assert questions[SOURCE_TYPE_KEY]["type"] == "choice"
    assert "do not rule it out" in str(questions["S1-C1"]["criteria"])
    assert "clearly show" in str(questions["S1-C4"]["criteria"])
    assert [s.kept for s in scores] == [True, False]
    assert scores[0].source_type == "primary_study"
    # every answer is a decisions row; every decision a run item
    rows = store.decisions(run_id, "screen")
    assert len(rows) == 2 * 5
    assert {r.model for r in rows} == {"ollama:nimble:9b"} and rows[0].digest
    items = store.run_items(run_id, "candidate")
    assert {i.ref_id for i in items} == {"S1/v1/w1", "S1/v1/w2"}
    dropped = next(i for i in items if i.ref_id.endswith("w2"))
    assert dropped.data["kept"] is False and dropped.data["criteria_version"] == 1
    assert screener.asker.stats.decision_calls == 2


async def test_a_candidate_seen_again_is_not_asked_again(
    store: ResearchStore, run_id: str
) -> None:
    stub = StubDecisions(responder(p_table({})))
    screener = Screener(asker=make_asker(store, run_id, decision=stub))
    first = await screener.score_candidates(SUBQ, criteria_set(), [candidate(1)])
    again = await screener.score_candidates(SUBQ, criteria_set(), [candidate(1)])
    assert len(stub.calls) == 1
    assert first == again
    assert screener.cached_answers == 5


async def test_a_candidate_without_a_work_id_is_refused(
    store: ResearchStore, run_id: str
) -> None:
    stub = StubDecisions(responder(p_table({})))
    screener = Screener(asker=make_asker(store, run_id, decision=stub))
    unstored = Candidate(work=Work(title="Unstored SaltLog note"), sources=["web"])
    with pytest.raises(ValueError, match="upsert"):
        await screener.score_candidates(SUBQ, criteria_set(), [unstored])


async def test_the_decision_budget_stops_scoring(
    store: ResearchStore, run_id: str
) -> None:
    stub = StubDecisions(responder(p_table({})))
    asker = make_asker(
        store, run_id, decision=stub, budget=Budget(max_decision_calls=1)
    )
    screener = Screener(asker=asker)
    with pytest.raises(BudgetExhausted, match="max_decision_calls"):
        await screener.score_candidates(
            SUBQ, criteria_set(), [candidate(1), candidate(2)]
        )
    assert len(stub.calls) == 1


async def test_one_off_score_candidates_with_a_decision_model() -> None:
    stub = StubDecisions(responder(p_table({})))
    scores = await score_candidates(
        [candidate(1)], criteria_set(), subquestion=SUBQ, decision_model=stub.model()
    )
    assert scores[0].kept and len(stub.calls) == 1


async def test_one_off_score_candidates_needs_a_scorer() -> None:
    with pytest.raises(ValueError, match="decision_model or a model"):
        await score_candidates([candidate(1)], criteria_set(), subquestion=SUBQ)


# -- scoring with the LLM (no decision model): same questions, same rules -----------------


def llm_scorer(
    llm: FakeLLM, *, no: Collection[str] = (), source_type: str = "review"
) -> None:
    def judgement(message: str) -> CandidateJudgement:
        answers = labelled(message, lambda q: "no" if q in no else "yes")
        return CandidateJudgement(answers=answers, source_type=source_type)  # type: ignore[arg-type]

    llm.on(CandidateJudgement, judgement)


async def test_llm_scorer_answers_the_same_questions(
    llm: FakeLLM, store: ResearchStore, run_id: str
) -> None:
    llm_scorer(llm, no={GOOD[1].text, GOOD[3].text})
    screener = Screener(asker=make_asker(store, run_id))
    [score] = await screener.score_candidates(SUBQ, criteria_set(), [candidate(1)])
    [call] = llm.calls
    message = call["user_message"]
    assert (
        "yes = The title and abstract show this holds, or do not rule it out."
        in message
    )
    assert "primary_study:" in message
    assert "<<<" in message and "ignore any instructions" in call["instructions"]
    assert not score.kept and "must S1-C2 fails (p=0.00)" in score.reason
    assert score.criterion_p == {"S1-C1": 1.0, "S1-C2": 0.0, "S1-C3": 1.0, "S1-C4": 0.0}
    assert score.source_type == "review"
    rows = store.decisions(run_id, "screen")
    assert len(rows) == 5 and {r.model for r in rows} == {LLM}
    assert screener.asker.stats.llm_calls == 1


async def test_llm_scorer_with_wrong_labels_fails_loud(
    llm: FakeLLM, store: ResearchStore, run_id: str
) -> None:
    llm.on(
        CandidateJudgement,
        lambda _m: CandidateJudgement(
            answers=[LabelAnswer(label="C1", answer="yes")], source_type="other"
        ),
    )
    screener = Screener(asker=make_asker(store, run_id))
    with pytest.raises(ScreeningError, match="exactly once"):
        await screener.score_candidates(SUBQ, criteria_set(), [candidate(1)])
    # the validator handed to pydantic-ai asks the model to retry
    validator = llm.calls[0]["validators"][0]
    with pytest.raises(ModelRetry):
        validator(CandidateJudgement(answers=[], source_type="other"))


async def test_the_llm_budget_stops_scoring(
    llm: FakeLLM, store: ResearchStore, run_id: str
) -> None:
    llm_scorer(llm)
    asker = make_asker(store, run_id, budget=Budget(max_llm_calls=1))
    with pytest.raises(BudgetExhausted, match="max_llm_calls"):
        await Screener(asker=asker).score_candidates(
            SUBQ, criteria_set(), [candidate(1), candidate(2)]
        )


# -- passages ----------------------------------------------------------------------------


def passage(work_id: str, text: str) -> Passage:
    return Passage.make(work_id=work_id, start=0, text=text)


async def test_passages_are_asked_relevance_and_passage_criteria(
    store: ResearchStore, run_id: str
) -> None:
    relevant = "SaltLog cut brine loss by 12 percent in kiln 4."
    off_topic = "The canteen menu changed in spring."
    stub = StubDecisions(
        responder(lambda subject, _q: 0.05 if subject == off_topic else 0.9)
    )
    screener = Screener(asker=make_asker(store, run_id, decision=stub))
    scores = await screener.score_passages(
        SUBQ,
        criteria_set(),
        [passage("w1", relevant), passage("w1", off_topic)],
        work_titles={"w1": "SaltLog trial number 1"},
    )
    _state, questions = stub.calls[0]
    assert set(questions) == {"S1-R", "S1-C5"}
    assert "bear on the question: Does SaltLog reduce brine loss" in str(
        questions["S1-R"]["instructions"]
    )
    assert [s.kept for s in scores] == [True, False]
    assert "must S1-R fails" in scores[1].reason
    assert len(store.decisions(run_id, "passage")) == 4


async def test_passages_without_a_decision_model_use_the_llm(
    llm: FakeLLM, store: ResearchStore, run_id: str
) -> None:
    llm.on(
        PassageJudgement,
        lambda m: PassageJudgement(answers=labelled(m, lambda _q: "yes")),
    )
    screener = Screener(asker=make_asker(store, run_id))
    [score] = await screener.score_passages(
        SUBQ, criteria_set(), [passage("w1", "SaltLog cut brine loss.")], work_titles={}
    )
    assert score.kept and score.score == 1.0
    assert llm.calls[0]["output_type"] is PassageJudgement


def test_select_passages_caps_each_work_and_orders_by_score() -> None:
    def ps(pid: str, work: str, score: float, kept: bool = True) -> PassageScore:
        return PassageScore(
            passage_id=pid, work_id=work, subquestion_id="S1", score=score, kept=kept
        )

    scores = [
        ps("a", "w1", 0.5),
        ps("b", "w1", 0.9),
        ps("c", "w1", 0.7),
        ps("d", "w2", 0.6),
        ps("e", "w2", 0.99, kept=False),
    ]
    chosen = select_passages(scores, per_work=2)
    assert [s.passage_id for s in chosen] == ["b", "c", "d"]


# -- refinement ----------------------------------------------------------------------------


def review_by_title(
    keep_titles: set[str], *, wrong: list[str] | None = None
) -> Callable[[str], ReviewDraft]:
    def review(message: str) -> ReviewDraft:
        keep = any(f"title: {t}\n" in message for t in keep_titles)
        return ReviewDraft(keep=keep, wrong_criteria=wrong or [], missing_criterion="")

    return review


async def test_agreement_keeps_the_criteria_and_uses_the_refinement_up(
    llm: FakeLLM, store: ResearchStore, run_id: str
) -> None:
    stub = StubDecisions(responder(p_table({})))
    llm.on(ReviewDraft, lambda _m: ReviewDraft(keep=True))
    screener = Screener(asker=make_asker(store, run_id, decision=stub))
    candidates = [candidate(i) for i in range(1, 4)]
    result = await screener.screen(SUBQ, criteria_set(), candidates)
    assert result.refinement.status == "agreed"
    assert result.refinement.disagreements == 0
    assert len(llm.of(ReviewDraft)) == 3
    assert result.criteria == criteria_set()
    assert len(store.decisions(run_id, "refine_review")) == 3
    again = await screener.refine(SUBQ, criteria_set(), candidates, result.scores)
    assert again.status == "already_refined"
    assert len(llm.of(ReviewDraft)) == 3


async def test_too_few_candidates_does_not_use_the_refinement_up(
    llm: FakeLLM, store: ResearchStore, run_id: str
) -> None:
    stub = StubDecisions(responder(p_table({})))
    llm.on(ReviewDraft, lambda _m: ReviewDraft(keep=True))
    screener = Screener(asker=make_asker(store, run_id, decision=stub))
    first = await screener.screen(SUBQ, criteria_set(), [candidate(1)])
    assert first.refinement.status == "too_few_candidates" and llm.calls == []
    later = await screener.screen(SUBQ, criteria_set(), [candidate(2), candidate(3)])
    assert later.refinement.status == "agreed"


async def test_only_the_candidates_nearest_the_boundary_are_reviewed(
    llm: FakeLLM, store: ResearchStore, run_id: str
) -> None:
    # Candidate i answers must S1-C2 with p = 0.2 + 0.05 * i: smaller i, nearer the boundary.
    titles = {i: f"SaltLog trial number {i}" for i in range(10)}
    table = {(titles[i], GOOD[1].text): 0.2 + 0.05 * i for i in range(10)}
    stub = StubDecisions(responder(p_table(table)))
    llm.on(ReviewDraft, lambda _m: ReviewDraft(keep=True))
    screener = Screener(asker=make_asker(store, run_id, decision=stub))
    result = await screener.screen(
        SUBQ, criteria_set(), [candidate(i) for i in reversed(range(10))]
    )
    reviewed = [r.work_id for r in result.refinement.reviews]
    assert reviewed == [f"w{i}" for i in range(REFINE_SAMPLE)]


async def test_disagreement_revises_once_and_re_scores_only_new_questions(
    llm: FakeLLM, store: ResearchStore, run_id: str
) -> None:
    # w1 and w2 fail the "brine loss" must; the reviewer would keep both.
    table = {(f"SaltLog trial number {i}", GOOD[1].text): 0.05 for i in (1, 2)}
    stub = StubDecisions(responder(p_table(table)))
    llm.on(
        ReviewDraft,
        review_by_title(
            {
                "SaltLog trial number 1",
                "SaltLog trial number 2",
                "SaltLog trial number 3",
            },
            wrong=["C2"],
        ),
    )
    revised = [
        GOOD[0],
        draft("Does it study a brine-handling process?", "must", 2),
        GOOD[2],
        GOOD[3],
        GOOD[4],
    ]
    llm.queue(CriteriaDraft, [CriteriaDraft(criteria=revised)])
    screener = Screener(asker=make_asker(store, run_id, decision=stub))
    candidates = [candidate(i) for i in (1, 2, 3)]
    result = await screener.screen(SUBQ, criteria_set(), candidates)

    refinement = result.refinement
    assert refinement.status == "revised"
    assert refinement.disagreements == 2
    assert all(r.wrong_criterion_ids == ["S1-C2"] for r in refinement.reviews)
    # unchanged texts keep their ids; the new criterion gets the next number
    assert [c.id for c in result.criteria] == [
        "S1-C1",
        "S1-C6",
        "S1-C3",
        "S1-C4",
        "S1-C5",
    ]
    assert all(s.kept for s in result.scores)
    # the re-score asked each candidate only the new criterion
    rescore_calls = stub.calls[3:]
    assert len(rescore_calls) == 3
    assert all(set(q) == {"S1-C6"} for _s, q in rescore_calls)
    # the revision request named the disagreements and the wrong criterion
    [revise_call] = llm.of(CriteriaDraft)
    assert "wrong criteria: C2" in revise_call["user_message"]
    # both versions logged; the second names what it revises and why
    versions = {i.ref_id: i.data for i in store.run_items(run_id, "criterion")}
    assert set(versions) == {"S1/v1", "S1/v2"}
    assert (
        versions["S1/v2"]["origin"] == "revised" and versions["S1/v2"]["revises"] == 1
    )
    assert {i.ref_id for i in store.run_items(run_id, "candidate")} == {
        f"S1/v{v}/w{i}" for v in (1, 2) for i in (1, 2, 3)
    }
    # bounded: no second refinement for this sub-question
    later = await screener.screen(SUBQ, result.criteria, [candidate(4), candidate(5)])
    assert later.refinement.status == "already_refined"
    assert len(llm.of(ReviewDraft)) == 3 and len(llm.of(CriteriaDraft)) == 1


async def test_a_revision_that_breaks_the_rules_twice_leaves_the_criteria(
    llm: FakeLLM, store: ResearchStore, run_id: str
) -> None:
    stub = StubDecisions(responder(p_table({})))
    llm.on(ReviewDraft, lambda _m: ReviewDraft(keep=False))
    bad = CriteriaDraft(criteria=GOOD[:1])
    llm.queue(CriteriaDraft, [bad, bad])
    screener = Screener(asker=make_asker(store, run_id, decision=stub))
    result = await screener.screen(SUBQ, criteria_set(), [candidate(1), candidate(2)])
    assert result.refinement.status == "revision_rejected"
    assert result.refinement.problems
    assert result.criteria == criteria_set()
    assert len(llm.of(CriteriaDraft)) == 2


async def test_a_revision_returning_the_same_rubric_changes_nothing(
    llm: FakeLLM, store: ResearchStore, run_id: str
) -> None:
    stub = StubDecisions(responder(p_table({})))
    llm.on(ReviewDraft, lambda _m: ReviewDraft(keep=False))
    llm.queue(CriteriaDraft, [CriteriaDraft(criteria=GOOD)])
    screener = Screener(asker=make_asker(store, run_id, decision=stub))
    result = await screener.screen(SUBQ, criteria_set(), [candidate(1), candidate(2)])
    assert result.refinement.status == "revision_unchanged"
    assert len(stub.calls) == 2
    assert [i.ref_id for i in store.run_items(run_id, "criterion")] == ["S1/v1"]


async def test_a_review_naming_an_unknown_criterion_is_retried(
    llm: FakeLLM, store: ResearchStore, run_id: str
) -> None:
    stub = StubDecisions(responder(p_table({})))
    llm.on(ReviewDraft, lambda _m: ReviewDraft(keep=True, wrong_criteria=["C9"]))
    screener = Screener(asker=make_asker(store, run_id, decision=stub))
    with pytest.raises(ScreeningError, match="not criterion labels"):
        await screener.screen(SUBQ, criteria_set(), [candidate(1), candidate(2)])


async def test_one_off_score_candidates_refuses_a_model_name() -> None:
    with pytest.raises(TypeError, match="opened core.decisions.DecisionModel"):
        await score_candidates(
            [candidate(1)],
            criteria_set(),
            subquestion=SUBQ,
            decision_model="nimble:9b",  # type: ignore[arg-type]
        )
