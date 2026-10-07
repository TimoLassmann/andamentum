"""Each baseline strategy on its own: the generic screen, rank fusion, BM25
passages, the caps, and the expansion seeds."""

from __future__ import annotations

import pytest

from andamentum.references import Work

from ..config import PRESETS
from ..criteria import Screener, ScreeningError
from ..identity import merge
from ..steps import Pair, read_targets, round_robin
from ..strategies import (
    RELEVANT_KEY,
    RRF_K,
    CitationExpand,
    LexicalPassages,
    NoExpand,
    RrfRank,
    ScoreRank,
    bm25_scores,
    item_state,
    lexical_terms,
    relevance_question,
    rrf_scores,
    screen_one,
)
from ..types import Candidate, CandidateScore, Passage, SubQuestion
from .engine_world import respond
from .llm_stub import ScriptedLLM, decisions, make_asker

SUBQ = SubQuestion(id="Q1", text="How much does the Apollo cache reduce latency?")
BASELINE = PRESETS["baseline"]


def _candidate(
    n: int, *, abstract: str = "", ranks: dict[str, int] | None = None
) -> Candidate:
    return Candidate(
        work=Work(
            doi=f"10.5555/w{n}",
            title=f"Apollo work {n}",
            abstract=abstract or f"Apollo cache study number {n}.",
            year=2024,
            venue="Journal of Invented Systems",
            work_id=f"w{n}",
        ),
        sources=["openalex"],
        snippet="",
        subquestion_ids=["Q1"],
        source_ranks=ranks or {},
    )


def _score(
    work_id: str, p: float, *, kept: bool = True, subq: str = "Q1"
) -> CandidateScore:
    return CandidateScore(
        work_id=work_id,
        subquestion_id=subq,
        criterion_p={RELEVANT_KEY: p},
        score=p,
        kept=kept,
    )


# -- the generic screen ---------------------------------------------------------------


def test_the_relevance_question_is_one_query_agnostic_noul() -> None:
    question = relevance_question(SUBQ)
    assert question.type == "noul"
    assert question.instructions == (
        "Does this item contain information that helps answer the question: "
        "How much does the Apollo cache reduce latency?"
    )
    assert not question.criteria  # no LLM-written rubric of any kind


def test_the_item_state_is_title_text_date_source() -> None:
    state = item_state(_candidate(1))
    assert set(state) == {"title", "text", "date", "source"}
    snippet_only = Candidate(
        work=Work(url="https://a.example/x", title="A page", work_id="w9"),
        sources=["web"],
        snippet="the search snippet",
    )
    assert item_state(snippet_only)["text"] == "the search snippet"


async def test_the_generic_screen_asks_one_question_and_drops_below_the_threshold() -> (
    None
):
    p_of = {"Apollo work 1": 0.9, "Apollo work 2": 0.5, "Apollo work 3": 0.49}

    def answer(state: object, instructions: str) -> float:
        assert isinstance(state, dict)
        return p_of[state["title"]]

    model, stub = decisions(answer)
    asker = make_asker(decision_model=model)
    screener = Screener(asker=asker)
    run = screen_one(BASELINE, criteria={}, screener=screener, question="q")
    candidates = [_candidate(1), _candidate(2), _candidate(3)]
    for c in candidates:
        asker.store.upsert_work(c.work)
    standing, scores = await run(SUBQ, candidates)

    assert standing == []
    assert len(stub.calls) == 3  # one request per candidate
    for state, questions in stub.calls:
        assert list(questions) == [RELEVANT_KEY]  # one question, nothing else
        assert (
            questions[RELEVANT_KEY]["instructions"]
            == relevance_question(SUBQ).instructions
        )
        assert isinstance(state, dict) and "question" not in state
    by_work = {s.work_id: s for s in scores}
    assert by_work["w1"].kept and by_work["w2"].kept  # 0.5 is not below 0.5
    assert not by_work["w3"].kept and "0.49" in by_work["w3"].reason
    assert scores[0].work_id == "w1"  # ranked kept first, best first
    items = asker.store.run_items(asker.run_id, "candidate")
    assert {i.ref_id for i in items} == {
        "Q1/generic/w1",
        "Q1/generic/w2",
        "Q1/generic/w3",
    }
    assert {d.step for d in asker.store.decisions(asker.run_id)} == {"screen"}


async def test_without_a_decision_model_the_llm_answers_the_same_question() -> None:
    llm = ScriptedLLM(respond)
    asker = make_asker(model=llm)
    screener = Screener(asker=asker)
    off_topic = Candidate(
        work=Work(
            doi="10.5555/s",
            title="SaltLog notes",
            abstract="About SaltLog.",
            work_id="ws",
        ),
        sources=["openalex"],
    )
    for c in (_candidate(1), off_topic):
        asker.store.upsert_work(c.work)
    run = screen_one(BASELINE, criteria={}, screener=screener, question="q")
    _, scores = await run(SUBQ, [_candidate(1), off_topic])
    assert len(llm.calls) == 2  # one call per candidate, the same single question
    for _instructions, prompt in llm.calls:
        assert (
            "C1: Does this item contain information that helps answer the "
            "question: How much does the Apollo" in prompt
        )
        assert "Sub-question:" not in prompt  # the question is in the question
    by_work = {s.work_id: s for s in scores}
    assert by_work["w1"].kept and by_work["w1"].score == 1.0
    assert not by_work["ws"].kept and by_work["ws"].score == 0.0


async def test_a_second_screen_of_the_same_candidate_asks_nothing() -> None:
    model, stub = decisions(lambda state, q: 0.8)
    asker = make_asker(decision_model=model)
    asker.store.upsert_work(_candidate(1).work)
    run = screen_one(
        BASELINE, criteria={}, screener=Screener(asker=asker), question="q"
    )
    await run(SUBQ, [_candidate(1)])
    await run(SUBQ, [_candidate(1)])
    assert len(stub.calls) == 1


async def test_an_answer_without_a_probability_is_an_error() -> None:
    from andamentum.core.decisions import Answer

    class Broken(Screener):
        async def ask(self, state, questions, *, step):  # type: ignore[no-untyped-def]
            return {
                RELEVANT_KEY: Answer(choice="yes", probabilities={}, confidence=1.0)
            }

    asker = make_asker(decision_model=decisions(lambda s, q: 0.5)[0])
    asker.store.upsert_work(_candidate(1).work)
    run = screen_one(BASELINE, criteria={}, screener=Broken(asker=asker), question="q")
    with pytest.raises(ScreeningError):
        await run(SUBQ, [_candidate(1)])


# -- rank fusion ------------------------------------------------------------------------


def test_rrf_sums_the_source_ranks_and_the_screen_rank() -> None:
    candidates = {
        "w1": _candidate(1, ranks={"openalex": 1, "europepmc": 3}),
        "w2": _candidate(2, ranks={"openalex": 2}),
        "w3": _candidate(3, ranks={}),
    }
    kept = [_score("w1", 0.6), _score("w2", 0.9), _score("w3", 0.9)]
    fused = rrf_scores(kept, candidates=candidates)
    # screen ranks: w2 and w3 tie at 1 (0.9), w1 is 3rd
    assert fused["w1"] == pytest.approx(
        1 / (RRF_K + 3) + 1 / (RRF_K + 1) + 1 / (RRF_K + 3)
    )
    assert fused["w2"] == pytest.approx(1 / (RRF_K + 1) + 1 / (RRF_K + 2))
    assert fused["w3"] == pytest.approx(1 / (RRF_K + 1))
    # two source lists outweigh one: fusion rewards agreement between lists
    assert fused["w1"] > fused["w2"] > fused["w3"]


def test_rrf_ranks_kept_works_only_and_takes_the_best_subquestion() -> None:
    candidates = {"w1": _candidate(1, ranks={"openalex": 1}), "w2": _candidate(2)}
    scores: dict[Pair, CandidateScore] = {
        ("w1", "Q1"): _score("w1", 0.9),
        ("w1", "Q2"): _score("w1", 0.9, subq="Q2"),
        ("w2", "Q1"): _score("w2", 0.2, kept=False),
    }
    kept = RrfRank().kept(scores, candidates=candidates)
    assert set(kept) == {"w1"}
    assert kept["w1"][1] == ["Q1", "Q2"]
    assert kept["w1"][0] == pytest.approx(2 / (RRF_K + 1))


def test_the_score_rank_is_the_best_screen_score() -> None:
    scores: dict[Pair, CandidateScore] = {
        ("w1", "Q1"): _score("w1", 0.6),
        ("w1", "Q2"): _score("w1", 0.8, subq="Q2"),
    }
    assert ScoreRank().kept(scores, candidates={}) == {"w1": (0.8, ["Q1", "Q2"])}


def test_merging_keeps_each_sources_best_rank() -> None:
    first = _candidate(1, ranks={"openalex": 4})
    second = _candidate(1, ranks={"openalex": 2, "europepmc": 7})
    (merged,) = merge([first, second])
    assert merged.source_ranks == {"openalex": 2, "europepmc": 7}


# -- BM25 passages ------------------------------------------------------------------------


def test_lexical_terms_fold_case_stopwords_and_plurals() -> None:
    assert lexical_terms("The Measurements of the CACHES and latency") == [
        "measurement",
        "cache",
        "latency",
    ]


def test_bm25_prefers_the_passage_that_matches_more_and_rarer_terms() -> None:
    docs = [
        "The Apollo cache reduced median latency from 120 ms to 45 ms.",
        "The weather was fine and the team had lunch.",
        "Latency was measured on every service.",
    ]
    scores = bm25_scores("How much does the Apollo cache reduce latency?", docs)
    assert scores[0] > scores[2] > scores[1] == 0.0


async def test_lexical_selection_takes_the_top_matches_and_asks_no_model() -> None:
    texts = [
        "Unrelated opening words about the history of the project team.",
        "Apollo cache latency fell by half in read-heavy services.",
        "More history, budgets and office moves.",
        "The Apollo cache needed no tuning.",
    ]
    passages = []
    position = 0
    for text in texts:
        passages.append(Passage.make(work_id="w1", start=position, text=text))
        position += len(text) + 2
    llm = ScriptedLLM(lambda i, p: {})
    asker = make_asker(model=llm)
    chosen = await LexicalPassages()(
        passages,
        subquestion=SUBQ,
        criteria=[],
        screener=Screener(asker=asker),
        limit=4,
        work_titles={},
    )
    assert [p.text for p in chosen] == [texts[1], texts[3]]  # zero-score ones left out
    assert llm.calls == [] and asker.stats.llm_calls == 0
    two = await LexicalPassages()(
        passages,
        subquestion=SUBQ,
        criteria=[],
        screener=Screener(asker=asker),
        limit=1,
        work_titles={},
    )
    assert [p.text for p in two] == [texts[1]]


# -- caps -------------------------------------------------------------------------------


def test_the_screen_cap_takes_the_first_of_each_subquestion() -> None:
    q1 = SubQuestion(id="Q1", text="a?")
    q2 = SubQuestion(id="Q2", text="b?")
    pairs = [(f"a{i}", "Q1") for i in range(5)] + [(f"b{i}", "Q2") for i in range(2)]
    chosen = round_robin(pairs, subquestions=[q1, q2], per_subquestion=3)
    assert [p for p in chosen if p[1] == "Q1"] == [
        ("a0", "Q1"),
        ("a1", "Q1"),
        ("a2", "Q1"),
    ]
    assert len([p for p in chosen if p[1] == "Q2"]) == 2


def test_the_read_cap_counts_the_whole_run_per_subquestion() -> None:
    kept = {f"w{i}": (1.0 - i / 10, ["Q1"]) for i in range(6)}
    kept["x"] = (0.5, ["Q2"])
    first = read_targets(
        kept, done=set(), excluded={}, covered=set(), limit=20, per_subquestion=3
    )
    assert [w for w, _ in first if w != "x"] == ["w0", "w1", "w2"]
    later = read_targets(
        kept,
        done={("w0", "Q1"), ("w1", "Q1")},
        excluded={},
        covered=set(),
        limit=20,
        per_subquestion=3,
    )
    assert [w for w, ids in later if "Q1" in ids] == ["w2"]  # one left of three
    assert read_targets(
        kept, done=set(), excluded={}, covered=set(), limit=20
    )  # no cap: as built


def test_the_baseline_expands_the_three_best_kept_works() -> None:
    candidates = {f"w{i}": _candidate(i) for i in range(6)}
    kept = {w: (i / 10, ["Q1"]) for i, w in enumerate(candidates)}
    seeds = CitationExpand().seeds(
        kept, candidates=candidates, expanded=set(), config=BASELINE
    )
    assert [w.work_id for w, _ in seeds] == ["w5", "w4", "w3"]
    assert (
        NoExpand().seeds(kept, candidates=candidates, expanded=set(), config=BASELINE)
        == []
    )
