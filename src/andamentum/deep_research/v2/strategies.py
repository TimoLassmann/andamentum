"""The swappable parts of a run: one small implementation per strategy name.

A ``ResearchConfig`` (``config.py``) names one strategy per slot; the registries
there map each name to an object defined here. The engine calls the slot, never
a particular implementation, so an experiment adds a strategy here, an entry in
the registry and a preset, and leaves the engine alone.

Slots and the strategies that exist:

- **screen** — ``generic``: ONE question per candidate, "Is this item relevant
  to the question: <sub-question>?", put to the decision model (or, without
  one, to the language model, which answers the same ``Question`` object), over
  the title, abstract or snippet, year and venue; dropped when
  p(relevant) < ``config.drop_below`` (``GENERIC_DROP_BELOW``). ``criteria``:
  the compiled rubric, ``Screener.score_candidates`` as built.
  ``generic_prefilter``: the candidates of a sub-question are ranked by BM25 of
  title and abstract (or snippet) against the sub-question, in code (the
  passages' BM25), and only the best ``PREFILTER_TOP`` are put to the generic
  question; the rest are dropped as "below lexical pre-filter".
- **queue** — the order in which waiting (work, sub-question) pairs are
  admitted to screening when a round cannot screen them all: ``arrival``
  (the order they were found in, as built; the default) or ``source_rank``
  (best by ``source_rank_score`` first, ties in arrival order).
- **verify** — ``model``: every report sentence's claim support is put to the
  model; ``code_first``: a sentence whose content tokens occur in its cited
  quotes (``verify.CODE_SUPPORT_MIN_OVERLAP``) counts as supported in code and
  only the others go to the model.
- **refine** — ``off``; ``on``: ``Screener.refine`` as built (needs criteria).
- **rank** — ``score``: a kept work's best screen score, as built; ``rrf``:
  reciprocal-rank fusion of each source's search rank and the screen score's
  rank, per sub-question.
- **passages** — ``lexical``: BM25 over the work's passages against the
  sub-question, in code (no model call); ``criteria``: ``read.select_passages``
  as built (the decision model scores the passage criteria, else one LLM pick).
- **expand** — ``citations``: references and citations of the best kept works,
  ``config.expand_seeds`` of them; ``off``.
- **facets** — ``on``: the compiled criteria travel with the evidence table
  (extraction tags findings with them, coverage counts them, the writer lists
  them), as built; ``off``: none do.

Nothing here imports ``config`` at run time (the registries import this module).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

from andamentum.core.decisions import Question
from andamentum.references import Work

from .criteria import MAX_STATE_CHARS, Screener, ScreeningError, rank
from .read import _STOPWORDS, _WORD, select_passages
from .steps import Pair, ScreenOne, expansion_seeds
from .types import Candidate, CandidateScore, Criterion, Passage, SubQuestion

if TYPE_CHECKING:
    from .config import ResearchConfig

# The generic screen drops a candidate whose p(relevant) is below this.
GENERIC_DROP_BELOW = 0.5
# The generic pre-filter sends this many candidates of a sub-question (the best
# by BM25 against the sub-question) to the generic question; the rest are
# dropped without a model request.
PREFILTER_TOP = 12
# The key of the generic screen's one question in a candidate's ``criterion_p``.
RELEVANT_KEY = "relevant"
# The constant of reciprocal-rank fusion (Cormack et al. 2009): each list adds
# 1 / (RRF_K + rank). Large enough that one list's top place does not dominate.
RRF_K = 60
# BM25 parameters (the usual defaults).
BM25_K1 = 1.5
BM25_B = 0.75

Kept = dict[str, tuple[float, list[str]]]


# -- protocols -------------------------------------------------------------------


class ScreenStrategy(Protocol):
    """Scores one sub-question's candidates, one ``CandidateScore`` each."""

    name: str
    needs_criteria: bool

    async def screen(
        self,
        subquestion: SubQuestion,
        candidates: list[Candidate],
        *,
        criteria: list[Criterion],
        screener: Screener,
        config: ResearchConfig,
    ) -> list[CandidateScore]: ...


class RefineStrategy(Protocol):
    """May revise the criteria after a screen; returns the criteria that stand
    and the scores under them."""

    name: str
    needs_criteria: bool

    async def refine(
        self,
        subquestion: SubQuestion,
        criteria: list[Criterion],
        candidates: list[Candidate],
        scores: list[CandidateScore],
        *,
        screener: Screener,
        question: str,
    ) -> tuple[list[Criterion], list[CandidateScore]]: ...


class RankStrategy(Protocol):
    """Each kept work's ranking value and the sub-questions it was kept for;
    a higher value is read, expanded and listed first."""

    name: str

    def kept(
        self,
        scores: Mapping[Pair, CandidateScore],
        *,
        candidates: Mapping[str, Candidate],
    ) -> Kept: ...


class QueueStrategy(Protocol):
    """The order waiting pairs are offered to screening in. The engine applies
    ``steps.round_robin`` (and its per-sub-question cap) to the result, so a
    strategy only decides which of a sub-question's pairs come first."""

    name: str

    def order(
        self, pairs: Sequence[Pair], *, candidates: Mapping[str, Candidate]
    ) -> list[Pair]: ...


class PassageStrategy(Protocol):
    """Chooses a work's passages for one sub-question (``read.PassageSelect``)."""

    name: str
    needs_criteria: bool

    async def __call__(
        self,
        passages: Sequence[Passage],
        *,
        subquestion: SubQuestion,
        criteria: Sequence[Criterion],
        screener: Screener,
        limit: int,
        work_titles: Mapping[str, str],
    ) -> list[Passage]: ...


class ExpandStrategy(Protocol):
    """Which kept works a round follows citations from (none when disabled)."""

    name: str
    enabled: bool

    def seeds(
        self,
        kept: Kept,
        *,
        candidates: Mapping[str, Candidate],
        expanded: set[str],
        config: ResearchConfig,
    ) -> list[tuple[Work, list[str]]]: ...


class VerifyStrategy(Protocol):
    """How a report sentence's claim support is settled: ``code_first`` says
    whether the token-overlap rule decides the sentences it can before the
    model sees the rest."""

    name: str
    code_first: bool


class FacetStrategy(Protocol):
    """The criteria that travel with the evidence table (extraction, coverage,
    the writer)."""

    name: str
    needs_criteria: bool

    def evidence(
        self, criteria: Mapping[str, list[Criterion]]
    ) -> dict[str, list[Criterion]]: ...


# -- screen ----------------------------------------------------------------------


def relevance_question(subquestion: SubQuestion) -> Question:
    """The generic screen's one question, the same for every kind of query."""
    text = subquestion.text.strip().rstrip("?").strip()
    return Question(
        type="noul", instructions=f"Is this item relevant to the question: {text}?"
    )


def item_state(candidate: Candidate) -> dict[str, object]:
    """What the generic screen shows of a candidate: title, abstract (or the
    search snippet), year and venue. The question is in the question itself."""
    work = candidate.work
    text = work.abstract or candidate.snippet
    return {
        "title": work.title,
        "abstract": text
        if len(text) <= MAX_STATE_CHARS
        else text[:MAX_STATE_CHARS] + " ...",
        "year": work.year,
        "venue": work.venue,
    }


class GenericScreen:
    """One relevance question per candidate; keep when p >= ``drop_below``."""

    name = "generic"
    needs_criteria = False

    async def screen(
        self,
        subquestion: SubQuestion,
        candidates: list[Candidate],
        *,
        criteria: list[Criterion],
        screener: Screener,
        config: ResearchConfig,
    ) -> list[CandidateScore]:
        question = {RELEVANT_KEY: relevance_question(subquestion)}
        scores: list[CandidateScore] = []
        for candidate in candidates:
            work_id = candidate.work.work_id
            if not work_id:
                raise ValueError(
                    f"candidate {candidate.work.title!r} has no work_id; "
                    "upsert it into the store before screening"
                )
            answers = await screener.ask(item_state(candidate), question, step="screen")
            p = answers[RELEVANT_KEY].p_true
            if p is None:
                raise ScreeningError(
                    f"a noul answer without a probability: {answers!r}"
                )
            kept = p >= config.drop_below
            score = CandidateScore(
                work_id=work_id,
                subquestion_id=subquestion.id,
                criterion_p={RELEVANT_KEY: p},
                score=p,
                kept=kept,
                reason=(
                    f"{'kept' if kept else 'dropped'}: p(relevant)={p:.2f} "
                    f"({'>=' if kept else '<'} {config.drop_below:.2f})"
                ),
            )
            screener.store.add_run_item(
                screener.run_id,
                "candidate",
                f"{subquestion.id}/generic/{work_id}",
                {**score.model_dump(mode="json"), "screen": self.name},
            )
            scores.append(score)
        return scores


def prefilter_text(candidate: Candidate) -> str:
    """What the lexical pre-filter reads of a candidate: title and abstract (or
    the search snippet)."""
    work = candidate.work
    return f"{work.title} {work.abstract or candidate.snippet}"


def prefilter_order(
    subquestion: SubQuestion, candidates: Sequence[Candidate]
) -> list[int]:
    """Indexes of *candidates*, best BM25 against the sub-question first; equal
    scores keep the order the candidates came in."""
    scores = bm25_scores(subquestion.text, [prefilter_text(c) for c in candidates])
    return sorted(range(len(candidates)), key=lambda i: (-scores[i], i))


class GenericPrefilterScreen(GenericScreen):
    """The generic screen over the ``PREFILTER_TOP`` candidates a lexical
    ranking puts first; every other candidate is dropped, in code, with the
    reason recorded (and logged as a ``candidate`` item like any other)."""

    name = "generic_prefilter"

    async def screen(
        self,
        subquestion: SubQuestion,
        candidates: list[Candidate],
        *,
        criteria: list[Criterion],
        screener: Screener,
        config: ResearchConfig,
    ) -> list[CandidateScore]:
        order = prefilter_order(subquestion, candidates)
        asked = {i for i in order[:PREFILTER_TOP]}
        answered = iter(
            await super().screen(
                subquestion,
                [c for i, c in enumerate(candidates) if i in asked],
                criteria=criteria,
                screener=screener,
                config=config,
            )
        )
        rank_of = {index: place for place, index in enumerate(order, start=1)}
        scores: list[CandidateScore] = []
        for index, candidate in enumerate(candidates):
            if index in asked:
                scores.append(next(answered))
                continue
            work_id = candidate.work.work_id
            if not work_id:
                raise ValueError(
                    f"candidate {candidate.work.title!r} has no work_id; "
                    "upsert it into the store before screening"
                )
            score = CandidateScore(
                work_id=work_id,
                subquestion_id=subquestion.id,
                score=0.0,
                kept=False,
                reason=(
                    f"dropped: below lexical pre-filter (BM25 rank {rank_of[index]} "
                    f"of {len(candidates)}; the best {PREFILTER_TOP} are asked)"
                ),
            )
            screener.store.add_run_item(
                screener.run_id,
                "candidate",
                f"{subquestion.id}/generic/{work_id}",
                {**score.model_dump(mode="json"), "screen": self.name},
            )
            scores.append(score)
        return scores


class CriteriaScreen:
    """The compiled rubric, scored as built (``Screener.score_candidates``)."""

    name = "criteria"
    needs_criteria = True

    async def screen(
        self,
        subquestion: SubQuestion,
        candidates: list[Candidate],
        *,
        criteria: list[Criterion],
        screener: Screener,
        config: ResearchConfig,
    ) -> list[CandidateScore]:
        return await screener.score_candidates(subquestion, criteria, candidates)


# -- refine ----------------------------------------------------------------------


class NoRefine:
    name = "off"
    needs_criteria = False

    async def refine(
        self,
        subquestion: SubQuestion,
        criteria: list[Criterion],
        candidates: list[Candidate],
        scores: list[CandidateScore],
        *,
        screener: Screener,
        question: str,
    ) -> tuple[list[Criterion], list[CandidateScore]]:
        return list(criteria), scores


class CriteriaRefine:
    """``Screener.refine`` as built: once per sub-question, boundary reviews,
    at most one revision and a re-score."""

    name = "on"
    needs_criteria = True

    async def refine(
        self,
        subquestion: SubQuestion,
        criteria: list[Criterion],
        candidates: list[Candidate],
        scores: list[CandidateScore],
        *,
        screener: Screener,
        question: str,
    ) -> tuple[list[Criterion], list[CandidateScore]]:
        refinement = await screener.refine(
            subquestion, criteria, candidates, scores, question=question
        )
        return refinement.criteria, refinement.scores


def screen_one(
    config: ResearchConfig,
    *,
    criteria: Mapping[str, list[Criterion]],
    screener: Screener,
    question: str,
) -> ScreenOne:
    """The run's screen then refine strategy for one sub-question, ranked kept
    first by score (``criteria.rank``). With the ``criteria`` screen and ``on``
    refinement this is exactly ``Screener.screen``."""
    screener_strategy = config.screen_strategy
    refiner = config.refine_strategy

    async def run(
        subquestion: SubQuestion, candidates: list[Candidate]
    ) -> tuple[list[Criterion], list[CandidateScore]]:
        own = list(criteria.get(subquestion.id, []))
        scores = await screener_strategy.screen(
            subquestion, candidates, criteria=own, screener=screener, config=config
        )
        standing, final = await refiner.refine(
            subquestion, own, candidates, scores, screener=screener, question=question
        )
        return standing, rank(final)

    return run


# -- queue -----------------------------------------------------------------------


def source_rank_score(candidate: Candidate) -> float:
    """Reciprocal-rank fusion of the candidate's places in the lists it was found
    in: the sum of ``1 / (RRF_K + rank)`` over ``source_ranks`` (a source's
    search results, or a citation hop, each its own list). The search rank
    only: no screening score is involved, so the rule is the same for every
    arm. It orders the screening queue (``source_rank``) and picks the works
    whose text a freeze prefetches (``bench.freeze.prefetch_targets``): one
    function for both."""
    return sum(1.0 / (RRF_K + rank) for rank in candidate.source_ranks.values())


class ArrivalQueue:
    """Waiting pairs in the order they were found in (the engine as built)."""

    name = "arrival"

    def order(
        self, pairs: Sequence[Pair], *, candidates: Mapping[str, Candidate]
    ) -> list[Pair]:
        return list(pairs)


class SourceRankQueue:
    """Waiting pairs best first by :func:`source_rank_score`; pairs with an equal
    score keep their arrival order (a stable sort)."""

    name = "source_rank"

    def order(
        self, pairs: Sequence[Pair], *, candidates: Mapping[str, Candidate]
    ) -> list[Pair]:
        return sorted(pairs, key=lambda pair: -source_rank_score(candidates[pair[0]]))


# -- rank ------------------------------------------------------------------------


class ScoreRank:
    """A kept work's best screen score over its sub-questions, as built."""

    name = "score"

    def kept(
        self,
        scores: Mapping[Pair, CandidateScore],
        *,
        candidates: Mapping[str, Candidate],
    ) -> Kept:
        kept: Kept = {}
        for (work_id, subquestion_id), score in scores.items():
            if not score.kept:
                continue
            best, ids = kept.get(work_id, (0.0, []))
            kept[work_id] = (max(best, score.score), [*ids, subquestion_id])
        return kept


def _competition_ranks(values: Mapping[str, float]) -> dict[str, int]:
    """1-based ranks, highest value first; equal values share the better rank."""
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    ranks: dict[str, int] = {}
    previous: float | None = None
    current = 0
    for position, (key, value) in enumerate(ordered, start=1):
        if value != previous:
            current, previous = position, value
        ranks[key] = current
    return ranks


def rrf_scores(
    kept_scores: Sequence[CandidateScore], *, candidates: Mapping[str, Candidate]
) -> dict[str, float]:
    """Reciprocal-rank fusion for one sub-question's kept candidates: the sum
    of ``1 / (RRF_K + rank)`` over every list a work is in, the lists being
    each source's search results (``Candidate.source_ranks``, the best place
    the work reached) and the screen scores (ties share a rank)."""
    fused: dict[str, float] = {}
    screen_ranks = _competition_ranks({s.work_id: s.score for s in kept_scores})
    for score in kept_scores:
        candidate = candidates.get(score.work_id)
        ranks = [screen_ranks[score.work_id]]
        if candidate is not None:
            ranks += list(candidate.source_ranks.values())
        fused[score.work_id] = sum(1.0 / (RRF_K + r) for r in ranks)
    return fused


class RrfRank:
    """Reciprocal-rank fusion of the source ranks and the screen score, per
    sub-question; a work kept for several takes its best."""

    name = "rrf"

    def kept(
        self,
        scores: Mapping[Pair, CandidateScore],
        *,
        candidates: Mapping[str, Candidate],
    ) -> Kept:
        by_subquestion: dict[str, list[CandidateScore]] = {}
        ids: dict[str, list[str]] = {}
        for (work_id, subquestion_id), score in scores.items():
            if not score.kept:
                continue
            by_subquestion.setdefault(subquestion_id, []).append(score)
            ids.setdefault(work_id, []).append(subquestion_id)
        best: dict[str, float] = {}
        for own in by_subquestion.values():
            for work_id, value in rrf_scores(own, candidates=candidates).items():
                best[work_id] = max(value, best.get(work_id, 0.0))
        return {
            work_id: (best[work_id], subquestion_ids)
            for work_id, subquestion_ids in ids.items()
        }


# -- passages --------------------------------------------------------------------


def lexical_terms(text: str) -> list[str]:
    """Words of *text* for BM25: casefolded, three letters or more, stopwords
    out, a plural ``s`` folded (``latencies`` stays, ``measurements`` ->
    ``measurement``)."""
    terms: list[str] = []
    for word in _WORD.findall(text.casefold()):
        if word in _STOPWORDS:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        terms.append(word)
    return terms


def bm25_scores(query: str, documents: Sequence[str]) -> list[float]:
    """Okapi BM25 of each document against *query*, with the documents as the
    collection (a work's own passages). The idf is the non-negative form,
    ``log(1 + (N - df + 0.5) / (df + 0.5))``."""
    tokenised = [lexical_terms(d) for d in documents]
    if not tokenised:
        return []
    wanted = set(lexical_terms(query))
    count = len(tokenised)
    average = sum(len(t) for t in tokenised) / count or 1.0
    frequency = Counter(term for terms in tokenised for term in set(terms))
    idf = {
        term: math.log(1 + (count - frequency[term] + 0.5) / (frequency[term] + 0.5))
        for term in wanted
    }
    scores: list[float] = []
    for terms in tokenised:
        counts = Counter(terms)
        norm = BM25_K1 * (1 - BM25_B + BM25_B * len(terms) / average)
        scores.append(
            sum(
                idf[term] * counts[term] * (BM25_K1 + 1) / (counts[term] + norm)
                for term in wanted
                if counts[term]
            )
        )
    return scores


class LexicalPassages:
    """BM25 of each passage against the sub-question; the best *limit* that
    share at least one term, best first. No model call."""

    name = "lexical"
    needs_criteria = False

    async def __call__(
        self,
        passages: Sequence[Passage],
        *,
        subquestion: SubQuestion,
        criteria: Sequence[Criterion],
        screener: Screener,
        limit: int,
        work_titles: Mapping[str, str],
    ) -> list[Passage]:
        scores = bm25_scores(subquestion.text, [p.text for p in passages])
        order = sorted(
            (i for i, value in enumerate(scores) if value > 0),
            key=lambda i: (-scores[i], i),
        )
        return [passages[i] for i in order[:limit]]


class CriteriaPassages:
    """``read.select_passages`` as built."""

    name = "criteria"
    needs_criteria = True

    async def __call__(
        self,
        passages: Sequence[Passage],
        *,
        subquestion: SubQuestion,
        criteria: Sequence[Criterion],
        screener: Screener,
        limit: int,
        work_titles: Mapping[str, str],
    ) -> list[Passage]:
        return await select_passages(
            passages,
            subquestion=subquestion,
            criteria=criteria,
            screener=screener,
            limit=limit,
            work_titles=work_titles,
        )


# -- expand ----------------------------------------------------------------------


class CitationExpand:
    """References and citations of the best kept works not followed yet."""

    name = "citations"
    enabled = True

    def seeds(
        self,
        kept: Kept,
        *,
        candidates: Mapping[str, Candidate],
        expanded: set[str],
        config: ResearchConfig,
    ) -> list[tuple[Work, list[str]]]:
        return expansion_seeds(
            kept, candidates=candidates, expanded=expanded, limit=config.expand_seeds
        )


class NoExpand:
    name = "off"
    enabled = False

    def seeds(
        self,
        kept: Kept,
        *,
        candidates: Mapping[str, Candidate],
        expanded: set[str],
        config: ResearchConfig,
    ) -> list[tuple[Work, list[str]]]:
        return []


# -- verify ----------------------------------------------------------------------


class ModelVerify:
    """Every sentence's claim support goes to the model."""

    name = "model"
    code_first = False


class CodeFirstVerify:
    """A sentence whose content tokens are in its quotes is supported in code;
    only the rest go to the model."""

    name = "code_first"
    code_first = True


# -- facets ----------------------------------------------------------------------


class CriteriaFacets:
    """The compiled criteria travel with the evidence table, as built."""

    name = "on"
    needs_criteria = True

    def evidence(
        self, criteria: Mapping[str, list[Criterion]]
    ) -> dict[str, list[Criterion]]:
        return dict(criteria)


class NoFacets:
    name = "off"
    needs_criteria = False

    def evidence(
        self, criteria: Mapping[str, list[Criterion]]
    ) -> dict[str, list[Criterion]]:
        return {}
