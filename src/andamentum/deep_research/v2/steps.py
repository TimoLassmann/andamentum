"""The work behind each node of the v2 graph, free of the graph engine.

``graph.py`` routes and assigns state; the functions here compute and return
values (dialect L2, L3). Each takes what it needs as arguments, so a step can be
tested without a graph. The bigger workers live in their own modules
(``criteria``, ``read``, ``coverage``, ``write``, ``verify``); this module holds
the steps that are only glue around them: planning, query writing, searching,
admitting candidates into the run, screening a round, citation expansion, the
choice of what to read, and conflict finding with its failures recorded.

Failure rules, the same in every step:

- a source that fails raises ``SourceError``; the step records why, marks the
  run degraded and goes on with the other sources (a dead source never reads
  as "nothing found");
- a model that fails raises one of ``ask.MODEL_ERRORS``; the step records it,
  marks the run degraded and falls back to a plainer path that invents nothing
  (the question as its own sub-question, the sub-question as its own query, one
  relevance criterion, an unscreened candidate left for the next round);
- a spent budget raises ``BudgetExhausted``; the step returns what it has and
  names the bound in ``spent``, and the graph stops the search with
  ``stop_reason='budget:<name>'``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry

from andamentum.core.decisions import Question
from andamentum.references import Work

from .ask import MODEL_ERRORS, Asker, BudgetExhausted
from .coverage import PairKey, find_conflicts
from .criteria import Screener
from .identity import merge, merge_pair
from .sources.base import ExpandingSource, SearchFilters, Source, SourceError
from .store import ResearchStore
from .types import (
    Candidate,
    CandidateScore,
    Conflict,
    Coverage,
    Criterion,
    Finding,
    SubQuestion,
    SubQuestionKind,
)

MIN_SUBQUESTIONS = 2
MAX_SUBQUESTIONS = 6
# Kept works whose references and citations are followed, per round, best first.
MAX_EXPAND_SEEDS = 5
# Results asked of one references or citations call.
EXPAND_PER_CALL = 20
# Queries already run that the query writer is shown, most recent last.
MAX_QUERIES_SHOWN = 30
# Language-model calls held back from searching and reading, per sub-question,
# so the report can still be written when the search spends the budget.
WRITE_RESERVE_PER_SUBQUESTION = 1
WRITE_RESERVE_FIXED = 2

Pair = tuple[str, str]  # (work_id, subquestion_id)

_KIND_OPTIONS: dict[str, str] = {
    "topical": "Find what is known about a topic: studies, evidence, explanations.",
    "navigational": "Find one particular paper, page or document named by its "
    "title, authors or name.",
    "metadata": "Find works by a property such as publication year, author, "
    "venue or open access.",
}
KIND_QUESTION = Question(
    type="choice",
    instructions="What kind of search does this sub-question need?",
    options=_KIND_OPTIONS,
)
_KINDS: tuple[SubQuestionKind, ...] = ("topical", "navigational", "metadata")


# -- plan ---------------------------------------------------------------------


class PlannedSubQuestion(BaseModel):
    text: str = Field(description="One sub-question, a full sentence ending in '?'.")
    kind: SubQuestionKind = Field(
        default="topical",
        description="topical (what is known), navigational (one named paper or "
        "page), or metadata (works by year, author, venue or open access)",
    )


class PlanDraft(BaseModel):
    subquestions: list[PlannedSubQuestion]


PLAN_INSTRUCTIONS = f"""\
You split a research question into {MIN_SUBQUESTIONS} to {MAX_SUBQUESTIONS} \
sub-questions. Together they must cover everything the question asks; each asks \
one thing that sources can answer on their own. Do not answer them. Do not add \
sub-questions the question does not need. Keep the question's own terms, names, \
populations, years and places in the sub-questions that need them."""


@dataclass
class PlanOutcome:
    subquestions: list[SubQuestion]
    degraded: list[str] = field(default_factory=list)
    spent: str = ""


def single_subquestion(question: str) -> list[SubQuestion]:
    """The fallback plan: the question as its one sub-question."""
    return [SubQuestion(id="Q1", text=question.strip())]


async def plan(question: str, *, asker: Asker) -> PlanOutcome:
    """Sub-questions for *question*: the LLM writes them; with a decision model
    each one's kind is its ``choice`` (else the LLM's field). A failed plan is
    the question as its one sub-question, recorded as degraded."""

    def well_formed(draft: PlanDraft) -> PlanDraft:
        texts = [s.text.strip() for s in draft.subquestions]
        if not MIN_SUBQUESTIONS <= len(texts) <= MAX_SUBQUESTIONS:
            raise ModelRetry(
                f"give {MIN_SUBQUESTIONS} to {MAX_SUBQUESTIONS} sub-questions, "
                f"not {len(texts)}"
            )
        if any(not t for t in texts):
            raise ModelRetry("every sub-question needs text")
        if len({t.casefold() for t in texts}) != len(texts):
            raise ModelRetry("the sub-questions repeat each other")
        return draft

    try:
        draft = await asker.llm(
            instructions=PLAN_INSTRUCTIONS,
            output_type=PlanDraft,
            user_message=f"QUESTION: {question.strip()}",
            validators=[well_formed],
        )
    except BudgetExhausted as spent:
        return PlanOutcome(single_subquestion(question), spent=spent.name)
    except MODEL_ERRORS as error:
        return PlanOutcome(
            single_subquestion(question),
            degraded=[
                f"planning failed ({type(error).__name__}: {error}); the question "
                "is searched as one sub-question"
            ],
        )
    planned = [
        SubQuestion(id=f"Q{n}", text=s.text.strip(), kind=s.kind)
        for n, s in enumerate(draft.subquestions, start=1)
    ]
    outcome = PlanOutcome(planned)
    if asker.decision_model is None:
        return outcome
    for position, subquestion in enumerate(planned):
        try:
            answers = await asker.decide(
                "plan_kind",
                state={"question": question, "sub_question": subquestion.text},
                questions={"kind": KIND_QUESTION},
            )
        except BudgetExhausted as spent:
            outcome.spent = spent.name
            break
        except MODEL_ERRORS as error:
            outcome.degraded.append(
                f"the kind of {subquestion.id} was not decided "
                f"({type(error).__name__}: {error}); the planner's kind stands"
            )
            continue
        chosen = answers["kind"].choice
        for kind in _KINDS:
            if chosen == kind:
                planned[position] = subquestion.model_copy(update={"kind": kind})
    return outcome


_YEAR = r"((?:19|20)\d{2})"
_SINCE = re.compile(rf"\b(since|from|after|newer than)\s+{_YEAR}\b", re.IGNORECASE)
_UNTIL = re.compile(
    rf"\b(before|until|up to|prior to|older than|through)\s+{_YEAR}\b", re.IGNORECASE
)
_RANGE = re.compile(rf"\b{_YEAR}\s*(?:-|–|to|and)\s*{_YEAR}\b", re.IGNORECASE)
_IN_YEAR = re.compile(rf"\b(?:in|during)\s+{_YEAR}\b", re.IGNORECASE)
_OPEN_ACCESS = re.compile(r"\bopen[- ]access\b", re.IGNORECASE)


def filters_for(subquestion: SubQuestion) -> SearchFilters | None:
    """Source filters a metadata sub-question states in words, read by code:
    a year range ("since 2020", "before 2015", "2018-2022", "in 2021") and
    open access. ``None`` for any other kind, or when nothing is stated."""
    if subquestion.kind != "metadata":
        return None
    text = subquestion.text
    year_from: int | None = None
    year_to: int | None = None
    if found := _RANGE.search(text):
        year_from, year_to = sorted((int(found.group(1)), int(found.group(2))))
    else:
        if found := _SINCE.search(text):
            year = int(found.group(2))
            year_from = year + 1 if found.group(1).lower() == "after" else year
        if found := _UNTIL.search(text):
            year = int(found.group(2))
            year_to = year - 1 if found.group(1).lower() == "before" else year
        if year_from is None and year_to is None and (found := _IN_YEAR.search(text)):
            year_from = year_to = int(found.group(1))
    open_access = bool(_OPEN_ACCESS.search(text))
    if year_from is None and year_to is None and not open_access:
        return None
    return SearchFilters(
        year_from=year_from, year_to=year_to, open_access_only=open_access
    )


# -- criteria -----------------------------------------------------------------


def fallback_criteria(subquestion: SubQuestion) -> list[Criterion]:
    """One relevance ``must``: the rubric when compiling the criteria failed."""
    return [
        Criterion(
            id=f"{subquestion.id}-C1",
            text=f"Does this result bear on the question: {subquestion.text.rstrip('?')}?",
            kind="must",
            weight=3,
            applies_to="candidate",
        )
    ]


@dataclass
class CriteriaOutcome:
    criteria: dict[str, list[Criterion]]
    degraded: list[str] = field(default_factory=list)
    spent: str = ""


async def compile_all(
    subquestions: Sequence[SubQuestion], *, screener: Screener, question: str
) -> CriteriaOutcome:
    """The compiled rubric of every sub-question; a failed compile falls back
    to :func:`fallback_criteria` and is recorded."""
    outcome = CriteriaOutcome({})
    for subquestion in subquestions:
        if outcome.spent:
            outcome.criteria[subquestion.id] = fallback_criteria(subquestion)
            continue
        try:
            outcome.criteria[subquestion.id] = await screener.compile(
                subquestion, question=question
            )
        except BudgetExhausted as spent:
            outcome.spent = spent.name
            outcome.criteria[subquestion.id] = fallback_criteria(subquestion)
        except MODEL_ERRORS as error:
            outcome.criteria[subquestion.id] = fallback_criteria(subquestion)
            outcome.degraded.append(
                f"criteria for {subquestion.id} were not compiled "
                f"({type(error).__name__}: {error}); one relevance criterion is used"
            )
    return outcome


# -- queries ------------------------------------------------------------------


class QueryDraft(BaseModel):
    subquestion: str = Field(description="The label of the sub-question, e.g. Q1.")
    query: str = Field(description="2 to 8 plain keywords, no quotes or operators.")


class QueriesDraft(BaseModel):
    queries: list[QueryDraft]


QUERY_INSTRUCTIONS = """\
You write search queries for sub-questions of a research question. For each \
sub-question listed, write up to the number of queries asked, each a few plain \
keywords (2 to 8 words, no quotes, no operators, no site: filters) that a \
scholarly database or a web search engine would match against titles and \
abstracts. Make the queries of one sub-question differ: synonyms, the \
technical term and the plain one, the population or setting, the outcome. Never \
repeat a query already run. Where a gap is named, aim a query at it."""


@dataclass
class QueriesOutcome:
    queries: dict[str, list[str]]
    degraded: list[str] = field(default_factory=list)
    spent: str = ""


def _query_key(query: str) -> str:
    return " ".join(query.casefold().split())


async def write_queries(
    question: str,
    *,
    targets: Sequence[SubQuestion],
    gaps: Mapping[str, str],
    done: Sequence[str],
    per_subquestion: int,
    asker: Asker,
) -> QueriesOutcome:
    """Up to *per_subquestion* new keyword queries for each target sub-question,
    one LLM call for all of them. *done* are the queries already run (never
    repeated); *gaps* names, per sub-question, what coverage still lacks. A
    target the model gives no new query keeps its own text as one, if unrun."""
    known = {s.id for s in targets}
    seen = {_query_key(q) for q in done}

    def known_labels(draft: QueriesDraft) -> QueriesDraft:
        unknown = sorted({q.subquestion.strip() for q in draft.queries} - known)
        if unknown:
            raise ModelRetry(
                f"unknown sub-question labels {unknown}; use {sorted(known)}"
            )
        return draft

    lines = [f"QUESTION: {question.strip()}", "", "SUB-QUESTIONS:"]
    for subquestion in targets:
        lines.append(f"{subquestion.id}: {subquestion.text}")
        if gap := gaps.get(subquestion.id):
            lines.append(f"  gap: {gap}")
    lines += ["", f"QUERIES PER SUB-QUESTION: at most {per_subquestion}"]
    if done:
        lines += ["", "ALREADY RUN:", *(f"- {q}" for q in done[-MAX_QUERIES_SHOWN:])]

    outcome = QueriesOutcome({s.id: [] for s in targets})
    try:
        draft = await asker.llm(
            instructions=QUERY_INSTRUCTIONS,
            output_type=QueriesDraft,
            user_message="\n".join(lines),
            validators=[known_labels],
        )
        written = [(q.subquestion.strip(), q.query.strip()) for q in draft.queries]
    except BudgetExhausted as spent:
        outcome.spent = spent.name
        written = []
    except MODEL_ERRORS as error:
        outcome.degraded.append(
            f"query writing failed ({type(error).__name__}: {error}); each "
            "sub-question is searched by its own text"
        )
        written = []
    for label, query in written:
        key = _query_key(query)
        own = outcome.queries.get(label)
        if own is None or not key or key in seen or len(own) >= per_subquestion:
            continue
        seen.add(key)
        own.append(" ".join(query.split()))
    if outcome.spent:
        return outcome
    for subquestion in targets:
        own = outcome.queries[subquestion.id]
        if not own and _query_key(subquestion.text) not in seen:
            seen.add(_query_key(subquestion.text))
            own.append(subquestion.text)
    return outcome


# -- search -------------------------------------------------------------------


@dataclass
class SearchOutcome:
    candidates: list[Candidate]
    degraded: list[str] = field(default_factory=list)
    queries_run: list[str] = field(default_factory=list)


async def search(
    queries: Mapping[str, Sequence[str]],
    *,
    sources: Sequence[Source],
    filters: Mapping[str, SearchFilters | None],
    limit: int,
) -> SearchOutcome:
    """Every query on every source, one call after another; results tagged
    with their sub-question and merged by identity. A failing source is
    recorded and the others go on."""
    found: list[Candidate] = []
    outcome = SearchOutcome([])
    for subquestion_id, own in queries.items():
        for query in own:
            outcome.queries_run.append(query)
            for source in sources:
                try:
                    results = await source.search(
                        query, limit=limit, filters=filters.get(subquestion_id)
                    )
                except SourceError as error:
                    outcome.degraded.append(
                        f"{source.name} search failed for {query!r}: {error.reason}"
                    )
                    continue
                found += [
                    c.model_copy(update={"subquestion_ids": [subquestion_id]})
                    for c in results
                ]
    outcome.candidates = merge(found)
    return outcome


@dataclass
class Admitted:
    """Candidates by stored work id after admission, and the pairs they bring."""

    candidates: dict[str, Candidate]
    pairs: list[Pair]


def admit(
    found: Sequence[Candidate],
    *,
    store: ResearchStore,
    known: Mapping[str, Candidate],
) -> Admitted:
    """Store each candidate's work (merged with what the store holds) and
    combine it with a candidate the run already has for that work. Returns the
    run's candidates afterwards and every ``(work_id, subquestion_id)`` pair the
    found candidates name, in their order; the caller drops pairs it has seen."""
    candidates = dict(known)
    pairs: list[Pair] = []
    for candidate in found:
        stored = store.upsert_work(candidate.work)
        fresh = candidate.model_copy(update={"work": stored})
        before = candidates.get(stored.work_id)
        combined = merge_pair(before, fresh) if before is not None else fresh
        candidates[stored.work_id] = combined.model_copy(update={"work": stored})
        pairs += [(stored.work_id, s) for s in candidate.subquestion_ids]
    return Admitted(candidates, list(dict.fromkeys(pairs)))


# -- screen -------------------------------------------------------------------


def round_robin(
    pairs: Sequence[Pair], *, subquestions: Sequence[SubQuestion]
) -> list[Pair]:
    """*pairs* interleaved across sub-questions (each sub-question's in their
    order), so a cap takes a fair share of each."""
    by_subquestion: dict[str, list[Pair]] = {s.id: [] for s in subquestions}
    for pair in pairs:
        by_subquestion.setdefault(pair[1], []).append(pair)
    ordered: list[Pair] = []
    depth = 0
    longest = max((len(v) for v in by_subquestion.values()), default=0)
    while depth < longest:
        ordered += [v[depth] for v in by_subquestion.values() if depth < len(v)]
        depth += 1
    return ordered


@dataclass
class ScreenOutcome:
    scores: list[CandidateScore] = field(default_factory=list)
    criteria: dict[str, list[Criterion]] = field(default_factory=dict)
    screened: list[Pair] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    spent: str = ""


async def screen(
    pairs: Sequence[Pair],
    *,
    candidates: Mapping[str, Candidate],
    subquestions: Sequence[SubQuestion],
    criteria: Mapping[str, list[Criterion]],
    screener: Screener,
    question: str,
    limit: int,
) -> ScreenOutcome:
    """Screen at most *limit* pairs, a fair share of each sub-question, through
    ``Screener.screen`` (score, then the once-per-sub-question refinement).
    ``criteria`` in the outcome holds the rubric that stands after refinement.
    Pairs not screened (over the cap, a failed model, a spent budget) are left
    for the caller to keep pending."""
    chosen = round_robin(pairs, subquestions=subquestions)[:limit]
    outcome = ScreenOutcome()
    for subquestion in subquestions:
        own = [p for p in chosen if p[1] == subquestion.id]
        if not own:
            continue
        try:
            screening = await screener.screen(
                subquestion,
                criteria[subquestion.id],
                [candidates[work_id] for work_id, _ in own],
                question=question,
            )
        except BudgetExhausted as spent:
            outcome.spent = spent.name
            break
        except MODEL_ERRORS as error:
            outcome.degraded.append(
                f"screening for {subquestion.id} failed "
                f"({type(error).__name__}: {error}); {len(own)} candidates wait "
                "for the next round"
            )
            continue
        outcome.scores += screening.scores
        outcome.criteria[subquestion.id] = screening.criteria
        outcome.screened += own
    return outcome


# -- expand -------------------------------------------------------------------


def is_scholarly(work: Work) -> bool:
    """A work a citation graph can place: it holds a scholarly identifier."""
    return any(
        (work.doi, work.arxiv_id, work.pmid, work.pmcid, work.openalex_id, work.s2_id)
    )


def expansion_seeds(
    kept: Mapping[str, tuple[float, list[str]]],
    *,
    candidates: Mapping[str, Candidate],
    expanded: set[str],
) -> list[tuple[Work, list[str]]]:
    """Kept scholarly works not yet expanded, best score first, at most
    ``MAX_EXPAND_SEEDS``. *kept* maps a work id to its best score and the
    sub-questions it was kept for."""
    ordered = sorted(kept.items(), key=lambda item: (-item[1][0], item[0]))
    seeds: list[tuple[Work, list[str]]] = []
    for work_id, (_, subquestion_ids) in ordered:
        candidate = candidates.get(work_id)
        if candidate is None or work_id in expanded or not is_scholarly(candidate.work):
            continue
        seeds.append((candidate.work, subquestion_ids))
        if len(seeds) >= MAX_EXPAND_SEEDS:
            break
    return seeds


@dataclass
class ExpandOutcome:
    candidates: list[Candidate] = field(default_factory=list)
    expanded: list[str] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


async def expand(
    seeds: Sequence[tuple[Work, list[str]]],
    *,
    sources: Sequence[ExpandingSource],
    limit: int,
) -> ExpandOutcome:
    """One hop back (references) and forward (citations) from each seed, until
    *limit* candidates are collected; each new candidate inherits its seed's
    sub-questions. A source that cannot place a work is a note; any other
    failure is degraded."""
    outcome = ExpandOutcome()
    for work, subquestion_ids in seeds:
        for source in sources:
            for direction in ("references", "citations"):
                room = limit - len(outcome.candidates)
                if room <= 0:
                    return outcome
                follow = (
                    source.references if direction == "references" else source.citations
                )
                try:
                    found = await follow(work, limit=min(room, EXPAND_PER_CALL))
                except SourceError as error:
                    message = (
                        f"{source.name} {direction} of "
                        f"{work.title or work.work_id}: {error.reason}"
                    )
                    if error.kind == "config":
                        outcome.notes.append(message)
                    else:
                        outcome.degraded.append(message)
                    continue
                outcome.candidates += [
                    c.model_copy(
                        update={
                            "subquestion_ids": list(subquestion_ids),
                            "sources": [f"{source.name}:{direction}"],
                        }
                    )
                    for c in found[:room]
                ]
        outcome.expanded.append(work.work_id)
    return outcome


# -- read ---------------------------------------------------------------------


def read_targets(
    kept: Mapping[str, tuple[float, list[str]]],
    *,
    done: set[Pair],
    excluded: Mapping[str, str],
    covered: set[str],
    limit: int,
) -> list[tuple[str, list[str]]]:
    """The works to read this round, with the sub-questions to read each for:
    kept pairs not read yet, works for uncovered sub-questions first, then by
    score; at most *limit* works."""
    wanted: list[tuple[bool, float, str, list[str]]] = []
    for work_id, (score, subquestion_ids) in kept.items():
        if work_id in excluded:
            continue
        open_ids = [s for s in subquestion_ids if (work_id, s) not in done]
        if not open_ids:
            continue
        only_covered = all(s in covered for s in open_ids)
        wanted.append((only_covered, -score, work_id, open_ids))
    wanted.sort()
    return [(work_id, ids) for _, _, work_id, ids in wanted[:limit]]


# -- coverage -----------------------------------------------------------------


@dataclass
class ConflictOutcome:
    conflicts: list[Conflict]
    degraded: list[str] = field(default_factory=list)


async def conflicts(
    findings: Sequence[Finding],
    *,
    works: Mapping[str, Work],
    asker: Asker,
    checked: dict[PairKey, Conflict | None],
) -> ConflictOutcome:
    """``coverage.find_conflicts``, with a failing model recorded: the pairs
    judged before the failure stand, the rest wait for the next round."""
    try:
        found = await find_conflicts(
            findings, works=works, asker=asker, checked=checked
        )
    except MODEL_ERRORS as error:
        found = [c for c in checked.values() if c is not None]
        for conflict in found:
            asker.store.add_run_item(
                asker.run_id, "conflict", "-".join(conflict.finding_ids), conflict
            )
        return ConflictOutcome(
            found,
            degraded=[
                f"conflict check failed ({type(error).__name__}: {error}); "
                "unjudged pairs were not checked"
            ],
        )
    return ConflictOutcome(found)


def coverage_gaps(coverage: Sequence[Coverage]) -> dict[str, str]:
    """What the query writer should aim at, per sub-question."""
    gaps: dict[str, str] = {}
    for entry in coverage:
        if entry.covered:
            continue
        gap = f"{entry.independent_sources} independent source(s) so far"
        if entry.note:
            gap += f"; {entry.note}"
        gaps[entry.subquestion_id] = gap
    return gaps


def write_reserve(subquestions: Sequence[SubQuestion]) -> int:
    """Language-model calls the search leaves for writing the report."""
    return WRITE_RESERVE_FIXED + WRITE_RESERVE_PER_SUBQUESTION * len(subquestions)


def unique(items: Sequence[str]) -> list[str]:
    """*items* without repeats, in order."""
    return list(dict.fromkeys(items))
