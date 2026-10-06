"""Query-dependent screening: an LLM writes the rubric, a decision model applies it.

Per sub-question an LLM compiles 3-8 atomic yes/no criteria (``must``,
``should``, ``exclude``; weight 1-3; judged on a candidate's title and abstract
or on one passage of its text). Every candidate is then put to the scorer in
ONE request: one ``noul`` question per criterion plus one ``choice`` for the
source type. The scorer is the decision model when one is given and the LLM
otherwise; both answer the same :class:`~andamentum.core.decisions.Question`
objects and return the same :class:`~andamentum.core.decisions.Answer`
records, so everything after the scorer is one code path.

:func:`combine` turns the answers into a decision in code, by rules that are
written down and logged per candidate (recall first: only a confident ``must``
miss or an ``exclude`` hit drops). Refinement runs at most once per
sub-question per run: an LLM reviews the candidates nearest the keep/drop
boundary, and when it disagrees with two or more of them it revises the
criteria once and the round is re-scored. Both criteria versions are logged.

Model calls go through the run's :class:`~.ask.Asker`, so they are counted
in ``RunStats`` and checked against ``Budget`` (an exhausted bound raises
``BudgetExhausted``). Every answer the scorer gives (decision model or LLM)
is written to the store's ``decisions`` table; every criteria version and
every candidate decision to ``run_items``. Calls are strictly sequential: one
candidate, one request, awaited before the next.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, JsonValue
from pydantic_ai import ModelRetry

from andamentum.core.decisions import Answer, DecisionModel, Question, question_hash

from .ask import UNTRUSTED, Asker, AskError, fence
from .store import ResearchStore
from .types import (
    SOURCE_TYPES,
    Budget,
    Candidate,
    CandidateScore,
    Criterion,
    CriterionKind,
    CriterionScope,
    Passage,
    RunStats,
    SourceType,
    SubQuestion,
)

# -- the rules, as named constants ----------------------------------------------

# A must criterion below this probability drops the candidate (a confident miss).
MUST_DROP_BELOW = 0.2
# An exclude criterion at or above this probability drops the candidate.
EXCLUDE_DROP_AT = 0.5

MIN_CRITERIA = 3
MAX_CRITERIA = 8
MAX_CRITERION_CHARS = 200
# compile_criteria asks once and, when the answer breaks a rule, once more.
COMPILE_ATTEMPTS = 2

# Refinement: at most this many candidates nearest the boundary are reviewed,
# and the criteria are revised when the reviewer disagrees on at least this many.
REFINE_SAMPLE = 6
REFINE_MIN_DISAGREEMENTS = 2

# Text put to a scorer: the abstract (or snippet, or passage) up to this many
# characters; local decision models have small context windows.
MAX_STATE_CHARS = 4000

# The LLM scorer answers yes or no. A language model gives no probability and
# none is invented (as in ``ask.Asker``): yes is 1.0, no is 0.0. The rules still
# hold, because each criterion says what yes means: a candidate ``must`` is
# "yes" unless the title and abstract rule it out, so only a confident miss
# drops.
LLM_ANSWER_P: dict[str, float] = {"yes": 1.0, "no": 0.0}

# The weight of the implicit passage question "does this passage bear on the
# sub-question?", asked of every passage as a must criterion.
PASSAGE_RELEVANCE_WEIGHT = 3

SOURCE_TYPE_KEY = "source_type"

# What a "yes" and a "no" mean, by scope and kind. A candidate must is phrased
# so that only a title and abstract that rule it out give a low probability:
# missing information keeps the candidate (recall first).
_OUTCOMES: dict[tuple[CriterionScope, CriterionKind], dict[str, str]] = {
    ("candidate", "must"): {
        "true": "The title and abstract show this holds, or do not rule it out.",
        "false": "The title and abstract show this does not hold.",
    },
    ("candidate", "should"): {
        "true": "The title and abstract show this holds.",
        "false": "The title and abstract do not show this.",
    },
    ("candidate", "exclude"): {
        "true": "The title and abstract clearly show this holds.",
        "false": "The title and abstract do not clearly show this.",
    },
    ("passage", "must"): {
        "true": "The passage shows this.",
        "false": "The passage does not show this.",
    },
    ("passage", "should"): {
        "true": "The passage shows this.",
        "false": "The passage does not show this.",
    },
    ("passage", "exclude"): {
        "true": "The passage clearly shows this.",
        "false": "The passage does not clearly show this.",
    },
}

SOURCE_TYPE_QUESTION = Question(
    type="choice",
    instructions="What kind of source is this?",
    options={
        "primary_study": "Original research with its own results: an experiment, "
        "trial, cohort, survey, measurement, or a new method evaluated.",
        "review": "A summary of other studies: a systematic review, "
        "meta-analysis, scoping or narrative review.",
        "guideline": "A guideline, standard, specification or official recommendation.",
        "news": "News or magazine reporting.",
        "vendor": "Product, company or marketing material.",
        "forum": "A forum post, Q&A answer, blog comment or social media post.",
        "dataset": "A dataset, database record or data descriptor.",
        "other": "Anything else: an editorial, letter, book chapter, thesis.",
    },
)

# Two properties joined into one question: "and/or", "as well as", or a
# conjunction followed by a new clause ("... and does it ...", "... or is ...").
_AUXILIARIES = (
    "is|are|was|were|does|do|did|has|have|had|can|could|will|would|should|"
    "may|might|must|it|they|this|these"
)
_COMPOUND = re.compile(
    rf"\band/or\b|\bas well as\b|\b(?:and|or),?\s+(?:also\s+)?(?:{_AUXILIARIES})\b",
    re.IGNORECASE,
)
_CRITERION_NUMBER = re.compile(r"-C(\d+)$")


class CriteriaError(AskError, ValueError):
    """The LLM's criteria broke a rule, also after the one re-ask."""

    def __init__(self, message: str, problems: list[str]) -> None:
        super().__init__(f"{message}: {'; '.join(problems)}")
        self.problems = problems


class ScreeningError(AskError, ValueError):
    """A scorer's answer does not fit the questions it was asked."""


# -- what the LLM fills (flat, for small models) ---------------------------------


class CriterionDraft(BaseModel):
    text: str = Field(description="One yes/no question about ONE property.")
    kind: CriterionKind = Field(
        description="must: a result failing it is useless; should: makes a "
        "result more useful (ranking); exclude: yes means drop the result."
    )
    weight: int = Field(ge=1, le=3, description="Importance, 1 (low) to 3 (high).")
    applies_to: CriterionScope = Field(
        description="candidate: judged from title and abstract; passage: judged "
        "from one paragraph of the full text."
    )


class CriteriaDraft(BaseModel):
    criteria: list[CriterionDraft]


class LabelAnswer(BaseModel):
    label: str = Field(description="The criterion label, e.g. C1.")
    answer: Literal["yes", "no"]


class CandidateJudgement(BaseModel):
    answers: list[LabelAnswer]
    source_type: SourceType


class PassageJudgement(BaseModel):
    answers: list[LabelAnswer]


class ReviewDraft(BaseModel):
    keep: bool = Field(description="Should this result be kept for the sub-question?")
    wrong_criteria: list[str] = Field(
        default_factory=list,
        description="Labels (e.g. C2) of criteria that misjudge this result.",
    )
    missing_criterion: str = Field(
        default="",
        description="A yes/no question the rubric lacks, or empty.",
    )


# -- records ----------------------------------------------------------------------


class PassageScore(BaseModel):
    """How one passage fared against a sub-question's passage criteria.

    ``criterion_p`` includes the implicit relevance question
    (:func:`relevance_criterion`), which every passage is asked.
    """

    passage_id: str
    work_id: str
    subquestion_id: str
    criterion_p: dict[str, float] = Field(default_factory=dict)
    score: float = Field(ge=0.0, le=1.0)
    kept: bool
    reason: str = ""


class Review(BaseModel):
    """The LLM's second opinion on one boundary candidate."""

    work_id: str
    kept: bool
    margin: float
    reviewer_keep: bool
    wrong_criterion_ids: list[str] = Field(default_factory=list)
    missing_criterion: str = ""

    @property
    def disagrees(self) -> bool:
        return self.kept != self.reviewer_keep


RefinementStatus = Literal[
    "already_refined",  # the once-per-sub-question refinement already ran
    "too_few_candidates",  # fewer than REFINE_MIN_DISAGREEMENTS; not used up
    "agreed",  # the reviewer disagreed on fewer than REFINE_MIN_DISAGREEMENTS
    "revised",  # criteria revised and the round re-scored
    "revision_unchanged",  # the revision returned the same criteria
    "revision_rejected",  # the revision broke a rule twice; the old criteria stand
]


class Refinement(BaseModel):
    subquestion_id: str
    status: RefinementStatus
    reviews: list[Review] = Field(default_factory=list)
    disagreements: int = 0
    criteria_before: list[Criterion] = Field(default_factory=list)
    criteria: list[Criterion] = Field(default_factory=list)
    scores: list[CandidateScore] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)


class Screening(BaseModel):
    """One round of screening for one sub-question: the criteria that stand
    after it, every candidate's decision (kept first, by score), and what
    refinement did."""

    subquestion_id: str
    criteria: list[Criterion]
    scores: list[CandidateScore]
    refinement: Refinement


# -- validation -------------------------------------------------------------------


def _normalised(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def criterion_problems(text: str) -> list[str]:
    """Why one criterion text is not an atomic yes/no question (empty if it is).

    A code check: it catches the common compound forms ("and/or", "as well
    as", a conjunction opening a second clause) but cannot judge meaning.
    """
    stripped = text.strip()
    problems: list[str] = []
    if not stripped:
        return ["a criterion is empty"]
    if stripped.count("?") != 1 or not stripped.endswith("?"):
        problems.append(f"{stripped!r} is not one question ending in '?'")
    if _COMPOUND.search(stripped):
        problems.append(f"{stripped!r} joins two properties; split it")
    if len(stripped) > MAX_CRITERION_CHARS:
        problems.append(
            f"{stripped[:60]!r}... is longer than {MAX_CRITERION_CHARS} characters"
        )
    return problems


def validate_criteria(drafts: list[CriterionDraft]) -> list[str]:
    """Every rule a set of criteria breaks (empty when it is usable)."""
    problems: list[str] = []
    if not MIN_CRITERIA <= len(drafts) <= MAX_CRITERIA:
        problems.append(
            f"{len(drafts)} criteria; write {MIN_CRITERIA} to {MAX_CRITERIA}"
        )
    seen: dict[frozenset[str], str] = {}
    for draft in drafts:
        problems.extend(criterion_problems(draft.text))
        words = frozenset(_normalised(draft.text).split())
        if words in seen:
            problems.append(f"{draft.text!r} duplicates {seen[words]!r}")
        else:
            seen[words] = draft.text
    if not any(d.kind == "must" and d.applies_to == "candidate" for d in drafts):
        problems.append(
            "no 'must' criterion judged on the candidate (title and abstract); "
            "screening needs at least one"
        )
    return problems


def _number(criterion_id: str) -> int:
    found = _CRITERION_NUMBER.search(criterion_id)
    return int(found.group(1)) if found else 0


def _assign_ids(
    subquestion: SubQuestion,
    drafts: list[CriterionDraft],
    previous: list[Criterion] | None = None,
) -> list[Criterion]:
    """Criteria with ids ``<subq>-C<n>``. A revised criterion with the same
    text as a previous one keeps its id; a new one gets the next free number."""
    by_text = {_normalised(c.text): c.id for c in previous or []}
    used = {_number(c.id) for c in previous or []}
    next_number = max(used, default=0) + 1
    criteria: list[Criterion] = []
    for draft in drafts:
        criterion_id = by_text.pop(_normalised(draft.text), None)
        if criterion_id is None:
            criterion_id = f"{subquestion.id}-C{next_number}"
            next_number += 1
        criteria.append(
            Criterion(
                id=criterion_id,
                text=draft.text.strip(),
                kind=draft.kind,
                weight=draft.weight,
                applies_to=draft.applies_to,
            )
        )
    return criteria


def relevance_criterion(subquestion: SubQuestion) -> Criterion:
    """The question every passage is asked: does it bear on the sub-question?"""
    return Criterion(
        id=f"{subquestion.id}-R",
        text=f"Does this passage bear on the question: {subquestion.text.rstrip(' ?')}?",
        kind="must",
        weight=PASSAGE_RELEVANCE_WEIGHT,
        applies_to="passage",
    )


# -- the questions both scorers answer --------------------------------------------


def criterion_question(criterion: Criterion) -> Question:
    """The ``noul`` question for one criterion, with what yes and no mean."""
    return Question(
        type="noul",
        instructions=criterion.text,
        criteria=_OUTCOMES[(criterion.applies_to, criterion.kind)],
    )


def _digest(value: object) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clip(text: str) -> str:
    return text if len(text) <= MAX_STATE_CHARS else text[:MAX_STATE_CHARS] + " ..."


def candidate_state(
    subquestion: SubQuestion, candidate: Candidate
) -> dict[str, object]:
    """What a scorer sees of a candidate: the sub-question and the work's
    title, abstract (or the search snippet), year, venue and where it was found."""
    work = candidate.work
    return {
        "question": subquestion.text,
        "title": work.title,
        "abstract": _clip(work.abstract or candidate.snippet),
        "year": work.year,
        "venue": work.venue,
        "found_by": sorted(candidate.sources),
    }


def passage_state(
    subquestion: SubQuestion, passage: Passage, *, work_title: str
) -> dict[str, object]:
    return {
        "question": subquestion.text,
        "work_title": work_title,
        "passage": _clip(passage.text),
    }


# -- combine: the decision, in code ------------------------------------------------


def _judge(
    criteria: list[Criterion], criterion_p: dict[str, float]
) -> tuple[float, bool, str]:
    """Score, keep and reason for one item from its answers to ``criteria``.

    1. any ``exclude`` with p >= EXCLUDE_DROP_AT drops;
    2. any ``must`` with p < MUST_DROP_BELOW drops;
    3. score = sum(weight * p) / sum(weight) over ``must`` and ``should``.
    """
    missing = [c.id for c in criteria if c.id not in criterion_p]
    if missing:
        raise ScreeningError(f"no answer for criteria {missing}")
    ranked = [c for c in criteria if c.kind != "exclude"]
    if not ranked:
        raise ScreeningError("no must or should criterion to score against")
    total = sum(c.weight for c in ranked)
    score = sum(c.weight * criterion_p[c.id] for c in ranked) / total
    score = min(1.0, max(0.0, score))

    excluded = [c for c in criteria if c.kind == "exclude"]
    hits = [c for c in excluded if criterion_p[c.id] >= EXCLUDE_DROP_AT]
    misses = [
        c for c in criteria if c.kind == "must" and criterion_p[c.id] < MUST_DROP_BELOW
    ]
    drops = [
        f"exclude {c.id} holds (p={criterion_p[c.id]:.2f}): {c.text}" for c in hits
    ] + [f"must {c.id} fails (p={criterion_p[c.id]:.2f}): {c.text}" for c in misses]
    if drops:
        return score, False, "dropped: " + "; ".join(drops)
    parts = [
        f"{kind} "
        + ", ".join(
            f"{c.id} {criterion_p[c.id]:.2f}" for c in criteria if c.kind == kind
        )
        for kind in ("must", "should", "exclude")
        if any(c.kind == kind for c in criteria)
    ]
    return score, True, f"kept: score {score:.2f} ({'; '.join(parts)})"


def _in_scope(criteria: list[Criterion], scope: CriterionScope) -> list[Criterion]:
    return [c for c in criteria if c.applies_to == scope]


def combine(
    *,
    work_id: str,
    subquestion_id: str,
    criteria: list[Criterion],
    criterion_p: dict[str, float],
    source_type: SourceType | None = None,
) -> CandidateScore:
    """The decision on one candidate from its answers, by the rules of :func:`_judge`.

    Only ``applies_to='candidate'`` criteria take part. Recall first: a
    candidate is dropped only by a confident ``must`` miss or an ``exclude``
    hit; everything else is kept and ranked by score. The source type is
    recorded, never used to drop.
    """
    scoped = _in_scope(criteria, "candidate")
    score, kept, reason = _judge(scoped, criterion_p)
    return CandidateScore(
        work_id=work_id,
        subquestion_id=subquestion_id,
        criterion_p={c.id: criterion_p[c.id] for c in scoped},
        source_type=source_type,
        score=score,
        kept=kept,
        reason=reason,
    )


def boundary_margin(score: CandidateScore, criteria: list[Criterion]) -> float:
    """How far, in probability, a candidate's decision is from flipping.

    Kept: the smallest distance of a ``must`` above MUST_DROP_BELOW or an
    ``exclude`` below EXCLUDE_DROP_AT. Dropped: the largest distance of a
    violated rule (every violation must flip to keep it).
    """
    distances: list[tuple[float, bool]] = []  # (distance, violated)
    for c in _in_scope(criteria, "candidate"):
        p = score.criterion_p.get(c.id)
        if p is None:
            continue
        if c.kind == "must":
            distances.append((abs(p - MUST_DROP_BELOW), p < MUST_DROP_BELOW))
        elif c.kind == "exclude":
            distances.append((abs(EXCLUDE_DROP_AT - p), p >= EXCLUDE_DROP_AT))
    if score.kept:
        return min((d for d, _ in distances), default=float("inf"))
    return max((d for d, violated in distances if violated), default=0.0)


def rank(scores: list[CandidateScore]) -> list[CandidateScore]:
    """Kept candidates first by score (highest first), then dropped ones; ties by work id."""
    return sorted(scores, key=lambda s: (not s.kept, -s.score, s.work_id))


def select_passages(scores: list[PassageScore], *, per_work: int) -> list[PassageScore]:
    """The kept passages, at most ``per_work`` of each work, best first."""
    chosen: list[PassageScore] = []
    taken: dict[str, int] = {}
    for s in sorted(scores, key=lambda s: (-s.score, s.passage_id)):
        if s.kept and taken.get(s.work_id, 0) < per_work:
            chosen.append(s)
            taken[s.work_id] = taken.get(s.work_id, 0) + 1
    return chosen


# -- prompts ------------------------------------------------------------------------

COMPILE_INSTRUCTIONS = f"""\
You write the screening rubric for one research sub-question: {MIN_CRITERIA} to \
{MAX_CRITERIA} yes/no criteria that a fast classifier will ask of every search \
result (a paper or a web page).

Each criterion:
- text: ONE question about ONE property, ending in '?'. Never join two \
properties with "and", "or", "and/or" or "as well as": write two criteria.
- kind:
  must    — a result that clearly fails it is useless for the sub-question \
(the topic, the population, the thing measured). Phrase it so YES is good.
  should  — makes a result more useful (study design, setting, recency). \
Used to rank, never to drop. YES is good.
  exclude — YES means drop the result (e.g. "Is it a retracted article?"). \
Use sparingly.
- weight: 1 to 3, how much it matters.
- applies_to:
  candidate — judged from the title and abstract alone, so ask only what a \
title and abstract can show.
  passage   — judged from one paragraph of the full text (e.g. "Does the \
passage report a numeric effect size?").

At least one must criterion applies to the candidate. Be specific to the \
sub-question, but general enough that a relevant result can satisfy it."""

REVISE_INSTRUCTIONS = (
    COMPILE_INSTRUCTIONS
    + f"""

You are REVISING a rubric. A reviewer read some results the rubric screened \
and disagreed with its decision on several. Keep the criteria that worked \
(same text), fix or drop the ones the reviews name as wrong, and add a missing \
one when the reviews show a gap. Return the full revised rubric. {UNTRUSTED}"""
)

SCORE_INSTRUCTIONS = f"""\
You screen one search result for a research sub-question. Answer every \
criterion, by its label, with yes or no, following what "yes" and "no" mean \
for that criterion. {UNTRUSTED}"""

CANDIDATE_SCORE_INSTRUCTIONS = (
    SCORE_INSTRUCTIONS + " Then choose the source type that fits best."
)

REVIEW_INSTRUCTIONS = f"""\
You check one decision of a screening rubric. Given a research sub-question, \
a search result, the rubric's criteria with the classifier's probability for \
each, and the decision (kept or dropped): say whether the result should be \
kept for the sub-question; list the labels of criteria that misjudge it; name \
a yes/no criterion the rubric lacks, if any. Keep a result that might help \
answer the sub-question (recall matters more than precision). {UNTRUSTED}"""


def _criteria_lines(criteria: list[Criterion], labels: dict[str, str]) -> str:
    return "\n".join(
        f"{labels[c.id]} [{c.kind}, weight {c.weight}, {c.applies_to}]: {c.text}"
        for c in criteria
    )


def _state_block(state: dict[str, object]) -> str:
    lines = [f"{key}: {value}" for key, value in state.items() if key != "question"]
    return fence("\n".join(lines))


def _labels(ids: list[str]) -> dict[str, str]:
    return {criterion_id: f"C{i}" for i, criterion_id in enumerate(ids, start=1)}


def _check_labels(answers: list[LabelAnswer], expected: set[str]) -> None:
    given = [a.label.strip().upper() for a in answers]
    if sorted(given) != sorted(expected):
        raise ModelRetry(
            f"answer each of {sorted(expected)} exactly once; you answered {given}"
        )


def _check_review_labels(review: ReviewDraft, expected: set[str]) -> None:
    unknown = sorted({w.strip().upper() for w in review.wrong_criteria} - expected)
    if unknown:
        raise ModelRetry(f"{unknown} are not criterion labels; use {sorted(expected)}")


# -- compiling ------------------------------------------------------------------------


@dataclass
class _Compiled:
    criteria: list[Criterion]
    rejected: list[list[str]]  # the problems of each rejected attempt


async def _draft_criteria(
    asker: Asker, *, instructions: str, message: str
) -> tuple[list[CriterionDraft], list[list[str]]]:
    """Ask for criteria; re-ask once, with the problems, when a rule is broken.

    Raises :class:`CriteriaError` when the last attempt breaks a rule too.
    """
    rejected: list[list[str]] = []
    for _attempt in range(COMPILE_ATTEMPTS):
        feedback = (
            "\n\nYour previous rubric broke these rules; fix them:\n- "
            + "\n- ".join(rejected[-1])
            if rejected
            else ""
        )
        output = await asker.llm(
            instructions=instructions,
            output_type=CriteriaDraft,
            user_message=message + feedback,
        )
        problems = validate_criteria(output.criteria)
        if not problems:
            return output.criteria, rejected
        rejected.append(problems)
    raise CriteriaError(
        f"criteria broke the rules {COMPILE_ATTEMPTS} times", rejected[-1]
    )


def _compile_message(subquestion: SubQuestion, *, question: str) -> str:
    context = f"The overall research question: {question}\n" if question else ""
    return f"{context}Sub-question ({subquestion.kind}): {subquestion.text}"


async def _compile(
    asker: Asker, subquestion: SubQuestion, *, question: str
) -> _Compiled:
    drafts, rejected = await _draft_criteria(
        asker,
        instructions=COMPILE_INSTRUCTIONS,
        message=_compile_message(subquestion, question=question),
    )
    return _Compiled(_assign_ids(subquestion, drafts), rejected)


# -- the screener ------------------------------------------------------------------------


def _model_id(model: Any) -> str:
    return model if isinstance(model, str) else type(model).__name__


def _p(answer: Answer) -> float:
    if answer.p_true is None:
        raise ScreeningError(f"a noul answer without a probability: {answer!r}")
    return answer.p_true


def _source_type(answer: Answer) -> SourceType:
    for source_type in SOURCE_TYPES:
        if answer.choice == source_type:
            return source_type
    raise ScreeningError(f"source type {answer.choice!r} is not one of {SOURCE_TYPES}")


def _finite(value: float) -> float:
    """JSON has no infinity: a margin with no rule to flip is recorded as 1.0."""
    return 1.0 if value == float("inf") else value


def _boundary_key(
    score: CandidateScore, criteria: list[Criterion]
) -> tuple[float, float, str]:
    """Nearest the boundary first; ties (common with the LLM's 0/1 answers) go
    to the lowest-scored kept and the highest-scored dropped candidates."""
    closeness = score.score if score.kept else 1.0 - score.score
    return (boundary_margin(score, criteria), closeness, score.work_id)


@dataclass(kw_only=True)
class Screener:
    """Screening for one run: compiles, scores, refines and logs.

    Holds the run's answer cache (an answer is reused when the same question
    is asked about the same state, so a re-score after a revision asks only
    the new criteria, and a candidate seen again costs nothing), the
    refinement bound (once per sub-question), and the criteria versions it
    has logged. The run's models, budget and counts live in ``asker``; its
    LLM writes the criteria and the reviews, and scores when there is no
    decision model.
    """

    asker: Asker
    cached_answers: int = 0
    _answers: dict[tuple[str, str], Answer] = field(default_factory=dict)
    _refined: set[str] = field(default_factory=set)
    _versions: dict[str, list[list[Criterion]]] = field(default_factory=dict)

    @property
    def store(self) -> ResearchStore:
        return self.asker.store

    @property
    def run_id(self) -> str:
        return self.asker.run_id

    # -- criteria versions ------------------------------------------------------

    def _version(
        self,
        subquestion: SubQuestion,
        criteria: list[Criterion],
        *,
        origin: str,
        detail: dict[str, JsonValue] | None = None,
    ) -> int:
        """The version number of this criteria set, recorded when first seen."""
        versions = self._versions.setdefault(subquestion.id, [])
        if criteria in versions:
            return versions.index(criteria) + 1
        versions.append(list(criteria))
        number = len(versions)
        self.store.add_run_item(
            self.run_id,
            "criterion",
            f"{subquestion.id}/v{number}",
            {
                "subquestion_id": subquestion.id,
                "version": number,
                "origin": origin,
                "criteria": [c.model_dump(mode="json") for c in criteria],
                **(detail or {}),
            },
        )
        return number

    async def compile(
        self, subquestion: SubQuestion, *, question: str = ""
    ) -> list[Criterion]:
        """:func:`compile_criteria` for this run, logged as version 1 with
        the problems of a rejected first attempt."""
        compiled = await _compile(self.asker, subquestion, question=question)
        self._version(
            subquestion,
            compiled.criteria,
            origin="compiled",
            detail={"rejected_attempts": [list(p) for p in compiled.rejected]},
        )
        return compiled.criteria

    # -- asking ---------------------------------------------------------------------

    async def ask(
        self, state: dict[str, object], questions: dict[str, Question], *, step: str
    ) -> dict[str, Answer]:
        """Answers to *questions* about *state*, through the run's scorer (the
        decision model, else the LLM), cached and recorded like every
        screening answer. A screen strategy with its own questions uses this."""
        return await self._ask(state, questions, step=step)

    async def _ask(
        self, state: dict[str, object], questions: dict[str, Question], *, step: str
    ) -> dict[str, Answer]:
        """Answers to ``questions`` about ``state``: cached ones reused, the
        rest asked in one request (decision model, else LLM) and recorded."""
        state_hash = _digest(state)
        answers: dict[str, Answer] = {}
        needed: dict[str, Question] = {}
        for key, q in questions.items():
            cached = self._answers.get((state_hash, question_hash(q)))
            if cached is None:
                needed[key] = q
            else:
                answers[key] = cached
                self.cached_answers += 1
        if not needed:
            return answers
        if self.asker.decision_model is not None:
            fresh = await self.asker.decide(step, state=state, questions=needed)
        else:
            started = time.monotonic()
            fresh = await self._ask_llm(state, needed)
            self.asker.record(
                step,
                model=_model_id(self.asker.model),
                digest="",
                state=state,
                questions=needed,
                answers=fresh,
                ms=(time.monotonic() - started) * 1000,
            )
        if set(fresh) != set(needed):
            raise ScreeningError(f"asked {sorted(needed)}, answered {sorted(fresh)}")
        for key, q in needed.items():
            self._answers[(state_hash, question_hash(q))] = fresh[key]
        return answers | fresh

    async def _ask_llm(
        self, state: dict[str, object], questions: dict[str, Question]
    ) -> dict[str, Answer]:
        """The LLM scorer: the same questions in one call; yes/no mapped by
        LLM_ANSWER_P, the source type as a choice with probability 1."""
        noul = [key for key, q in questions.items() if q.type == "noul"]
        labels = _labels(noul)
        expected = set(labels.values())
        lines = []
        for key in noul:
            q = questions[key]
            outcomes = q.criteria or {}
            lines.append(
                f"{labels[key]}: {q.instructions}\n"
                f"    yes = {outcomes.get('true', 'it holds')}\n"
                f"    no = {outcomes.get('false', 'it does not hold')}"
            )
        with_type = SOURCE_TYPE_KEY in questions
        # A state without the sub-question (the generic screen asks it in the
        # question itself) is shown without that line.
        context = (
            f"Sub-question: {state['question']}\n\n" if "question" in state else ""
        )
        message = f"{context}Result:\n{_state_block(state)}\n\nCriteria:\n" + "\n".join(
            lines
        )
        output_type: type[CandidateJudgement] | type[PassageJudgement] = (
            PassageJudgement
        )
        instructions = SCORE_INSTRUCTIONS
        if with_type:
            message += "\n\nSource types:\n" + "\n".join(
                f"{name}: {text}"
                for name, text in questions[SOURCE_TYPE_KEY].option_descriptions.items()
            )
            output_type = CandidateJudgement
            instructions = CANDIDATE_SCORE_INSTRUCTIONS

        def complete(
            output: CandidateJudgement | PassageJudgement,
        ) -> CandidateJudgement | PassageJudgement:
            _check_labels(output.answers, expected)
            return output

        output = await self.asker.llm(
            instructions=instructions,
            output_type=output_type,
            user_message=message,
            validators=[complete],
        )
        try:
            complete(output)
        except ModelRetry as error:
            raise ScreeningError(str(error)) from error
        by_label = {a.label.strip().upper(): a.answer for a in output.answers}
        answers: dict[str, Answer] = {}
        for key in noul:
            p = LLM_ANSWER_P[by_label[labels[key]]]
            answers[key] = Answer(
                choice="yes" if p >= 0.5 else "no",
                probabilities={"yes": p, "no": 1.0 - p},
                confidence=max(p, 1.0 - p),
                p_true=p,
            )
        if isinstance(output, CandidateJudgement):
            answers[SOURCE_TYPE_KEY] = Answer(
                choice=output.source_type,
                probabilities={
                    t: (1.0 if t == output.source_type else 0.0) for t in SOURCE_TYPES
                },
                confidence=1.0,
            )
        return answers

    # -- scoring ------------------------------------------------------------------

    async def score_candidates(
        self,
        subquestion: SubQuestion,
        criteria: list[Criterion],
        candidates: list[Candidate],
    ) -> list[CandidateScore]:
        """One request per candidate (its criteria as ``noul`` questions plus
        the source type as a ``choice``), one after the other; each decision by
        :func:`combine`, logged with its margin. Candidates must carry the
        ``work_id`` the store gave them."""
        scoped = _in_scope(criteria, "candidate")
        version = self._version(subquestion, criteria, origin="given")
        questions = {c.id: criterion_question(c) for c in scoped}
        questions[SOURCE_TYPE_KEY] = SOURCE_TYPE_QUESTION
        scores: list[CandidateScore] = []
        for candidate in candidates:
            work_id = candidate.work.work_id
            if not work_id:
                raise ValueError(
                    f"candidate {candidate.work.title!r} has no work_id; "
                    "upsert it into the store before screening"
                )
            answers = await self._ask(
                candidate_state(subquestion, candidate), questions, step="screen"
            )
            source_type = _source_type(answers[SOURCE_TYPE_KEY])
            score = combine(
                work_id=work_id,
                subquestion_id=subquestion.id,
                criteria=scoped,
                criterion_p={c.id: _p(answers[c.id]) for c in scoped},
                source_type=source_type,
            )
            self.store.add_run_item(
                self.run_id,
                "candidate",
                f"{subquestion.id}/v{version}/{work_id}",
                {
                    **score.model_dump(mode="json"),
                    "criteria_version": version,
                    "margin": _finite(boundary_margin(score, scoped)),
                },
            )
            scores.append(score)
        return scores

    async def score_passages(
        self,
        subquestion: SubQuestion,
        criteria: list[Criterion],
        passages: list[Passage],
        *,
        work_titles: dict[str, str],
    ) -> list[PassageScore]:
        """Passage selection: each passage is asked the ``applies_to='passage'``
        criteria plus :func:`relevance_criterion`, one request per passage, and
        judged by the same rules as a candidate. ``work_titles`` maps a work id
        to its title (context for the scorer). Pick with :func:`select_passages`."""
        scoped = [relevance_criterion(subquestion), *_in_scope(criteria, "passage")]
        questions = {c.id: criterion_question(c) for c in scoped}
        scores: list[PassageScore] = []
        for passage in passages:
            state = passage_state(
                subquestion, passage, work_title=work_titles.get(passage.work_id, "")
            )
            answers = await self._ask(state, questions, step="passage")
            criterion_p = {c.id: _p(answers[c.id]) for c in scoped}
            score, kept, reason = _judge(scoped, criterion_p)
            scores.append(
                PassageScore(
                    passage_id=passage.passage_id,
                    work_id=passage.work_id,
                    subquestion_id=subquestion.id,
                    criterion_p=criterion_p,
                    score=score,
                    kept=kept,
                    reason=reason,
                )
            )
        return scores

    # -- refinement ---------------------------------------------------------------

    async def refine(
        self,
        subquestion: SubQuestion,
        criteria: list[Criterion],
        candidates: list[Candidate],
        scores: list[CandidateScore],
        *,
        question: str = "",
    ) -> Refinement:
        """Review the REFINE_SAMPLE candidates nearest the keep/drop boundary;
        when the reviewer disagrees on REFINE_MIN_DISAGREEMENTS or more, revise
        the criteria once and re-score ``candidates``. Runs at most once per
        sub-question: a later call returns ``already_refined`` and asks nothing.
        A round with fewer than REFINE_MIN_DISAGREEMENTS candidates does not
        use the refinement up."""
        base = Refinement(
            subquestion_id=subquestion.id,
            status="already_refined",
            criteria_before=criteria,
            criteria=criteria,
            scores=scores,
        )
        if subquestion.id in self._refined:
            return base
        if len(scores) < REFINE_MIN_DISAGREEMENTS:
            return base.model_copy(update={"status": "too_few_candidates"})
        self._refined.add(subquestion.id)

        scoped = _in_scope(criteria, "candidate")
        by_work = {c.work.work_id: c for c in candidates}
        boundary = sorted(scores, key=lambda s: _boundary_key(s, scoped))[
            :REFINE_SAMPLE
        ]
        labels = _labels([c.id for c in criteria])
        ids_by_label = {label: cid for cid, label in labels.items()}
        reviews: list[Review] = []
        for score in boundary:
            candidate = by_work.get(score.work_id)
            if candidate is None:
                raise ValueError(f"no candidate for scored work {score.work_id}")
            draft = await self._review(
                subquestion, criteria, candidate, score, labels=labels
            )
            reviews.append(
                Review(
                    work_id=score.work_id,
                    kept=score.kept,
                    margin=_finite(boundary_margin(score, scoped)),
                    reviewer_keep=draft.keep,
                    wrong_criterion_ids=[
                        ids_by_label[w.strip().upper()] for w in draft.wrong_criteria
                    ],
                    missing_criterion=draft.missing_criterion.strip(),
                )
            )
        disagreements = sum(r.disagrees for r in reviews)
        reviewed = base.model_copy(
            update={"reviews": reviews, "disagreements": disagreements}
        )
        if disagreements < REFINE_MIN_DISAGREEMENTS:
            return reviewed.model_copy(update={"status": "agreed"})

        try:
            revised = await self._revise(
                subquestion, criteria, reviews, by_work, question=question
            )
        except CriteriaError as error:
            return reviewed.model_copy(
                update={"status": "revision_rejected", "problems": error.problems}
            )
        if revised == criteria:
            return reviewed.model_copy(update={"status": "revision_unchanged"})
        before = self._version(subquestion, criteria, origin="given")
        self._version(
            subquestion,
            revised,
            origin="revised",
            detail={
                "revises": before,
                "reviews": [r.model_dump(mode="json") for r in reviews],
            },
        )
        rescored = await self.score_candidates(subquestion, revised, candidates)
        return reviewed.model_copy(
            update={"status": "revised", "criteria": revised, "scores": rescored}
        )

    async def _review(
        self,
        subquestion: SubQuestion,
        criteria: list[Criterion],
        candidate: Candidate,
        score: CandidateScore,
        *,
        labels: dict[str, str],
    ) -> ReviewDraft:
        """One LLM review of one decision, recorded as a ``refine_review`` decision."""
        expected = set(labels.values())
        state = candidate_state(subquestion, candidate)
        answered = "\n".join(
            f"{labels[c.id]} [{c.kind}]: p={score.criterion_p[c.id]:.2f}"
            for c in criteria
            if c.id in score.criterion_p
        )
        message = (
            f"Sub-question: {subquestion.text}\n\nResult:\n{_state_block(state)}\n\n"
            f"Criteria:\n{_criteria_lines(criteria, labels)}\n\n"
            f"The classifier's probability that each holds:\n{answered}\n\n"
            f"Decision: {'kept' if score.kept else 'dropped'}; {score.reason}"
        )

        def known_labels(output: ReviewDraft) -> ReviewDraft:
            _check_review_labels(output, expected)
            return output

        started = time.monotonic()
        output = await self.asker.llm(
            instructions=REVIEW_INSTRUCTIONS,
            output_type=ReviewDraft,
            user_message=message,
            validators=[known_labels],
        )
        try:
            known_labels(output)
        except ModelRetry as error:
            raise ScreeningError(str(error)) from error
        self.store.log_decision(
            self.run_id,
            step="refine_review",
            model=_model_id(self.asker.model),
            digest="",
            question_hash=_digest([REVIEW_INSTRUCTIONS, [c.text for c in criteria]]),
            state_hash=_digest(state),
            answer=output.model_dump(mode="json"),
            ms=(time.monotonic() - started) * 1000,
        )
        return output

    async def _revise(
        self,
        subquestion: SubQuestion,
        criteria: list[Criterion],
        reviews: list[Review],
        by_work: dict[str, Candidate],
        *,
        question: str,
    ) -> list[Criterion]:
        labels = _labels([c.id for c in criteria])
        notes = []
        for r in reviews:
            if not r.disagrees:
                continue
            wrong = ", ".join(labels[w] for w in r.wrong_criterion_ids) or "none named"
            notes.append(
                f"- {by_work[r.work_id].work.title!r}: the rubric "
                f"{'kept' if r.kept else 'dropped'} it, the reviewer would "
                f"{'keep' if r.reviewer_keep else 'drop'} it; wrong criteria: {wrong}"
                + (f"; missing: {r.missing_criterion}" if r.missing_criterion else "")
            )
        message = (
            _compile_message(subquestion, question=question)
            + f"\n\nCurrent rubric:\n{_criteria_lines(criteria, labels)}"
            + "\n\nReviews that disagreed:\n"
            + fence("\n".join(notes))
        )
        drafts, _rejected = await _draft_criteria(
            self.asker, instructions=REVISE_INSTRUCTIONS, message=message
        )
        return _assign_ids(subquestion, drafts, previous=criteria)

    # -- one round --------------------------------------------------------------------

    async def screen(
        self,
        subquestion: SubQuestion,
        criteria: list[Criterion],
        candidates: list[Candidate],
        *,
        question: str = "",
    ) -> Screening:
        """Score the round's candidates, refine once per sub-question, and
        return the criteria that stand with the ranked decisions. *question* is
        the overall research question, context for a revision of the criteria."""
        scores = await self.score_candidates(subquestion, criteria, candidates)
        refinement = await self.refine(
            subquestion, criteria, candidates, scores, question=question
        )
        return Screening(
            subquestion_id=subquestion.id,
            criteria=refinement.criteria,
            scores=rank(refinement.scores),
            refinement=refinement,
        )


# -- one-off entry points (outside a run) ---------------------------------------------


def _one_off_asker(*, model: Any, decision_model: DecisionModel | None) -> Asker:
    """An Asker over a throwaway in-memory store, for a call outside a run."""
    store = ResearchStore.in_memory()
    run_id = store.start_run(
        question="", mode="screen", config={}, models={"model": _model_id(model)}
    )
    return Asker(
        model=model,
        store=store,
        run_id=run_id,
        budget=Budget(),
        stats=RunStats(),
        decision_model=decision_model,
    )


async def compile_criteria(
    sub_question: SubQuestion, *, model: Any, question: str = ""
) -> list[Criterion]:
    """The screening rubric for one sub-question: one LLM call (two when the
    first answer breaks a rule). Raises :class:`CriteriaError` when the second
    answer breaks a rule too. ``question`` is the overall research question,
    given as context. Inside a run use :meth:`Screener.compile`, which logs."""
    asker = _one_off_asker(model=model, decision_model=None)
    return (await _compile(asker, sub_question, question=question)).criteria


async def score_candidates(
    candidates: list[Candidate],
    criteria: list[Criterion],
    *,
    subquestion: SubQuestion,
    decision_model: DecisionModel | None = None,
    model: Any = None,
) -> list[CandidateScore]:
    """Score candidates against criteria outside a run (no refinement):
    the decision model when given, else the LLM ``model``. Inside a run use
    :meth:`Screener.score_candidates` or :meth:`Screener.screen`."""
    if decision_model is None and model is None:
        raise ValueError("score_candidates needs a decision_model or a model")
    if decision_model is not None and not isinstance(decision_model, DecisionModel):
        raise TypeError(
            "decision_model must be an opened core.decisions.DecisionModel "
            f"(await DecisionModel.open(name)), not {type(decision_model).__name__}"
        )
    screener = Screener(
        asker=_one_off_asker(model=model, decision_model=decision_model)
    )
    return await screener.score_candidates(subquestion, criteria, candidates)
