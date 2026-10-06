"""READ: from a kept work to findings whose quotes are verified in code.

For each work kept for a sub-question:

1. **Retraction.** A work with a DOI is looked up in Crossref (the
   ``CrossrefSource`` adapter over ``references``, through the store's cache);
   a retracted one is excluded and listed, and never read.
2. **Text.** Each ``TextSource`` is asked in turn (the caller orders them:
   Europe PMC, arXiv, OpenAlex's open-access link, the web) and the first that
   holds text wins; a source caches what it fetches in the store, so a work is
   fetched and extracted once. What we end up holding is recorded on the work as
   ``access``: ``full_text``, ``abstract`` (a source's or the record's own),
   ``metadata`` (nothing to read) or ``unreadable`` (a source failed and nothing
   else was left). A work whose licence forbids redistribution (Semantic
   Scholar) is never read from an abstract. A source's failure is noted on the
   reading; one other than a routine refusal also marks the run degraded.
3. **Passages.** The text is cut into passages of about 1,200 characters on
   paragraph boundaries. A passage is an exact slice of the text, so a
   quote can be located in it and the offsets mean something.
4. **Selection.** The passages most likely to answer the sub-question are
   chosen: a decision model is asked each one the sub-question's ``applies_to=
   'passage'`` criteria (``criteria.Screener.score_passages``); without a
   decision model the language model picks, in one call. A cheap word-overlap
   pre-filter bounds the number of passages either looks at.
5. **Extraction.** The language model reads the selected passages and returns
   ``{claim, quote, criterion_ids}``; the schema is flat.
6. **Quote check, in code.** A finding is kept only if its quote occurs in the
   passage it names: exactly, then ignoring case, whitespace, quotation-mark
   and dash style. The stored quote is the passage's own text at that span, so a
   kept quote is always verbatim. Any other finding is dropped and recorded as
   a ``quote_missing`` flag.

Every step runs one after another. A spent budget stops the loop and is
reported on the outcome (``stopped``), keeping what was read so far.
"""

from __future__ import annotations

import re
import time
import unicodedata
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence, Set
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry

from andamentum.references import (
    Work,
    WorkAccess,
    WorkRetraction,
    merge_works,
)

from .ask import MODEL_ERRORS, UNTRUSTED, Asker, BudgetExhausted, fence
from .criteria import Screener
from .criteria import select_passages as select_scored
from .sources.base import SourceError, TextSource
from .sources.crossref import CrossrefSource
from .store import ResearchStore
from .types import Criterion, Finding, Passage, SubQuestion, VerifyFlag

PASSAGE_TARGET_CHARS = 1200
PASSAGE_MAX_CHARS = 1800
PASSAGE_MIN_CHARS = 80
# How many passages the decision model or the language model looks at, at most.
DECISION_SCORE_CAP = 60
LLM_PICK_CAP = 20
# A quote shorter than this proves too little to be a finding.
QUOTE_MIN_CHARS = 20
MAX_FINDINGS_PER_SELECTION = 8

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_BLANK_LINE = re.compile(r"\n[ \t]*\n\s*")
_WORD = re.compile(r"[^\W_]{3,}")
_STOPWORDS = frozenset(
    "the and for are was were with that this from have has had not but can will "
    "its their there which what when where who how why does did into than then "
    "also been being such any all more most other some only over under about "
    "between these those they them our out one two may might could would should".split()
)


class PassageSelect(Protocol):
    """How a work's passages are chosen for one sub-question (the run's
    ``passages`` strategy). :func:`select_passages` is the criteria one."""

    def __call__(
        self,
        passages: Sequence[Passage],
        *,
        subquestion: SubQuestion,
        criteria: Sequence[Criterion],
        screener: Screener,
        limit: int,
        work_titles: Mapping[str, str],
    ) -> Awaitable[list[Passage]]: ...


@dataclass(kw_only=True)
class ReadDeps:
    """What reading needs besides the work: the run's models and the network.

    ``select`` chooses the passages (``None``: :func:`select_passages`);
    ``passages_per_work`` is how many it may choose (``None``: the budget's).
    """

    screener: Screener
    crossref: CrossrefSource
    providers: Sequence[TextSource]
    fetched_bytes: Callable[[], int] | None = None
    select: PassageSelect | None = None
    passages_per_work: int | None = None

    @property
    def asker(self) -> Asker:
        return self.screener.asker

    @property
    def store(self) -> ResearchStore:
        return self.screener.asker.store


@dataclass(frozen=True)
class ReadTarget:
    """A kept work and the sub-questions it was kept for."""

    work: Work
    subquestion_ids: Sequence[str]


class WorkReading(BaseModel):
    """What reading one work produced."""

    work: Work
    excluded: str = ""
    passages: list[Passage] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    dropped: list[VerifyFlag] = Field(default_factory=list)
    done: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)


class ReadOutcome(BaseModel):
    """The readings of a round. ``stopped`` is the spent budget, if one stopped it."""

    readings: list[WorkReading] = Field(default_factory=list)
    stopped: str = ""

    @property
    def findings(self) -> list[Finding]:
        return [f for r in self.readings for f in r.findings]

    @property
    def passages(self) -> list[Passage]:
        return [p for r in self.readings for p in r.passages]

    @property
    def excluded(self) -> list[tuple[str, str]]:
        """``(work_id, reason)`` of every work left out, retracted ones."""
        return [(r.work.work_id, r.excluded) for r in self.readings if r.excluded]

    @property
    def dropped(self) -> list[VerifyFlag]:
        return [d for r in self.readings for d in r.dropped]

    @property
    def degraded(self) -> list[str]:
        return [d for r in self.readings for d in r.degraded]

    @property
    def done(self) -> set[tuple[str, str]]:
        """``(work_id, subquestion_id)`` pairs read, to skip on the next round."""
        return {(r.work.work_id, s) for r in self.readings for s in r.done}


# -- passages ---------------------------------------------------------------


def split_passages(
    text: str, *, work_id: str, retrieved_at: float = 0.0
) -> list[Passage]:
    """Cut *text* into passages of about ``PASSAGE_TARGET_CHARS``, exact slices.

    Paragraph boundaries are kept whenever a paragraph fits; a longer one is
    cut at sentence ends, and a sentence longer than ``PASSAGE_MAX_CHARS`` at
    whitespace. Passages hold neighbouring paragraphs up to the target, so
    no passage is a lone heading. Repeated text within a work is one passage.
    """
    units: list[tuple[int, int]] = []
    for start, end in _paragraph_spans(text):
        units.extend(_fit(text, start, end))
    spans: list[tuple[int, int]] = []
    for start, end in units:
        if spans and end - spans[-1][0] <= PASSAGE_TARGET_CHARS:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < PASSAGE_MIN_CHARS:
        tail = spans.pop()
        if tail[1] - spans[-1][0] <= PASSAGE_MAX_CHARS:
            spans[-1] = (spans[-1][0], tail[1])
    passages: dict[str, Passage] = {}
    for start, end in spans:
        piece = text[start:end]
        if piece.strip():
            passage = Passage.make(
                work_id=work_id, start=start, text=piece, retrieved_at=retrieved_at
            )
            passages.setdefault(passage.passage_id, passage)
    return list(passages.values())


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    position = 0
    for found in [*_BLANK_LINE.finditer(text), None]:
        stop = found.start() if found else len(text)
        spans.append((position, stop))
        position = found.end() if found else len(text)
    return [_trim(text, s, e) for s, e in spans if text[s:e].strip()]


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _fit(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """A paragraph as units no longer than ``PASSAGE_MAX_CHARS``."""
    if end - start <= PASSAGE_MAX_CHARS:
        return [(start, end)]
    sentences: list[tuple[int, int]] = []
    position = start
    for found in _SENTENCE_END.finditer(text, start, end):
        sentences.append((position, found.start()))
        position = found.end()
    sentences.append((position, end))
    units: list[tuple[int, int]] = []
    for s_start, s_end in sentences:
        while s_end - s_start > PASSAGE_MAX_CHARS:
            cut = text.rfind(" ", s_start, s_start + PASSAGE_MAX_CHARS)
            cut = cut if cut > s_start else s_start + PASSAGE_MAX_CHARS
            units.append(_trim(text, s_start, cut))
            s_start = cut
            while s_start < s_end and text[s_start].isspace():
                s_start += 1
        if s_end > s_start:
            units.append((s_start, s_end))
    # Pack the sentences back up to the target, so units are not one-liners.
    packed: list[tuple[int, int]] = []
    for unit in units:
        if packed and unit[1] - packed[-1][0] <= PASSAGE_TARGET_CHARS:
            packed[-1] = (packed[-1][0], unit[1])
        else:
            packed.append(unit)
    return packed


# -- quotes -----------------------------------------------------------------

_QUOTE_FOLD = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "−": "-",
        " ": " ",
        "­": None,
        "​": None,
    }
)


def _fold(text: str) -> tuple[str, list[int]]:
    """*text* case-, width-, quote- and whitespace-folded, with the original
    index of every folded character."""
    folded: list[str] = []
    origin: list[int] = []
    gap = True  # no leading space
    for index, char in enumerate(text):
        for piece in (
            unicodedata.normalize("NFKC", char).translate(_QUOTE_FOLD).casefold()
        ):
            if piece.isspace():
                if gap:
                    continue
                piece, gap = " ", True
            else:
                gap = False
            folded.append(piece)
            origin.append(index)
    if folded and folded[-1] == " ":
        folded.pop()
        origin.pop()
    return "".join(folded), origin


def locate_quote(quote: str, text: str) -> tuple[int, int] | None:
    """Where *quote* occurs in *text*: exactly, else ignoring case, whitespace,
    quotation-mark and dash style. ``(start, end)`` in *text*, or ``None``."""
    needle = quote.strip()
    if not needle:
        return None
    at = text.find(needle)
    if at >= 0:
        return at, at + len(needle)
    folded_needle, _ = _fold(needle)
    if not folded_needle:
        return None
    folded_text, origin = _fold(text)
    at = folded_text.find(folded_needle)
    if at < 0:
        return None
    return origin[at], origin[at + len(folded_needle) - 1] + 1


def quote_is_long_enough(quote: str) -> bool:
    return len(_fold(quote)[0]) >= QUOTE_MIN_CHARS


# -- selection --------------------------------------------------------------


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.casefold()) if w not in _STOPWORDS}


def prefilter(passages: Sequence[Passage], *, query: str, cap: int) -> list[Passage]:
    """The *cap* passages sharing the most terms with *query*, in document order.

    Cheap and deterministic: it only bounds how many passages a model reads.
    """
    if len(passages) <= cap:
        return list(passages)
    wanted = _terms(query)
    scored = sorted(
        range(len(passages)),
        key=lambda i: (-len(wanted & _terms(passages[i].text)), i),
    )
    return [passages[i] for i in sorted(scored[:cap])]


def _passage_criteria(criteria: Sequence[Criterion]) -> list[Criterion]:
    return [c for c in criteria if c.applies_to == "passage"]


class _PassagePick(BaseModel):
    passages: list[int]


async def select_passages(
    passages: Sequence[Passage],
    *,
    subquestion: SubQuestion,
    criteria: Sequence[Criterion],
    screener: Screener,
    limit: int,
    work_titles: Mapping[str, str],
) -> list[Passage]:
    """The up-to-*limit* passages most likely to answer *subquestion*, best first.

    With a decision model the screener asks each candidate passage the
    sub-question's ``passage`` criteria and the implicit relevance question
    (``criteria.Screener.score_passages``), and the best-scored kept ones are
    taken (``criteria.select_passages``). Without a decision model the language
    model reads the candidate passages together and picks, which costs one call
    instead of one per passage.
    """
    asker = screener.asker
    own = _passage_criteria(criteria)
    query = " ".join([subquestion.text, *(c.text for c in own)])
    if asker.decision_model is None:
        pool = prefilter(passages, query=query, cap=LLM_PICK_CAP)
        return await _pick_with_llm(pool, subquestion, own, asker, limit)
    pool = prefilter(passages, query=query, cap=DECISION_SCORE_CAP)
    scores = await screener.score_passages(
        subquestion, list(criteria), pool, work_titles=dict(work_titles)
    )
    by_id = {p.passage_id: p for p in pool}
    return [by_id[s.passage_id] for s in select_scored(scores, per_work=limit)]


async def _pick_with_llm(
    pool: Sequence[Passage],
    subquestion: SubQuestion,
    own: Sequence[Criterion],
    asker: Asker,
    limit: int,
) -> list[Passage]:
    if not pool:
        return []
    numbered = "\n\n".join(
        f"[{n}]\n{fence(p.text)}" for n, p in enumerate(pool, start=1)
    )
    facets = "\n".join(f"- {c.text}" for c in own if c.kind != "exclude")
    avoid = "\n".join(f"- {c.text}" for c in own if c.kind == "exclude")

    def in_range(output: _PassagePick) -> _PassagePick:
        bad = [n for n in output.passages if not 1 <= n <= len(pool)]
        if bad:
            raise ModelRetry(
                f"passage numbers must be between 1 and {len(pool)}: {bad}"
            )
        return output

    picked = await asker.llm(
        instructions=(
            "You choose which passages of a document to read closely. Choose the "
            f"passages most likely to contain direct evidence for the question, at most {limit}, "
            "and return their numbers, best first. Return an empty list when none "
            "is relevant. " + UNTRUSTED
        ),
        output_type=_PassagePick,
        user_message=(
            f"QUESTION: {subquestion.text}\n"
            + (f"LOOK FOR:\n{facets}\n" if facets else "")
            + (f"SKIP PASSAGES THAT ARE:\n{avoid}\n" if avoid else "")
            + f"\nPASSAGES:\n\n{numbered}"
        ),
        validators=[in_range],
    )
    chosen = list(dict.fromkeys(picked.passages))[:limit]
    return [pool[n - 1] for n in chosen]


# -- extraction -------------------------------------------------------------


class _RawFinding(BaseModel):
    passage: int
    claim: str
    quote: str
    criteria: list[str] = Field(default_factory=list)


class _Extraction(BaseModel):
    findings: list[_RawFinding]


_EXTRACT_INSTRUCTIONS = (
    "You extract findings from passages of one document, to answer a question. "
    "A finding is a claim the passage makes that bears on the question, with the "
    "exact words that make it. Rules: the claim is one self-contained statement "
    "(name what it is about; no 'the study', 'this', 'they'); the quote is copied "
    "from the passage word for word, in one piece, with no ellipsis and no "
    "changes, long enough to show the claim; 'passage' is the number of the "
    "passage the quote is in; 'criteria' lists the ids of the listed criteria "
    "the finding gives evidence for, and may be empty. Report only what the "
    "passages say. Return an empty list when no passage bears on the question. "
    + UNTRUSTED
)


async def extract_findings(
    selected: Sequence[Passage],
    *,
    subquestion: SubQuestion,
    criteria: Sequence[Criterion],
    asker: Asker,
    ids: Iterator[int],
) -> tuple[list[Finding], list[VerifyFlag]]:
    """Findings from *selected*, each quote checked in code; and the drops."""
    if not selected:
        return [], []
    listed = [c for c in criteria if c.kind != "exclude"]
    facets = "\n".join(f"- {c.id}: {c.text}" for c in listed)
    numbered = "\n\n".join(
        f"[{n}]\n{fence(p.text)}" for n, p in enumerate(selected, start=1)
    )
    out = await asker.llm(
        instructions=_EXTRACT_INSTRUCTIONS,
        output_type=_Extraction,
        user_message=(
            f"QUESTION: {subquestion.text}\n"
            + (f"CRITERIA:\n{facets}\n" if facets else "")
            + f"\nPASSAGES:\n\n{numbered}"
        ),
    )
    valid_criteria = {c.id for c in listed}
    findings: list[Finding] = []
    dropped: list[VerifyFlag] = []
    seen: set[tuple[str, str]] = set()
    for raw in out.findings[: MAX_FINDINGS_PER_SELECTION * len(selected)]:
        located = _verified(raw, selected)
        if located is None:
            detail = (
                "quote not in the passage it names"
                if quote_is_long_enough(raw.quote)
                else f"quote shorter than {QUOTE_MIN_CHARS} characters"
            )
            dropped.append(
                VerifyFlag(
                    kind="quote_missing",
                    ref=selected[0].work_id,
                    detail=f"{detail}: {raw.quote[:120]!r}",
                )
            )
            continue
        passage, quote = located
        if (passage.passage_id, quote) in seen or not raw.claim.strip():
            continue
        seen.add((passage.passage_id, quote))
        findings.append(
            Finding(
                finding_id=f"F{next(ids)}",
                subquestion_id=subquestion.id,
                claim=raw.claim.strip(),
                quote=quote,
                passage_id=passage.passage_id,
                work_id=passage.work_id,
                criterion_ids=[
                    c for c in dict.fromkeys(raw.criteria) if c in valid_criteria
                ],
            )
        )
    return findings, dropped


def _verified(
    raw: _RawFinding, selected: Sequence[Passage]
) -> tuple[Passage, str] | None:
    """The passage and verbatim quote for *raw*, or ``None`` when its quote is in
    none of the selected passages. The named passage is tried first."""
    if not quote_is_long_enough(raw.quote):
        return None
    named = selected[raw.passage - 1] if 1 <= raw.passage <= len(selected) else None
    order = ([named] if named else []) + [p for p in selected if p is not named]
    for passage in order:
        span = locate_quote(raw.quote, passage.text)
        if span is not None:
            return passage, passage.text[span[0] : span[1]]
    return None


# -- one work ---------------------------------------------------------------


async def check_retraction(
    work: Work, *, crossref: CrossrefSource
) -> tuple[WorkRetraction | None, str]:
    """The work's retraction state from Crossref, and a note on a failed check.

    ``None`` means not checked (no DOI, an unknown DOI, Crossref failed, or an
    offline run's snapshot does not hold the answer), never "fine". Only a
    failure is a note; an offline miss is counted by the source client.
    """
    if not work.doi:
        return None, ""
    try:
        found = await crossref.retraction_status(work.doi)
    except SourceError as exc:
        if exc.kind == "offline":
            return None, ""
        return None, f"retraction check failed for {work.doi}: {exc.reason}"
    return (None if found == "unknown" else found), ""


async def _load_text(
    work: Work, deps: ReadDeps
) -> tuple[str, WorkAccess, list[str], list[str]]:
    """The text to read, what it is, notes on attempts that failed, and the
    failures that mean a source misbehaved (not a routine refusal).

    Each provider is asked in turn; the first that holds text for the work wins.
    A source caches what it fetches in the store, so a work is fetched and
    extracted once however often it is read.
    """
    notes: list[str] = []
    degraded: list[str] = []
    failed = False
    for provider in deps.providers:
        _sync_fetched_bytes(deps)
        if deps.asker.stats.fetched_bytes >= deps.asker.budget.max_fetch_bytes:
            raise BudgetExhausted("max_fetch_bytes")
        try:
            fetched = await provider.fetch_text(work)
        except SourceError as exc:
            notes.append(f"{provider.name}: {exc.reason}")
            if exc.kind == "offline":
                continue  # not in the snapshot: not a failure, nothing fetched
            failed = True
            if exc.kind != "refused":
                degraded.append(
                    f"{provider.name} could not give the text of "
                    f"{work.title or work.work_id}: {exc.reason}"
                )
            continue
        finally:
            _sync_fetched_bytes(deps)
        if fetched is None:
            continue
        if fetched.access == "abstract" and not work.redistributable:
            continue  # the licence forbids using the record's abstract
        return fetched.text, fetched.access, notes, degraded
    if work.abstract.strip() and work.redistributable:
        return work.abstract, "abstract", notes, degraded
    return "", ("unreadable" if failed else "metadata"), notes, degraded


def _sync_fetched_bytes(deps: ReadDeps) -> None:
    if deps.fetched_bytes is not None:
        deps.asker.stats.fetched_bytes = deps.fetched_bytes()


async def load_work(work: Work, *, deps: ReadDeps) -> tuple[WorkReading, list[Passage]]:
    """Steps 1-3 for one work, no language model: the retraction check, the
    text, and every passage of it, stored. A retracted work has ``excluded``
    set and no passages; so does a work with nothing to read.
    """
    store = deps.store
    if not work.work_id:
        work = store.upsert_work(work)
    reading = WorkReading(work=work)

    checked, problem = await check_retraction(work, crossref=deps.crossref)
    if problem:
        reading.degraded.append(problem)
    if checked is not None:
        stricter = merge_works(work, work.model_copy(update={"retracted": checked}))
        if stricter.retracted != work.retracted:
            store.set_work_state(work.work_id, retracted=stricter.retracted)
            work = reading.work = work.model_copy(
                update={"retracted": stricter.retracted}
            )
    if work.retracted == "retracted":
        reading.excluded = f"retracted: {work.title or work.work_id}"
        return reading, []

    text, access, notes, failures = await _load_text(work, deps)
    reading.notes.extend(notes)
    reading.degraded.extend(failures)
    store.set_work_state(work.work_id, access=access)
    reading.work = work.model_copy(update={"access": access})
    if not text:
        return reading, []
    passages = split_passages(text, work_id=work.work_id, retrieved_at=time.time())
    store.add_passages(passages)
    return reading, passages


async def read_work(
    work: Work,
    *,
    subquestions: Sequence[SubQuestion],
    criteria: Mapping[str, Sequence[Criterion]],
    deps: ReadDeps,
    ids: Iterator[int],
    passages_per_work: int | None = None,
) -> WorkReading:
    """Read one work for each of *subquestions*; nothing for a retracted work.

    *criteria* maps a sub-question id to its criteria; *ids* yields the numbers
    for ``F<n>`` finding ids. Raises ``BudgetExhausted`` when a bound is spent.
    """
    asker = deps.asker
    store = deps.store
    limit = (
        passages_per_work or deps.passages_per_work or asker.budget.passages_per_work
    )
    choose = deps.select or select_passages
    reading, all_passages = await load_work(work, deps=deps)
    work = reading.work
    if not all_passages:
        return reading
    used: dict[str, Passage] = {}
    for subquestion in subquestions:
        own = criteria.get(subquestion.id, ())
        selected = await choose(
            all_passages,
            subquestion=subquestion,
            criteria=own,
            screener=deps.screener,
            limit=limit,
            work_titles={work.work_id: work.title},
        )
        findings, dropped = await extract_findings(
            selected, subquestion=subquestion, criteria=own, asker=asker, ids=ids
        )
        for finding in findings:
            store.add_run_item(asker.run_id, "finding", finding.finding_id, finding)
        for number, flag in enumerate(dropped, start=1):
            store.add_run_item(
                asker.run_id,
                "flag",
                f"{work.work_id}:{subquestion.id}:{number}",
                flag,
            )
        reading.findings.extend(findings)
        reading.dropped.extend(dropped)
        reading.done.append(subquestion.id)
        used.update({p.passage_id: p for p in selected})
    reading.passages = list(used.values())
    return reading


async def read_works(
    targets: Sequence[ReadTarget],
    *,
    subquestions: Sequence[SubQuestion],
    criteria: Mapping[str, Sequence[Criterion]],
    deps: ReadDeps,
    ids: Iterator[int],
    skip: Set[tuple[str, str]] = frozenset(),
) -> ReadOutcome:
    """Read each target, one after another.

    *skip* holds ``(work_id, subquestion_id)`` pairs already read (an earlier
    round's ``ReadOutcome.done``), so nothing is read twice. A spent budget
    ends the loop with ``stopped`` set and the readings so far kept. A model
    that fails on one work (``MODEL_ERRORS``) is recorded on that work's
    reading as degraded and the loop goes on; the work is not marked done, so
    a later round may read it again.
    """
    by_id = {s.id: s for s in subquestions}
    outcome = ReadOutcome()
    for target in targets:
        wanted = [
            by_id[s]
            for s in target.subquestion_ids
            if s in by_id and (target.work.work_id, s) not in skip
        ]
        if not wanted:
            continue
        try:
            outcome.readings.append(
                await read_work(
                    target.work,
                    subquestions=wanted,
                    criteria=criteria,
                    deps=deps,
                    ids=ids,
                )
            )
        except BudgetExhausted as spent:
            outcome.stopped = spent.name
            break
        except MODEL_ERRORS as error:
            outcome.readings.append(
                WorkReading(
                    work=target.work,
                    degraded=[
                        f"reading {target.work.title or target.work.work_id} failed: "
                        f"{type(error).__name__}: {error}"
                    ],
                )
            )
    return outcome
