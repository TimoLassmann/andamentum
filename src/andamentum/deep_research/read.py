"""READ: from a chosen search result to findings whose quotes are checked in code.

For one page (spec section 1.5):

1. **Retraction.** A work with a DOI is looked up in Crossref; a retracted one is
   excluded and never read.
2. **Text.** Each ``TextSource`` is asked in turn (the caller orders them) and the
   first that holds text wins; the web source falls back to the Chrome that
   searches when the plain fetch fails. Else the record's redistributable
   abstract, else the search snippet (when it has ``SNIPPET_MIN_CHARS``), else
   nothing: the page is unreadable and the engine picks a reserve.
3. **Window.** The most relevant ``window_chars`` of the text, assembled from the
   best BM25 passages against the sub-question and the round's queries, in
   document order, stored as passages (exact slices, so offsets mean something).
4. **Notes.** One model call returns findings: a claim and the words that state
   it. Code locates every quote in the window. A quote that cannot be found
   keeps its finding with ``quote_verified=False``: nothing is dropped silently.

A source's failure is recorded on the reading and never escapes; a model failure
on the notes call is recorded as ``degraded`` and the page has no findings;
``BudgetExhausted`` propagates to the engine.
"""

from __future__ import annotations

import re
import time
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel

from andamentum.references import Work, WorkAccess, WorkRetraction, merge_works

from . import runlog
from .ask import MODEL_ERRORS, UNTRUSTED, Asker, BudgetExhausted, fence
from .lexical import bm25_scores
from .sources.base import SourceError, TextSource
from .sources.browser_fetch import CONTENT_TYPE as CHROME_CONTENT_TYPE
from .sources.crossref import CrossrefSource
from .types import (
    DEFAULT_WINDOW_CHARS,
    Candidate,
    PageReading,
    PageVia,
    Passage,
    RawFinding,
    SubQuestion,
)

PASSAGE_TARGET_CHARS = 1200
PASSAGE_MAX_CHARS = 1800
PASSAGE_MIN_CHARS = 80
# Findings kept from one page.
MAX_FINDINGS_PER_PAGE = 8
# A search snippet shorter than this is not read as a page.
SNIPPET_MIN_CHARS = 80
# A run is degraded by unreadable pages only when more than this share of the
# pages tried could not be read.
UNREADABLE_DEGRADE_SHARE = 0.5

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_BLANK_LINE = re.compile(r"\n[ \t]*\n\s*")


@dataclass(kw_only=True)
class ReadDeps:
    """What reading needs besides the work: the run's model and the sources.

    ``fetched_bytes`` lets the ``max_fetch_bytes`` bound see what the source
    client has fetched (e.g. ``lambda: client.fetched_bytes - before``).
    """

    asker: Asker
    crossref: CrossrefSource
    providers: Sequence[TextSource]
    window_chars: int = DEFAULT_WINDOW_CHARS
    fetched_bytes: Callable[[], int] | None = None


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


def build_window(
    text: str,
    *,
    work_id: str,
    query: str,
    max_chars: int,
    retrieved_at: float = 0.0,
    skip: Sequence[Passage] = (),
) -> list[Passage]:
    """The most relevant parts of *text*, at most about *max_chars* characters, as
    passages in document order: best BM25 passages first until the limit, adjacent
    ones merged into one span, a text within the limit whole. Exact slices. Parts
    that overlap a passage in *skip* (an earlier window) are left out."""
    body = text.strip()
    if not body:
        return []
    if len(body) <= max_chars:
        start = len(text) - len(text.lstrip())
        return [
            Passage.make(
                work_id=work_id, start=start, text=body, retrieved_at=retrieved_at
            )
        ]
    passages = [
        p
        for p in split_passages(text, work_id=work_id, retrieved_at=retrieved_at)
        if not any(p.start < s.end and s.start < p.end for s in skip)
    ]
    scores = bm25_scores(query, [p.text for p in passages])
    best_first = sorted(range(len(passages)), key=lambda i: (-scores[i], i))
    chosen: list[int] = []
    used = 0
    for index in best_first:
        size = len(passages[index].text)
        if used + size <= max_chars:
            chosen.append(index)
            used += size
    spans: list[tuple[int, int]] = []
    for index in sorted(chosen):
        p = passages[index]
        if spans and not text[spans[-1][1] : p.start].strip():
            spans[-1] = (spans[-1][0], p.end)
        else:
            spans.append((p.start, p.end))
    return [
        Passage.make(
            work_id=work_id,
            start=start,
            text=text[start:end],
            retrieved_at=retrieved_at,
        )
        for start, end in spans
    ]


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
        " ": " ",
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


# Markdown a web page's text keeps and a reader does not see: a citation
# marker ``[[3]](url)`` (dropped whole), a link ``[text](url)`` or image (its
# text kept), emphasis and code marks, table pipes, and a table's separator row.
_MD_CITATION = re.compile(r"\[\[[^\]]*\]\]\([^)]*\)")
_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_MD_MARKS = frozenset("*_`")
_TABLE_PIPE = "|"
_SEPARATOR_ROW = re.compile(r"^[ \t|:\-]+$", re.MULTILINE)
_LITERAL_ESCAPES = re.compile(r"\\[ntr]")


def _visible(text: str) -> tuple[str, list[int]]:
    """*text* as a reader sees it (Markdown link targets, citation markers,
    emphasis marks, table pipes and table separator rows left out), folded as
    :func:`_fold` folds it and with no whitespace at all, with the original
    index of every character kept."""
    keep = [True] * len(text)
    for match in _MD_CITATION.finditer(text):
        keep[match.start() : match.end()] = [False] * (match.end() - match.start())
    for match in _MD_LINK.finditer(text):
        if not keep[match.start()]:
            continue
        keep[match.start() : match.end()] = [False] * (match.end() - match.start())
        keep[match.start(1) : match.end(1)] = [True] * (match.end(1) - match.start(1))
    for match in _SEPARATOR_ROW.finditer(text):
        if "-" in match.group():
            keep[match.start() : match.end()] = [False] * (match.end() - match.start())
    folded: list[str] = []
    origin: list[int] = []
    for index, char in enumerate(text):
        if not keep[index] or char in _MD_MARKS or char == _TABLE_PIPE:
            continue
        for piece in (
            unicodedata.normalize("NFKC", char).translate(_QUOTE_FOLD).casefold()
        ):
            if piece.isspace() or piece in _MD_MARKS or piece == _TABLE_PIPE:
                continue
            folded.append(piece)
            origin.append(index)
    return "".join(folded), origin


def locate_quote(quote: str, text: str) -> tuple[int, int] | None:
    """Where *quote* occurs in *text*: exactly, else ignoring case, width, quote
    marks, dashes and whitespace, else in the text a reader sees (Markdown links
    read as their text, citation markers, emphasis marks and table pipes left
    out, whitespace ignored: a page converted to Markdown often loses the space
    next to a link). A literal backslash-n or backslash-t in the quote reads as a
    space. The characters must still occur in order, and the span returned is the
    original text's. ``(start, end)`` in *text*, or ``None``. A quote with an
    ellipsis is not matched: verbatim means verbatim."""
    raw = quote.strip()
    if not raw:
        return None
    at = text.find(raw)
    if at >= 0:
        return at, at + len(raw)
    needle = _LITERAL_ESCAPES.sub(" ", raw).strip()
    if not needle:
        return None
    if needle != raw:
        at = text.find(needle)
        if at >= 0:
            return at, at + len(needle)
    folded_needle, _ = _fold(needle)
    if not folded_needle:
        return None
    folded_text, origin = _fold(text)
    at = folded_text.find(folded_needle)
    if at >= 0:
        return origin[at], origin[at + len(folded_needle) - 1] + 1
    visible_needle, _ = _visible(needle)
    if not visible_needle:
        return None
    visible_text, origin = _visible(text)
    at = visible_text.find(visible_needle)
    if at < 0:
        return None
    return origin[at], origin[at + len(visible_needle) - 1] + 1


def check_quote(
    quote: str, passages: Sequence[Passage], *, named: int | None = None
) -> tuple[Passage, str] | None:
    """The passage that holds *quote* and the passage's own words at that span, or
    ``None``. The passage numbered *named* (1-based) is tried first."""
    first = (
        passages[named - 1]
        if named is not None and 1 <= named <= len(passages)
        else None
    )
    order = ([first] if first is not None else []) + [
        p for p in passages if p is not first
    ]
    for passage in order:
        span = locate_quote(quote, passage.text)
        if span is not None:
            return passage, passage.text[span[0] : span[1]]
    return None


# -- retraction and text ------------------------------------------------------


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


def unreadable_degraded(*, attempted: int, unreadable: int) -> list[str]:
    """The degraded line for a run whose pages mostly could not be read: one
    entry when more than ``UNREADABLE_DEGRADE_SHARE`` of the *attempted* pages were
    *unreadable*, else none."""
    if attempted and unreadable / attempted > UNREADABLE_DEGRADE_SHARE:
        return [
            f"{unreadable} of {attempted} pages chosen for reading could not be read "
            f"(more than {UNREADABLE_DEGRADE_SHARE:.0%})"
        ]
    return []


def _sync_fetched_bytes(deps: ReadDeps) -> None:
    if deps.fetched_bytes is not None:
        deps.asker.stats.fetched_bytes = deps.fetched_bytes()


@dataclass
class _Text:
    """What the text step found: the text, how it was got, what went wrong."""

    text: str = ""
    via: PageVia = "none"
    access: WorkAccess = "metadata"
    url: str = ""
    notes: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    status: int | None = None
    error: str = ""


async def _load_text(candidate: Candidate, deps: ReadDeps) -> _Text:
    """Each provider in turn (the first that holds text wins), then the
    redistributable abstract, then a long enough snippet.

    A source's failure is noted; one other than a routine refusal (or an offline
    miss) is also listed in ``failed``: the page is unreadable if nothing else
    is left, but a single page does not make the run degraded.
    """

    def before_fetch() -> None:
        _sync_fetched_bytes(deps)
        if deps.asker.stats.fetched_bytes >= deps.asker.budget.max_fetch_bytes:
            raise BudgetExhausted("max_fetch_bytes")

    found, had_failure = await provider_text(
        candidate.work,
        deps.providers,
        before_fetch=before_fetch,
        after_fetch=lambda: _sync_fetched_bytes(deps),
    )
    if found.text:
        return found
    work = candidate.work
    if work.abstract.strip() and work.redistributable:
        found.text, found.via, found.access = work.abstract, "abstract", "abstract"
    elif work.redistributable and len(candidate.snippet.strip()) >= SNIPPET_MIN_CHARS:
        found.text, found.via = candidate.snippet, "snippet"
        found.access = "metadata"
    else:
        found.access = "unreadable" if had_failure else "metadata"
    return found


async def provider_text(
    work: Work,
    providers: Sequence[TextSource],
    *,
    before_fetch: Callable[[], None] = lambda: None,
    after_fetch: Callable[[], None] = lambda: None,
) -> tuple[_Text, bool]:
    """The text of *work* from the first provider that holds it, and whether a
    provider failed on the way. An abstract the licence forbids is skipped.
    ``before_fetch`` runs before each request (it may raise to stop) and
    ``after_fetch`` after it."""
    found = _Text()
    notes, failed = found.notes, found.failed
    had_failure = False
    for provider in providers:
        before_fetch()
        try:
            fetched = await provider.fetch_text(work)
        except SourceError as exc:
            notes.append(f"{provider.name}: {exc.reason}")
            found.status, found.error = exc.status, exc.reason
            if exc.kind == "offline":
                continue  # not in the snapshot: not a failure, nothing fetched
            had_failure = True
            if exc.kind != "refused":
                failed.append(
                    f"{provider.name} could not give the text of "
                    f"{work.title or work.work_id}: {exc.reason}"
                )
            continue
        finally:
            after_fetch()
        if fetched is None:
            continue
        if fetched.access == "abstract" and not work.redistributable:
            continue  # the licence forbids using the record's abstract
        found.text = fetched.text
        found.access = fetched.access
        found.url = fetched.url
        found.status, found.error = None, ""
        if fetched.access == "abstract":
            found.via = "abstract"
        elif (
            "chrome-cdp" in fetched.content_type
            or CHROME_CONTENT_TYPE in fetched.content_type
        ):
            found.via = "chrome"
        else:
            found.via = "fetched"
        break
    return found, had_failure


def _span(passage: Passage) -> list[int]:
    return [passage.start, passage.end]


async def _load(
    candidate: Candidate,
    *,
    subquestion: SubQuestion,
    queries: Sequence[str],
    deps: ReadDeps,
) -> tuple[PageReading, str]:
    """``load_page`` and the text the window was cut from."""
    asker = deps.asker
    store = asker.store
    log = asker.log
    work = candidate.work
    if not work.work_id:
        work = store.upsert_work(work)
    reading = PageReading(work=work, subquestion_id=subquestion.id)

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
        if log is not None:
            log.event(
                "excluded",
                f"excluded: {reading.excluded}",
                step="Read",
                work_id=work.work_id,
                reason=reading.excluded,
            )
        return reading, ""

    found = await _load_text(candidate.model_copy(update={"work": work}), deps)
    reading.notes.extend(found.notes)
    reading.failed_notes.extend(found.failed)
    reading.via = found.via
    reading.status = found.status if found.via == "none" else None
    reading.error = found.error if found.via == "none" else ""
    store.set_work_state(work.work_id, access=found.access)
    reading.work = work = work.model_copy(update={"access": found.access})
    query = " ".join([subquestion.text, *queries])
    if found.text:
        reading.passages = build_window(
            found.text,
            work_id=work.work_id,
            query=query,
            max_chars=deps.window_chars,
            retrieved_at=time.time(),
        )
        store.add_passages(reading.passages)
    if log is not None:
        shown, truncated = runlog.clipped(found.text)
        log.event(
            "fetched",
            f"{'read' if found.via != 'none' else 'could not read'} "
            f"{work.title or work.work_id} ({found.via}, {len(found.text)} characters)",
            step="Read",
            work_id=work.work_id,
            url=work.url or "",
            title=work.title,
            via=found.via,
            ok=found.via != "none",
            chars=len(found.text),
            status=reading.status,
            error=reading.error,
            truncated=truncated,
            text=shown,
        )
        if found.via == "none":
            reasons = [*found.notes, *found.failed] or [
                "no text source holds this page"
            ]
            log.event(
                "unreadable",
                f"unreadable: {work.title or work.work_id}",
                step="Read",
                work_id=work.work_id,
                url=work.url or "",
                title=work.title,
                reasons=list(dict.fromkeys(reasons)),
            )
    return reading, found.text


async def load_page(
    candidate: Candidate,
    *,
    subquestion: SubQuestion,
    queries: Sequence[str],
    deps: ReadDeps,
    round_number: int = 1,
) -> PageReading:
    """Steps without the notes model: retraction check, text, window, passages
    stored, events ``fetched`` / ``unreadable`` / ``excluded`` logged. ``findings``
    is empty. Raises ``BudgetExhausted`` when ``max_fetch_bytes`` is spent."""
    reading, _text = await _load(
        candidate, subquestion=subquestion, queries=queries, deps=deps
    )
    return reading


# -- notes ------------------------------------------------------------------


class _Note(BaseModel):
    passage: int
    claim: str
    quote: str


class _Notes(BaseModel):
    findings: list[_Note]


NOTES_INSTRUCTIONS = (
    "You read passages of ONE document to find evidence for a question. Extract the "
    "findings: specific statements the passages make that bear on the question, "
    "each with the exact words that state it.\n"
    '- claim: one self-contained statement (name what it is about; no "the study", '
    '"this", "they"), with the figures, names and dates as the passage gives '
    "them.\n"
    "- quote: copied from the passage character for character, in one piece, from "
    "the start of the sentence, table row or list item to its end, long enough to "
    "show the claim. No ellipsis. Do not write line breaks as \\n or tabs as \\t. "
    "Do not change punctuation or spacing. For a table, copy the row as written, "
    "with its cells and separators.\n"
    "- passage: the number of the passage the quote is in.\n"
    "Report what the passages say even when it contradicts what you expect, and "
    "even when the document looks unreliable; do not add anything from your own "
    "knowledge. A document mostly about something else can still hold a usable "
    "statement: judge content, not topic. Return an empty list only when no "
    "passage says anything that bears on the question.\n" + UNTRUSTED
)


def _notes_message(
    passages: Sequence[Passage], *, title: str, subquestion: SubQuestion
) -> str:
    numbered = "\n\n".join(
        f"[{n}]\n{fence(p.text)}" for n, p in enumerate(passages, start=1)
    )
    return (
        f"QUESTION: {subquestion.text}\n"
        f"DOCUMENT TITLE:\n{fence(title or '(untitled)')}\n\n"
        f"PASSAGES:\n\n{numbered}"
    )


def _squash(text: str) -> str:
    return " ".join(text.split())


async def read_page(
    candidate: Candidate,
    *,
    subquestion: SubQuestion,
    queries: Sequence[str],
    deps: ReadDeps,
    round_number: int = 1,
) -> PageReading:
    """``load_page`` and then the notes: one model call, findings with the soft quote
    check, events ``notes_input`` / ``finding_noted`` / ``notes`` logged. A page too
    long for one window whose first window gave no finding is read once more over
    the best parts left (the window is chosen without knowing where the answer is).
    A model failure (``ask.MODEL_ERRORS``) on the notes call is recorded in
    ``degraded`` and the page has no findings. Raises ``BudgetExhausted`` when a
    bound is spent."""
    reading, text = await _load(
        candidate, subquestion=subquestion, queries=queries, deps=deps
    )
    if not reading.readable:
        return reading
    first = list(reading.passages)
    failed = await _note_passages(
        reading, first, subquestion=subquestion, deps=deps, source_chars=len(text)
    )
    if reading.findings or failed or sum(len(p.text) for p in first) >= len(text):
        return reading
    more = build_window(
        text,
        work_id=reading.work.work_id,
        query=" ".join([subquestion.text, *queries]),
        max_chars=deps.window_chars,
        retrieved_at=time.time(),
        skip=first,
    )
    if not more:
        return reading
    deps.asker.store.add_passages(more)
    reading.passages.extend(more)
    await _note_passages(
        reading, more, subquestion=subquestion, deps=deps, source_chars=len(text)
    )
    return reading


async def _note_passages(
    reading: PageReading,
    passages: Sequence[Passage],
    *,
    subquestion: SubQuestion,
    deps: ReadDeps,
    source_chars: int,
) -> bool:
    """One notes call over *passages* of the page; its findings are added to
    *reading*. ``True`` when the model failed."""
    asker = deps.asker
    log = asker.log
    work = reading.work
    title = work.title or work.work_id
    message = _notes_message(passages, title=work.title, subquestion=subquestion)
    if log is not None:
        log.event(
            "notes_input",
            f"notes input for {title}: {sum(len(p.text) for p in passages)} "
            f"of {source_chars} characters",
            step="Notes",
            work_id=work.work_id,
            url=work.url or "",
            title=work.title,
            subquestion=subquestion.id,
            source_chars=source_chars,
            window_chars=sum(len(p.text) for p in passages),
            window_spans=[_span(p) for p in passages],
            text=message,
        )
    error = ""
    try:
        out = await asker.llm(
            step="Notes",
            instructions=NOTES_INSTRUCTIONS,
            output_type=_Notes,
            user_message=message,
        )
    except MODEL_ERRORS as exc:
        error = f"{type(exc).__name__}: {exc}"
        reading.degraded.append(f"notes for {title} failed: {error}")
        out = _Notes(findings=[])
    seen = {(f.passage_id, f.quote) for f in reading.findings}
    before = len(reading.findings)
    empty = 0
    surplus = 0
    for note in out.findings:
        claim, shown = _squash(note.claim), _squash(note.quote)
        if not claim or not shown:
            empty += 1
            continue
        located = check_quote(note.quote, passages, named=note.passage)
        if located is not None:
            passage, quote = located
            verified = True
        else:
            named = (
                passages[note.passage - 1]
                if 1 <= note.passage <= len(passages)
                else passages[0]
            )
            passage, quote, verified = named, shown, False
        if (passage.passage_id, quote) in seen:
            continue
        if len(reading.findings) >= MAX_FINDINGS_PER_PAGE:
            surplus += 1
            continue
        seen.add((passage.passage_id, quote))
        reading.findings.append(
            RawFinding(
                subquestion_id=subquestion.id,
                claim=claim,
                quote=quote,
                quote_verified=verified,
                passage_id=passage.passage_id,
                work_id=work.work_id,
            )
        )
    added = reading.findings[before:]
    if empty:
        reading.notes.append(f"{empty} finding(s) with an empty claim or quote ignored")
    if surplus:
        reading.notes.append(
            f"{surplus} finding(s) beyond the {MAX_FINDINGS_PER_PAGE} kept per page"
        )
    unverified = sum(1 for f in added if not f.quote_verified)
    if log is not None:
        for f in added:
            log.event(
                "finding_noted",
                f"finding from {title}: {f.claim}",
                step="Notes",
                subquestion=subquestion.id,
                work_id=work.work_id,
                title=work.title,
                claim=f.claim,
                quote=f.quote,
                verified=f.quote_verified,
            )
        log.event(
            "notes",
            f"{len(added)} finding(s) from {title}, {unverified} unverified",
            step="Notes",
            work_id=work.work_id,
            url=work.url or "",
            subquestion=subquestion.id,
            findings=len(added),
            unverified=unverified,
            error=error,
        )
    return bool(error)
