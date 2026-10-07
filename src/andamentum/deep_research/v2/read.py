"""READ: from a chosen search result to findings whose quotes are checked in code.

Package b implements this module (spec section 1.5). Steps for one page:
retraction check, text (a text source; for the web the plain fetch, then the
same Chrome over CDP), a window of the most relevant ~``window_chars`` characters
stored as passages, then one model call for the findings, then the soft quote
check. The public names below are the contract package a (the engine), the MCP
server and the bench call.

The old implementation is ``git show bbdb010:src/andamentum/deep_research/v2/read.py``
(``split_passages``, ``locate_quote``, ``_fold``, ``_visible``, ``check_retraction``,
``_load_text`` and ``load_work`` carry over; the criteria, screener and passage
selection do not).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from andamentum.references import Work, WorkRetraction

from .ask import Asker
from .sources.base import TextSource
from .sources.crossref import CrossrefSource
from .types import (
    DEFAULT_WINDOW_CHARS,
    Candidate,
    Passage,
    PageReading,
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


def split_passages(
    text: str, *, work_id: str, retrieved_at: float = 0.0
) -> list[Passage]:
    """Cut *text* into passages of about ``PASSAGE_TARGET_CHARS``, exact slices."""
    raise NotImplementedError("package b")


def build_window(
    text: str, *, work_id: str, query: str, max_chars: int, retrieved_at: float = 0.0
) -> list[Passage]:
    """The most relevant parts of *text*, at most about *max_chars* characters, as
    passages in document order: best BM25 passages first until the limit, adjacent
    ones merged into one span, a text within the limit whole. Exact slices."""
    raise NotImplementedError("package b")


def locate_quote(quote: str, text: str) -> tuple[int, int] | None:
    """Where *quote* occurs in *text*: exactly, else ignoring case, width, quote
    marks, dashes and whitespace, else in the text a reader sees (Markdown links,
    citation markers, emphasis marks and table pipes left out). A literal
    backslash-n or backslash-t in the quote reads as a space. ``(start, end)`` in
    *text*, or ``None``."""
    raise NotImplementedError("package b")


def check_quote(
    quote: str, passages: Sequence[Passage], *, named: int | None = None
) -> tuple[Passage, str] | None:
    """The passage that holds *quote* and the passage's own words at that span, or
    ``None``. The passage numbered *named* (1-based) is tried first."""
    raise NotImplementedError("package b")


async def check_retraction(
    work: Work, *, crossref: CrossrefSource
) -> tuple[WorkRetraction | None, str]:
    """The work's retraction state from Crossref, and a note on a failed check."""
    raise NotImplementedError("package b")


def unreadable_degraded(*, attempted: int, unreadable: int) -> list[str]:
    """The degraded line for a run whose pages mostly could not be read: one entry
    when more than ``UNREADABLE_DEGRADE_SHARE`` of the *attempted* pages were
    *unreadable*, else none."""
    raise NotImplementedError("package b")


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
    raise NotImplementedError("package b")


async def read_page(
    candidate: Candidate,
    *,
    subquestion: SubQuestion,
    queries: Sequence[str],
    deps: ReadDeps,
    round_number: int = 1,
) -> PageReading:
    """``load_page`` and then the notes: one model call, findings with the soft quote
    check, events ``notes_input`` / ``finding_noted`` / ``notes`` logged. A model
    failure (``ask.MODEL_ERRORS``) on the notes call is recorded in ``degraded`` and
    the page has no findings. Raises ``BudgetExhausted`` when a bound is spent."""
    raise NotImplementedError("package b")
