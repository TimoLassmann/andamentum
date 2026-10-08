"""deep_research: an evidence-first research loop, web and scholarly.

``research(question, *, mode, model, ...)`` runs the engine and returns a
``ResearchResult``: sub-questions, the works found, verbatim passages, findings
(each with the quote behind it, checked in code or marked unverified), the
answer the research reached for each sub-question, and a report that cites
findings by id, verified. The loop is plan, then per sub-question: queries,
search, choose what to read, read, take notes, reflect (is it answered?), then
one write and a code-only verify. ``find_works(sentence, *, model, ...)`` is
the finder: one sentence in, the works it could cite out, ranked (metadata
only). See ``CLAUDE.md`` for the design.
"""

from .ask import MODEL_ERRORS, Asker, BudgetExhausted  # noqa: F401  (importable, not exported)
from .engine import research
from .errors import ResearchError
from .identity import merge  # noqa: F401  (importable, not exported)
from .finder import FINDER_SOURCES, FinderModelError, FinderResult, find_works
from .novelty import NoveltyResearch, run_novelty_check
from .read import locate_quote
from .sources import (
    SourceClient,  # noqa: F401  (importable, not exported)
    SourceConfig,
    SourceError,
    SourceSet,  # noqa: F401
    build_sources,  # noqa: F401
)
from .store import ResearchStore, StoreError
from .write import ReportWriterError
from .types import (
    Budget,
    Candidate,
    Depth,
    Finding,
    LogEvent,
    Mode,
    Passage,
    ResearchResult,
    RunLog,
    RunStats,
    SubQuestion,
    SubQuestionResult,
    VerifyFlag,
)

__all__ = [
    "FINDER_SOURCES",
    "Budget",
    "Candidate",
    "Depth",
    "FinderModelError",
    "FinderResult",
    "Finding",
    "LogEvent",
    "Mode",
    "NoveltyResearch",
    "Passage",
    "ReportWriterError",
    "ResearchError",
    "ResearchResult",
    "ResearchStore",
    "RunLog",
    "RunStats",
    "SourceConfig",
    "SourceError",
    "StoreError",
    "SubQuestion",
    "SubQuestionResult",
    "VerifyFlag",
    "find_works",
    "locate_quote",
    "research",
    "run_novelty_check",
]
