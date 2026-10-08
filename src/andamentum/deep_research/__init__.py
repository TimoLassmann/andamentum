"""deep_research: an evidence-first research loop, web and scholarly.

``research(question, *, mode, model, ...)`` runs the engine and returns a
``ResearchResult``: sub-questions, the works found, verbatim passages, findings
(each with the quote behind it, checked in code or marked unverified), the
answer the research reached for each sub-question, and a report that cites
findings by id, verified. The loop is plan, then per sub-question: queries,
search, choose what to read, read, take notes, reflect (is it answered?), then
one write and a code-only verify. See ``CLAUDE.md`` for the design.
"""

from .ask import MODEL_ERRORS, Asker, AskError, BudgetExhausted
from .engine import research
from .identity import merge
from .novelty import NoveltyResearch, run_novelty_check
from .read import locate_quote
from .sources import (
    SourceClient,
    SourceConfig,
    SourceError,
    SourceSet,
    build_sources,
)
from .store import ResearchStore, StoreError
from .types import (
    Budget,
    Candidate,
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
    "MODEL_ERRORS",
    "AskError",
    "Asker",
    "Budget",
    "BudgetExhausted",
    "Candidate",
    "Finding",
    "LogEvent",
    "Mode",
    "NoveltyResearch",
    "Passage",
    "ResearchResult",
    "ResearchStore",
    "RunLog",
    "RunStats",
    "SourceClient",
    "SourceConfig",
    "SourceError",
    "SourceSet",
    "StoreError",
    "SubQuestion",
    "SubQuestionResult",
    "VerifyFlag",
    "build_sources",
    "locate_quote",
    "merge",
    "research",
    "run_novelty_check",
]
