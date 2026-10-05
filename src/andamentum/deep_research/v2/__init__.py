"""deep_research v2: an evidence-first research engine, web and scholarly.

``research(question, *, mode, model, ...)`` runs the engine and returns a
``ResearchResult``: sub-questions, criteria, works, verbatim passages, findings
with checked quotes, coverage, conflicts, and a report that cites findings by
id, verified. See ``CLAUDE.md`` for the design and what each module owns.
"""

from .ask import MODEL_ERRORS, Asker, AskError, BudgetExhausted
from .coverage import compute_coverage, decide_stop, find_conflicts, record_coverage
from .criteria import (
    CriteriaError,
    PassageScore,
    Refinement,
    Screener,
    Screening,
    ScreeningError,
    combine,
    compile_criteria,
    score_candidates,
)
from .graph import research
from .identity import merge
from .novelty import NoveltyResearch, run_novelty_check_v2
from .read import ReadDeps, ReadTarget, load_work, read_work, read_works
from .sources import SourceConfig, SourceError, SourceSet, build_sources
from .store import ResearchStore, StoreError
from .verify import verify_report
from .write import write_report
from .types import (
    Budget,
    Candidate,
    CandidateScore,
    Conflict,
    Coverage,
    Criterion,
    Finding,
    Passage,
    ResearchResult,
    RunStats,
    SubQuestion,
    VerifyFlag,
)

__all__ = [
    "MODEL_ERRORS",
    "NoveltyResearch",
    "AskError",
    "Asker",
    "Budget",
    "BudgetExhausted",
    "Candidate",
    "CandidateScore",
    "Conflict",
    "Coverage",
    "CriteriaError",
    "Criterion",
    "Finding",
    "Passage",
    "PassageScore",
    "ReadDeps",
    "ReadTarget",
    "Refinement",
    "ResearchResult",
    "ResearchStore",
    "RunStats",
    "Screener",
    "Screening",
    "ScreeningError",
    "StoreError",
    "SourceConfig",
    "SourceError",
    "SourceSet",
    "SubQuestion",
    "VerifyFlag",
    "build_sources",
    "combine",
    "compile_criteria",
    "compute_coverage",
    "decide_stop",
    "find_conflicts",
    "load_work",
    "merge",
    "read_work",
    "read_works",
    "record_coverage",
    "research",
    "run_novelty_check_v2",
    "score_candidates",
    "verify_report",
    "write_report",
]
