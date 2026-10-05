"""The R0 benchmark harness for deep_research v2 (spec section 9).

Run it with ``uv run python -m andamentum.deep_research.v2.bench --help``. The
datasets are downloaded into a directory the caller names and never committed;
the live runs need models and, for the web set, a SearxNG instance, so the
tests of this package use tiny synthetic fixtures and stub models, and the one
test that needs the network carries the ``research_eval`` marker.

- ``web_set``: 15 questions with checkable answers.
- ``litsearch``, ``synergy``: the two public datasets (downloaders and loaders).
- ``metrics``, ``results``: the metrics, the result records and the tables.
- ``arms``, ``screening``, ``judging``, ``runner``: the systems under test and
  how they are run and checked.
"""

from .litsearch import LitSearchSet, load_litsearch
from .metrics import (
    claim_support,
    retrieval_score,
    screening_score,
    url_integrity,
)
from .results import BenchRun, LitSearchRun, ScreeningRun, WebRun, render_markdown
from .synergy import ScreeningReview, load_review
from .web_set import WEB_QUESTIONS, WebQuestion, answer_correct

__all__ = [
    "BenchRun",
    "LitSearchRun",
    "LitSearchSet",
    "ScreeningReview",
    "ScreeningRun",
    "WEB_QUESTIONS",
    "WebQuestion",
    "WebRun",
    "answer_correct",
    "claim_support",
    "load_litsearch",
    "load_review",
    "render_markdown",
    "retrieval_score",
    "screening_score",
    "url_integrity",
]
