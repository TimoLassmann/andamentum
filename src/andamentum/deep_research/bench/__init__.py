"""The benchmark harness for deep_research v2.

Run it with ``uv run python -m andamentum.deep_research.bench --help``. The
datasets are downloaded into a directory the caller names and never committed;
the live runs need models and a web search (a Chrome over CDP, or SearxNG), so
the tests of this package use tiny synthetic fixtures and stub arms.

- ``experiment``, ``experiment_run``, ``registry``, ``verdicts``: the
  pre-registered comparison of arms (runs of ``research()``) on the same items,
  and its rules.
- ``simpleqa``, ``litsearch_local``, ``metrics``: the sets and their metrics.
- ``attribution``: where a lost answer was lost, from the run logs.
"""

from .attribution import trace_litsearch, trace_simpleqa
from .metrics import retrieval_score

__all__ = ["retrieval_score", "trace_litsearch", "trace_simpleqa"]
