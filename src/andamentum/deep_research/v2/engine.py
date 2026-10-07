"""The v2 engine: plan, then per sub-question a loop of search, choose, read,
take notes and reflect, then one write and a code-only verify.

Package a (loop core) implements this module; the signature below is the
contract the bench, the novelty check, compose and the CLI call.
See the spec (sections 1.2, 1.3, 1.4, 1.7, 1.10) for the behaviour.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from pydantic_ai.models import Model

from .sources import SourceClient, SourceConfig
from .store import ResearchStore
from .types import Budget, Mode, ResearchResult

# Language-model calls the search leaves for writing the report.
WRITE_RESERVE = 2


async def research(
    question: str,
    *,
    mode: Mode,
    model: str | Model,
    store: Path | ResearchStore | None = None,
    budget: Budget = Budget(),
    sources: SourceConfig = SourceConfig(),
    source_client: SourceClient | None = None,
    http_cache: ResearchStore | None = None,
    progress: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> ResearchResult:
    """Research *question* and return the evidence table with a verified report.

    ``mode`` picks the sources: ``web`` (a Chrome over CDP, ``sources.browser_cdp_url``,
    or SearxNG, ``sources.searxng_url``; one of them is required), ``scholarly``
    (``sources.scholarly``) or ``both``. ``model`` is the language model (a
    pydantic-ai model id or object); it is the only model the engine uses.

    ``store`` is the research store: ``None`` keeps everything in memory for this
    run; a path opens (or creates) a persistent one; an open ``ResearchStore`` is
    used as it is. ``source_client`` is the HTTP client the sources share (built
    for the store when not given; one passed must belong to the open store passed
    as ``store``). ``http_cache`` is a shared read-through HTTP cache (a
    ``ResearchStore`` kept apart from the run's own), applied to a store this call
    opens; it cannot be combined with an open ``store``. ``progress`` receives one
    line per step; ``clock`` is the clock ``budget.max_wall_s`` is measured on.

    Every run fills ``result.log`` (``RunLog``). A spent budget ends the search with
    ``stop_reason='budget:<name>'`` and still writes and verifies the report from
    what was found. A failing source or model is recorded in ``degraded``. Anything
    else raises: the run is recorded as failed in the store first, and the log so
    far is the error's ``run_log`` attribute.
    """
    raise NotImplementedError("package a: the research loop")
