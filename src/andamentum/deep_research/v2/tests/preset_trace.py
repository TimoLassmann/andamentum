"""A run's trace over the invented world, made comparable across code versions.

``trace(...)`` runs one scenario of ``engine_world`` and returns every model
request (each language-model call's instructions and prompt, each decision
request's state and questions) and the result's evidence table, with every
store-assigned identifier (work, run and passage ids, which are random or
derived from them) replaced by the work's DOI or URL. Two runs that ask the
same questions in the same order and return the same evidence give equal
traces. ``test_presets.py`` compares the ``criteria`` preset's trace with the
one recorded from the engine as first built (commit cf61b6e), in
``fixtures/criteria_preset_trace.json``.

The helpers take the modules they drive as arguments, so the same code can
trace an older copy of the engine.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import Any

_WORK_ID = re.compile(r"\bw_[0-9a-f]{12}\b")
_RUN_ID = re.compile(r"\br_[0-9a-f]{12}\b")
_DIGEST = re.compile(r"\b[0-9a-f]{64}\b")

# (mode, with a decision model)
SCENARIOS: tuple[tuple[str, bool], ...] = (
    ("scholarly", True),
    ("scholarly", False),
    ("both", True),
    ("web", False),
)


@contextmanager
def counted_uuids() -> Iterator[None]:
    """``uuid.uuid4`` as a counter while the block runs: the store names works
    and runs by it, and a tie between two works is broken by that name, so a
    trace is reproducible only with names that are."""
    counter = itertools.count(1)
    real = uuid.uuid4
    uuid.uuid4 = lambda: uuid.UUID(int=next(counter) << 80)  # type: ignore[assignment]
    try:
        yield
    finally:
        uuid.uuid4 = real


def _short(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def normaliser(names: dict[str, str]) -> Callable[[str], str]:
    """Replaces work ids by *names* (work id -> DOI or URL), and run ids and
    content digests by fixed tokens."""

    def normal(text: str) -> str:
        text = _WORK_ID.sub(lambda m: names.get(m.group(), "w_?"), text)
        text = _RUN_ID.sub("r_?", text)
        return _DIGEST.sub("sha_?", text)

    return normal


def _dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


async def trace(
    *,
    mode: str,
    decision: bool,
    run: Callable[..., Awaitable[Any]],
    world: Any,
    store_type: Any,
    source_config: Any,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One scenario: *run* is a ``research`` function, *world* the
    ``engine_world`` module, *store_type* ``ResearchStore``, *source_config*
    ``SourceConfig``; *extra* keyword arguments go to *run* (a ``config``)."""
    store = store_type.in_memory()
    llm = world.world_llm()
    decision_model, stub = world.world_decisions() if decision else (None, None)
    with counted_uuids():
        result = await run(
            world.QUESTION,
            mode=mode,
            model=llm.model,
            decision_model=decision_model,
            store=store,
            sources=source_config(searxng_url=world.SEARXNG, scholarly=("openalex",)),
            source_client=world.World().client(store),
            **(extra or {}),
        )
    names: dict[str, str] = {}
    for work in store.iter_works():
        names[work.work_id] = work.doi or work.url or work.title
    for work in result.works:
        names.setdefault(work.work_id, work.doi or work.url or work.title)
    normal = normaliser(names)
    llm_calls = [
        {"instructions": _short(i), "head": i[:40], "prompt": _short(normal(p))}
        for i, p in llm.calls
    ]
    decision_calls = (
        [
            {"state": _short(normal(_dumps(state))), "questions": _short(_dumps(q))}
            for state, q in stub.calls
        ]
        if stub is not None
        else []
    )
    by_id = {f.finding_id: f for f in result.findings}
    return {
        "mode": mode,
        "decision": decision,
        "llm_calls": llm_calls,
        "decision_calls": decision_calls,
        "subquestions": [s.model_dump(mode="json") for s in result.subquestions],
        "criteria": {
            k: [c.model_dump(mode="json") for c in v]
            for k, v in sorted(result.criteria.items())
        },
        "works": [names.get(w.work_id, "?") for w in result.works],
        "excluded": sorted(names.get(k, "?") for k in result.excluded),
        "queries": list(result.queries),
        "findings": [
            {
                "id": f.finding_id,
                "subquestion": f.subquestion_id,
                "work": names.get(f.work_id, "?"),
                "claim": f.claim,
                "quote": f.quote,
                "criteria": f.criterion_ids,
            }
            for f in by_id.values()
        ],
        "coverage": [c.model_dump(mode="json") for c in result.coverage],
        "conflicts": [c.model_dump(mode="json") for c in result.conflicts],
        "report": normal(result.report_markdown),
        "verification": [
            {"kind": v.kind, "ref": normal(v.ref), "detail": normal(v.detail)}
            for v in result.verification
        ],
        "stop_reason": result.stop_reason,
        "degraded": [normal(d) for d in result.degraded],
        "llm_count": result.stats.llm_calls,
        "decision_count": result.stats.decision_calls,
    }
