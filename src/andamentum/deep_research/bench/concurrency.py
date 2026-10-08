"""How many items an experiment run may process at once.

A local Ollama model answers one request at a time, so every local model is run
with concurrency 1. A model served by Ollama's cloud (a ``-cloud`` tag, such as
``gemma4:31b-cloud``) takes several requests at once; the cap is
``MAX_CLOUD_CONCURRENCY`` (4, a decision of 2026-10-06).

``resolve_concurrency`` is the one check, used by the library entry points and
the ``--concurrency`` flag: a request above 1 for a model that is not a
``-cloud`` model is refused with the reason, never lowered silently.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import TypeVar

from pydantic_ai.models import Model

MAX_CLOUD_CONCURRENCY = 4
CLOUD_SUFFIX = "-cloud"

T = TypeVar("T")


def model_label(model: str | Model) -> str:
    """The model's id: the string itself, or a model object's name."""
    return model if isinstance(model, str) else str(model.model_name)


def is_cloud_model(model: str | Model) -> bool:
    """Is *model* an Ollama cloud model (the name ends in ``-cloud``)?

    A model id string must carry the ``ollama:`` prefix; a model object is
    judged by its ``model_name``.
    """
    if isinstance(model, str):
        return model.startswith("ollama:") and model.endswith(CLOUD_SUFFIX)
    return str(model.model_name).endswith(CLOUD_SUFFIX)


def resolve_concurrency(requested: int, *, models: Iterable[str | Model | None]) -> int:
    """*requested* when it is allowed for every model given, else ``ValueError``.

    ``None`` entries (an arm that calls no language model) are ignored. With
    no model at all, any value up to the cap is allowed.
    """
    if requested < 1:
        raise ValueError(f"concurrency must be at least 1, not {requested}")
    if requested > MAX_CLOUD_CONCURRENCY:
        raise ValueError(
            f"concurrency {requested} is above the cap of {MAX_CLOUD_CONCURRENCY}"
        )
    if requested > 1:
        local = sorted(
            {model_label(m) for m in models if m is not None and not is_cloud_model(m)}
        )
        if local:
            raise ValueError(
                f"concurrency {requested} is refused for local model(s) "
                f"{', '.join(local)}: a local Ollama server answers one request at "
                f"a time; only '{CLOUD_SUFFIX}' models may run concurrently"
            )
    return requested


async def run_bounded(
    jobs: Sequence[Callable[[], Awaitable[T]]], *, concurrency: int
) -> list[T]:
    """Run *jobs* with at most *concurrency* in flight; results in job order.

    Jobs start in order, each by a free worker. When one raises, no job starts
    after it, the ones in flight are cancelled and the error propagates (a job
    that must not abort the rest catches its own errors).
    """
    results: dict[int, T] = {}
    upcoming = iter(range(len(jobs)))
    stopped = False

    async def worker() -> None:
        nonlocal stopped
        for index in upcoming:
            if stopped:
                return
            try:
                results[index] = await jobs[index]()
            except BaseException:
                stopped = True
                raise

    workers = [
        asyncio.ensure_future(worker()) for _ in range(min(concurrency, len(jobs)))
    ]
    try:
        await asyncio.gather(*workers)
    except BaseException:
        stopped = True
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        raise
    return [results[index] for index in range(len(jobs))]
