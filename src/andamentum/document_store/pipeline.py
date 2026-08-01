"""Harvest-backed source conversion — the ONLY place ``andamentum.harvest`` is imported.

The store core is deliberately harvest-free: :func:`process_pending` takes a
``convert_fn`` rather than importing a converter, so a caller that already has
markdown never pulls in Docling/trafilatura. This module is the optional adapter
that fills that parameter in with :func:`andamentum.harvest.extract`, following
the same single-glue-point pattern as ``figures.scribe_glue`` and
``proofread.cli``.

Nothing here is privileged. It is a convenience, not a gateway — an app may call
the core directly and inject its own converter:

    from andamentum.document_store import process_pending
    await process_pending("brain", model=..., embedding_model=...,
                          convert_fn=my_own_converter)

Usage with harvest:

    from andamentum.document_store.pipeline import ingest_source, drain

    await ingest_source("brain", "~/papers/big.pdf")   # queued, no conversion yet
    await drain("brain", model=..., embedding_model=...)  # converts + enriches
"""

from __future__ import annotations

from pathlib import Path

from .public import (
    ConvertFn,
    ProcessMode,
    ProcessReport,
    ingest_source as _core_ingest_source,
    process_pending as _core_process_pending,
)

__all__ = ["harvest_convert_fn", "ingest_source", "drain"]


def harvest_convert_fn() -> ConvertFn:
    """Return an async ``(source) -> markdown`` backed by ``harvest.extract``.

    Imported lazily so that merely importing this module does not pull in
    Docling — the cost is paid only when a conversion actually runs.

    Raises:
        ImportError: If harvest's extraction backends are not installed.
    """

    async def _convert(source: str) -> str:
        from andamentum.harvest import extract

        # harvest.extract takes str | Path; a local path is passed as Path so
        # its format detection uses the extension, not URL heuristics.
        target: str | Path = source
        if "://" not in source:
            target = Path(source).expanduser()
        return await extract(target)

    return _convert


async def ingest_source(
    database: str,
    source: str,
    title: str | None = None,
    metadata: dict | None = None,
    *,
    model: str | None = None,
    embedding_model: str | None = None,
    process: ProcessMode = "defer",
) -> str:
    """Queue (or immediately convert + enrich) a source file/URL using harvest.

    Thin wrapper over :func:`document_store.ingest_source` with
    ``convert_fn`` pre-filled. See that function for the argument semantics.
    """
    return await _core_ingest_source(
        database,
        source,
        title,
        metadata,
        convert_fn=harvest_convert_fn(),
        model=model,
        embedding_model=embedding_model,
        process=process,
    )


async def drain(database: str, **kwargs) -> ProcessReport:
    """Run :func:`document_store.process_pending` with harvest wired in.

    Accepts every keyword :func:`process_pending` does (``model``,
    ``embedding_model``, ``should_continue``, ``on_progress``, ``max_docs``,
    ``max_seconds``); ``convert_fn`` defaults to harvest but can be overridden.
    """
    kwargs.setdefault("convert_fn", harvest_convert_fn())
    return await _core_process_pending(database, **kwargs)
