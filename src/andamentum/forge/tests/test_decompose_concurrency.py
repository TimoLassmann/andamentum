"""decompose fans out model calls (list jobs per area, declare per node) — bounded.

A local (Ollama) model serialises inference, so the fan-outs must run strictly one call at
a time; ``concurrency`` lets a cloud model overlap them. Driven by a stub sink that records
how many calls are in flight (no live model).
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from andamentum.core import AgentDefinition
from andamentum.forge.decompose import decompose
from andamentum.forge.schemas import ForgeWhy

from .conftest import ScriptedSink

_WHY = ForgeWhy(
    purpose="Summarize a document into bullets.",
    boundary_in="a document",
    boundary_out="bullet points",
)


class _CountingSink(ScriptedSink):
    """Counts in-flight ``list_jobs`` / ``declare_node`` calls and the peak."""

    in_flight = 0
    peak = 0

    async def run(self, defn: AgentDefinition, **kwargs: object) -> BaseModel:
        fanned_out = defn.name in ("list_jobs", "declare_node")
        if fanned_out:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            await asyncio.sleep(0.01)  # let any concurrent sibling start
        try:
            return await super().run(defn, **kwargs)
        finally:
            if fanned_out:
                self.in_flight -= 1


def _sink() -> _CountingSink:
    areas = ["a", "b", "c"]
    return _CountingSink(
        why=_WHY,
        areas=areas,
        jobs_by_area={area: [f"Job {area} one.", f"Job {area} two."] for area in areas},
    )


async def test_decompose_default_is_strictly_serial() -> None:
    """Regression: both gather sites used to launch every call at once."""
    sink = _sink()
    await decompose(_WHY, sink.areas, sink=sink, max_jobs_per_area=5, max_nodes=18)
    assert sink.peak == 1


async def test_decompose_overlaps_calls_when_concurrency_allows() -> None:
    sink = _sink()
    await decompose(
        _WHY, sink.areas, sink=sink, max_jobs_per_area=5, max_nodes=18, concurrency=8
    )
    assert sink.peak > 1
