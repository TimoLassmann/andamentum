"""Worker: frame the problem into its 2–4 big concerns (areas).

The second design head. Bounded by ``max_areas`` (dialect Law 5 — the fan-out width
traces to a Deps value), truncating loudly via the returned note rather than silently.
"""

from __future__ import annotations

from .agents import FRAME, AgentSink
from .schemas import ForgeAreas, ForgeWhy


async def frame(
    why: ForgeWhy, *, sink: AgentSink, max_areas: int
) -> tuple[list[str], list[str]]:
    """Return ``(areas, notes)`` — the concern areas (capped) and any advisory notes."""
    out = await sink.run(
        FRAME,
        purpose=why.purpose,
        boundary_in=why.boundary_in,
        boundary_out=why.boundary_out,
    )
    assert isinstance(out, ForgeAreas)
    areas = [a.strip() for a in out.areas if a.strip()]
    notes: list[str] = []
    if len(areas) > max_areas:
        notes.append(f"frame: {len(areas)} areas proposed; capped to {max_areas}")
        areas = areas[:max_areas]
    if not areas:
        areas = ["core"]
        notes.append("frame: no areas proposed; using a single 'core' area")
    return areas, notes
