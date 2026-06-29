"""Worker: frame the problem into its 2–4 big concerns (areas).

The second design head. Bounded by ``max_areas`` (dialect Law 5 — the fan-out width
traces to a Deps value), truncating loudly via the returned note rather than silently.
"""

from __future__ import annotations

from .agents import FRAME, AgentSink
from .schemas import ForgeAreas, ForgeWhy


async def frame(
    why: ForgeWhy,
    *,
    sink: AgentSink,
    max_areas: int,
    plan_feedback: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Return ``(areas, notes)`` — the concern areas (capped) and any advisory notes."""
    out = await sink.run(
        FRAME,
        purpose=why.purpose,
        boundary_in=why.boundary_in,
        boundary_out=why.boundary_out,
        plan_feedback=_frame_feedback_text(plan_feedback),
    )
    assert isinstance(out, ForgeAreas)
    areas = [a.strip() for a in out.areas if a.strip()]
    if not areas:
        # Fail loud: no fabricated default. An empty framing means the brief did not
        # yield distinct concerns — surface it, never invent a placeholder area.
        raise ValueError(
            "frame produced no areas; the brief did not yield distinct concerns — design incomplete"
        )
    notes: list[str] = []
    if len(areas) > max_areas:
        notes.append(f"frame: {len(areas)} areas proposed; capped to {max_areas}")
        areas = areas[:max_areas]
    return areas, notes


def _frame_feedback_text(concerns: list[str] | None) -> str:
    """Turn plan-manager concerns into framing feedback — possibly a NEW concern each."""
    if not concerns:
        return ""
    return (
        "A prior plan review found the design missed part of the goal. Reconsider whether a "
        "NEW distinct concern is needed for each of these (only if genuinely separable):\n"
        + "\n".join(f"- {c}" for c in concerns)
    )
