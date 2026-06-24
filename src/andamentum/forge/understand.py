"""Worker: restate the brief as a problem — purpose and boundaries.

The first design head. Engine-free: takes the brief + the agent Port, returns the
typed ``ForgeWhy``. (Dialect Law 2 — delete the graph engine and this still runs.)
"""

from __future__ import annotations

from .agents import UNDERSTAND, AgentSink
from .schemas import ForgeWhy


async def understand(brief: str, *, sink: AgentSink) -> ForgeWhy:
    out = await sink.run(UNDERSTAND, brief=brief)
    assert isinstance(out, ForgeWhy)
    return out
