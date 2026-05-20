"""Public entrypoint for the Strunk lens.

Filled in after the graph is wired (see ``graph.py``). Until then this
stub keeps the package importable so the scaffold pyright clean.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...schemas import Finding
    from ...structural.types import SectionRef
    from .state import StrunkLensDeps


async def run_strunk_lens(
    section: "SectionRef",
    *,
    deps: "StrunkLensDeps",
) -> "list[Finding]":
    """Run the full Strunk sub-graph against one section.

    Returns a list of ``whetstone.schemas.Finding`` objects ready to
    merge into the main review pool. Each finding's ``perspective``
    field is set to ``"strunk"`` and the ``source`` is ``"investigate"``.
    """
    # Real implementation is wired once the graph is in place (task #10).
    from .graph import run_strunk_lens as _impl

    return await _impl(section, deps=deps)
