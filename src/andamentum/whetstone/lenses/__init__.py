"""Whetstone lenses with internal structure.

The seven persona lenses (rigorous, writer, methodology, statistician,
consistency, overclaim, claim_evidence) live in ``whetstone/agents/`` as
single-prompt entries in ``LENS_PROMPTS``. They are appropriate for
fuzzy, judgement-heavy review where the persona prompt is the value.

This subpackage hosts lenses whose internal structure is a
pydantic-graph DAG of small, narrow agents — used when the lens checks
a closed set of named rules. Each lens's sub-graph clearly separates
deterministic operations (regex, dictionary lookup) from LLM agent
calls, with a ``NodeKind`` ClassVar on every node and a topology
introspection helper for static testing.

Currently registered:
  - ``strunk`` — Elements of Style rule-based review.

``SUBGRAPH_LENS_ENTRYPOINTS`` is the dispatch table consulted by
``whetstone.nodes.critical_read._run_lens``: when a lens name is a key
here, the prompt-based path is skipped and the lens's entrypoint is
invoked instead. Each entrypoint adapts the per-lens deps (e.g.
``StrunkLensDeps``) onto whetstone's run-level ``ReviewDeps``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..schemas import Finding
    from ..structural.types import SectionRef


async def _run_strunk(
    section: "SectionRef", *, model: Any
) -> "list[Finding]":
    """Adapter: build a runner + deps, then call ``run_strunk_lens``.

    Defers all strunk imports to call time so importing the lenses
    package doesn't drag in the strunk sub-graph (and through it,
    proofread / pydantic-graph) at module-load time.
    """
    from andamentum.core.agents import AgentRunner

    from .strunk import run_strunk_lens
    from .strunk.state import StrunkLensDeps

    runner = AgentRunner(model=model)
    deps = StrunkLensDeps(executor=runner)
    return await run_strunk_lens(section, deps=deps)


# Lens name → entrypoint that takes (section, *, model) and returns
# ``list[Finding]``. The shape matches whetstone.nodes.critical_read's
# expectation for a per-section lens read.
SUBGRAPH_LENS_ENTRYPOINTS: dict[
    str, Callable[..., Awaitable["list[Finding]"]]
] = {
    "strunk": _run_strunk,
}

SUBGRAPH_LENS_NAMES: frozenset[str] = frozenset(SUBGRAPH_LENS_ENTRYPOINTS)


__all__ = ["SUBGRAPH_LENS_ENTRYPOINTS", "SUBGRAPH_LENS_NAMES"]
