"""ResolveDemands — re-run ambiguous cases on a stronger model.

Kind: control
Reads: state.demands
Writes: state.findings (when a demand resolves into a verdict)
Successor: Aggregate

Phase A behaviour: no-op. The node passes straight through to
``Aggregate``. Demands accumulate in ``state.demands`` for diagnostics
but no escalation is attempted, because Phase A intentionally leaves
the per-rule model assignment unconfigured.

Phase 4 will replace this stub with: group demands by rule, look up
the escalation model from ``deps.model_for_rule`` (or a global
stronger-model default), re-run the agent for each demand, and
convert successful re-runs into ``StrunkFinding`` rows. Demands that
remain ``unsure`` after escalation become low-confidence ``StrunkFinding``
entries so the user still sees them — fallibilism over false silence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from pydantic_graph import BaseNode, GraphRunContext

from ....schemas import Finding
from ..kinds import NodeKind
from ..state import StrunkLensDeps, StrunkLensState

if TYPE_CHECKING:
    from .aggregate import Aggregate


@dataclass
class ResolveDemands(BaseNode[StrunkLensState, StrunkLensDeps, list[Finding]]):
    """Phase A: pass through. Phase 4: escalate demands on a stronger model."""

    kind: ClassVar[NodeKind] = NodeKind.CONTROL
    reads: ClassVar[frozenset[str]] = frozenset({"demands"})
    writes: ClassVar[frozenset[str]] = frozenset({"findings"})

    async def run(
        self,
        ctx: GraphRunContext[StrunkLensState, StrunkLensDeps],
    ) -> "Aggregate":
        # Phase A: no-op. Real implementation lives in Phase 4.
        from .aggregate import Aggregate

        return Aggregate()
