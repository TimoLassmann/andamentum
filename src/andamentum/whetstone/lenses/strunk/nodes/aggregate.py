"""Aggregate — convert StrunkFindings to whetstone.Findings and terminate.

Kind: control
Reads: state.findings, state.section
Writes: (returns End[list[Finding]] to graph; does not mutate state)
Successor: End[list[Finding]]

Each ``StrunkFinding`` produced by a rule node carries char offsets and
the offending span. The conversion here:

* slices the verbatim span from ``section.text`` (so the persisted
  ``Quote.text`` always reflects what the document actually says, not
  what the LLM echoed back);
* attaches a single ``Quote`` per finding pinning it to the section;
* sets ``source="investigate"`` and ``perspective="strunk"`` so the
  whetstone renderer treats Strunk findings the same as any lens hit;
* sorts by ``(char_start, rule_number)`` for stable presentation.

Phase A does not yet dedup overlapping spans (R11 and R13 flagging
the same sentence span is rare in practice and easy to add when a
fixture shows the case).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pydantic_graph import BaseNode, End, GraphRunContext

from ....schemas import Finding, Quote
from ..kinds import NodeKind
from ..models import StrunkFinding
from ..state import StrunkLensDeps, StrunkLensState


def _to_whetstone_findings(
    raw: list[StrunkFinding],
    section_id: str,
    section_text: str,
) -> list[Finding]:
    """Convert sorted raw Strunk findings into public Finding objects."""
    out: list[Finding] = []
    sorted_raw = sorted(raw, key=lambda f: (f.char_start, f.rule_number))
    for r in sorted_raw:
        verbatim = section_text[r.char_start : r.char_end]
        rationale = r.rationale
        if r.suggested_replacement:
            rationale = f"{rationale} Suggested rewrite: {r.suggested_replacement!r}."
        out.append(
            Finding(
                title=r.title,
                severity=r.severity,
                confidence=r.confidence,
                rationale=rationale,
                quotes=[
                    Quote(
                        section_id=section_id,
                        char_start=r.char_start,
                        char_end=r.char_end,
                        text=verbatim,
                    )
                ],
                sections_involved=[section_id],
                source="investigate",
                perspective="strunk",
                category=r.category,
            )
        )
    return out


@dataclass
class Aggregate(BaseNode[StrunkLensState, StrunkLensDeps, list[Finding]]):
    """Convert ``state.findings`` to ``list[Finding]`` and terminate."""

    kind: ClassVar[NodeKind] = NodeKind.CONTROL
    reads: ClassVar[frozenset[str]] = frozenset({"findings", "section"})
    writes: ClassVar[frozenset[str]] = frozenset()

    async def run(
        self,
        ctx: GraphRunContext[StrunkLensState, StrunkLensDeps],
    ) -> End[list[Finding]]:
        result = _to_whetstone_findings(
            ctx.state.findings,
            ctx.state.section.id,
            ctx.state.section.text,
        )
        return End(result)
