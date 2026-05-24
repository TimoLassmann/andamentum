"""Light consolidation — roll up near-duplicate findings into one.

After the gap loop, several findings can make essentially the same point about
different parts of the document (e.g. nine separate "this LLM step is
operationally under-specified" comments, or one overstated claim flagged four
times). A single agent groups the near-duplicates; a deterministic merge keeps
the most-severe member's anchor and a unified issue statement, so the reader
sees one comment instead of nine.

Conservative by design: only clear same-point groups merge; everything else
passes through untouched. On any agent error the findings are returned
unconsolidated — degraded, never wrong.
"""

from __future__ import annotations

import logging
from typing import cast

from pydantic import BaseModel, Field

from andamentum.core.agents import AgentDefinition, build_pydantic_ai_agent
from andamentum.core.models import resolve_model

from .review import Finding

logger = logging.getLogger("andamentum.whetstone.v3")

_SEVERITY_RANK = {"minor": 0, "moderate": 1, "major": 2}


class _Group(BaseModel):
    member_indices: list[int] = Field(
        description="Indices (from the numbered list) of findings that make the "
        "same underlying point. Only include groups of two or more."
    )
    merged_issue: str = Field(
        description="One issue statement capturing the shared point, noting that "
        "it recurs across the document where relevant."
    )


class _Consolidation(BaseModel):
    groups: list[_Group] = Field(default_factory=list)


_PROMPT = """You are tidying a set of review findings before they are shown to \
the author. Some findings make essentially the SAME underlying point, possibly \
about different parts of the document — for example several separate notes that \
a method step is under-specified, or repeated versions of one overstated claim.

Each finding is shown with its criterion, severity, section id, verbatim quote, \
and issue text. Use ALL of these to decide whether two findings are the same \
point. Two findings can have similar issue text but anchor on very different \
quotes/sections (= distinct), or have different issue text but anchor on the \
same span (= likely duplicates). The quote and section_id are the structural \
ground truth; the issue text is one author's phrasing.

Group together findings a reader would see as IDENTICAL — same underlying \
critique, same recommended action. For each group of TWO OR MORE, write one \
merged issue statement that captures the shared point.

Be aggressively conservative. Defaults:
  • When in doubt, DON'T merge — leave findings distinct.
  • Two findings about the SAME paragraph but raising DIFFERENT concerns \
(e.g. "missing baseline" vs "missing uncertainty") are NOT duplicates.
  • Two findings about the same conceptual issue (e.g. "claim overstated") \
in DIFFERENT sections may be the same point — but only if a reader fixing one \
would naturally fix the other. If fixing them requires different edits, keep \
them separate.
  • Two findings with similar issue text but anchored on DIFFERENT quotes \
that point at structurally different problems (e.g. one says the math drops a \
square root, the other says a stated bound is inconsistent with bias \
correction) are NOT duplicates, even if both fall under the same section.

Most distinct-looking findings ARE distinct. Most groups should have exactly \
two members; groups of 4+ should be rare. If you can't articulate the shared \
point in one sentence that covers ALL members' quotes, the findings probably \
aren't duplicates.

You return only the group structure (which indices belong together) and the \
merged issue statement. You do NOT pick or rewrite quotes — the system anchors \
the merged finding on the most-severe member's quote deterministically."""


def _severity_of(f: Finding) -> int:
    return _SEVERITY_RANK.get(f.severity, 1)


async def consolidate(findings: list[Finding], *, agent_model: str) -> list[Finding]:
    """Merge near-duplicate findings into one comment each. Returns a list no
    longer than the input; distinct findings are preserved unchanged."""
    if len(findings) < 2:
        return findings

    defn = AgentDefinition(
        name="v3_consolidate",
        prompt=_PROMPT,
        output_model=_Consolidation,
        retries=2,
        output_retries=2,
    )
    agent = build_pydantic_ai_agent(defn, resolve_model(agent_model))
    # Per-line format includes quote + section so the agent can tell apart
    # two findings with similar issue text but different anchors. The agent
    # NEVER picks or rewrites quotes (the docx anchor uses Finding.quote as
    # an exact-match anchor — silently breaks if the quote changes).
    numbered = "\n".join(
        f"  [{i}] ({f.criterion}/{f.severity}, section={f.span.section_id if f.span else '?'}) "
        f"{f.issue}\n      quote: {f.quote!r}"
        for i, f in enumerate(findings)
    )
    try:
        result = await agent.run(f"FINDINGS:\n{numbered}\n\nGroup the near-duplicates.")
        groups = cast(_Consolidation, result.output).groups
    except Exception as exc:
        logger.warning("[v3.consolidate] crashed, leaving findings as-is: %s", exc)
        return findings

    n = len(findings)
    used: set[int] = set()
    merged: list[Finding] = []
    for g in groups:
        idxs = [
            i for i in dict.fromkeys(g.member_indices) if 0 <= i < n and i not in used
        ]
        if len(idxs) < 2:
            continue
        members = [findings[i] for i in idxs]
        used.update(idxs)
        # Anchor the merged comment on the most-severe member.
        primary = max(members, key=_severity_of)
        merged.append(
            Finding(
                criterion=primary.criterion,
                issue=g.merged_issue.strip() or primary.issue,
                quote=primary.quote,
                severity=primary.severity,
                span=primary.span,
            )
        )

    passthrough = [f for i, f in enumerate(findings) if i not in used]
    if merged:
        logger.info(
            "[v3.consolidate] %d finding(s) → %d (merged %d group(s))",
            n,
            len(merged) + len(passthrough),
            len(merged),
        )
    # Gate re-orders by severity afterwards, so order here is not load-bearing.
    return merged + passthrough
