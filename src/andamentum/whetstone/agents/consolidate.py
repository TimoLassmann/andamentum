"""Consolidate agent: judge whether two findings are the same issue.

The deterministic substrate proposes candidate pairs (overlapping anchors
or similar claims). This agent does the one thing the substrate cannot:
decide whether two findings that sit near each other are *the same issue*
(merge them) or *distinct issues that merely co-locate* (keep both).

The schema is the smallest possible — a single binary field — so the
smallest local models fill it reliably. Merge groups are rebuilt from the
pairwise verdicts by union-find in the node; the agent never has to emit a
partition.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ._definition import AgentDefinition

CONSOLIDATE_PROMPT = """You are deduplicating reviewer comments on a document.

You are shown TWO findings (each: title, rationale, the quoted text it
refers to, and its section). They were flagged as possibly redundant.

Decide ONE thing:

  relation = "same"
    → They are the SAME underlying issue. One merged comment would fully
      cover both. (e.g. two reviewers both saying a claim is unsupported.)

  relation = "distinct"
    → They are DIFFERENT issues that merely sit near each other or share a
      quote. Both deserve their own comment. (e.g. one flags a passive-voice
      construction, the other flags that the same sentence overclaims.)

Judge by the ISSUE each finding raises, not by how close their quotes are.
Same wording about the same problem → same. Different problems → distinct.
When genuinely unsure, choose "distinct" (keeping a real second issue is
safer than hiding it)."""


class SameOrDistinct(BaseModel):
    """consolidate_agent's flat output — one binary decision."""

    relation: Literal["same", "distinct"] = Field(
        description=(
            "'same' = the two findings are one underlying issue; merge them. "
            "'distinct' = different issues that merely co-locate; keep both. "
            "Default to 'distinct' when unsure."
        ),
    )


def _build() -> AgentDefinition:
    return AgentDefinition(
        name="consolidate",
        prompt=CONSOLIDATE_PROMPT,
        output_model=SameOrDistinct,
        retries=2,
        output_retries=2,
    )


CONSOLIDATE_AGENT = _build()
