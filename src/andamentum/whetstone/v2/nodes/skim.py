"""Node: Skim.

Reads the DocumentMap + abstract + conclusion and emits the initial
hypothesis queue. In Phase 5, runs once per perspective in parallel —
all hypotheses go into the SAME shared queue (per the design decision
made during the PRD discussion).

Also enriches the DocumentMap by replacing the deterministic gist with
the agent-written one (where the agent emitted one).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_graph import BaseNode, GraphRunContext

from ..agents import build_pydantic_ai_agent, SkimOutput
from ..deps import ReviewDeps
from ..investigators import classify_hypothesis
from ..schemas import Hypothesis, ReviewResult, SectionCard
from ..state import ReviewState

if TYPE_CHECKING:
    from .investigate import InvestigateLoop
    from ..structural.types import SectionRef


# How much abstract / conclusion text to include in the skim prompt.
# The DocumentMap is the structural backbone; abstract + conclusion give
# the agent the paper's framing without sending the whole doc.
_ABSTRACT_TITLES = ("abstract",)
_CONCLUSION_TITLES = ("conclusion", "conclusions", "discussion", "summary")


@dataclass
class Skim(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    """Read structural skeleton, emit hypotheses, enrich DocumentMap."""

    async def run(
        self, ctx: GraphRunContext[ReviewState, ReviewDeps]
    ) -> "InvestigateLoop":
        ctx.state.current_phase = "skim"

        abstract_text = _find_section_text(ctx.state.sections, _ABSTRACT_TITLES)
        conclusion_text = _find_section_text(ctx.state.sections, _CONCLUSION_TITLES)

        # Run skim_agent once per perspective (Phase 5: panel mode).
        outputs = await asyncio.gather(
            *[
                _run_one(ctx.deps, ctx.state.document_map, abstract_text, conclusion_text, p)
                for p in ctx.state.perspectives
            ]
        )
        ctx.state.llm_calls += len(outputs)

        # Merge enriched gists. If multiple perspectives offered gists for
        # the same section, the first non-empty one wins (deterministic).
        enriched_by_id: dict[str, str] = {}
        for output, _perspective in outputs:
            for sec in output.enriched_sections:
                if sec.section_id and sec.one_line_gist and sec.section_id not in enriched_by_id:
                    enriched_by_id[sec.section_id] = sec.one_line_gist
        if enriched_by_id:
            ctx.state.document_map = [
                SectionCard(
                    section_id=card.section_id,
                    title=card.title,
                    one_line_gist=enriched_by_id.get(card.section_id, card.one_line_gist),
                )
                for card in ctx.state.document_map
            ]

        # Push hypotheses onto the shared queue, tagged by perspective and
        # classified for the investigator registry.
        for output, perspective in outputs:
            for h in output.hypotheses:
                hypothesis = Hypothesis(
                    text=h.text,
                    priority=h.priority,
                    relevant_section_ids=h.relevant_section_ids,
                    perspective=perspective if len(ctx.state.perspectives) > 1 else None,
                )
                hypothesis.investigation_type = classify_hypothesis(hypothesis)
                ctx.state.hypotheses.append(hypothesis)

        from .investigate import InvestigateLoop

        return InvestigateLoop()


async def _run_one(
    deps: ReviewDeps,
    document_map,
    abstract_text: str,
    conclusion_text: str,
    perspective: str,
) -> tuple[SkimOutput, str]:
    """One skim_agent call. Returns (output, perspective_label)."""
    map_lines = "\n".join(
        f"  • {c.section_id} — {c.title}: {c.one_line_gist}" for c in document_map
    )
    prompt = f"""PERSPECTIVE: {perspective}

DOCUMENT MAP ({len(document_map)} sections):
{map_lines}

ABSTRACT (if any):
{abstract_text or "(no abstract section identified)"}

CONCLUSION (if any):
{conclusion_text or "(no conclusion section identified)"}

Now: enrich each section's one-line gist AND emit 5–15 hypotheses to investigate."""

    agent = build_pydantic_ai_agent("skim", deps.model)
    result = await agent.run(prompt)
    from typing import cast

    return cast(SkimOutput, result.output), perspective


def _find_section_text(
    sections: "list[SectionRef]", title_keywords: tuple[str, ...]
) -> str:
    """Return the text of the first section whose title contains a keyword."""
    for s in sections:
        title_low = (s.title or "").lower()
        if any(kw in title_low for kw in title_keywords):
            return s.text
    return ""
