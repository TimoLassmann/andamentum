"""Node: ExtractKeywords (panel mode).

Single LLM call. Reads ``state.markdown`` (or the document_map summary
when the markdown is large) and produces 3-5 academic disciplines into
``state.disciplines``.

If the caller pre-supplied disciplines via ``panel_disciplines`` we
skip the LLM call entirely and copy them in. This makes panel mode
deterministic for tests and lets advanced callers override the
discipline pool.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from pydantic_graph import BaseNode, GraphRunContext

from ..agents import KeywordExtractionOutput, build_pydantic_ai_agent
from ..deps import ReviewDeps
from ..schemas import ReviewResult
from ..state import ReviewState

if TYPE_CHECKING:
    from .generate_expert_panel import GenerateExpertPanel


logger = logging.getLogger("andamentum.whetstone.v2")

# Hard cap on prompt size — keep small enough for any model.
_MAX_PROMPT_CHARS = 12_000


@dataclass
class ExtractKeywords(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    """Identify 3-5 academic disciplines for the document."""

    async def run(
        self, ctx: GraphRunContext[ReviewState, ReviewDeps]
    ) -> "GenerateExpertPanel":
        ctx.state.current_phase = "extract_keywords"

        if ctx.state.panel_disciplines:
            # Caller provided disciplines explicitly — skip the LLM.
            ctx.state.disciplines = list(ctx.state.panel_disciplines)
            logger.info(
                "[panel] disciplines provided by caller — skipping extraction "
                "(%d disciplines)",
                len(ctx.state.disciplines),
            )
        else:
            logger.info("[panel] extracting disciplines from document")
            ctx.state.disciplines = await _extract_disciplines(ctx.deps, ctx.state)
            ctx.state.llm_calls += 1
            logger.info(
                "[panel] extracted %d discipline(s): %s",
                len(ctx.state.disciplines),
                ", ".join(ctx.state.disciplines),
            )

        from .generate_expert_panel import GenerateExpertPanel

        return GenerateExpertPanel()


async def _extract_disciplines(deps: ReviewDeps, state: ReviewState) -> list[str]:
    """Run the keyword_extractor agent and return its disciplines."""
    document_view = _build_document_view(state)
    prompt = f"""DOCUMENT TO ANALYSE:
{document_view}

Identify 3-5 academic disciplines that would provide the most valuable
and diverse perspectives for reviewing this work. Order from most to
least relevant. Use specific discipline names, not generic ones."""

    agent = build_pydantic_ai_agent("extract_keywords", deps.model)
    result = await agent.run(prompt)
    output = cast(KeywordExtractionOutput, result.output)
    # Strip blanks, dedupe while preserving order.
    seen: set[str] = set()
    disciplines: list[str] = []
    for d in output.disciplines:
        d_clean = d.strip()
        if d_clean and d_clean not in seen:
            seen.add(d_clean)
            disciplines.append(d_clean)
    return disciplines


def _build_document_view(state: ReviewState) -> str:
    """Build a compact document view for the keyword-extractor prompt.

    Prefers the document_map (titles + gists) when available; falls
    back to the truncated markdown otherwise. Keeps the prompt small
    enough for any model.
    """
    if state.document_map:
        lines = [
            f"  • {c.title}: {c.one_line_gist}".rstrip(": ") for c in state.document_map
        ]
        view = "Document map:\n" + "\n".join(lines)
        if state.markdown:
            head = state.markdown[: _MAX_PROMPT_CHARS // 2]
            view += f"\n\nDocument excerpt (first {len(head)} chars):\n{head}"
        return view
    # No document map — use truncated markdown directly.
    if len(state.markdown) <= _MAX_PROMPT_CHARS:
        return state.markdown
    head = state.markdown[:_MAX_PROMPT_CHARS]
    return f"{head}\n\n[... truncated, original was {len(state.markdown)} chars ...]"
