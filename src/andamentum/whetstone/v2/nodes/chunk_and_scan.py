"""Node 2: ChunkAndScan — split into sections + extract structural facts.

Three pieces of work, all deterministic:

  1. Run ``chunker.extract_units`` on the markdown to get section spans.
  2. Walk the sections through every structural extractor (citations,
     terms, numerics, cross-references) to populate StructuralFacts.
     Build the deterministic DocumentMap.
  3. Synthesise deterministic Findings from StructuralFacts.

If a model is supplied (deps.model is not None), flow into Skim → the
LLM-driven phases. Otherwise End immediately with deterministic-only
findings — that's "Phase 1 mode", preserved as a valid public path so
callers who don't want LLM costs can still get the cheap output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

from pydantic_graph import BaseNode, End, GraphRunContext

from andamentum.chunker import extract_units

from ..deps import ReviewDeps
from ..schemas import ReviewMetrics, ReviewResult
from ..state import ReviewState
from ..structural.citations import extract_citations
from ..structural.crossrefs import extract_cross_references
from ..structural.deterministic_findings import synthesize_deterministic_findings
from ..structural.document_map import build_document_map
from ..structural.numerics import extract_numeric_claims
from ..structural.terms import extract_term_glossary
from ..structural.types import SectionRef, StructuralFacts

if TYPE_CHECKING:
    from .critical_read import CriticalRead
    from .custom_reviewer import CustomReviewer
    from .extract_checkable_items import ExtractCheckableItems
    from .extract_keywords import ExtractKeywords


logger = logging.getLogger("andamentum.whetstone.v2")


@dataclass
class ChunkAndScan(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    """Chunk the markdown, extract structural facts, emit deterministic findings."""

    async def run(
        self, ctx: GraphRunContext[ReviewState, ReviewDeps]
    ) -> "Union[CriticalRead, ExtractKeywords, ExtractCheckableItems, CustomReviewer, End[ReviewResult]]":
        ctx.state.current_phase = "scan"
        logger.info("[scan] chunking %d chars into sections", len(ctx.state.markdown))

        # ── 1. Chunk into sections ────────────────────────────────────
        chunking = await extract_units(
            ctx.state.markdown,
            target_min_chars=ctx.deps.target_min_chars,
            target_max_chars=ctx.deps.target_max_chars,
            embedding_fn=ctx.deps.embedding_fn,
        )
        ctx.state.sections = [
            SectionRef(
                id=f"sec_{i:03d}",
                title=unit.title,
                text=unit.text,
                char_start=unit.source_start,
                char_end=unit.source_end,
            )
            for i, unit in enumerate(chunking.units, start=1)
        ]
        logger.info("[scan] %d sections produced", len(ctx.state.sections))

        # ── 2. Extract structural facts ───────────────────────────────
        ctx.state.structural_facts = StructuralFacts(
            citation_graph=extract_citations(ctx.state.sections),
            term_glossary=extract_term_glossary(ctx.state.sections),
            numeric_claims=extract_numeric_claims(ctx.state.sections),
            cross_references=extract_cross_references(ctx.state.sections),
        )

        # ── 3. Build document map (deterministic; Skim node will enrich) ──
        ctx.state.document_map = build_document_map(ctx.state.sections)

        # ── 4. Synthesise deterministic findings ──────────────────────
        ctx.state.deterministic_findings = synthesize_deterministic_findings(
            sections=ctx.state.sections,
            facts=ctx.state.structural_facts,
            markdown=ctx.state.markdown,
        )
        logger.info(
            "[scan] %d deterministic finding(s)",
            len(ctx.state.deterministic_findings),
        )

        # ── 5. Branch: with model → next LLM phase; without → End now ──
        if ctx.deps.model is None:
            ctx.state.current_phase = "done"
            logger.info("[scan] no model — skipping LLM phases (--no-llm mode)")
            return End(_build_result(ctx.state))

        if ctx.state.mode == "panel":
            from .extract_keywords import ExtractKeywords

            return ExtractKeywords()

        if ctx.state.mode == "guidelines":
            from .extract_checkable_items import ExtractCheckableItems

            return ExtractCheckableItems()

        if ctx.state.mode == "custom":
            from .custom_reviewer import CustomReviewer

            return CustomReviewer()

        from .critical_read import CriticalRead

        return CriticalRead()


def _build_result(state: ReviewState) -> ReviewResult:
    """Assemble the final ReviewResult from the populated state."""
    return ReviewResult(
        summary="",  # Phase 4 fills this in
        findings=list(state.findings),
        deterministic_findings=list(state.deterministic_findings),
        edits=list(state.edits),
        author_questions=list(state.author_questions),
        document_map=list(state.document_map),
        metrics=ReviewMetrics(
            llm_calls=state.llm_calls,
            wall_seconds=0.0,  # api.py wraps this with a timer
            deterministic_findings_count=len(state.deterministic_findings),
            investigated_findings_count=len(state.findings),
            challenged_findings_count=len(state.challenged_findings),
            edits_count=len(state.edits),
            sections_processed=len(state.sections),
            reflection_rounds_used=state.reflection_round,
        ),
    )
