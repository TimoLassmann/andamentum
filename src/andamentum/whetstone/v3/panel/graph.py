"""Panel-mode graph for v3.

Five-node pydantic-graph chain — same shape as v2's panel pipeline,
minus the harvest/chunk substrate (run_panel_v3 receives already-
harvested markdown the same way run_review_v3 does):

    Sectionize → ExtractKeywords → GenerateExpertPanel → ExpertReview
       (D)            (A)                    (A)               (A)
                  → PanelSynthesise → End[ReviewResult]
                          (A)

D = deterministic, A = agent. ExtractKeywords is skipped (no LLM call)
when the caller supplies ``panel_disciplines`` explicitly. The two
fan-out nodes (GenerateExpertPanel, ExpertReview) use a 2-concurrent
semaphore, matching v2's calibration.

Output: a ``ReviewResult`` whose panel-specific fields
(``expert_profiles``, ``expert_reviews``, ``panel_synthesis``) are
populated; the criterion-cascade fields (``findings``, ``edits``,
``author_questions``) stay empty. The existing renderers detect panel
output via ``bool(result.expert_profiles or result.expert_reviews)``
and route accordingly — no renderer changes needed.

Cost shape: ``2N + 2`` LLM calls per run (default 10 at N=4):
  - 1 extract_keywords (skipped if --panel-disciplines supplied)
  - N expert_generator (one per discipline, semaphore=2)
  - N expert_reviewer (one per expert, semaphore=2)
  - 1 panel_synthesise

Per-node failure isolation: profile / review generation crashes leave
None in the result list (filtered out); panel synthesis is loud-fail-
safe — a synthesis crash still returns the per-expert reviews with a
failure note in the summary.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import cast

from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from andamentum.core.agents import build_pydantic_ai_agent
from andamentum.core.models import resolve_model

from ...schemas import (
    ExpertProfile,
    ExpertReview,
    PanelSynthesis,
    ReviewMetrics,
    ReviewResult,
)
from ..model import Section
from ..sectionize import sectionize
from .agents import (
    EXPERT_GENERATOR_DEFN,
    EXPERT_REVIEWER_DEFN,
    EXTRACT_KEYWORDS_DEFN,
    PANEL_SYNTHESISE_DEFN,
    KeywordExtractionOutput,
)

logger = logging.getLogger("andamentum.whetstone.v3.panel")


_MAX_CONCURRENT_GENERATIONS: int = 2
_MAX_CONCURRENT_REVIEWS: int = 2
# Cap on how much markdown reaches the per-expert review prompt. v2 uses
# 30k chars; keep parity so panel output is bit-comparable across the
# v2 → v3 cutover for the same draft + same expert profiles.
_DOCUMENT_VIEW_MAX_CHARS: int = 30_000


@dataclass
class PanelDeps:
    agent_model: str
    n_experts: int = 4
    panel_disciplines: list[str] = field(default_factory=list)


@dataclass
class PanelState:
    source: str
    sections: list[Section] = field(default_factory=list)
    disciplines: list[str] = field(default_factory=list)
    expert_profiles: list[ExpertProfile] = field(default_factory=list)
    expert_reviews: list[ExpertReview] = field(default_factory=list)
    panel_synthesis: PanelSynthesis | None = None
    llm_calls: int = 0
    summary: str = ""


# ── Node 1: Sectionize ─────────────────────────────────────────────────────


@dataclass
class Sectionize(BaseNode[PanelState, PanelDeps, ReviewResult]):
    async def run(
        self, ctx: GraphRunContext[PanelState, PanelDeps]
    ) -> "ExtractKeywords":
        ctx.state.sections = sectionize(ctx.state.source)
        return ExtractKeywords()


# ── Node 2: ExtractKeywords (skipped when caller supplies disciplines) ─────


@dataclass
class ExtractKeywords(BaseNode[PanelState, PanelDeps, ReviewResult]):
    async def run(
        self, ctx: GraphRunContext[PanelState, PanelDeps]
    ) -> "GenerateExpertPanel":
        if ctx.deps.panel_disciplines:
            ctx.state.disciplines = list(ctx.deps.panel_disciplines)
            logger.info(
                "[panel] disciplines provided by caller — skipping extraction"
                " (%d disciplines)",
                len(ctx.state.disciplines),
            )
            return GenerateExpertPanel()

        document_view = _build_document_view(ctx.state)
        agent = build_pydantic_ai_agent(
            EXTRACT_KEYWORDS_DEFN, resolve_model(ctx.deps.agent_model)
        )
        result = await agent.run(
            "Identify 3-5 relevant academic disciplines for reviewing:\n\n"
            f"{document_view}"
        )
        ctx.state.disciplines = cast(KeywordExtractionOutput, result.output).disciplines
        ctx.state.llm_calls += 1
        logger.info(
            "[panel] extracted %d discipline(s): %s",
            len(ctx.state.disciplines),
            ", ".join(ctx.state.disciplines),
        )
        return GenerateExpertPanel()


# ── Node 3: GenerateExpertPanel (N parallel, semaphore=2) ──────────────────


@dataclass
class GenerateExpertPanel(BaseNode[PanelState, PanelDeps, ReviewResult]):
    async def run(
        self, ctx: GraphRunContext[PanelState, PanelDeps]
    ) -> "ExpertReviewPhase":
        disciplines = list(ctx.state.disciplines)[: ctx.deps.n_experts]
        if not disciplines:
            logger.warning(
                "[panel] no disciplines available — skipping panel generation"
            )
            return ExpertReviewPhase()

        sem = asyncio.Semaphore(_MAX_CONCURRENT_GENERATIONS)

        async def _gen_one(discipline: str) -> ExpertProfile | None:
            async with sem:
                try:
                    agent = build_pydantic_ai_agent(
                        EXPERT_GENERATOR_DEFN,
                        resolve_model(ctx.deps.agent_model),
                    )
                    result = await agent.run(
                        f"Generate a fictional expert biosketch for the discipline: "
                        f"{discipline}"
                    )
                    profile = cast(ExpertProfile, result.output)
                    # Defensive: agents occasionally drop the discipline field.
                    if not profile.discipline:
                        profile = profile.model_copy(update={"discipline": discipline})
                    return profile
                except Exception as exc:
                    logger.warning(
                        "[panel] profile generation for %r failed: %s",
                        discipline,
                        exc,
                    )
                    return None

        results = await asyncio.gather(*[_gen_one(d) for d in disciplines])
        ctx.state.expert_profiles = [p for p in results if p is not None]
        ctx.state.llm_calls += len(ctx.state.expert_profiles)
        logger.info(
            "[panel] generated %d/%d profile(s) successfully",
            len(ctx.state.expert_profiles),
            len(disciplines),
        )
        return ExpertReviewPhase()


# ── Node 4: ExpertReviewPhase (N parallel, semaphore=2) ────────────────────


@dataclass
class ExpertReviewPhase(BaseNode[PanelState, PanelDeps, ReviewResult]):
    """Named ``ExpertReviewPhase`` (not ``ExpertReview``) to avoid a name
    collision with the schema type of the same name."""

    async def run(
        self, ctx: GraphRunContext[PanelState, PanelDeps]
    ) -> "PanelSynthesisPhase":
        profiles = list(ctx.state.expert_profiles)
        if not profiles:
            logger.warning(
                "[panel] no expert profiles — skipping reviews and synthesis"
            )
            return PanelSynthesisPhase()

        sem = asyncio.Semaphore(_MAX_CONCURRENT_REVIEWS)
        document_view = _build_document_view(ctx.state)

        async def _review_one(profile: ExpertProfile) -> ExpertReview | None:
            async with sem:
                try:
                    agent = build_pydantic_ai_agent(
                        EXPERT_REVIEWER_DEFN,
                        resolve_model(ctx.deps.agent_model),
                    )
                    result = await agent.run(
                        f"EXPERT BIOSKETCH:\n"
                        f"  Name: {profile.name}\n"
                        f"  Position: {profile.position}\n"
                        f"  Education: {profile.education}\n"
                        f"  Contributions: {profile.contributions}\n"
                        f"  Research: {profile.research}\n"
                        f"  Discipline: {profile.discipline}\n\n"
                        f"DOCUMENT TO REVIEW:\n{document_view}"
                    )
                    review = cast(ExpertReview, result.output)
                    if not review.expert_name:
                        review = review.model_copy(update={"expert_name": profile.name})
                    if not review.discipline:
                        review = review.model_copy(
                            update={"discipline": profile.discipline}
                        )
                    return review
                except Exception as exc:
                    logger.warning(
                        "[panel] review by %r (%s) failed: %s",
                        profile.name,
                        profile.discipline,
                        exc,
                    )
                    return None

        results = await asyncio.gather(*[_review_one(p) for p in profiles])
        ctx.state.expert_reviews = [r for r in results if r is not None]
        ctx.state.llm_calls += len(ctx.state.expert_reviews)
        logger.info(
            "[panel] %d/%d review(s) completed",
            len(ctx.state.expert_reviews),
            len(profiles),
        )
        return PanelSynthesisPhase()


# ── Node 5: PanelSynthesisPhase (1 call, loud-fail-safe) ───────────────────


@dataclass
class PanelSynthesisPhase(BaseNode[PanelState, PanelDeps, ReviewResult]):
    """Named ``PanelSynthesisPhase`` (not ``PanelSynthesise``) to avoid a
    name collision with the schema type of the same name."""

    async def run(
        self, ctx: GraphRunContext[PanelState, PanelDeps]
    ) -> End[ReviewResult]:
        reviews = list(ctx.state.expert_reviews)
        if not reviews:
            logger.warning("[panel] no expert reviews — skipping synthesis")
            ctx.state.summary = (
                "[Panel synthesis skipped — no expert reviews were produced.]"
            )
            return End(_build_result(ctx.state))

        try:
            review_block = "\n\n---\n\n".join(_format_expert_review(r) for r in reviews)
            agent = build_pydantic_ai_agent(
                PANEL_SYNTHESISE_DEFN, resolve_model(ctx.deps.agent_model)
            )
            result = await agent.run(f"EXPERT REVIEWS TO SYNTHESISE:\n\n{review_block}")
            ctx.state.panel_synthesis = cast(PanelSynthesis, result.output)
            ctx.state.llm_calls += 1
            # Renderers prefer panel_synthesis.review_summary; keep the
            # top-level summary blank so neither double-renders.
            ctx.state.summary = ""
        except Exception as exc:
            logger.warning("[panel] synthesis call failed: %s", exc)
            ctx.state.summary = (
                "[Panel synthesis call failed — per-expert reviews are still "
                f"valid below.]\nFailure: {exc}"
            )

        return End(_build_result(ctx.state))


# ── Helpers ────────────────────────────────────────────────────────────────


def _build_document_view(state: PanelState) -> str:
    """Build a panel-friendly document view: section-title outline +
    truncated markdown body. Matches v2's _build_document_view shape."""
    outline = "\n".join(f"  • {s.title}" for s in state.sections if s.title)
    body = state.source[:_DOCUMENT_VIEW_MAX_CHARS]
    truncated = (
        ""
        if len(state.source) <= _DOCUMENT_VIEW_MAX_CHARS
        else ("\n\n[...document truncated for review...]")
    )
    if outline:
        return f"DOCUMENT OUTLINE:\n{outline}\n\nDOCUMENT BODY:\n{body}{truncated}"
    return f"DOCUMENT BODY:\n{body}{truncated}"


def _format_expert_review(r: ExpertReview) -> str:
    """One expert review formatted for the synthesiser's prompt."""
    return (
        f"EXPERT: {r.expert_name} ({r.discipline})\n"
        f"Overall score: {r.overall_score}/10\n"
        f"Recommendation: {r.recommendation}\n"
        f"Rigor: {r.scientific_rigor_score}/10 — {r.scientific_rigor_justification}\n"
        f"Methodology: {r.methodology_score}/10 — {r.methodology_justification}\n"
        f"Novelty: {r.novelty_score}/10 — {r.novelty_justification}\n"
        f"Clarity: {r.clarity_score}/10 — {r.clarity_justification}\n"
        f"Strengths:\n  - " + "\n  - ".join(r.strengths) + "\n"
        "Weaknesses:\n  - " + "\n  - ".join(r.weaknesses) + "\n"
        f"Overall assessment: {r.overall_assessment}\n"
        f"Recommendation rationale: {r.recommendation_justification}"
    )


def _build_result(state: PanelState) -> ReviewResult:
    return ReviewResult(
        summary=state.summary,
        findings=[],
        deterministic_findings=[],
        edits=[],
        author_questions=[],
        document_map=[],
        expert_profiles=list(state.expert_profiles),
        expert_reviews=list(state.expert_reviews),
        panel_synthesis=state.panel_synthesis,
        metrics=ReviewMetrics(
            llm_calls=state.llm_calls,
            wall_seconds=0.0,
            sections_processed=len(state.sections),
        ),
    )


# ── Graph + public entry ───────────────────────────────────────────────────


panel_graph_v3 = Graph(
    nodes=[
        Sectionize,
        ExtractKeywords,
        GenerateExpertPanel,
        ExpertReviewPhase,
        PanelSynthesisPhase,
    ]
)


async def run_panel_v3(
    markdown: str,
    *,
    model: str,
    n_experts: int = 4,
    panel_disciplines: list[str] | None = None,
) -> ReviewResult:
    """Run a v3 panel review over already-harvested markdown.

    Returns a `ReviewResult` whose panel-specific fields are populated;
    the criterion-cascade fields stay empty.

    Parameters
    ----------
    markdown:
        Pre-harvested document body (use `harvest.extract` to get this
        from a PDF / DOCX / URL).
    model:
        pydantic-ai model id (e.g. "openai:gpt-5.4-nano",
        "ollama:gemma4:31b-nvfp4").
    n_experts:
        Cap on parallel expert generations. Default 4. If
        `panel_disciplines` is supplied, this is the upper bound on
        how many disciplines from that list become experts.
    panel_disciplines:
        Optional explicit list of disciplines. When supplied, the
        keyword-extraction LLM call is skipped (saving one call) and
        these are used directly.
    """
    deps = PanelDeps(
        agent_model=model,
        n_experts=n_experts,
        panel_disciplines=list(panel_disciplines) if panel_disciplines else [],
    )
    state = PanelState(source=markdown)
    result = await panel_graph_v3.run(Sectionize(), state=state, deps=deps)
    return result.output
