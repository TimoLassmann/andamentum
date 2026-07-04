"""Panel-mode graph for v3.

Five-node pydantic-graph chain — same shape as v2's panel pipeline,
minus the harvest/chunk substrate (run_panel_v3 receives already-
harvested markdown the same way run_review_v3 does):

    Sectionize → ExtractKeywords → GenerateExpertPanel → ExpertReview
       (D)            (A)                    (A)               (A)
                  → PanelSynthesise → End[ReviewResult]
                          (A)

D = deterministic, A = agent. ExtractKeywords is skipped (no LLM call)
when the caller supplies ``panel_disciplines`` explicitly. Each node is
thin — read surfaces, call one engine-free worker, assign, return the
successor; the fan-outs (2-concurrent semaphore, matching v2's
calibration) live inside the workers (``generate_panel``,
``review_experts``) and the node's assignment is the sole State-write
site for each gather join.

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
Call counting flows through the shared ``v3._metrics`` contextvar (the
workers bump it after each agent run), same as the main review graph.

Per-node failure isolation: profile / review generation crashes leave
holes the workers filter out; panel synthesis is loud-fail-safe — a
synthesis crash still returns the per-expert reviews with a failure
note in the summary. When the failure rate of either fan-out reaches
``PanelDeps.soft_failure_threshold``, the result is flagged
``degraded`` with a reason — a mostly-failed run is not green.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from ...schemas import ExpertProfile, ExpertReview, PanelSynthesis, ReviewResult
from ..model import Section
from ..sectionize import sectionize
from .build_result import build_panel_result
from .extract_keywords import extract_keywords
from .generate_panel import generate_panel
from .review_experts import review_experts
from .synthesise_panel import synthesise_panel

logger = logging.getLogger("andamentum.whetstone.v3.panel")


@dataclass(frozen=True)
class PanelDeps:
    agent_model: str
    n_experts: int = 4
    panel_disciplines: list[str] = field(default_factory=list)
    # Aggregate-failure gate: when this fraction (or more) of attempted
    # profile generations or expert reviews fail, the result is flagged
    # degraded — a panel that lost most of its experts is not green.
    soft_failure_threshold: float = 0.5


@dataclass
class PanelState:
    # ── inputs
    source: str
    # ── artifacts
    sections: list[Section] = field(default_factory=list)
    disciplines: list[str] = field(default_factory=list)
    expert_profiles: list[ExpertProfile] = field(default_factory=list)
    expert_reviews: list[ExpertReview] = field(default_factory=list)
    panel_synthesis: PanelSynthesis | None = None
    summary: str = ""
    # ── flow control
    profiles_attempted: int = 0
    reviews_attempted: int = 0


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

        ctx.state.disciplines = await extract_keywords(
            ctx.state.source, ctx.state.sections, model=ctx.deps.agent_model
        )
        return GenerateExpertPanel()


# ── Node 3: GenerateExpertPanel (N parallel, semaphore in the worker) ──────


@dataclass
class GenerateExpertPanel(BaseNode[PanelState, PanelDeps, ReviewResult]):
    async def run(
        self, ctx: GraphRunContext[PanelState, PanelDeps]
    ) -> "ExpertReviewPhase":
        disciplines = list(ctx.state.disciplines)[: ctx.deps.n_experts]
        ctx.state.profiles_attempted = len(disciplines)
        if not disciplines:
            logger.warning(
                "[panel] no disciplines available — skipping panel generation"
            )
            return ExpertReviewPhase()

        ctx.state.expert_profiles = await generate_panel(
            disciplines, model=ctx.deps.agent_model
        )
        return ExpertReviewPhase()


# ── Node 4: ExpertReviewPhase (N parallel, semaphore in the worker) ────────


@dataclass
class ExpertReviewPhase(BaseNode[PanelState, PanelDeps, ReviewResult]):
    """Named ``ExpertReviewPhase`` (not ``ExpertReview``) to avoid a name
    collision with the schema type of the same name."""

    async def run(
        self, ctx: GraphRunContext[PanelState, PanelDeps]
    ) -> "PanelSynthesisPhase":
        profiles = list(ctx.state.expert_profiles)
        ctx.state.reviews_attempted = len(profiles)
        if not profiles:
            logger.warning(
                "[panel] no expert profiles — skipping reviews and synthesis"
            )
            return PanelSynthesisPhase()

        ctx.state.expert_reviews = await review_experts(
            profiles,
            ctx.state.source,
            ctx.state.sections,
            model=ctx.deps.agent_model,
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
            return End(_result(ctx))

        # On success the failure note is "" — renderers prefer
        # panel_synthesis.review_summary; a blank top-level summary keeps
        # neither from double-rendering.
        ctx.state.panel_synthesis, ctx.state.summary = await synthesise_panel(
            reviews, model=ctx.deps.agent_model
        )
        return End(_result(ctx))


def _result(ctx: GraphRunContext[PanelState, PanelDeps]) -> ReviewResult:
    """Assemble the End payload from the surfaces (thin adapter — the
    real assembly + degradation gate live in ``build_panel_result``)."""
    from .._metrics import current as current_counters

    counters = current_counters()
    return build_panel_result(
        summary=ctx.state.summary,
        sections_processed=len(ctx.state.sections),
        expert_profiles=ctx.state.expert_profiles,
        expert_reviews=ctx.state.expert_reviews,
        panel_synthesis=ctx.state.panel_synthesis,
        profiles_attempted=ctx.state.profiles_attempted,
        reviews_attempted=ctx.state.reviews_attempted,
        llm_calls=counters.llm_calls if counters else 0,
        soft_failure_threshold=ctx.deps.soft_failure_threshold,
    )


# ── Graph + public entry ───────────────────────────────────────────────────


# linear pipeline — topology test exempt
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
    confirm_own_draft: bool = False,
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
    confirm_own_draft:
        Panel mode simulates peer review, which makes it tempting to point
        at material shared with you in confidence. Like `review_document`,
        this entry point refuses to run on text carrying a confidentiality
        marker unless the caller affirms the draft is their own. The
        `andamentum-whetstone panel` CLI maps `--i-am-the-author` here.
    """
    if not confirm_own_draft:
        from andamentum.whetstone._confidentiality import check_confidentiality

        check_confidentiality(markdown)

    # Per-run metric counters live in the shared v3 contextvar; the
    # panel workers bump them after each agent run. Read at End time.
    from .._metrics import start_run

    start_run()

    deps = PanelDeps(
        agent_model=model,
        n_experts=n_experts,
        panel_disciplines=list(panel_disciplines) if panel_disciplines else [],
    )
    state = PanelState(source=markdown)
    result = await panel_graph_v3.run(Sectionize(), state=state, deps=deps)
    return result.output
