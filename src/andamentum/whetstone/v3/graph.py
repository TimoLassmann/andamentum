"""The v3 review graph — clean pydantic-graph, one job per node.

Linear flow with the gap loop encapsulated in its own node:

    Sectionize(D) → ExtractClaims(A) → BuildModel(D) → ReviewCriteria(A)
      → VerifyFindings(D) → GapLoop(A/D) → Gate(D) → Synthesise(A)
      → CritiqueRevise(A) → Finalize(D) → End[ReviewResult]

Each node wraps a tested phase function; the deterministic/agent split is kept
at the node level. Entry point: `run_review_v3(markdown, *, model=…)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from ..schemas import ReviewResult
from .criteria import Criterion, criterion_set_for
from .extract import build_claims
from .digest import build_document_model
from .gaps import gap_loop
from .gate import gate_and_aggregate
from .model import Claim, DocumentModel, Section
from .review import Finding, run_criteria, verify_findings
from .sectionize import sectionize
from .synth import StructuredReview, critique_and_revise, synthesise, to_review_result


@dataclass
class V3Deps:
    agent_model: str
    cap: int = 2  # gap-loop round cap
    criteria: list[Criterion] = field(
        default_factory=lambda: criterion_set_for("academic")
    )
    raw_text_budget: int = 30_000  # include full text in criterion context if it fits


@dataclass
class V3State:
    source: str
    sections: list[Section] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    document_model: DocumentModel | None = None
    findings: list[Finding] = field(default_factory=list)
    review: StructuredReview | None = None


@dataclass
class Sectionize(BaseNode[V3State, V3Deps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[V3State, V3Deps]) -> "ExtractClaims":
        ctx.state.sections = sectionize(ctx.state.source)
        return ExtractClaims()


@dataclass
class ExtractClaims(BaseNode[V3State, V3Deps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[V3State, V3Deps]) -> "BuildModel":
        ctx.state.claims = await build_claims(
            ctx.state.sections, ctx.state.source, model=ctx.deps.agent_model
        )
        return BuildModel()


@dataclass
class BuildModel(BaseNode[V3State, V3Deps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[V3State, V3Deps]) -> "ReviewCriteria":
        ctx.state.document_model = build_document_model(
            ctx.state.source, ctx.state.sections, ctx.state.claims
        )
        return ReviewCriteria()


@dataclass
class ReviewCriteria(BaseNode[V3State, V3Deps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[V3State, V3Deps]) -> "VerifyFindings":
        model = ctx.state.document_model
        assert model is not None
        full_text = (
            ctx.state.source
            if len(ctx.state.source) <= ctx.deps.raw_text_budget
            else None
        )
        ctx.state.findings = await run_criteria(
            ctx.deps.criteria,
            model,
            agent_model=ctx.deps.agent_model,
            full_text=full_text,
        )
        return VerifyFindings()


@dataclass
class VerifyFindings(BaseNode[V3State, V3Deps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[V3State, V3Deps]) -> "GapLoop":
        model = ctx.state.document_model
        assert model is not None
        ctx.state.findings = verify_findings(ctx.state.findings, model)
        return GapLoop()


@dataclass
class GapLoop(BaseNode[V3State, V3Deps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[V3State, V3Deps]) -> "Gate":
        model = ctx.state.document_model
        assert model is not None
        ctx.state.findings = await gap_loop(
            model,
            ctx.state.findings,
            agent_model=ctx.deps.agent_model,
            cap=ctx.deps.cap,
        )
        return Gate()


@dataclass
class Gate(BaseNode[V3State, V3Deps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[V3State, V3Deps]) -> "Synthesise":
        ctx.state.findings = gate_and_aggregate(ctx.state.findings)
        return Synthesise()


@dataclass
class Synthesise(BaseNode[V3State, V3Deps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[V3State, V3Deps]) -> "CritiqueRevise":
        model = ctx.state.document_model
        assert model is not None
        ctx.state.review = await synthesise(
            model, ctx.state.findings, agent_model=ctx.deps.agent_model
        )
        return CritiqueRevise()


@dataclass
class CritiqueRevise(BaseNode[V3State, V3Deps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[V3State, V3Deps]) -> "Finalize":
        model = ctx.state.document_model
        assert model is not None and ctx.state.review is not None
        ctx.state.review = await critique_and_revise(
            model, ctx.state.review, agent_model=ctx.deps.agent_model
        )
        return Finalize()


@dataclass
class Finalize(BaseNode[V3State, V3Deps, ReviewResult]):
    async def run(self, ctx: GraphRunContext[V3State, V3Deps]) -> End[ReviewResult]:
        model = ctx.state.document_model
        assert model is not None and ctx.state.review is not None
        return End(to_review_result(model, ctx.state.findings, ctx.state.review))


review_graph_v3 = Graph(
    nodes=[
        Sectionize,
        ExtractClaims,
        BuildModel,
        ReviewCriteria,
        VerifyFindings,
        GapLoop,
        Gate,
        Synthesise,
        CritiqueRevise,
        Finalize,
    ]
)


async def run_review_v3(
    markdown: str,
    *,
    model: str,
    cap: int = 2,
    document_type: str = "academic",
) -> ReviewResult:
    """Run the v3 review over already-harvested markdown. Returns a
    `ReviewResult` the existing renderers consume unchanged."""
    deps = V3Deps(agent_model=model, cap=cap, criteria=criterion_set_for(document_type))
    state = V3State(source=markdown)
    result = await review_graph_v3.run(Sectionize(), state=state, deps=deps)
    return result.output


async def review_document_v3(
    source: str,
    *,
    model: str,
    cap: int = 2,
    document_type: str = "academic",
) -> ReviewResult:
    """v3 entry from a source path/URL: harvest → markdown → review.

    Opt-in alongside the v2 `review_document`; v2 remains the default until v3
    is validated. Raw markdown can be passed directly to `run_review_v3`.
    """
    from pathlib import Path

    from andamentum.harvest import extract

    md = await extract(Path(source)) if Path(source).exists() else source
    return await run_review_v3(md, model=model, cap=cap, document_type=document_type)
