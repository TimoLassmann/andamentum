"""The forge meta-pipeline: a brief becomes a recipe-validated, rendered agentic system.

This is the orchestration file — the *only* engine-aware layer (dialect Law 2). Every
step is thin: read the surfaces, call one engine-free worker, assign the result, return
a typed successor. The shape:

    Understand → Frame → Decompose → Compile ─┬─ dest? → Render → Verify → Finish → End
                                              └─ design-only ───────────→ Finish → End

The one branch (Compile → Render | Finish) routes on ``deps.dest`` — an
operator-trusted predicate, never model output (Law 4). Caps are module constants
(Law 5). The model never drives flow (Law 6): the heads only fill ``why`` / ``areas`` /
the typed board; the spine compiles, renders, and verifies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from .agents import AgentSink, CoreAgentSink
from .compile_spec import compile_spec
from .decompose import decompose
from .frame import frame
from .render import render
from .schemas import DesignPlan, ForgeResult, ForgeWhy, VerificationReport
from .spec import SystemSpec
from .understand import understand
from .verify import verify_package

# Fan-out bounds — the recipe keeps a system "well under 20 steps" (Law 5).
MAX_AREAS = 5
MAX_JOBS_PER_AREA = 5
MAX_NODES = 18


@dataclass(frozen=True)
class ForgeDeps:
    """Injected, never mutated mid-run: the model handle, the agent Port, the output
    destination, and the design caps."""

    model: str
    sink: AgentSink
    dest: Path | None = None
    max_areas: int = MAX_AREAS
    max_jobs_per_area: int = MAX_JOBS_PER_AREA
    max_nodes: int = MAX_NODES


@dataclass
class ForgeState:
    """Run-scoped record of the design as it is produced."""

    # ── inputs
    brief: str
    # ── artifacts (T | None until produced)
    why: ForgeWhy | None = None
    areas: list[str] = field(default_factory=list)
    plan: DesignPlan | None = None
    spec: SystemSpec | None = None
    rendered_files: list[str] = field(default_factory=list)
    report: VerificationReport | None = None
    # ── flow-control
    notes: list[str] = field(default_factory=list)


Ctx = GraphRunContext[ForgeState, ForgeDeps]


@dataclass
class Understand(BaseNode[ForgeState, ForgeDeps, ForgeResult]):
    """Restate the brief as a problem — purpose and boundaries."""

    async def run(self, ctx: Ctx) -> Frame:
        ctx.state.why = await understand(ctx.state.brief, sink=ctx.deps.sink)
        return Frame()


@dataclass
class Frame(BaseNode[ForgeState, ForgeDeps, ForgeResult]):
    """Frame the problem into its 2–4 big concerns."""

    async def run(self, ctx: Ctx) -> Decompose:
        why = ctx.state.why
        assert why is not None  # topology guarantees Understand ran first
        areas, notes = await frame(
            why, sink=ctx.deps.sink, max_areas=ctx.deps.max_areas
        )
        ctx.state.areas = areas
        ctx.state.notes.extend(notes)
        return Decompose()


@dataclass
class Decompose(BaseNode[ForgeState, ForgeDeps, ForgeResult]):
    """Decompose each area into a fully-typed node board."""

    async def run(self, ctx: Ctx) -> Compile:
        why = ctx.state.why
        assert why is not None
        plan, notes = await decompose(
            why,
            ctx.state.areas,
            sink=ctx.deps.sink,
            max_jobs_per_area=ctx.deps.max_jobs_per_area,
            max_nodes=ctx.deps.max_nodes,
        )
        ctx.state.plan = plan
        ctx.state.notes.extend(notes)
        return Compile()


@dataclass
class Compile(BaseNode[ForgeState, ForgeDeps, ForgeResult]):
    """Compile the board into a recipe-validated SystemSpec (deterministic).

    Branches on whether a destination was given: render the package, or finish with
    the spec alone (design-only mode).
    """

    async def run(self, ctx: Ctx) -> Render | Finish:
        plan = ctx.state.plan
        assert plan is not None
        ctx.state.spec = compile_spec(plan)
        return Render() if ctx.deps.dest is not None else Finish()


@dataclass
class Render(BaseNode[ForgeState, ForgeDeps, ForgeResult]):
    """Render the spec into a runnable package (deterministic, no LLM)."""

    async def run(self, ctx: Ctx) -> Verify:
        dest = ctx.deps.dest
        spec = ctx.state.spec
        assert dest is not None and spec is not None
        paths = render(spec, dest)
        ctx.state.rendered_files = [str(p) for p in paths]
        return Verify()


@dataclass
class Verify(BaseNode[ForgeState, ForgeDeps, ForgeResult]):
    """Verify the rendered package: parses, imports, assembles, re-validates."""

    async def run(self, ctx: Ctx) -> Finish:
        dest = ctx.deps.dest
        spec = ctx.state.spec
        assert dest is not None and spec is not None
        ctx.state.report = verify_package(spec, dest)
        return Finish()


@dataclass
class Finish(BaseNode[ForgeState, ForgeDeps, ForgeResult]):
    """Assemble the ForgeResult (Assembly law: deterministic code builds the object)."""

    async def run(self, ctx: Ctx) -> End[ForgeResult]:
        spec = ctx.state.spec
        assert spec is not None
        return End(
            ForgeResult(
                spec=spec,
                design_only=ctx.deps.dest is None,
                rendered_files=ctx.state.rendered_files,
                report=ctx.state.report,
                notes=ctx.state.notes,
            )
        )


graph = Graph(nodes=[Understand, Frame, Decompose, Compile, Render, Verify, Finish])


async def run_forge(
    brief: str,
    *,
    model: str,
    dest: Path | None = None,
    sink: AgentSink | None = None,
    max_areas: int = MAX_AREAS,
    max_jobs_per_area: int = MAX_JOBS_PER_AREA,
    max_nodes: int = MAX_NODES,
) -> ForgeResult:
    """Design (and optionally render + verify) an agentic system from a brief.

    Validates the brief at the door (Input law), builds State + Deps, runs the graph
    to ``End``, and returns the ``ForgeResult``. ``sink`` is the agent Port — left
    ``None`` it is built from ``model`` via ``core.AgentRunner``; tests pass a stub.
    When ``dest`` is ``None`` the result is design-only (no package rendered).
    """
    if not brief or not brief.strip():
        raise ValueError("brief must not be blank")

    resolved_sink: AgentSink = sink if sink is not None else CoreAgentSink(model)
    deps = ForgeDeps(
        model=model,
        sink=resolved_sink,
        dest=dest,
        max_areas=max_areas,
        max_jobs_per_area=max_jobs_per_area,
        max_nodes=max_nodes,
    )
    out = await graph.run(Understand(), state=ForgeState(brief=brief), deps=deps)
    return out.output
