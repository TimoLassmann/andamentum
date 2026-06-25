"""The forge meta-pipeline: a brief becomes a recipe-validated, built, audited system.

This is the orchestration file — the *only* engine-aware layer (dialect Law 2). Every
step is thin: read the surfaces, call one engine-free worker, assign, return a typed
successor. The whole authoring pipeline, in one graph:

    Understand → Frame → Decompose → Compile → Render → Verify → Build → Audit → Finish → End
        └──── design heads ────┘      det      det      det     agents  sandbox+
                                                                 (static  agents
                                                                  gates)

The branches all route on ``deps.stop_after`` / ``deps.dest`` — operator-trusted
predicates, never model output (Law 4). Caps are Deps fields / module constants (Law 5).
The model never drives flow (Law 6). The sandbox and the agent runner are Deps Ports, so
the whole pipeline runs under stubs with no container and no live model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from .agents import AgentSink, CoreAgentSink
from .audit import audit_system
from .build import build_system
from .compile_spec import compile_spec
from .decompose import decompose
from .frame import frame
from .render import render
from .sandbox import SandboxPort, make_sandbox
from .schemas import (
    AuditReport,
    BuildReport,
    DesignPlan,
    DesignReport,
    ForgeResult,
    ForgeWhy,
    VerificationReport,
)
from .spec import SystemSpec
from .understand import understand
from .verify import verify_package

# Fan-out bounds — the recipe keeps a system "well under 20 steps" (Law 5).
MAX_AREAS = 5
MAX_JOBS_PER_AREA = 5
MAX_NODES = 18
# Per-node authoring: draft + (cap-1) repairs (the prior-art sweet spot).
ATTEMPT_CAP = 3

#: How far the pipeline runs. Each is a superset of the previous.
_STAGES = ("design", "render", "build", "audit")


@dataclass(frozen=True)
class ForgeDeps:
    """Injected, never mutated mid-run: the model handle, the two Ports (agents +
    sandbox), the output destination, how far to run, and the caps."""

    model: str
    sink: AgentSink
    sandbox: SandboxPort
    dest: Path | None = None
    stop_after: str = "audit"
    max_areas: int = MAX_AREAS
    max_jobs_per_area: int = MAX_JOBS_PER_AREA
    max_nodes: int = MAX_NODES
    attempt_cap: int = ATTEMPT_CAP


@dataclass
class ForgeState:
    """Run-scoped record of the design as it is produced."""

    # ── inputs
    brief: str
    # ── artifacts (T | None until produced)
    why: ForgeWhy | None = None
    areas: list[str] = field(default_factory=list)
    plan: DesignPlan | None = None
    design_report: DesignReport | None = None
    spec: SystemSpec | None = None
    rendered_files: list[str] = field(default_factory=list)
    report: VerificationReport | None = None
    build: BuildReport | None = None
    audit: AuditReport | None = None
    # ── flow-control
    notes: list[str] = field(default_factory=list)


Ctx = GraphRunContext[ForgeState, ForgeDeps]


def _wants(deps: ForgeDeps, stage: str) -> bool:
    """True if the run should reach ``stage`` — and a destination exists to render into."""
    return deps.dest is not None and _STAGES.index(deps.stop_after) >= _STAGES.index(
        stage
    )


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
        plan, design_report, notes = await decompose(
            why,
            ctx.state.areas,
            sink=ctx.deps.sink,
            max_jobs_per_area=ctx.deps.max_jobs_per_area,
            max_nodes=ctx.deps.max_nodes,
        )
        ctx.state.plan = plan
        ctx.state.design_report = design_report
        ctx.state.notes.extend(notes)
        return Compile()


@dataclass
class Compile(BaseNode[ForgeState, ForgeDeps, ForgeResult]):
    """Compile the board into a recipe-validated SystemSpec (deterministic).

    Branches on whether a destination was given and the run wants rendering: render the
    package, or finish design-only.
    """

    async def run(self, ctx: Ctx) -> Render | Finish:
        plan = ctx.state.plan
        assert plan is not None
        ctx.state.spec = compile_spec(plan)
        return Render() if _wants(ctx.deps, "render") else Finish()


@dataclass
class Render(BaseNode[ForgeState, ForgeDeps, ForgeResult]):
    """Render the spec into a runnable skeleton package (deterministic, no LLM)."""

    async def run(self, ctx: Ctx) -> Verify:
        dest = ctx.deps.dest
        spec = ctx.state.spec
        assert dest is not None and spec is not None
        paths = render(spec, dest)
        ctx.state.rendered_files = [str(p) for p in paths]
        return Verify()


@dataclass
class Verify(BaseNode[ForgeState, ForgeDeps, ForgeResult]):
    """Cheap deterministic render-stage verdict (parses, imports, assembles, recipe)."""

    async def run(self, ctx: Ctx) -> Build | Finish:
        dest = ctx.deps.dest
        spec = ctx.state.spec
        assert dest is not None and spec is not None
        ctx.state.report = verify_package(spec, dest)
        return Build() if _wants(ctx.deps, "build") else Finish()


@dataclass
class Build(BaseNode[ForgeState, ForgeDeps, ForgeResult]):
    """Agents author every node body, statically gated (stage 3)."""

    async def run(self, ctx: Ctx) -> Audit | Finish:
        dest = ctx.deps.dest
        spec = ctx.state.spec
        assert dest is not None and spec is not None
        ctx.state.build = await build_system(
            spec, dest / spec.name, sink=ctx.deps.sink, attempt_cap=ctx.deps.attempt_cap
        )
        return Audit() if _wants(ctx.deps, "audit") else Finish()


@dataclass
class Audit(BaseNode[ForgeState, ForgeDeps, ForgeResult]):
    """Run the built system in the sandbox and review it end-to-end (stage 4)."""

    async def run(self, ctx: Ctx) -> Finish:
        dest = ctx.deps.dest
        spec = ctx.state.spec
        assert dest is not None and spec is not None
        ctx.state.audit = await audit_system(
            spec,
            ctx.state.brief,
            dest,
            sink=ctx.deps.sink,
            sandbox=ctx.deps.sandbox,
            build=ctx.state.build,
        )
        return Finish()


@dataclass
class Finish(BaseNode[ForgeState, ForgeDeps, ForgeResult]):
    """Assemble the ForgeResult (Assembly law: deterministic code builds the object)."""

    async def run(self, ctx: Ctx) -> End[ForgeResult]:
        spec = ctx.state.spec
        assert spec is not None
        if ctx.state.audit is not None:
            stage = "audit"
        elif ctx.state.build is not None:
            stage = "build"
        elif ctx.state.report is not None:
            stage = "render"
        else:
            stage = "design"
        return End(
            ForgeResult(
                spec=spec,
                stage_reached=stage,
                rendered_files=ctx.state.rendered_files,
                design_report=ctx.state.design_report,
                report=ctx.state.report,
                build=ctx.state.build,
                audit=ctx.state.audit,
                notes=ctx.state.notes,
            )
        )


graph = Graph(
    nodes=[Understand, Frame, Decompose, Compile, Render, Verify, Build, Audit, Finish]
)


async def run_forge(
    brief: str,
    *,
    model: str,
    dest: Path | None = None,
    stop_after: str = "audit",
    sandbox_backend: str = "podman",
    sink: AgentSink | None = None,
    sandbox: SandboxPort | None = None,
    max_areas: int = MAX_AREAS,
    max_jobs_per_area: int = MAX_JOBS_PER_AREA,
    max_nodes: int = MAX_NODES,
    attempt_cap: int = ATTEMPT_CAP,
) -> ForgeResult:
    """Design, render, build, and audit an agentic system from a brief.

    Validates the brief at the door (Input law), builds State + Deps, runs the graph to
    ``End``, and returns the ``ForgeResult``. ``sink`` (agents) and ``sandbox`` (code
    execution) are Deps Ports — left ``None`` they are built from ``model`` /
    ``sandbox_backend``; tests pass stubs. ``stop_after`` ∈ {design, render, build,
    audit} bounds how far the pipeline runs (default the full build). With ``dest=None``
    the run is design-only.
    """
    if not brief or not brief.strip():
        raise ValueError("brief must not be blank")
    if stop_after not in _STAGES:
        raise ValueError(f"stop_after must be one of {_STAGES}, not {stop_after!r}")

    resolved_sink: AgentSink = sink if sink is not None else CoreAgentSink(model)
    resolved_sandbox: SandboxPort = (
        sandbox if sandbox is not None else make_sandbox(sandbox_backend)
    )
    deps = ForgeDeps(
        model=model,
        sink=resolved_sink,
        sandbox=resolved_sandbox,
        dest=dest,
        stop_after=stop_after,
        max_areas=max_areas,
        max_jobs_per_area=max_jobs_per_area,
        max_nodes=max_nodes,
        attempt_cap=attempt_cap,
    )
    out = await graph.run(Understand(), state=ForgeState(brief=brief), deps=deps)
    return out.output
