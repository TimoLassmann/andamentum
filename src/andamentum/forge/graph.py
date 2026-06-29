"""The forge meta-pipeline: a brief becomes a recipe-validated, built, audited system.

This is the orchestration file — the *only* engine-aware layer (dialect Law 2). Every
step is thin: read the surfaces, call one engine-free worker, assign, return a typed
successor. The whole authoring pipeline, in one graph:

    Understand → Frame → Decompose → Compile → Review → Render → Verify → Build → Audit → Finish → End
        └──── design heads ────┘      det     plan-mgr   det      det     agents  sandbox+
                                               (⇄ Frame)                  (static  agents
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
from .reporter import ForgeReporter, NoopReporter
from .sandbox import SandboxPort, make_sandbox
from .review import plan_board, review_plan
from .schemas import (
    AuditReport,
    BuildReport,
    DesignPlan,
    DesignReport,
    ForgeResult,
    ForgeWhy,
    PlanVerdict,
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
# Plan-manager feedback loop: the design is redesigned at most this many times before the
# pipeline fails loud with the unresolved concerns (Law 5 — bound is a named constant).
MAX_PLAN_REVIEW_ROUNDS = 2

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
    reporter: ForgeReporter = field(default_factory=NoopReporter)


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
    plan_review: PlanVerdict | None = None
    spec: SystemSpec | None = None
    rendered_files: list[str] = field(default_factory=list)
    report: VerificationReport | None = None
    build: BuildReport | None = None
    audit: AuditReport | None = None
    # ── flow-control
    notes: list[str] = field(default_factory=list)
    plan_review_rounds: int = 0
    plan_feedback: list[str] = field(default_factory=list)


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
            why,
            sink=ctx.deps.sink,
            max_areas=ctx.deps.max_areas,
            plan_feedback=ctx.state.plan_feedback,
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
            plan_feedback=ctx.state.plan_feedback,
        )
        ctx.state.plan = plan
        ctx.state.design_report = design_report
        ctx.state.notes.extend(notes)
        return Compile()


@dataclass
class Compile(BaseNode[ForgeState, ForgeDeps, ForgeResult]):
    """Compile the board into a recipe-validated SystemSpec (deterministic).

    Single-successor: the plan must pass the plan manager (Review) before any render.
    """

    async def run(self, ctx: Ctx) -> Review:
        plan = ctx.state.plan
        assert plan is not None
        ctx.state.spec = compile_spec(plan)
        return Review()


@dataclass
class Review(BaseNode[ForgeState, ForgeDeps, ForgeResult]):
    """Plan manager (Tier 1b): does the planned board serve the goal?

    On serves_goal (or no surviving concerns) → proceed (Render if the run wants it, else
    Finish). On reject → loop back to Frame carrying the concerns, bounded by
    MAX_PLAN_REVIEW_ROUNDS; at the cap, fail loud with the unresolved concerns (never a
    silent pass). The Review→Frame back-edge is a declared, cap-bounded cycle (L5).
    """

    async def run(self, ctx: Ctx) -> Frame | Render | Finish:
        plan = ctx.state.plan
        assert plan is not None
        board = plan_board(plan.nodes)
        jobs = [n.job for n in plan.nodes]
        verdict = await review_plan(plan.why, board, jobs, sink=ctx.deps.sink)

        if verdict.serves_goal or not verdict.uncovered_concerns:
            ctx.state.plan_review = verdict
            return Render() if _wants(ctx.deps, "render") else Finish()

        ctx.state.plan_review_rounds += 1
        if ctx.state.plan_review_rounds >= MAX_PLAN_REVIEW_ROUNDS:
            raise ValueError(
                "plan review did not converge after "
                f"{MAX_PLAN_REVIEW_ROUNDS} redesign rounds; the plan still does not serve "
                "the goal (surfaced, never silently passed). Unresolved concerns:\n"
                + "\n".join(f"- {c}" for c in verdict.uncovered_concerns)
            )
        ctx.state.plan_feedback = verdict.uncovered_concerns
        return Frame()


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
            spec,
            dest / spec.name,
            sink=ctx.deps.sink,
            attempt_cap=ctx.deps.attempt_cap,
            reporter=ctx.deps.reporter,
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
            reporter=ctx.deps.reporter,
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
                plan_review=ctx.state.plan_review,
                report=ctx.state.report,
                build=ctx.state.build,
                audit=ctx.state.audit,
                notes=ctx.state.notes,
            )
        )


graph = Graph(
    nodes=[
        Understand,
        Frame,
        Decompose,
        Compile,
        Review,
        Render,
        Verify,
        Build,
        Audit,
        Finish,
    ]
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
    reporter: ForgeReporter | None = None,
) -> ForgeResult:
    """Design, render, build, and audit an agentic system from a brief.

    Validates the brief at the door (Input law), builds State + Deps, runs the graph to
    ``End``, and returns the ``ForgeResult``. ``sink`` (agents) and ``sandbox`` (code
    execution) are Deps Ports — left ``None`` they are built from ``model`` /
    ``sandbox_backend``; tests pass stubs. ``stop_after`` ∈ {design, render, build,
    audit} bounds how far the pipeline runs (default the full build). With ``dest=None``
    the run is design-only. ``reporter`` (a Port, default silent) receives progress events
    as each stage runs — the CLI installs a live dashboard on ``--verbose``.
    """
    if not brief or not brief.strip():
        raise ValueError("brief must not be blank")
    if stop_after not in _STAGES:
        raise ValueError(f"stop_after must be one of {_STAGES}, not {stop_after!r}")

    resolved_sink: AgentSink = sink if sink is not None else CoreAgentSink(model)
    resolved_sandbox: SandboxPort = (
        sandbox if sandbox is not None else make_sandbox(sandbox_backend)
    )
    resolved_reporter: ForgeReporter = (
        reporter if reporter is not None else NoopReporter()
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
        reporter=resolved_reporter,
    )
    state = ForgeState(brief=brief)
    planned = _planned_stages(deps)
    planned_set = set(planned)
    resolved_reporter.planned(stages=planned)

    # Drive the graph node-by-node (Law 2: still all engine-aware code, in this one file)
    # so the reporter can light each stage as it runs and read its one-line summary off the
    # state once it finishes. graph.iter yields each node *before* it runs; by the time the
    # next node is yielded the previous one has completed and the state reflects its work.
    # Only the displayed stages emit events — internal nodes (Finish) are not stages.
    current = ""
    prev_name = ""
    try:
        async with graph.iter(Understand(), state=state, deps=deps) as run:
            async for node in run:
                if prev_name:
                    resolved_reporter.stage_finished(
                        name=prev_name, detail=_stage_detail(prev_name, state)
                    )
                    prev_name = ""
                if isinstance(node, End):
                    break
                current = type(node).__name__
                if current in planned_set:
                    resolved_reporter.stage_started(name=current)
                    prev_name = current
        result = run.result
        assert result is not None
    except Exception as exc:  # surface which stage failed, then re-raise (fail loud)
        resolved_reporter.stage_failed(name=current, error=str(exc))
        raise
    out = result.output
    resolved_reporter.run_finished(works=out.works, stage_reached=out.stage_reached)
    return out


def _planned_stages(deps: ForgeDeps) -> list[str]:
    """The stages this run will visit, in order — for the reporter's checklist. Mirrors the
    node routing (Review→Render→Verify→Build→Audit, each gated by ``_wants``)."""
    stages = ["Understand", "Frame", "Decompose", "Compile", "Review"]
    if _wants(deps, "render"):
        stages += ["Render", "Verify"]
    if _wants(deps, "build"):
        stages.append("Build")
    if _wants(deps, "audit"):
        stages.append("Audit")
    return stages


def _stage_detail(name: str, state: ForgeState) -> str:
    """The one-line summary shown next to a finished stage, read off the run state."""
    if name == "Understand":
        return "purpose + boundaries" if state.why is not None else ""
    if name == "Frame":
        return f"{len(state.areas)} concern(s): " + " · ".join(state.areas)
    if name == "Decompose" and state.plan is not None:
        n = len(state.plan.nodes)
        rep = state.design_report
        tail = "clean" if rep is None or rep.clean else f"{len(rep.findings)} findings"
        return f"{n} steps · {tail}"
    if name == "Compile" and state.spec is not None:
        s = state.spec
        heads = sum(1 for nd in s.nodes if nd.kind.value == "head")
        return f"{len(s.nodes)} nodes ({heads} head · {len(s.nodes) - heads} spine) · {len(s.agents)} agents"
    if name == "Review" and state.plan_review is not None:
        pr = state.plan_review
        return (
            "plan manager: serves the goal ✓"
            if pr.serves_goal
            else f"{len(pr.uncovered_concerns)} concern(s) — redesigning"
        )
    if name == "Render":
        return f"{len(state.rendered_files)} files"
    if name == "Verify" and state.report is not None:
        r = state.report
        passed = sum(1 for c in r.checks if c.passed)
        return f"{passed}/{len(r.checks)} checks"
    if name == "Build" and state.build is not None:
        b = state.build
        extra = f" · {len(b.concerns)} concern(s)" if b.concerns else ""
        return f"{len(b.filled)} authored · {len(b.unfillable)} unfillable{extra}"
    if name == "Audit" and state.audit is not None:
        return "WORKS" if state.audit.works else "INCOMPLETE"
    return ""
