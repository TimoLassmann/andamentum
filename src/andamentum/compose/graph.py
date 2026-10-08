"""The compose orchestrator: a pydantic-graph state machine over a bundle.

    Ingest -> CompileBrief -> Research -> Evidence -> Plan -> Draft -> Check
        Research -> Evidence      skipped unless the run was started with --research
        Check -> Finish           goal met twice running, or the round cap
        Check -> Check            goal met once: re-check the same text
        Check -> Revise -> Check  revise the sections the checks named
        Revise -> Finish          no progress
    Draft -> Finish               nothing could be drafted (unresolvable)

The loop is bounded by ``ComposeDeps.max_rounds`` (dialect L5). The bundle on
disk is the durable store (L1): every node saves what it produced, and a node
whose output is already there is skipped, so a resumed run continues where the
last one stopped. Nodes only route and assign state; the work is in workers.

The goal rule is the whetstone goal loop's: a round that meets the goal counts
only when the next check pass over the same text meets it too. Grounding is
cached by section hash, so the second pass pays only for the reader and critic
calls, which are sampled and can disagree with themselves.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_graph import BaseNode, End, GraphBuilder, GraphRunContext

from andamentum.compose.bundle import Bundle

from . import store
from .assess import Assessment, assess
from .brief import compile_brief
from .checks.deterministic import to_fill_items
from .checks.grounding import Unsupported
from .evidence import extract_facts, ingest_sources, is_tabular, stored_source_path
from .genres import Genre, genre_for
from .log import ActionLog
from .plan import MAX_OUTLINE_AMENDMENTS, amend_outline, make_outline
from .render import ProvenanceRequired, render_outputs
from .research import (
    RESEARCH_DB,
    ResearchSetup,
    ResearchSetupError,
    degraded_notes,
    research_facts,
    research_sources,
    run_research_v2,
)
from .revise import revise_section
from .schemas import (
    Brief,
    ComposeResult,
    Fact,
    Outline,
    RoundRecord,
    Roles,
    Source,
    StopReason,
)
from .steps import (
    add_figure,
    call_counts,
    document_hash,
    first_sentences,
    keep_figures,
    plan_for,
    sync_sections,
    write_report,
)
from .write import draft_section

if TYPE_CHECKING:
    from andamentum.deep_research.v2 import ResearchResult

DEFAULT_MAX_ROUNDS = 4
GOAL_STREAK = 2  # consecutive passing check passes the goal needs


@dataclass(frozen=True)
class ComposeDeps:
    """What a run is given."""

    roles: Roles
    bundle_dir: Path
    task: str
    genre_spec: str | None  # a pack name or YAML path; None = the brief picks
    source_paths: tuple[str, ...]
    log: ActionLog
    progress: Callable[[str], None]
    max_rounds: int = DEFAULT_MAX_ROUNDS
    research: ResearchSetup | None = None
    research_requested: bool = False  # the bundle was started with --research
    confirm_own_sources: bool = False
    provenance: bool = True
    formats: tuple[str, ...] | None = None


@dataclass
class ComposeState:
    """The run's artefacts. Written only by nodes."""

    brief: Brief | None = None
    genre: Genre | None = None
    sources: list[Source] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    research: ResearchResult | None = None
    outline: Outline | None = None
    rounds: list[RoundRecord] = field(default_factory=list)
    first_round: int = 0  # rounds already on disk when this run started
    hashes: list[str] = field(default_factory=list)
    grounding_cache: dict[str, list[Unsupported]] = field(default_factory=dict)
    last: Assessment | None = None
    streak: int = 0
    amendments: int = 0
    open_questions: list[str] = field(default_factory=list)
    stop_reason: StopReason | None = None


Ctx = GraphRunContext[ComposeState, ComposeDeps]


def _note(state: ComposeState, text: str) -> None:
    if text not in state.open_questions:
        state.open_questions.append(text)


def require_provenance(genre: Genre, *, provenance: bool) -> None:
    """Refuse early, before any model call, what render would refuse at the end."""
    if not provenance and genre.provenance_required:
        raise ProvenanceRequired(
            f"genre {genre.name!r} requires provenance markers; "
            "they cannot be switched off"
        )


def _tabular(state: ComposeState) -> list[Source]:
    return [s for s in state.sources if is_tabular(s.path)]


def _text_sources(state: ComposeState) -> list[Source]:
    return [s for s in state.sources if not is_tabular(s.path)]


def _need(state: ComposeState) -> tuple[Brief, Genre]:
    assert state.brief is not None and state.genre is not None  # topology
    return state.brief, state.genre


@dataclass
class Ingest(BaseNode[ComposeState, ComposeDeps, ComposeResult]):
    """Harvest every source into the bundle (skipped when already done)."""

    async def run(self, ctx: Ctx) -> CompileBrief:
        deps, state = ctx.deps, ctx.state
        if store.has(deps.bundle_dir, store.SOURCES):
            state.sources = store.load_sources(deps.bundle_dir)
            return CompileBrief()
        paths = list(deps.source_paths)
        deps.progress(f"Ingest: {len(paths)} source(s)")
        sources = await ingest_sources(
            paths,
            bundle_dir=deps.bundle_dir,
            confirm_own_sources=deps.confirm_own_sources,
        )
        state.sources = sources
        store.save_sources(deps.bundle_dir, sources)
        return CompileBrief()


@dataclass
class CompileBrief(BaseNode[ComposeState, ComposeDeps, ComposeResult]):
    """Compile the task into the frozen brief (skipped when brief.json exists)."""

    async def run(self, ctx: Ctx) -> Research:
        deps, state = ctx.deps, ctx.state
        bundle = Bundle.open(deps.bundle_dir)
        if store.has(deps.bundle_dir, store.BRIEF):
            state.brief = store.load_brief(deps.bundle_dir)
        else:
            deps.progress("Brief: compiling the task")
            source_text = {
                s.id: stored_source_path(deps.bundle_dir, s).read_text(encoding="utf-8")
                for s in _text_sources(state)
            }
            preset = genre_for(deps.genre_spec) if deps.genre_spec else None
            state.brief = await compile_brief(
                deps.task,
                genre=preset,
                sources=state.sources,
                source_text=source_text,
                model=deps.roles.drafter,
                log=deps.log,
            )
            store.save_brief(deps.bundle_dir, state.brief)
            bundle.set_title(state.brief.title)
        spec = deps.genre_spec or str(bundle.metadata.get("genre") or state.brief.genre)
        state.genre = genre_for(spec)
        if bundle.metadata.get("genre") != spec:
            bundle.set_metadata(genre=spec)
        require_provenance(state.genre, provenance=deps.provenance)
        return Research()


@dataclass
class Research(BaseNode[ComposeState, ComposeDeps, ComposeResult]):
    """Run research once, in the genre's mode, and add its works as sources.

    Skipped unless the bundle was started with ``--research``. The research
    result is saved whole (``evidence/research.json``), so a resumed run reuses
    it; each work with a finding becomes a source whose text is the passages
    the findings quote (see ``research.py``).
    """

    async def run(self, ctx: Ctx) -> Evidence:
        deps, state = ctx.deps, ctx.state
        if not deps.research_requested:
            return Evidence()
        _brief, genre = _need(state)
        if store.has(deps.bundle_dir, store.RESEARCH):
            result = store.load_research(deps.bundle_dir)
        else:
            if deps.research is None:
                raise ResearchSetupError(
                    "this bundle was started with --research and the research did "
                    "not finish; give the research settings again to continue"
                )
            result = await run_research_v2(
                deps.task,
                genre=genre,
                setup=deps.research,
                model=deps.roles.drafter,
                database=deps.bundle_dir / RESEARCH_DB,
                progress=deps.progress,
            )
            store.save_research(deps.bundle_dir, result)
            deps.log.call(
                node="Research",
                role="drafter",
                model=deps.roles.drafter,
                prompt="deep_research v2",
                input_text=deps.task,
                output_text=result.run_id,
                input_tokens=result.stats.input_tokens,
                output_tokens=result.stats.output_tokens,
            )
            deps.log.event(
                node="Research",
                event="research",
                run_id=result.run_id,
                mode=result.mode,
                works=len(result.works),
                findings=len(result.findings),
                stop_reason=result.stop_reason,
                degraded=len(result.degraded),
            )
        state.research = result
        for note in degraded_notes(result):
            _note(state, note)
        if not any(s.kind == "research" and s.work_id for s in state.sources):
            added = research_sources(
                result,
                bundle_dir=deps.bundle_dir,
                first_number=len(state.sources) + 1,
            )
            state.sources = [*state.sources, *added]
            store.save_sources(deps.bundle_dir, state.sources)
            deps.progress(f"Research: {len(added)} source(s) with findings")
        return Evidence()


@dataclass
class Evidence(BaseNode[ComposeState, ComposeDeps, ComposeResult]):
    """Extract quote-checked facts, one source at a time (skipped when saved)."""

    async def run(self, ctx: Ctx) -> Plan:
        deps, state = ctx.deps, ctx.state
        brief, genre = _need(state)
        if store.has(deps.bundle_dir, store.FACTS):
            state.facts = store.load_facts(deps.bundle_dir)
        else:
            facts: list[Fact] = []
            for source in _text_sources(state):
                deps.progress(f"Evidence: reading {source.title}")
                text = stored_source_path(deps.bundle_dir, source).read_text(
                    encoding="utf-8"
                )
                if source.kind == "research" and source.work_id:
                    assert state.research is not None  # Research saved the sources
                    found, dropped = research_facts(
                        state.research,
                        source,
                        text,
                        first_number=len(facts) + 1,
                    )
                    for note in dropped:
                        deps.log.event(
                            node="evidence", event="finding_dropped", note=note
                        )
                    facts += found
                    continue
                facts += await extract_facts(
                    source,
                    text,
                    brief=brief,
                    first_number=len(facts) + 1,
                    model=deps.roles.drafter,
                    log=deps.log,
                )
            state.facts = facts
            store.save_facts(deps.bundle_dir, facts)
        if not state.facts and genre.cites:
            _note(
                state,
                "No source facts were available, so claims are unsourced and the "
                "grounding check was skipped.",
            )
        return Plan()


@dataclass
class Plan(BaseNode[ComposeState, ComposeDeps, ComposeResult]):
    """Make the outline and lay out empty sections (skipped when saved)."""

    async def run(self, ctx: Ctx) -> Draft:
        deps, state = ctx.deps, ctx.state
        brief, genre = _need(state)
        if store.has(deps.bundle_dir, store.OUTLINE):
            state.outline = store.load_outline(deps.bundle_dir)
        else:
            deps.progress("Plan: outlining")
            state.outline = await make_outline(
                brief,
                genre,
                state.facts,
                tabular_sources=_tabular(state),
                model=deps.roles.drafter,
                log=deps.log,
            )
            store.save_outline(deps.bundle_dir, state.outline)
        sync_sections(Bundle.open(deps.bundle_dir), state.outline)
        return Draft()


async def _draft_planned(ctx: Ctx) -> int:
    """Draft every section still ``planned``, in order. Returns how many."""
    deps, state = ctx.deps, ctx.state
    brief, genre = _need(state)
    outline = state.outline
    assert outline is not None
    bundle = Bundle.open(deps.bundle_dir)
    done = 0
    for section in bundle.sections():
        if section.status != "planned":
            continue
        deps.progress(f"Draft: {section.title}")
        summaries = [
            f"{s.title}: {first_sentences(s.body)}"
            for s in bundle.sections()
            if s.order < section.order and s.status != "planned"
        ]
        plan = plan_for(outline, title=section.title)
        body = await draft_section(
            plan,
            brief=brief,
            genre=genre,
            outline=outline,
            facts=state.facts,
            previous_summaries=summaries,
            model=deps.roles.drafter,
            log=deps.log,
        )
        bundle.write_section(
            section.id,
            body=body,
            expected_revision=section.revision,
            status="drafted",
        )
        done += 1
        if plan.figure is not None:
            problem = add_figure(
                bundle, section.id, figure=plan.figure, sources=state.sources
            )
            if problem is not None:
                _note(state, problem)
                deps.log.event(node="Draft", event="figure_failed", problem=problem)
    return done


@dataclass
class Draft(BaseNode[ComposeState, ComposeDeps, ComposeResult]):
    """Draft sections one by one (only the ones still planned: resume skips the rest)."""

    async def run(self, ctx: Ctx) -> Check | Finish:
        await _draft_planned(ctx)
        sections = Bundle.open(ctx.deps.bundle_dir).sections()
        if not any(s.body.strip() for s in sections):
            ctx.state.stop_reason = "unresolvable"
            _note(ctx.state, "No section could be drafted.")
            return Finish()
        return Check()


@dataclass
class Check(BaseNode[ComposeState, ComposeDeps, ComposeResult]):
    """Run every check over the current text and decide: stop, re-check or revise."""

    async def run(self, ctx: Ctx) -> Revise | Finish | Check:
        deps, state = ctx.deps, ctx.state
        brief, genre = _need(state)
        bundle = Bundle.open(deps.bundle_dir)
        round_no = len(state.rounds) + 1
        deps.progress(f"Check: round {round_no}")
        result = await assess(
            bundle,
            brief=brief,
            genre=genre,
            outline=state.outline,
            facts=state.facts,
            sources=state.sources,
            roles=deps.roles,
            log=deps.log,
            round_no=round_no,
            grounding_cache=state.grounding_cache,
        )
        state.last = result
        state.hashes.append(document_hash(bundle.sections()))
        state.streak = state.streak + 1 if result.goal_met else 0
        state.rounds.append(
            RoundRecord(
                round=round_no,
                checks=result.checks,
                work=result.routed.items,
                changed_sections=[],
                goal_met=result.goal_met,
            )
        )
        store.save_rounds(deps.bundle_dir, state.rounds)
        failed = [c.name for c in result.checks if not c.passed]
        deps.progress(
            f"Check: round {round_no} "
            + ("goal met" if result.goal_met else f"open: {', '.join(failed)}")
        )
        if state.streak >= GOAL_STREAK:
            state.stop_reason = "goal_met"
            return Finish()
        if len(state.rounds) - state.first_round >= deps.max_rounds:
            state.stop_reason = "cap_reached"
            return Finish()
        return Check() if result.goal_met else Revise()


@dataclass
class Revise(BaseNode[ComposeState, ComposeDeps, ComposeResult]):
    """Rewrite each section the checks named, once; amend the outline if asked."""

    async def run(self, ctx: Ctx) -> Check | Finish:
        deps, state = ctx.deps, ctx.state
        brief, genre = _need(state)
        assert state.last is not None and state.outline is not None  # topology
        routed = state.last.routed
        bundle = Bundle.open(deps.bundle_dir)
        changed: list[str] = []

        if routed.outline:
            if state.amendments < MAX_OUTLINE_AMENDMENTS:
                deps.progress("Revise: amending the outline")
                state.outline = await amend_outline(
                    state.outline,
                    routed.outline,
                    brief=brief,
                    genre=genre,
                    facts=state.facts,
                    tabular_sources=_tabular(state),
                    model=deps.roles.drafter,
                    log=deps.log,
                )
                state.amendments += 1
                store.save_outline(deps.bundle_dir, state.outline)
                sync_sections(bundle, state.outline)
                before = {s.id for s in bundle.sections() if s.status != "planned"}
                await _draft_planned(ctx)
                changed += [
                    s.id
                    for s in Bundle.open(deps.bundle_dir).sections()
                    if s.id not in before
                ]
            else:
                for item in routed.outline:
                    _note(state, f"Needs a structural change: {item.reason}")

        bundle = Bundle.open(deps.bundle_dir)
        for section in bundle.sections():
            items = routed.by_section.get(section.id)
            if not items or section.id in changed:
                continue
            deps.progress(f"Revise: {section.title}")
            new_body = await revise_section(
                section.body,
                items,
                plan=plan_for(state.outline, title=section.title),
                brief=brief,
                genre=genre,
                facts=state.facts,
                model=deps.roles.drafter,
                log=deps.log,
            )
            new_body = keep_figures(section.body, new_body)
            if new_body.strip() != section.body.strip():
                bundle.write_section(
                    section.id,
                    body=new_body,
                    expected_revision=section.revision,
                    status="revised",
                )
                changed.append(section.id)

        state.rounds[-1].changed_sections = changed
        store.save_rounds(deps.bundle_dir, state.rounds)
        new_hash = document_hash(Bundle.open(deps.bundle_dir).sections())
        if not changed or new_hash in state.hashes[-2:]:
            state.stop_reason = "no_progress"
            return Finish()
        return Check()


@dataclass
class Finish(BaseNode[ComposeState, ComposeDeps, ComposeResult]):
    """Render the outputs, write the report, build the result."""

    async def run(self, ctx: Ctx) -> End[ComposeResult]:
        deps, state = ctx.deps, ctx.state
        brief, genre = _need(state)
        stop = state.stop_reason
        assert stop is not None  # every edge into Finish sets it
        bundle = Bundle.open(deps.bundle_dir)
        outputs: list[Path] = []
        if stop != "unresolvable":
            deps.progress("Finish: rendering")
            outputs = render_outputs(
                bundle,
                brief=brief,
                genre=genre,
                facts=state.facts,
                sources=state.sources,
                model=deps.roles.drafter,
                formats=deps.formats,
                provenance=deps.provenance,
            )
        if state.last is not None:
            for text in to_fill_items(state.last.checks):
                _note(state, text)
        deps.log.event(node="Finish", event="stop", stop_reason=stop)
        write_report(
            deps.bundle_dir,
            title=brief.title,
            stop_reason=stop,
            rounds=state.rounds,
            open_questions=state.open_questions,
            roles=deps.roles,
            usage=deps.log.usage(),
            calls=call_counts(deps.log),
            outputs=outputs,
        )
        return End(
            ComposeResult(
                bundle=deps.bundle_dir,
                outputs=outputs,
                goal_met=stop == "goal_met",
                stop_reason=stop,
                rounds=state.rounds,
                open_questions=state.open_questions,
                brief=brief,
                usage=deps.log.usage(),
            )
        )


_builder = GraphBuilder[ComposeState, ComposeDeps, Ingest, ComposeResult](
    state_type=ComposeState,
    deps_type=ComposeDeps,
    input_type=Ingest,
    output_type=ComposeResult,
)
_builder.add(
    _builder.edge_from(_builder.start_node).to(Ingest),
    _builder.node(Ingest),
    _builder.node(CompileBrief),
    _builder.node(Research),
    _builder.node(Evidence),
    _builder.node(Plan),
    _builder.node(Draft),
    _builder.node(Check),
    _builder.node(Revise),
    _builder.node(Finish),
)
graph = _builder.build()


async def run_graph(deps: ComposeDeps) -> ComposeResult:
    """Run (or resume) the pipeline over ``deps.bundle_dir``."""
    rounds = store.load_rounds(deps.bundle_dir)
    state = ComposeState(rounds=rounds, first_round=len(rounds))
    return await graph.run(state=state, deps=deps, inputs=Ingest())
