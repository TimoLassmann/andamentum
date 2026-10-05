"""Public entry points: ``write_document`` and ``compose_bundle``.

A run is a function of its inputs: a task (and sources) in, a finished document
in a bundle folder out. Nothing is sent anywhere.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from andamentum.compose.bundle import Bundle

from . import store
from .assess import Assessment, assess
from .genres import genre_for
from .graph import (
    DEFAULT_MAX_ROUNDS,
    ComposeDeps,
    require_provenance,
    run_graph,
)
from .log import ActionLog
from .render import render_outputs
from .research import ResearchSetup
from .schemas import ComposeResult, Roles

VALID_FORMATS = ("docx", "md", "html", "txt")
TITLE_CHARS = 60


def _quiet(_line: str) -> None:
    return None


def _check_formats(formats: Sequence[str] | None) -> tuple[str, ...] | None:
    if formats is None:
        return None
    for fmt in formats:
        if fmt not in VALID_FORMATS:
            raise ValueError(
                f"unknown output format {fmt!r}; valid: {', '.join(VALID_FORMATS)}"
            )
    return tuple(formats)


def _check_rounds(max_rounds: int) -> None:
    if max_rounds < 1:
        raise ValueError(f"max_rounds must be at least 1, got {max_rounds}")


def _source_ref(source: str) -> str:
    """Store local files as absolute paths so a resume works from any directory."""
    path = Path(source)
    return str(path.resolve()) if path.exists() else source


def _genre_ref(genre: str) -> str:
    path = Path(genre)
    return str(path.resolve()) if path.suffix.lower() in {".yaml", ".yml"} else genre


def _deps(
    bundle: Bundle,
    *,
    roles: Roles,
    max_rounds: int,
    confirm_own_sources: bool,
    provenance: bool,
    formats: tuple[str, ...] | None,
    progress: Callable[[str], None],
    research: ResearchSetup | None,
) -> ComposeDeps:
    meta = bundle.metadata
    genre = meta.get("genre")
    sources = meta.get("sources")
    return ComposeDeps(
        roles=roles,
        bundle_dir=bundle.path,
        task=str(meta.get("task", "")),
        genre_spec=str(genre) if genre else None,
        source_paths=tuple(str(p) for p in sources)
        if isinstance(sources, list)
        else (),
        log=ActionLog(bundle.path),
        progress=progress,
        max_rounds=max_rounds,
        research=research,
        research_requested=bool(meta.get("research", False)),
        confirm_own_sources=confirm_own_sources,
        provenance=provenance,
        formats=formats,
    )


async def write_document(
    task: str,
    *,
    out: str | Path,
    roles: Roles,
    genre: str | None = None,
    sources: Sequence[str] = (),
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    research: ResearchSetup | None = None,
    confirm_own_sources: bool = False,
    provenance: bool = True,
    formats: Sequence[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> ComposeResult:
    """Write a finished document from a task. No approval step.

    The brief is compiled, frozen and saved in the bundle so it can be
    inspected afterwards. *out* must not exist or must be empty; use
    ``compose_bundle`` to continue an existing bundle. *genre* is a pack name or
    a YAML path; ``None`` lets the brief compiler choose.

    *research* turns on research: ``deep_research.v2`` runs once, in the mode
    the genre names, over ``<out>/research.db``, and its findings (claims with
    verbatim quotes from real passages) become sources and facts. ``None``
    runs no research. See ``compose.research``.
    """
    if not task.strip():
        raise ValueError("task is empty")
    _check_rounds(max_rounds)
    wanted = _check_formats(formats)
    if genre is not None:
        require_provenance(genre_for(genre), provenance=provenance)
    bundle = Bundle.create(
        out,
        title=task.strip()[:TITLE_CHARS],
        metadata={
            "task": task,
            "genre": _genre_ref(genre) if genre else None,
            "sources": [_source_ref(s) for s in sources],
            "research": research is not None,
            "models": roles.model_dump(),
        },
    )
    return await run_graph(
        _deps(
            bundle,
            roles=roles,
            max_rounds=max_rounds,
            confirm_own_sources=confirm_own_sources,
            provenance=provenance,
            formats=wanted,
            progress=progress or _quiet,
            research=research,
        )
    )


async def compose_bundle(
    bundle: str | Path,
    *,
    roles: Roles,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    confirm_own_sources: bool = False,
    provenance: bool = True,
    formats: Sequence[str] | None = None,
    progress: Callable[[str], None] | None = None,
    research: ResearchSetup | None = None,
) -> ComposeResult:
    """Continue an existing bundle: finished steps are skipped, checks resume.

    *max_rounds* counts the check rounds of this run, not of earlier runs.
    *research* is needed only when the bundle was started with research and it
    did not finish; a finished research result is reused as it is.
    """
    _check_rounds(max_rounds)
    opened = Bundle.open(bundle)
    opened.set_metadata(models=roles.model_dump())
    return await run_graph(
        _deps(
            opened,
            roles=roles,
            max_rounds=max_rounds,
            confirm_own_sources=confirm_own_sources,
            provenance=provenance,
            formats=_check_formats(formats),
            progress=progress or _quiet,
            research=research,
        )
    )


def recorded_roles(bundle: str | Path) -> Roles | None:
    """The models the bundle's last run used, if it recorded them."""
    models = Bundle.open(bundle).metadata.get("models")
    return Roles.model_validate(models) if models else None


async def check_bundle(
    bundle: str | Path,
    *,
    roles: Roles,
    progress: Callable[[str], None] | None = None,
) -> Assessment:
    """One check pass over a bundle, no revision (``andamentum-compose check``)."""
    opened = Bundle.open(bundle)
    brief = store.load_brief(opened.path)
    genre = genre_for(str(opened.metadata.get("genre") or brief.genre))
    log = ActionLog(opened.path)
    rounds = store.load_rounds(opened.path)
    (progress or _quiet)(f"Check: round {len(rounds) + 1}")
    return await assess(
        opened,
        brief=brief,
        genre=genre,
        outline=store.load_outline(opened.path)
        if store.has(opened.path, store.OUTLINE)
        else None,
        facts=store.load_facts(opened.path)
        if store.has(opened.path, store.FACTS)
        else [],
        sources=store.load_sources(opened.path)
        if store.has(opened.path, store.SOURCES)
        else [],
        roles=roles,
        log=log,
        round_no=len(rounds) + 1,
        grounding_cache={},
    )


def render_bundle(
    bundle: str | Path,
    *,
    model: str,
    formats: Sequence[str] | None = None,
    provenance: bool = True,
) -> list[Path]:
    """Render a bundle's current text to ``out/`` again (no model calls)."""
    opened = Bundle.open(bundle)
    brief = store.load_brief(opened.path)
    genre = genre_for(str(opened.metadata.get("genre") or brief.genre))
    require_provenance(genre, provenance=provenance)
    sources = (
        store.load_sources(opened.path) if store.has(opened.path, store.SOURCES) else []
    )
    facts = store.load_facts(opened.path) if store.has(opened.path, store.FACTS) else []
    return render_outputs(
        opened,
        brief=brief,
        genre=genre,
        facts=facts,
        sources=sources,
        model=model,
        formats=_check_formats(formats),
        provenance=provenance,
    )


def bundle_status(bundle: str | Path) -> str:
    """A short human summary of where a bundle stands."""
    opened = Bundle.open(bundle)
    lines = [f"{opened.title}  ({opened.path})"]
    for name, label in (
        (store.SOURCES, "sources"),
        (store.BRIEF, "brief"),
        (store.FACTS, "facts"),
        (store.OUTLINE, "outline"),
    ):
        lines.append(
            f"  {label}: {'done' if store.has(opened.path, name) else 'missing'}"
        )
    for section in opened.sections():
        words = len(section.body.split())
        lines.append(
            f"  {section.id}  {section.status:<8} {words:>5} words  {section.title}"
        )
    rounds = store.load_rounds(opened.path)
    if rounds:
        last = rounds[-1]
        failed = [c.name for c in last.checks if not c.passed]
        lines.append(
            f"  rounds: {len(rounds)}; last: "
            + ("goal met" if last.goal_met else f"open: {', '.join(failed)}")
        )
    outputs = (
        sorted((opened.path / "out").glob("*"))
        if (opened.path / "out").exists()
        else []
    )
    lines.append(f"  outputs: {', '.join(p.name for p in outputs) or 'none'}")
    return "\n".join(lines)
