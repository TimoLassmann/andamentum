"""``draft_document``: write into a new Word file, or into an existing one.

The compose workers do the writing (brief, evidence, outline, draft, revise);
this module sequences them around a ``.docx`` instead of a bundle folder:

1. a project folder ``state/`` beside ``out`` (the harness store; or *project*) and a scratch
   folder inside it for the workers' artefacts;
2. sources, result files and the text of the document being extended become
   compose sources; one call per source extracts quote-checked facts
   (``[[fN]]`` markers in the drafts point at them); literature enters only
   after drafting, through the cite loop (``cite.py``);
3. an outline, then one draft per target section; every sentence is analysed
   (``claims``): what it rests on, which numbers bind to result files;
4. a build writes the sections into a copy of the base document as tracked
   changes (``assemble``), then the harness ``check`` runs on it; an error in
   a drafted paragraph sends that section to the compose reviser and the
   document is rebuilt, up to ``max_rounds`` times;
5. what is left becomes anchored comments signed "andamentum compose", the
   statement on the use of AI is added, the file is stamped and written to
   ``out``, the claims ledger is saved and a report says what happened.

``out`` is always a new file: the input of ``--into`` and every source are only
read. Model calls are strictly sequential.
"""

from __future__ import annotations

import re
import zipfile
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import docx
import httpx
from pydantic import BaseModel, Field

from andamentum.compose import store as bundle_store
from andamentum.compose.brief import compile_brief
from andamentum.compose.evidence import (
    extract_facts,
    ingest_sources,
    stored_source_path,
)
from andamentum.compose.genres import Genre, genre_for
from andamentum.compose.log import ActionLog
from andamentum.compose.plan import make_outline
from andamentum.compose.research import (
    RESEARCH_DB,
    attribute,
    degraded_notes,
    research_facts,
    research_sources,
)
from andamentum.compose.revise import revise_section
from andamentum.compose.schemas import (
    Brief,
    Fact,
    Outline,
    Roles,
    SectionPlan,
    Source,
    WorkItem,
)
from andamentum.compose.steps import first_sentences
from andamentum.compose.write import draft_section
from andamentum.whetstone import stamp_provenance

from ..errors import WordUsageError
from ..project import project_dir, write_copy
from ..bindings.ledger import Ledger
from ..bindings.results import Catalogue, load_catalogue
from ..check import CheckParams, CheckReport, comment_ops, run_check
from ..docx.reader import DocxReadError
from ..docx.writer import DocxAdapter
from ..kernel import (
    Diagnostic,
    DocumentView,
    InsertParagraph,
    ProjectStore,
    Transaction,
)
from ..kernel.ops import Operation
from .assemble import (
    AUTHOR,
    BuiltRound,
    Placement,
    SectionDraft,
    Styles,
    build_round,
    normalised_title,
    place_sections,
    resolve_styles,
    section_spans,
)
from .body import parse_body
from .claims import (
    LabelCache,
    SourceInfo,
    analyse_section,
    claim_entries,
    record_claims,
)
from .disclosure import HEADING as AI_HEADING
from .disclosure import ai_use_statement
from .cite import CiteTools, Rewrite, cite_sentences
from .works import ResearchRun

DEFAULT_MAX_ROUNDS = 2
EXIT_OK, EXIT_OPEN, EXIT_USAGE = 0, 1, 2
RESULT_SUFFIXES = (".csv", ".tsv", ".json")
NAMED_SECTION_WORDS = 250
HEADING_LEVEL = 1
WORK_DIR = "compose"
TEMPLATE_TYPE = "wordprocessingml.template.main+xml"
DOCUMENT_TYPE = "wordprocessingml.document.main+xml"
TABLE_NOTE = (
    "a table in a draft was written as one paragraph per row; the operation set "
    "has no table"
)
FIGURE_NOTE = "a figure the outline planned was not drawn: figures are not written into Word documents"


class DraftUsageError(WordUsageError, ValueError):
    """A bad invocation (missing file, impossible combination): exit code 2."""


class RoundNote(BaseModel):
    round: int
    errors: int
    warnings: int
    revised: list[str]


class DraftResult(BaseModel):
    output: Path
    report: Path
    project: Path
    ledger: Path
    sections: list[str]
    rounds: list[RoundNote]
    citations: int
    works: list[str]
    numbers_bound: int
    open_errors: list[str]
    author_questions: list[str]
    notes: list[str]
    refused: list[str]
    usage: dict[str, int]
    statement: str
    counts: dict[str, int] = Field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        """0 when nothing is left for the authors, else 1: a check error, a
        refused operation, or a question (an unverified work, a placeholder,
        an interpretation nobody has backed with data)."""
        if self.open_errors or self.refused or self.author_questions:
            return EXIT_OPEN
        return EXIT_OK


@dataclass
class _Evidence:
    sources: list[Source]
    facts: list[Fact]
    infos: dict[str, SourceInfo]
    unverified: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    rewrites: list[Rewrite] = field(default_factory=list)


@dataclass
class _Run:
    """What one run holds: its project, its models and the collaborators it was given."""

    task: str
    out: Path
    roles: Roles
    store: ProjectStore
    log: ActionLog
    work: Path
    research_db: Path
    results: Path | None
    offline: bool
    mailto: str | None
    progress: Callable[[str], None]
    transport: httpx.AsyncBaseTransport | None
    url_check: Callable[[str], tuple[bool, str]] | None
    catalogue: Catalogue | None = None
    labels: LabelCache = field(default_factory=dict)
    taken: set[str] = field(default_factory=set)

    def step(self, step: str, *, role: str, model: str, detail: str = "") -> None:
        self.store.append_event(
            {
                "type": "draft_step",
                "step": step,
                "role": role,
                "model": model,
                "detail": detail,
            }
        )


# ------------------------------------------------------------------- setup


def _validate(
    task: str,
    *,
    out: Path,
    into: Path | None,
    template: Path | None,
    results: Path | None,
    sources: Sequence[str],
    max_rounds: int,
) -> None:
    if not task.strip():
        raise DraftUsageError("the task is empty")
    if out.suffix.lower() != ".docx":
        raise DraftUsageError(f"--out must be a .docx file, got {out}")
    if max_rounds < 0:
        raise DraftUsageError(f"max_rounds must not be negative, got {max_rounds}")
    if into is not None and template is not None:
        raise DraftUsageError("--into and --template cannot be combined")
    if into is not None:
        if not into.is_file() or into.suffix.lower() != ".docx":
            raise DraftUsageError(f"--into {into} is not a .docx file")
        if into.resolve() == out.resolve():
            raise DraftUsageError(
                "--out must not be the --into file; it is never edited"
            )
    if out.exists():
        raise DraftUsageError(
            f"{out} exists; write never overwrites a file (it may hold edits). "
            "Remove it or choose another --out"
        )
    if template is not None and (
        not template.is_file() or template.suffix.lower() not in {".docx", ".dotx"}
    ):
        raise DraftUsageError(f"--template {template} is not a .docx or .dotx file")
    if results is not None and not results.is_dir():
        raise DraftUsageError(f"--results {results} is not a folder")
    for source in sources:
        if not source.startswith(("http://", "https://")) and not Path(source).exists():
            raise DraftUsageError(f"source not found: {source}")
        if Path(source).exists() and Path(source).resolve() == out.resolve():
            raise DraftUsageError("--out must not be one of the sources")


def _openable_template(template: Path, *, scratch: Path) -> Path:
    """A ``.dotx`` read as a ``.docx`` (python-docx refuses the template content type)."""
    if template.suffix.lower() == ".docx":
        return template
    target = scratch / f"{template.stem}.docx"
    with zipfile.ZipFile(template) as source, zipfile.ZipFile(target, "w") as copy:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "[Content_Types].xml":
                data = data.replace(TEMPLATE_TYPE.encode(), DOCUMENT_TYPE.encode())
            copy.writestr(item, data)
    return target


def write_base_document(
    path: Path, *, template: Path | None, title: str, scratch: Path
) -> None:
    """A new document at *path*: the template's styles and page setup, a title, no body.

    The template's own body text is dropped (a placeholder is not content).
    """
    source = _openable_template(template, scratch=scratch) if template else None
    document = docx.Document(str(source)) if source else docx.Document()
    body = document.element.body
    for child in list(body):
        if not child.tag.endswith("}sectPr"):
            body.remove(child)
    has_title = "title" in {s.name.lower() for s in document.styles if s.name}
    document.add_paragraph(title, style="Title" if has_title else None)
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(path))


def _existing_text(view: DocumentView) -> str:
    lines = [
        ("#" * (p.heading_level or 1) + " " if p.heading_level else "") + p.text
        for p in view.paragraphs
        if p.text.strip()
    ]
    return "\n\n".join(lines) + "\n"


def _result_files(results: Path | None) -> list[Path]:
    if results is None:
        return []
    return sorted(
        p
        for p in results.rglob("*")
        if p.is_file() and p.suffix.lower() in RESULT_SUFFIXES
    )


async def _gather_evidence(
    run: _Run,
    *,
    sources: Sequence[str],
    existing: Path | None,
    brief_genre: Genre | None,
    research: ResearchRun | None,
    confirm_own_sources: bool,
) -> tuple[Brief, Genre, _Evidence]:
    paths: list[str] = [str(existing)] if existing else []
    paths += [str(Path(s).resolve()) if Path(s).exists() else s for s in sources]
    first_result = len(paths)
    result_files = _result_files(run.results)
    paths += [str(p) for p in result_files]
    run.progress(f"Evidence: {len(paths)} source(s)")
    ingested = await ingest_sources(
        paths, bundle_dir=run.work, confirm_own_sources=confirm_own_sources
    )
    infos: dict[str, SourceInfo] = {}
    for index, source in enumerate(ingested):
        if index >= first_result and run.results is not None:
            relative = str(result_files[index - first_result].relative_to(run.results))
            source = source.model_copy(update={"label": relative})
            ingested[index] = source
            infos[source.id] = SourceInfo(source=source, results_file=relative)
        else:
            if existing is not None and index == 0:
                source = source.model_copy(
                    update={"label": "the existing document text"}
                )
                ingested[index] = source
            infos[source.id] = SourceInfo(source=source)

    text_of = {
        s.id: stored_source_path(run.work, s).read_text(encoding="utf-8")
        for s in ingested
    }
    brief = await compile_brief(
        run.task,
        genre=brief_genre,
        sources=ingested,
        source_text=text_of,
        model=run.roles.drafter,
        log=run.log,
    )
    run.step("brief", role="drafter", model=run.roles.drafter)
    genre = genre_for(brief.genre)
    bundle_store.save_brief(run.work, brief)

    evidence = _Evidence(sources=list(ingested), facts=[], infos=infos)
    for source in evidence.sources:
        run.progress(f"Evidence: reading {source.title}")
        found_facts = await extract_facts(
            source,
            text_of[source.id],
            brief=brief,
            first_number=len(evidence.facts) + 1,
            model=run.roles.drafter,
            log=run.log,
        )
        if evidence.infos[source.id].work is not None:
            # The drafter sees only the statement: say whose finding it is, or a
            # published result gets written up as the author's own.
            found_facts = [
                fact.model_copy(
                    update={"statement": attribute(fact.statement, title=source.title)}
                )
                for fact in found_facts
            ]
        evidence.facts += found_facts
        run.step("evidence", role="drafter", model=run.roles.drafter, detail=source.id)
    if research is not None:
        await _add_research(run, research, genre=genre, evidence=evidence)
    bundle_store.save_sources(run.work, evidence.sources)
    bundle_store.save_facts(run.work, evidence.facts)
    return brief, genre, evidence


async def _add_research(
    run: _Run, research: ResearchRun, *, genre: Genre, evidence: _Evidence
) -> None:
    """Run the research and add each verified work with a finding as a source.

    Facts come straight from the findings: their quotes were checked against
    stored passages by the engine and are located again in the source text
    here, so this adds no model call.
    """
    found = await research(run.task, genre=genre, database=run.research_db)
    result = found.result
    run.step("research", role="drafter", model=run.roles.drafter, detail=result.mode)
    evidence.unverified += [u for u in found.unverified if u not in evidence.unverified]
    evidence.notes += found.notes + degraded_notes(result)
    added = research_sources(
        result,
        bundle_dir=run.work,
        first_number=len(evidence.sources) + 1,
        only=set(found.works),
    )
    for source in added:
        text = stored_source_path(run.work, source).read_text(encoding="utf-8")
        facts, dropped = research_facts(
            result, source, text, first_number=len(evidence.facts) + 1
        )
        evidence.sources.append(source)
        evidence.infos[source.id] = SourceInfo(
            source=source, work=found.works[source.work_id]
        )
        evidence.facts += facts
        evidence.notes += dropped


# ---------------------------------------------------------------- targets


def choose_targets(
    outline: Outline, *, task: str, view: DocumentView | None
) -> tuple[Outline, set[str]]:
    """Which sections to write, and the outline including any named extra ones.

    A new document gets every planned section. Into an existing document: the
    sections the task names (an outline title or an existing heading appearing in
    the task); with none named, the planned sections that are missing or empty.
    A named heading the outline does not know is added to it with a bare plan.
    """
    titles = [p.title for p in outline.sections]
    if view is None:
        return outline, set(titles)
    spans = section_spans(view)
    text = " ".join(task.casefold().split())

    def named(title: str) -> bool:
        return bool(re.search(rf"\b{re.escape(normalised_title(title))}\b", text))

    headings = [p.text for p in view.paragraphs if p.heading_level is not None]
    chosen = {t for t in titles if named(t)}
    extra = [
        h
        for h in headings
        if named(h) and normalised_title(h) not in {normalised_title(t) for t in titles}
    ]
    chosen |= set(extra)
    if not chosen:
        for title in titles:
            span = spans.get(normalised_title(title))
            if span is None or span[0] == span[1]:
                chosen.add(title)
    if extra:
        outline = Outline(
            sections=outline.sections
            + [
                SectionPlan(
                    title=h,
                    purpose=f"The section '{h}' of the document, as the task asks.",
                    key_points=[],
                    fact_ids=[],
                    target_words=NAMED_SECTION_WORDS,
                )
                for h in extra
            ]
        )
    if not chosen:
        raise DraftUsageError(
            "nothing to draft: every planned section already has text; name the "
            "sections to write in the task"
        )
    return outline, chosen


def _existing_summary(view: DocumentView, title: str) -> str:
    spans = section_spans(view)
    span = spans.get(normalised_title(title))
    if span is None or span[1] <= span[0]:
        return ""
    text = " ".join(p.text for p in view.paragraphs[span[0] + 1 : span[1] + 1])
    return first_sentences(text)


# ------------------------------------------------------------ analysis/draft


async def _refresh(run: _Run, section: SectionDraft, evidence: _Evidence) -> None:
    """Parse the section's body and analyse its sentences (after a draft or revision)."""
    run.taken -= {s.binding.anchor_id for s in section.analysis.number_spots()}
    section.paragraphs = parse_body(section.body)
    section.analysis = await analyse_section(
        section.paragraphs,
        facts={f.id: f for f in evidence.facts},
        infos=evidence.infos,
        catalogue=run.catalogue,
        taken_anchors=run.taken,
        model=run.roles.judge,
        cache=run.labels,
        log=run.log,
    )
    run.step("claims", role="judge", model=run.roles.judge, detail=section.title)


async def _draft_all(
    run: _Run,
    *,
    brief: Brief,
    genre: Genre,
    outline: Outline,
    evidence: _Evidence,
    targets: set[str],
    placements: dict[str, Placement],
    view: DocumentView | None,
) -> list[SectionDraft]:
    drafts: list[SectionDraft] = []
    summaries: list[str] = []
    for plan in outline.sections:
        if plan.title not in targets:
            if view is not None:
                existing = _existing_summary(view, plan.title)
                if existing:
                    summaries.append(f"{plan.title}: {existing}")
            continue
        run.progress(f"Draft: {plan.title}")
        body = await draft_section(
            plan,
            brief=brief,
            genre=genre,
            outline=outline,
            facts=evidence.facts,
            previous_summaries=summaries,
            model=run.roles.drafter,
            log=run.log,
        )
        run.step("draft", role="drafter", model=run.roles.drafter, detail=plan.title)
        section = SectionDraft(
            key=f"s{len(drafts) + 1}",
            title=plan.title,
            body=body,
            plan=plan,
            placement=placements[plan.title],
        )
        await _refresh(run, section, evidence)
        drafts.append(section)
        summaries.append(f"{plan.title}: {first_sentences(body)}")
    if not any(d.paragraphs for d in drafts):
        raise DraftUsageError(
            "no section could be drafted (the model returned no text)"
        )
    return drafts


async def _cite(
    run: _Run, tools: CiteTools, *, drafts: list[SectionDraft], evidence: _Evidence
) -> None:
    """The cite loop over the drafted sections; changed sections are analysed again."""
    outcome = await cite_sentences(
        drafts,
        sources=evidence.sources,
        facts=evidence.facts,
        infos=evidence.infos,
        tools=tools,
        bundle_dir=run.work,
        model=run.roles.judge,
        progress=run.progress,
        log=run.log,
    )
    if outcome.checked > 0:
        run.step(
            "cite",
            role="judge",
            model=run.roles.judge,
            detail=(
                f"{outcome.checked} checked, {len(outcome.cited)} cited, "
                f"{len(outcome.changed)} changed"
            ),
        )
    evidence.rewrites += outcome.rewrites
    evidence.notes += outcome.notes
    for section in drafts:
        if section.key in outcome.changed_sections:
            await _refresh(run, section, evidence)
    bundle_store.save_sources(run.work, evidence.sources)
    bundle_store.save_facts(run.work, evidence.facts)


# ----------------------------------------------------------- check & repair


async def _check(run: _Run, docx_path: Path, *, round_no: int) -> CheckReport:
    folder = run.work / f"round-{round_no}"
    params = CheckParams(
        file=docx_path,
        results=run.results,
        model=run.roles.judge if run.results is not None else None,
        out=folder / "checked.docx",
        report=folder / "checked.md",
        project=run.store.path,
        mailto=run.mailto,
        offline=run.offline,
    )
    done = await run_check(params, transport=run.transport, url_check=run.url_check)
    return done.report


def _work_items(
    report: CheckReport, owner: dict[str, str]
) -> dict[str, list[WorkItem]]:
    """Check errors inside drafted paragraphs, as reviser work items by section."""
    items: dict[str, list[WorkItem]] = {}
    for diagnostic in report.diagnostics:
        key = owner.get(diagnostic.paragraph_id or "")
        if diagnostic.severity != "error" or key is None:
            continue
        reason = diagnostic.message
        if diagnostic.evidence:
            reason += f" ({diagnostic.evidence[0]})"
        items.setdefault(key, []).append(
            WorkItem(
                level="section", section_id=key, reason=reason, source=diagnostic.check
            )
        )
    return items


def typed_citation_errors(sections: list[SectionDraft]) -> list[str]:
    """One line per citation-shaped string the drafter typed, by section."""
    return [
        f"{section.title}: typed citation {typed.text!r} was not inserted by the "
        "harness and is not verified"
        for section in sections
        for typed in section.analysis.typed
    ]


def _typed_work_items(sections: list[SectionDraft]) -> dict[str, list[WorkItem]]:
    """Typed citations as reviser work items by section."""
    items: dict[str, list[WorkItem]] = {}
    for section in sections:
        for typed in section.analysis.typed:
            items.setdefault(section.key, []).append(
                WorkItem(
                    level="section",
                    section_id=section.key,
                    reason=f"the text types the citation {typed.text!r}; works are "
                    "cited only with a [[fN]] marker on a fact, never written out. "
                    "Remove the typed citation and keep the claim only if a fact "
                    "supports it",
                    source="typed_citation",
                )
            )
    return items


async def _build_and_check(
    run: _Run,
    *,
    base: Path,
    sections: list[SectionDraft],
    styles: Styles,
    evidence: _Evidence,
    document_notes: list[str],
    round_no: int,
) -> tuple[BuiltRound, CheckReport]:
    built = build_round(
        base=base,
        round_dir=run.work / f"round-{round_no}",
        sections=sections,
        styles=styles,
        facts={f.id: f for f in evidence.facts},
        infos=evidence.infos,
        document_notes=document_notes,
        reason=f"write: round {round_no}",
        rewrites=evidence.rewrites,
    )
    run.progress(f"Check: round {round_no}")
    return built, await _check(run, built.docx, round_no=round_no)


async def _repair(
    run: _Run,
    *,
    base: Path,
    sections: list[SectionDraft],
    styles: Styles,
    evidence: _Evidence,
    brief: Brief,
    genre: Genre,
    max_rounds: int,
    document_notes: list[str],
) -> tuple[BuiltRound, CheckReport, list[RoundNote]]:
    """Build, check; revise the sections with errors and rebuild, at most *max_rounds* times."""
    built, report = await _build_and_check(
        run,
        base=base,
        sections=sections,
        styles=styles,
        evidence=evidence,
        document_notes=document_notes,
        round_no=0,
    )
    rounds = [_round_note(0, report, [], typed=len(typed_citation_errors(sections)))]
    by_key = {s.key: s for s in sections}
    for round_no in range(1, max_rounds + 1):
        work = _work_items(report, built.owner)
        for key, typed in _typed_work_items(sections).items():
            work.setdefault(key, []).extend(typed)
        if not work:
            break
        revised: list[str] = []
        for key, items in work.items():
            section = by_key[key]
            run.progress(f"Revise: {section.title}")
            body = await revise_section(
                section.body,
                items,
                plan=section.plan,
                brief=brief,
                genre=genre,
                facts=evidence.facts,
                model=run.roles.drafter,
                log=run.log,
            )
            run.step(
                "revise", role="drafter", model=run.roles.drafter, detail=section.title
            )
            if body.strip() == section.body.strip():
                continue
            section.body = body
            await _refresh(run, section, evidence)
            revised.append(section.title)
        if not revised:
            break
        built, report = await _build_and_check(
            run,
            base=base,
            sections=sections,
            styles=styles,
            evidence=evidence,
            document_notes=document_notes,
            round_no=round_no,
        )
        rounds.append(
            _round_note(
                round_no, report, revised, typed=len(typed_citation_errors(sections))
            )
        )
    return built, report, rounds


def _round_note(
    round_no: int, report: CheckReport, revised: list[str], *, typed: int
) -> RoundNote:
    return RoundNote(
        round=round_no,
        errors=report.counts.get("error", 0) + typed,
        warnings=report.counts.get("warning", 0),
        revised=revised,
    )


# ------------------------------------------------------------------ finish


def _statement_ops(view: DocumentView, text: str, *, styles: Styles) -> list[Operation]:
    """The AI-use paragraph, under its heading (added when the document has none)."""
    spans = section_spans(view)
    span = spans.get(normalised_title(AI_HEADING))
    if span is not None:
        return [
            InsertParagraph(
                after_paragraph_id=view.paragraphs[span[1]].id,
                text=text,
                style=styles.body,
            )
        ]
    last = view.paragraphs[-1].id
    return [
        InsertParagraph(after_paragraph_id=last, text=AI_HEADING, style=styles.heading),
        InsertParagraph(after_paragraph_id=last, text=text, style=styles.body),
    ]


def _ours(diagnostic: Diagnostic, *, owner: dict[str, str], new_document: bool) -> bool:
    if new_document:
        return True
    return diagnostic.paragraph_id in owner


def _author_questions(sections: list[SectionDraft], evidence: _Evidence) -> list[str]:
    questions: list[str] = []
    for section in sections:
        for paragraph in section.paragraphs:
            for start, end in paragraph.to_fill():
                label = paragraph.text[start:end].removeprefix("[TO FILL:").rstrip("]")
                questions.append(f"{section.title}: needs the author: {label.strip()}")
        count = sum(1 for s in section.analysis.sentences if s.kind == "interpretation")
        if count:
            questions.append(
                f"{section.title}: {count} interpretation claim(s) need the authors' judgement"
            )
    questions += [
        f"not cited, could not be verified: {line}" for line in evidence.unverified
    ]
    return questions


def _report_markdown(result: DraftResult, *, task: str) -> str:
    lines = [f"# Write report: {result.output.name}", "", f"Task: {task}", ""]
    lines += [
        f"- output: `{result.output}`",
        f"- project: `{result.project}`; claims ledger: `{result.ledger}`",
        f"- sections written: {', '.join(result.sections)}",
        f"- citations inserted: {result.citations} ({len(result.works)} work(s)); "
        f"numbers bound: {result.numbers_bound}",
        f"- check errors left: {len(result.open_errors)}; "
        f"warnings: {result.counts.get('warning', 0)}",
        f"- exit code: {result.exit_code}",
        "",
        "## Rounds",
        "",
    ]
    for note in result.rounds:
        revised = f"; revised {', '.join(note.revised)}" if note.revised else ""
        lines.append(
            f"- round {note.round}: {note.errors} error(s), {note.warnings} warning(s){revised}"
        )
    lines += ["", "## Open check errors", ""]
    lines += [f"- {e}" for e in result.open_errors] or ["- none"]
    lines += ["", "## Questions for the authors", ""]
    lines += [f"- {q}" for q in result.author_questions] or ["- none"]
    if result.notes or result.refused:
        lines += ["", "## Notes", ""]
        lines += [f"- {n}" for n in result.notes]
        lines += [f"- the writer refused: {r}" for r in result.refused]
    lines += ["", "## Statement on the use of AI", "", result.statement, ""]
    usage = ", ".join(
        f"{role} {tokens}" for role, tokens in sorted(result.usage.items())
    )
    lines += ["## Model use (tokens)", "", usage or "not reported", ""]
    return "\n".join(lines)


async def _finish(
    run: _Run,
    *,
    built: BuiltRound,
    report: CheckReport,
    rounds: list[RoundNote],
    sections: list[SectionDraft],
    evidence: _Evidence,
    styles: Styles,
    facts: dict[str, Fact],
    new_document: bool,
    project_id: str,
    roles: Roles,
) -> DraftResult:
    adapter = DocxAdapter()
    view = built.view
    remaining = [
        d
        for d in report.diagnostics
        if d.severity in ("error", "warning")
        and _ours(d, owner=built.owner, new_document=new_document)
    ]
    open_errors = [
        f"{d.check}: {d.message}" for d in remaining if d.severity == "error"
    ] + typed_citation_errors(sections)
    run.store.append_event(
        {
            "type": "draft_run",
            "task": run.task,
            "citations": built.citations,
            "numbers_bound": len(built.bindings),
            "works": built.works,
            "open_errors": len(open_errors),
        }
    )
    statement = ai_use_statement(run.store.read_events())
    ops: list[Operation] = list(comment_ops(view, remaining))
    ops += _statement_ops(view, statement, styles=styles)
    write_copy(
        adapter,
        view,
        Transaction(
            ops=ops,
            author=AUTHOR,
            reason="write: remaining issues and AI statement",
            source="write",
        ),
        out=run.out,
        project_id=project_id,
    )
    stamp_provenance(run.out, model=roles.drafter)
    final = adapter.open(run.out)
    if final.project_id != project_id:
        raise DocxReadError(f"{run.out} lost its project id when it was stamped")
    run.store.snapshot_view(
        final,
        message=f"write: {len(sections)} section(s), {built.citations} citation(s)",
    )

    bound = {b.anchor_id for b in built.bindings}
    entries = [
        entry
        for section in sections
        for entry in claim_entries(
            section.title,
            section.analysis,
            facts=facts,
            infos=evidence.infos,
            bound=bound,
        )
    ]
    ledger = Ledger(run.store)
    record_claims(run.store, entries)
    if built.bindings:
        ledger.record(built.bindings)

    notes = list(evidence.notes) + built.notes
    for section in sections:
        if not section.paragraphs:
            notes.append(
                f"{section.title}: the model returned no text for this section"
            )
        notes += [
            f"{section.title}: marker for an unknown fact ignored: {m}"
            for m in section.analysis.unknown_markers
        ]
        notes += [f"{section.title}: {n}" for n in section.analysis.ambiguous_numbers]
        if any(p.table_row for p in section.paragraphs) and TABLE_NOTE not in notes:
            notes.append(TABLE_NOTE)
        if section.plan.figure is not None and FIGURE_NOTE not in notes:
            notes.append(FIGURE_NOTE)
    questions = _author_questions(sections, evidence) + open_errors
    result = DraftResult(
        output=run.out,
        report=run.out.with_name(f"{run.out.stem}.report.md"),
        project=run.store.path,
        ledger=run.store.path / "claims.yaml",
        sections=[s.title for s in sections],
        rounds=rounds,
        citations=built.citations,
        works=built.works,
        numbers_bound=len(built.bindings),
        open_errors=open_errors,
        author_questions=questions,
        notes=notes,
        refused=built.refused,
        usage=run.log.usage(),
        statement=statement,
        counts=dict(report.counts),
    )
    result.report.write_text(_report_markdown(result, task=run.task), encoding="utf-8")
    result.report.with_suffix(".json").write_text(
        result.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return result


# ------------------------------------------------------------------- entry


async def draft_document(
    task: str,
    *,
    out: str | Path,
    roles: Roles,
    into: str | Path | None = None,
    genre: str | None = None,
    sources: Sequence[str] = (),
    results: str | Path | None = None,
    research: ResearchRun | None = None,
    cite: CiteTools | None = None,
    template: str | Path | None = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    offline: bool = False,
    confirm_own_sources: bool = False,
    mailto: str | None = None,
    project: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    url_check: Callable[[str], tuple[bool, str]] | None = None,
) -> DraftResult:
    """Draft *task* into the new ``.docx`` *out* (a copy of *into* when given).

    *roles* names the model for each job (the drafter writes, the judge labels
    claims and runs the numbers check; the critic and reader roles are part of
    the compose roles and unused here). *research* runs
    ``deep_research`` once in the genre's mode over ``<project>/research.db``; *project* is the folder that keeps the ledger and logs (default ``state/`` beside *out*)
    and adds each verified work that has a finding (its quotes are checked
    passages) as a source; without it nothing is searched. *cite* runs the cite
    loop after the draft (``cite.py``: sentences that cite nothing get a paper
    that was read and supports them; the papers read go into its document
    store); without it no sentence is cited after drafting. *genre* is a compose genre
    pack name or YAML path; ``None`` lets the brief compiler choose.

    Returns a :class:`DraftResult`; ``exit_code`` is 0 when no check error is
    left in the drafted text and nothing was refused, 1 otherwise. Raises
    :class:`DraftUsageError` for a bad invocation.
    """
    out_path = Path(out)
    into_path = Path(into) if into is not None else None
    template_path = Path(template) if template is not None else None
    results_path = Path(results) if results is not None else None
    _validate(
        task,
        out=out_path,
        into=into_path,
        template=template_path,
        results=results_path,
        sources=sources,
        max_rounds=max_rounds,
    )
    preset = genre_for(genre) if genre else None

    project = project_dir(out_path, Path(project) if project is not None else None)
    work = project / WORK_DIR
    work.mkdir(parents=True, exist_ok=True)
    adapter = DocxAdapter()
    base = into_path or project / "base.docx"
    if into_path is None:
        docx.Document().save(str(base))  # a stand-in until the brief names the title
    try:
        base_view = adapter.open(base)
        if into_path is not None and not base_view.paragraphs:
            raise DraftUsageError(f"{base} has no paragraphs to write after")
        store = ProjectStore(project)
        meta = store.init(base, project_id=base_view.project_id)
    except (DocxReadError, ValueError) as exc:
        raise DraftUsageError(str(exc)) from exc
    run = _Run(
        task=task,
        out=out_path,
        roles=roles,
        store=store,
        log=ActionLog(work),
        work=work,
        research_db=project / RESEARCH_DB,
        results=results_path,
        offline=offline,
        mailto=mailto,
        progress=progress or (lambda _line: None),
        transport=transport,
        url_check=url_check,
        catalogue=load_catalogue(results_path) if results_path else None,
        taken={a.id for a in base_view.anchors},
    )

    existing = None
    if into_path is not None:
        existing = work / "existing-document.md"
        existing.write_text(_existing_text(base_view), encoding="utf-8")
    brief, genre_pack, evidence = await _gather_evidence(
        run,
        sources=sources,
        existing=existing,
        brief_genre=preset,
        research=research,
        confirm_own_sources=confirm_own_sources,
    )

    run.progress("Plan: outlining")
    outline = await make_outline(
        brief,
        genre_pack,
        evidence.facts,
        tabular_sources=[],
        model=roles.drafter,
        log=run.log,
    )
    run.step("outline", role="drafter", model=roles.drafter)
    outline, targets = choose_targets(
        outline, task=task, view=base_view if into_path else None
    )
    bundle_store.save_outline(run.work, outline)

    if into_path is None:
        write_base_document(
            base, template=template_path, title=brief.title, scratch=work
        )
        base_view = adapter.open(base)
    styles = resolve_styles(
        base, base_view, heading_level=_heading_level(base_view, new=into_path is None)
    )
    placements = place_sections([p.title for p in outline.sections], targets, base_view)
    drafts = await _draft_all(
        run,
        brief=brief,
        genre=genre_pack,
        outline=outline,
        evidence=evidence,
        targets=targets,
        placements=placements,
        view=base_view if into_path else None,
    )
    if cite is not None and genre_pack.cites:
        await _cite(run, cite, drafts=drafts, evidence=evidence)
    document_notes = [
        f"Not cited, could not be verified: {line}" for line in evidence.unverified
    ]
    built, report, rounds = await _repair(
        run,
        base=base,
        sections=drafts,
        styles=styles,
        evidence=evidence,
        brief=brief,
        genre=genre_pack,
        max_rounds=max_rounds,
        document_notes=document_notes,
    )
    return await _finish(
        run,
        built=built,
        report=report,
        rounds=rounds,
        sections=drafts,
        evidence=evidence,
        styles=styles,
        facts={f.id: f for f in evidence.facts},
        new_document=into_path is None,
        project_id=meta.id,
        roles=roles,
    )


def _heading_level(view: DocumentView, *, new: bool) -> int:
    """Level of the section headings: 1 in a new document, else the document's top level."""
    if new:
        return HEADING_LEVEL
    levels = Counter(p.heading_level for p in view.paragraphs if p.heading_level)
    return min(levels) if levels else HEADING_LEVEL


__all__ = [
    "DEFAULT_MAX_ROUNDS",
    "DraftResult",
    "DraftUsageError",
    "RoundNote",
    "draft_document",
    "write_base_document",
]
