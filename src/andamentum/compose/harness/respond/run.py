"""The ``respond`` verb: comments and reviews as a work queue.

``andamentum-word respond FILE.docx [--reviews FILE] [--model M]`` never
modifies FILE. It

1. collects the items: every unresolved comment thread not started by the
   harness, and every point of the reviews file (split by a model, each point
   verified by quoting);
2. triages each item into ``edit``, ``answer``, ``author`` or ``done`` (the
   decision model's top category when ``--decision-model`` is given, and then a
   ``done`` or ``author`` item needs no language-model call; otherwise the
   language model);
3. for ``edit``, has the paragraph rewritten, turns the rewrite into word-level
   span operations, applies them to a staged copy and runs ``check`` on it: an
   edit that introduces an error in the paragraph it changed is not written;
4. writes a copy with the tracked changes, a reply in each thread (a human
   comment is never resolved), comments for what needs the authors, and
   comments for check errors in paragraphs collaborators changed since the
   harness last read the document;
5. with ``--reviews``, writes ``response-letter.docx`` and ``.md`` beside the
   copy (refusing, exit 2, when they exist: they may hold hand edits); writes the report; appends events and snapshots the document.

Exit codes: 0 everything handled, 4 written but open issues remain (author
items, unreadable review points, refused operations, check errors in
collaborators' changes), 2 usage error, 3 crash (raised to the caller).
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field

from andamentum.compose.log import ActionLog
from andamentum.core.decisions import DecisionModel, DecisionModelUnavailable

from ..check import CheckParams, CheckUsageError, run_check
from ..docx.reader import DocxReadError
from ..docx.writer import DocxAdapter
from ..kernel import (
    ApplyResult,
    CommentOp,
    Diagnostic,
    DocumentView,
    Operation,
    Paragraph,
    ProjectStore,
    Reply,
    Transaction,
    Verb,
)
from ..kernel.models import open_decision_model
from ..errors import WordUsageError
from ..project import project_dir, refuse_taken, write_copy, written_files
from .collab import changed_paragraph_ids, previous_texts
from .edits import EditRefused, SpanOp, span_ops
from .items import Item, comment_items, read_reviews, review_items
from .letter import LetterEntry, default_letter, existing_letters, write_letter
from .quotes import find_in_paragraph, find_in_view
from .triage import (
    CATEGORIES,
    Category,
    DecisionRecord,
    propose_answer,
    propose_edit,
    triage_with_decision,
    verified_quotes,
)

AUTHOR = "andamentum respond"
EXIT_OK, EXIT_OPEN, EXIT_USAGE = 0, 1, 2
_GATE_DIR = "respond-gate"


class RespondUsageError(WordUsageError):
    """A bad invocation (missing file, no model for the work): exit code 2."""


class RespondParams(BaseModel):
    file: Path = Field(
        json_schema_extra={"positional": True}, description="the .docx to respond in"
    )
    reviews: Path | None = Field(
        default=None, description="reviewers' report (.txt, .md, .docx or .pdf)"
    )
    model: str | None = Field(
        default=None, description="model id for triage, edits and answers"
    )
    decision_model: str | None = Field(
        default=None,
        description="ollama decision model that triages instead of the language model",
    )
    out: Path = Field(description="the responded copy to write (a .docx path)")
    report: Path | None = Field(
        default=None, description="markdown report (default: beside --out)"
    )
    letter: Path | None = Field(
        default=None,
        description="the response letter's .docx, with the .md beside it "
        "(default: response-letter.docx beside --out)",
    )
    project: Path | None = Field(
        default=None, description="folder that keeps the ledger and logs (default: state/ beside the output)"
    )
    results: Path | None = Field(
        default=None, description="folder of result files, for the numbers check"
    )
    mailto: str | None = Field(
        default=None, description="contact address for Crossref's polite pool"
    )
    offline: bool = Field(default=False, description="make no network calls")


class Outcome(BaseModel):
    """What happened to one item."""

    model_config = ConfigDict(extra="forbid")

    item: Item
    proposed: Category  # what triage chose
    category: Category  # what it ended as, after verification and the gate
    reason: str
    reply: str | None = None
    paragraph_index: int | None = None
    section: str = ""
    new_text: str | None = None
    evidence: list[str] = Field(default_factory=list)
    decision: DecisionRecord | None = None
    explanation: str = (
        ""  # for an edit: what changed and why, as written to the commenter
    )
    demoted: str = ""  # why an edit, answer or done became author
    paragraph_id: str | None = None
    span: tuple[int, int] | None = None


class RespondReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str
    project_id: str
    output: str
    letter: list[str]
    outcomes: list[Outcome]
    skipped_threads: list[str]
    unverified_points: list[str]
    collaborator_baseline: str  # "compared with the snapshot of ..." or why not
    collaborator_changes: int
    collaborator_errors: list[Diagnostic]
    gate_rejections: list[str]
    counts: dict[str, int]
    operations_applied: int
    operations_refused: list[str]
    model_calls: int
    input_tokens: int
    output_tokens: int
    seconds: float

    @property
    def exit_code(self) -> int:
        open_issues = (
            self.counts.get("author", 0)
            or self.unverified_points
            or self.operations_refused
            or self.collaborator_errors
        )
        return EXIT_OPEN if open_issues else EXIT_OK


class RespondRun(BaseModel):
    report: RespondReport
    report_md: Path
    report_json: Path
    output: Path
    letter_docx: Path | None
    letter_md: Path | None


# ------------------------------------------------------------------- helpers


def _sections(view: DocumentView) -> dict[str, str]:
    """Paragraph id to the heading it sits under."""
    heading = ""
    out: dict[str, str] = {}
    for paragraph in view.paragraphs:
        if paragraph.heading_level is not None:
            heading = paragraph.text
        out[paragraph.id] = heading
    return out


def _place(outcome: Outcome, paragraph: Paragraph, sections: dict[str, str]) -> None:
    outcome.paragraph_id = paragraph.id
    outcome.paragraph_index = paragraph.index
    outcome.section = sections.get(paragraph.id, "")


def _where(outcome: Outcome) -> str:
    if outcome.paragraph_index is None:
        return ""
    section = f'section "{outcome.section}", ' if outcome.section else ""
    return f"{section}paragraph {outcome.paragraph_index}"


def _pointers(view: DocumentView, quotes: list[str]) -> str:
    parts: list[str] = []
    for quote in quotes:
        found = find_in_view(view, quote)
        where = f" (paragraph {found[0].index})" if found else ""
        parts.append(f'"{quote.strip()}"{where}')
    return "; ".join(parts)


def _demote(outcome: Outcome, why: str) -> None:
    outcome.category = "author"
    outcome.demoted = why
    outcome.new_text = None
    outcome.evidence = []
    outcome.reply = f"Needs the authors: {why}."


def _reply_for(outcome: Outcome, view: DocumentView) -> None:
    if outcome.category == "edit":
        outcome.reply = (
            f"Changed in {_where(outcome)}: {outcome.explanation} "
            "The change is tracked; accept or reject it as you see fit."
        )
    elif outcome.category == "author":
        outcome.reply = f"Needs the authors: {outcome.reason.rstrip('.')}."
    elif outcome.category == "done":
        outcome.reply = (
            f"Already addressed: {outcome.reason.rstrip('.')}. "
            f"See {_pointers(view, outcome.evidence)}."
        )


def _clamped_comment(
    view: DocumentView, diagnostic: Diagnostic, text: str
) -> CommentOp | None:
    sizes = {p.id: len(p.text) for p in view.paragraphs}
    if (
        diagnostic.paragraph_id not in sizes
        or diagnostic.start is None
        or diagnostic.end is None
    ):
        return None
    size = sizes[diagnostic.paragraph_id]
    start = min(diagnostic.start, size)
    end = min(max(diagnostic.end, start + 1), size)
    if end <= start:
        return None
    return CommentOp(
        paragraph_id=str(diagnostic.paragraph_id), start=start, end=end, text=text
    )


def _error_counts(diagnostics: list[Diagnostic]) -> Counter[tuple[str, str, str]]:
    return Counter(
        (d.check, d.paragraph_id or "", str(d.data.get("kind", "")))
        for d in diagnostics
        if d.severity == "error"
    )


def new_errors(
    before: list[Diagnostic], after: list[Diagnostic], *, paragraph_ids: set[str]
) -> dict[str, list[Diagnostic]]:
    """Errors in *after* that *before* did not have, per paragraph of *paragraph_ids*."""
    known = _error_counts(before)
    seen: Counter[tuple[str, str, str]] = Counter()
    found: dict[str, list[Diagnostic]] = {}
    for diagnostic in after:
        if (
            diagnostic.severity != "error"
            or diagnostic.paragraph_id not in paragraph_ids
        ):
            continue
        key = (
            diagnostic.check,
            diagnostic.paragraph_id or "",
            str(diagnostic.data.get("kind", "")),
        )
        seen[key] += 1
        if seen[key] > known[key]:
            found.setdefault(key[1], []).append(diagnostic)
    return found


def _apply(
    adapter: DocxAdapter,
    view: DocumentView,
    ops: list[Operation],
    *,
    reason: str,
    output: Path,
    project_id: str | None,
) -> ApplyResult:
    """Apply *ops* to *output*; *project_id* is written when the document has none."""
    transaction = Transaction(ops=ops, author=AUTHOR, reason=reason, source="respond")
    return write_copy(adapter, view, transaction, out=output, project_id=project_id)


# ------------------------------------------------------------------- the run


class _Gate:
    """``check`` over a staged copy, in a folder of its own inside the project."""

    def __init__(
        self,
        params: RespondParams,
        store: ProjectStore,
        *,
        transport: httpx.AsyncBaseTransport | None,
        url_check: Callable[[str], tuple[bool, str]] | None,
    ) -> None:
        self.params = params
        self.folder = store.path / _GATE_DIR
        self.transport = transport
        self.url_check = url_check

    async def diagnostics(
        self, file: Path, *, project_id: str | None
    ) -> list[Diagnostic]:
        self.folder.mkdir(parents=True, exist_ok=True)
        marker = self.folder / "project.yaml"
        if project_id is not None and marker.is_file():
            if ProjectStore(self.folder).meta().id != project_id:
                marker.unlink()  # a gate left by a document that had no id yet
        run = await run_check(
            CheckParams(
                file=file,
                results=self.params.results,
                model=self.params.model,
                out=self.folder / f"{file.stem}.checked.docx",
                report=self.folder / f"{file.stem}.checked.md",
                project=self.folder,
                mailto=self.params.mailto,
                offline=self.params.offline,
            ),
            transport=self.transport,
            url_check=self.url_check,
        )
        return run.report.diagnostics


def _outputs(params: RespondParams) -> tuple[Path, Path]:
    source = params.file
    out = params.out
    if out.resolve() == source.resolve():
        raise RespondUsageError(
            "the output must not be the input file; respond never edits it"
        )
    files = written_files(out, params.report)
    refuse_taken(files, error=RespondUsageError)
    return out, files[1]


def _validate(params: RespondParams) -> None:
    if not params.file.is_file() or params.file.suffix.lower() != ".docx":
        raise RespondUsageError(f"{params.file} is not a .docx file")
    if params.reviews is not None and not params.reviews.is_file():
        raise RespondUsageError(f"--reviews {params.reviews} is not a file")
    if params.results is not None and not params.results.is_dir():
        raise RespondUsageError(f"--results {params.results} is not a folder")
    if params.decision_model and not params.model:
        raise RespondUsageError("--decision-model needs --model for the follow-up work")


async def run_respond(
    params: RespondParams,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    url_check: Callable[[str], tuple[bool, str]] | None = None,
    decision_handle: DecisionModel | None = None,
) -> RespondRun:
    """Run the respond verb. ``transport``, ``url_check`` and ``decision_handle``
    serve recorded answers in tests; production leaves them unset
    (``--decision-model`` is opened and checked)."""
    if decision_handle is not None and not params.decision_model:
        params = params.model_copy(update={"decision_model": decision_handle.id})
    _validate(params)
    try:
        decision, owned = await open_decision_model(
            params.decision_model, injected=decision_handle
        )
    except DecisionModelUnavailable as exc:
        raise RespondUsageError(str(exc)) from exc
    try:
        return await _respond(
            params, transport=transport, url_check=url_check, decision=decision
        )
    finally:
        if owned and decision is not None:
            await decision.aclose()


async def _respond(
    params: RespondParams,
    *,
    transport: httpx.AsyncBaseTransport | None,
    url_check: Callable[[str], tuple[bool, str]] | None,
    decision: DecisionModel | None,
) -> RespondRun:
    started = time.monotonic()
    out, report_md = _outputs(params)
    if params.reviews is not None:
        kept = existing_letters(params.letter or default_letter(out.parent))
        if kept:
            raise RespondUsageError(
                f"{', '.join(str(p) for p in kept)} already exists and may hold your "
                "edits; move or rename it, or give --out in another folder"
            )
    adapter = DocxAdapter()
    try:
        view = adapter.open(params.file)
    except DocxReadError as exc:
        raise RespondUsageError(str(exc)) from exc
    store = ProjectStore(project_dir(out, params.project))
    try:
        meta = store.init(params.file, project_id=view.project_id)
    except ValueError as exc:
        raise RespondUsageError(str(exc)) from exc

    log = ActionLog(store.path)
    calls_before = len(log.records())
    previous = previous_texts(store)
    sections = _sections(view)
    paragraphs = {p.id: p for p in view.paragraphs}

    # -- items
    items, skipped = comment_items(view)
    unverified: list[str] = []
    if params.reviews is not None:
        if not params.model:
            raise RespondUsageError("--reviews needs --model to split the report")
        reviews_text = await read_reviews(params.reviews)
        review_list, unverified = await review_items(
            reviews_text, model=params.model, log=log
        )
        items += review_list
    if items and not params.model:
        raise RespondUsageError(
            f"{len(items)} item(s) to respond to; pass --model MODEL"
        )

    # -- triage and the per-category work, one item at a time
    outcomes: list[Outcome] = []
    current: dict[str, str] = {}  # paragraph id -> text after the edits so far
    edit_items: dict[str, list[Outcome]] = {}
    for item in items:
        outcome = await _work_item(
            item,
            view,
            params=params,
            decision=decision,
            log=log,
            sections=sections,
            paragraphs=paragraphs,
            current=current,
            edit_items=edit_items,
        )
        outcomes.append(outcome)

    # -- stage the edits, verify, gate by check
    changed = changed_paragraph_ids(view, previous) if previous is not None else []
    gate = _Gate(params, store, transport=transport, url_check=url_check)
    baseline: list[Diagnostic] = []
    gate_rejections: list[str] = []
    if edit_items or changed:
        baseline = await gate.diagnostics(params.file, project_id=view.project_id)
    ops_by_paragraph = {
        pid: span_ops(paragraphs[pid], text, fields=view.fields)
        for pid, text in current.items()
        if text != paragraphs[pid].text and edit_items.get(pid)
    }
    if ops_by_paragraph:
        dropped = await _stage_and_gate(
            adapter,
            view,
            ops_by_paragraph,
            current=current,
            gate=gate,
            baseline=baseline,
            project_id=meta.id,
        )
        for pid, why in dropped.items():
            gate_rejections.append(f"paragraph {paragraphs[pid].index}: {why}")
            for outcome in edit_items[pid]:
                _demote(outcome, f"the edit was not written ({why})")
            del ops_by_paragraph[pid]

    # -- the transaction
    ops: list[Operation] = [op for group in ops_by_paragraph.values() for op in group]
    for outcome in outcomes:
        if (
            outcome.category == "author"
            and outcome.item.source == "review"
            and outcome.paragraph_id is not None
            and outcome.span is not None
        ):
            ops.append(
                CommentOp(
                    paragraph_id=outcome.paragraph_id,
                    start=outcome.span[0],
                    end=outcome.span[1],
                    text=f"{outcome.item.label}: {outcome.reply}",
                )
            )
    collaborator_errors = [
        d for d in baseline if d.severity == "error" and d.paragraph_id in set(changed)
    ]
    for diagnostic in collaborator_errors:
        text = f"Changed since the last look. [{diagnostic.severity}] {diagnostic.check}: {diagnostic.message}"
        text += "".join(f"\nEvidence: {e}" for e in diagnostic.evidence[:4])
        comment = _clamped_comment(view, diagnostic, text)
        if comment is not None:
            ops.append(comment)
    for outcome in outcomes:
        if outcome.item.comment_id is not None and outcome.reply:
            ops.append(
                Reply(parent_comment_id=outcome.item.comment_id, text=outcome.reply)
            )
    result = _apply(
        adapter,
        view,
        ops,
        reason="respond to comments and reviews",
        output=out,
        project_id=meta.id,
    )

    # -- the letter
    letter_docx: Path | None = None
    letter_md: Path | None = None
    if params.reviews is not None:
        entries = [
            _letter_entry(o, view) for o in outcomes if o.item.source == "review"
        ]
        letter_docx, letter_md = write_letter(
            params.letter or default_letter(out.parent), entries, unanswered=unverified
        )

    counts = Counter(o.category for o in outcomes)
    records = log.records()[calls_before:]
    report = RespondReport(
        file=str(params.file),
        project_id=meta.id,
        output=str(out),
        letter=[str(p) for p in (letter_docx, letter_md) if p is not None],
        outcomes=outcomes,
        skipped_threads=skipped,
        unverified_points=unverified,
        collaborator_baseline=(
            "no earlier snapshot of this document; collaborators' changes not checked"
            if previous is None
            else f"compared with the last snapshot: {len(changed)} paragraph(s) changed"
        ),
        collaborator_changes=len(changed),
        collaborator_errors=collaborator_errors,
        gate_rejections=gate_rejections,
        counts={c: counts.get(c, 0) for c in CATEGORIES},
        operations_applied=result.applied,
        operations_refused=result.refused,
        model_calls=len(records),
        input_tokens=sum(_tokens(r, "input_tokens") for r in records),
        output_tokens=sum(_tokens(r, "output_tokens") for r in records),
        seconds=round(time.monotonic() - started, 2),
    )
    report_json = report_md.with_suffix(".json")
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    report_md.write_text(_markdown(report), encoding="utf-8")
    store.append_event(
        {
            "type": "transaction",
            "source": "respond",
            "author": AUTHOR,
            "ops": len(ops),
            "applied": result.applied,
            "refused": result.refused,
            "output": str(out),
        }
    )
    store.append_event(
        {
            "type": "respond_run",
            "model": params.model,
            "decision_model": params.decision_model,
            "counts": report.counts,
            "edited_paragraphs": sorted(ops_by_paragraph),
            "reviews": str(params.reviews) if params.reviews else None,
            "model_calls": report.model_calls,
            "report": str(report_md),
        }
    )
    store.snapshot_view(
        view,
        message=f"respond: {counts.get('edit', 0)} edit(s), "
        f"{counts.get('answer', 0)} answer(s), {counts.get('author', 0)} for the authors",
    )
    return RespondRun(
        report=report,
        report_md=report_md,
        report_json=report_json,
        output=out,
        letter_docx=letter_docx,
        letter_md=letter_md,
    )


def _tokens(record: dict[str, object], key: str) -> int:
    value = record.get(key)
    return value if isinstance(value, int) else 0


async def _work_item(
    item: Item,
    view: DocumentView,
    *,
    params: RespondParams,
    decision: DecisionModel | None,
    log: ActionLog,
    sections: dict[str, str],
    paragraphs: dict[str, Paragraph],
    current: dict[str, str],
    edit_items: dict[str, list[Outcome]],
) -> Outcome:
    model = params.model
    assert model is not None  # checked before any item is worked
    anchored = paragraphs.get(item.paragraph_id or "")
    triage, decision_record = await triage_with_decision(
        item,
        anchored,
        view,
        model=model,
        log=log,
        decision=decision,
    )
    outcome = Outcome(
        item=item,
        proposed=triage.category,
        category=triage.category,
        reason=triage.reason,
        decision=decision_record,
    )
    target = anchored
    span: tuple[int, int] | None = None
    if triage.target_quote:
        found = find_in_view(view, triage.target_quote)
        if found is not None and (target is None or found[0].id == target.id):
            target, span = found[0], (found[1], found[2])
    if target is not None:
        _place(outcome, target, sections)
        if span is None:
            span = _anchor_span(item, target)
        outcome.span = span

    if triage.category == "edit":
        if target is None:
            _demote(outcome, "the passage it refers to could not be located")
            return outcome
        base = current.get(target.id, target.text)
        proposal = await propose_edit(item, target, base, view, model=model, log=log)
        try:
            span_ops(target, proposal.new_text, fields=view.fields)
        except EditRefused as exc:
            _demote(outcome, str(exc))
            return outcome
        current[target.id] = proposal.new_text.strip()
        outcome.new_text = proposal.new_text.strip()
        edit_items.setdefault(target.id, []).append(outcome)
        outcome.explanation = proposal.explanation.strip()
        _reply_for(outcome, view)
    elif triage.category == "answer":
        answer_out = await propose_answer(item, view, model=model, log=log)
        evidence = verified_quotes(view, answer_out.evidence_quotes)
        if not answer_out.answerable or not evidence or not answer_out.answer.strip():
            _demote(
                outcome,
                "the document does not answer it with a passage that can be quoted",
            )
            return outcome
        outcome.evidence = evidence
        outcome.reply = f"{answer_out.answer.strip()} See {_pointers(view, evidence)}."
        outcome.reason = answer_out.answer.strip()
    elif triage.category == "done":
        evidence = verified_quotes(view, [triage.evidence_quote])
        if not evidence:
            _demote(outcome, "no passage that already addresses it could be quoted")
            return outcome
        outcome.evidence = evidence
        _reply_for(outcome, view)
    else:
        _reply_for(outcome, view)
    return outcome


def _anchor_span(item: Item, paragraph: Paragraph) -> tuple[int, int] | None:
    if not item.anchor_text:
        return None
    return find_in_paragraph(paragraph, item.anchor_text)


async def _stage_and_gate(
    adapter: DocxAdapter,
    view: DocumentView,
    ops_by_paragraph: dict[str, list[SpanOp]],
    *,
    current: dict[str, str],
    gate: _Gate,
    baseline: list[Diagnostic],
    project_id: str,
) -> dict[str, str]:
    """Apply the edits to a staged copy; return the paragraphs that must not be written, with why.

    A paragraph is dropped when the writer refused part of its edit (it would
    be half applied), or when ``check`` finds an error in it that the original
    did not have.
    """
    staged = gate.folder / "staged.docx"
    gate.folder.mkdir(parents=True, exist_ok=True)
    flat: list[Operation] = [op for ops in ops_by_paragraph.values() for op in ops]
    _apply(
        adapter,
        view,
        flat,
        reason="staged for the check gate",
        output=staged,
        project_id=None,
    )
    after = adapter.open(staged)
    texts = {p.id: p.text for p in after.paragraphs}
    dropped: dict[str, str] = {}
    for pid in ops_by_paragraph:
        if texts.get(pid) != current[pid]:
            dropped[pid] = "the writer refused part of the change"
    diagnostics = await gate.diagnostics(staged, project_id=view.project_id)
    introduced = new_errors(
        baseline, diagnostics, paragraph_ids=set(ops_by_paragraph) - set(dropped)
    )
    for pid, found in introduced.items():
        dropped[pid] = "check found a new error: " + "; ".join(
            f"{d.check}: {d.message}" for d in found[:2]
        )
    return dropped


def _letter_entry(outcome: Outcome, view: DocumentView) -> LetterEntry:
    where = _where(outcome)
    if outcome.category == "edit":
        response = outcome.explanation
        location = f"{where}; tracked change by {AUTHOR}"
    elif outcome.category == "answer":
        response = outcome.reply or ""
        location = _pointers(view, outcome.evidence)
    elif outcome.category == "done":
        response = outcome.reply or ""
        location = _pointers(view, outcome.evidence)
    else:
        response = f"[Authors to complete] {outcome.reply or ''}".strip()
        location = where
    return LetterEntry(
        label=outcome.item.label,
        point=outcome.item.text,
        response=response,
        location=location,
    )


def _markdown(report: RespondReport) -> str:
    lines = [
        f"# Respond report: {Path(report.file).name}",
        "",
        f"- project: `{report.project_id}`",
        f"- copy: `{report.output}` ({report.operations_applied} operation(s) applied, "
        f"{len(report.operations_refused)} refused)",
        "- items: " + ", ".join(f"{n} {c}" for c, n in report.counts.items()),
        f"- collaborators' changes: {report.collaborator_baseline}; "
        f"{len(report.collaborator_errors)} check error(s)",
        f"- model calls: {report.model_calls} "
        f"({report.input_tokens} input / {report.output_tokens} output tokens)",
        f"- time: {report.seconds} s",
    ]
    if report.letter:
        lines.append("- response letter: " + ", ".join(f"`{p}`" for p in report.letter))
    lines.append("")
    needs = [o for o in report.outcomes if o.category == "author"]
    if needs:
        lines += ["## Needs the authors", ""]
        lines += [f"- {o.item.label}: {o.reply}" for o in needs]
        lines.append("")
    if report.unverified_points:
        lines += ["## Review points that could not be verified by quoting", ""]
        lines += [f"- {label}" for label in report.unverified_points]
        lines.append("")
    if report.gate_rejections:
        lines += ["## Edits not written", ""]
        lines += [f"- {text}" for text in report.gate_rejections]
        lines.append("")
    if report.collaborator_errors:
        lines += ["## Check errors in collaborators' changes", ""]
        lines += [f"- {d.check}: {d.message}" for d in report.collaborator_errors]
        lines.append("")
    if report.operations_refused:
        lines += ["## Operations the writer refused", ""]
        lines += [f"- {text}" for text in report.operations_refused]
        lines.append("")
    lines += ["## Items", ""]
    for outcome in report.outcomes:
        lines.append(
            f"- **{outcome.category}** {outcome.item.label} ({_where(outcome) or 'no location'})"
            + (
                f", triage said {outcome.proposed}"
                if outcome.proposed != outcome.category
                else ""
            )
        )
        if outcome.decision is not None:
            decided = outcome.decision
            kind = (
                "decision model (no language-model triage call)"
                if decided.saved_llm_call
                else "decision model"
            )
            lines.append(f"  - {kind}: {decided.choice} at {decided.probability:.2f}")
        if outcome.reply:
            lines.append(f"  - reply: {outcome.reply}")
    if report.skipped_threads:
        lines += ["", "## Threads left alone", ""]
        lines += [f"- {text}" for text in report.skipped_threads]
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------ the verb


def _handle(params: RespondParams) -> int:
    try:
        run = asyncio.run(run_respond(params))
    except (RespondUsageError, CheckUsageError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    report = run.report
    print(f"copy: {run.output}")
    print(f"report: {run.report_md} and {run.report_json}")
    for path in report.letter:
        print(f"response letter: {path}")
    print(", ".join(f"{n} {c}" for c, n in report.counts.items()))
    return report.exit_code


RESPOND_VERB = Verb(
    name="respond",
    summary="answer the comments in a .docx and a reviewers' report: tracked edits, "
    "replies, a response letter (example: respond paper.docx --reviews reviews.txt --model M)",
    params=RespondParams,
    handler=_handle,
)

VERBS: tuple[Verb, ...] = (RESPOND_VERB,)
