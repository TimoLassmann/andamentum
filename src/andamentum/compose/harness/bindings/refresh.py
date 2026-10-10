"""``refresh``: recompute the bound numbers of a document from the result files.

``andamentum-word refresh FILE.docx --results DIR`` never modifies FILE. For
every number bound by ``bind-numbers`` (a hidden content control plus a ledger
entry in ``claims.yaml``) it reads the stored value again and writes it in the
shape the text has: the decimals, percent sign, thousands separator and minus
sign of the number as it stands now. A changed number is a tracked
replacement inside its control; a number that rounds to the same text is left
alone. Numbers that were never bound are not touched.

Each sentence that holds a changed number gets one comment signed
"andamentum refresh" (``value changed 48 -> 52; check the claim still holds``).
When the change moves a p-value across .05, or a confidence interval across
its null value, the comment is an error and ``refresh`` exits 1 (see
``significance.py`` for what is recognised).

Also reported, never silently skipped: a bound number whose key is gone from the
result files (error, nothing replaced), a binding whose control was edited away
from the document (warning), a bound number whose text is no longer a number, and
a number whose text differs from what was recorded at the last bind or refresh
(someone edited it; it is replaced from the results all the same and the comment
says so). The ledger keeps the new text and value for every replacement the copy
carries.

Exit codes: 0 done without errors, 1 errors (a flip or a missing result),
2 usage error.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from pydantic import BaseModel, Field

from ..kernel import Anchor, CommentOp, DocumentView, Replace, Transaction, Verb
from .common import (
    EXIT_ERRORS,
    EXIT_OK,
    EXIT_USAGE,
    UsageError,
    open_document,
    output_path,
)
from ..project import refuse_taken, write_copy, written_files
from .ledger import Ledger, NumberBinding
from .results import load_catalogue
from .significance import Change, find_flips
from .written import format_like, parse_written

AUTHOR = "andamentum refresh"


class RefreshParams(BaseModel):
    file: Path = Field(
        json_schema_extra={"positional": True}, description="the .docx to refresh"
    )
    results: Path = Field(description="folder of result files (csv, tsv, json)")
    out: Path = Field(description="the refreshed copy to write (a .docx path)")
    report: Path | None = Field(
        default=None, description="markdown report (default: beside --out)"
    )
    project: Path | None = Field(
        default=None, description="folder that keeps the ledger and logs (default: state/ beside the output)"
    )


class ChangeItem(BaseModel):
    anchor_id: str
    paragraph: int
    old: str
    new: str
    file: str
    key: str
    sentence: str
    edited_by_hand: bool
    applied: bool


class FlipItem(BaseModel):
    paragraph: int
    kind: str
    message: str


class ProblemItem(BaseModel):
    anchor_id: str
    severity: str
    message: str


class RefreshReport(BaseModel):
    file: str
    output: str
    project_id: str
    project_dir: str
    changes: list[ChangeItem]
    flips: list[FlipItem]
    problems: list[ProblemItem]
    unchanged: int
    unbound_anchors: list[str]
    comments_written: int
    refused: list[str]
    seconds: float

    @property
    def errors(self) -> int:
        return len(self.flips) + sum(p.severity == "error" for p in self.problems)

    @property
    def exit_code(self) -> int:
        return EXIT_ERRORS if self.errors else EXIT_OK


class RefreshRun(BaseModel):
    report: RefreshReport
    report_md: Path
    report_json: Path
    output: Path


class _Planned(BaseModel):
    """One changed number: the report row, the new binding and where its sentence is."""

    item: ChangeItem
    binding: NumberBinding
    paragraph_id: str
    sentence: tuple[int, int]


def run_refresh(params: RefreshParams) -> RefreshRun:
    """Run ``refresh``; the programmatic entry the verb handler wraps."""
    started = time.monotonic()
    out = output_path(params.file, params.out)
    refuse_taken(written_files(params.out, params.report), error=UsageError)
    adapter, view, store, project_id = open_document(
        params.file, out=params.out, project=params.project, results=params.results
    )
    catalogue = load_catalogue(params.results)
    ledger = Ledger(store)
    bindings = ledger.bindings()
    anchors = {a.id: a for a in view.anchors if a.kind == "number"}

    planned, changes, problems, unchanged = _plan(view, bindings, anchors, catalogue)
    flips = [
        (paragraph_id, flip)
        for paragraph_id, found in changes.items()
        for flip in find_flips(_text(view, paragraph_id), found)
    ]
    replaces = [
        Replace(
            paragraph_id=p.paragraph_id,
            start=anchors[p.item.anchor_id].start,
            end=anchors[p.item.anchor_id].end,
            text=p.item.new,
            expect=p.item.old,
        )
        for p in planned
    ]
    comments = _comments(planned, flips)
    transaction = Transaction(
        ops=[*replaces, *comments],
        author=AUTHOR,
        reason="refresh bound numbers from the result files",
        source="refresh",
    )
    result = write_copy(adapter, view, transaction, out=out, project_id=project_id)

    written = adapter.open(out)
    texts = {p.id: p.text for p in written.paragraphs}
    after = {a.id: a for a in written.anchors}
    for p in planned:
        anchor = after.get(p.item.anchor_id)
        p.item.applied = (
            anchor is not None
            and texts[anchor.paragraph_id][anchor.start : anchor.end] == p.item.new
        )
    ledger.record([p.binding for p in planned if p.item.applied])

    index = {p.id: p.index for p in view.paragraphs}
    report = RefreshReport(
        file=str(params.file),
        output=str(out),
        project_id=project_id,
        project_dir=str(store.path),
        changes=[p.item for p in planned],
        flips=[
            FlipItem(paragraph=index[pid], kind=f.kind, message=f.message)
            for pid, f in flips
        ],
        problems=problems,
        unchanged=unchanged,
        unbound_anchors=sorted(set(anchors) - set(bindings)),
        comments_written=sum(
            1 for c in written.comments if c.author == AUTHOR and c.parent_id is None
        ),
        refused=list(result.refused),
        seconds=round(time.monotonic() - started, 2),
    )
    report_md = params.report or out.with_suffix(".md")
    report_json = report_md.with_suffix(".json")
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    report_md.write_text(_markdown(report), encoding="utf-8")
    store.append_event(
        {
            "type": "transaction",
            "source": "refresh",
            "author": AUTHOR,
            "ops": len(transaction.ops),
            "applied": result.applied,
            "refused": result.refused,
            "output": str(out),
        }
    )
    store.snapshot_view(
        view, message=f"refresh: {len(planned)} changed, {len(flips)} flip(s)"
    )
    return RefreshRun(
        report=report, report_md=report_md, report_json=report_json, output=out
    )


def _text(view: DocumentView, paragraph_id: str) -> str:
    return next(p.text for p in view.paragraphs if p.id == paragraph_id)


def _plan(
    view: DocumentView,
    bindings: dict[str, NumberBinding],
    anchors: dict[str, Anchor],
    catalogue,  # Catalogue
) -> tuple[list[_Planned], dict[str, list[Change]], list[ProblemItem], int]:
    """The changed numbers, their changes by paragraph, the problems, the unchanged count."""
    paragraphs = {p.id: p for p in view.paragraphs}
    planned: list[_Planned] = []
    changes: dict[str, list[Change]] = {}
    problems: list[ProblemItem] = []
    unchanged = 0

    def problem(anchor_id: str, severity: str, message: str) -> None:
        problems.append(
            ProblemItem(anchor_id=anchor_id, severity=severity, message=message)
        )

    for anchor_id, binding in bindings.items():
        anchor = anchors.get(anchor_id)
        if anchor is None:
            problem(
                anchor_id,
                "warning",
                f"the control is no longer in the document (was "
                f"{binding.written!r}); not refreshed",
            )
            continue
        entry = catalogue.get(binding.key)
        if entry is None:
            problem(
                anchor_id,
                "error",
                f"{binding.key!r} is not in the result files any more; the text "
                f"still says {binding.written!r}",
            )
            continue
        if binding.context and entry.context != binding.context:
            problem(
                anchor_id,
                "error",
                f"{binding.key!r} now belongs to row {entry.context!r}, not "
                f"{binding.context!r}: the result file was reordered; not refreshed",
            )
            continue
        paragraph = paragraphs[anchor.paragraph_id]
        current = paragraph.text[anchor.start : anchor.end]
        template = parse_written(current)
        if template is None:
            problem(
                anchor_id,
                "warning",
                "the bound text was deleted; not refreshed"
                if not current
                else f"the bound text {current!r} is not a number; not refreshed",
            )
            continue
        new = format_like(template, entry.value * binding.scale)
        if new == current:
            unchanged += 1
            continue
        shown = parse_written(new)
        assert shown is not None  # format_like writes what parse_written reads
        changes.setdefault(anchor.paragraph_id, []).append(
            Change(
                start=anchor.start,
                end=anchor.end,
                old=template.value,
                new=shown.value,
                old_text=current,
                new_text=new,
            )
        )
        found = paragraph.text.find(binding.sentence)
        inside = (
            found >= 0
            and found <= anchor.start
            and anchor.end <= found + len(binding.sentence)
        )
        planned.append(
            _Planned(
                item=ChangeItem(
                    anchor_id=anchor_id,
                    paragraph=paragraph.index,
                    old=current,
                    new=new,
                    file=binding.file,
                    key=binding.key,
                    sentence=binding.sentence,
                    edited_by_hand=current != binding.written,
                    applied=False,
                ),
                binding=binding.model_copy(
                    update={"written": new, "value": entry.value}
                ),
                paragraph_id=anchor.paragraph_id,
                sentence=(found, found + len(binding.sentence))
                if inside
                else (anchor.start, anchor.end),
            )
        )
    return planned, changes, problems, unchanged


def _comments(planned: list[_Planned], flips: list) -> list[CommentOp]:
    """One comment per sentence holding a changed number; an error when it flipped."""
    groups: dict[tuple[str, int, int], list[_Planned]] = {}
    for p in planned:
        groups.setdefault((p.paragraph_id, *p.sentence), []).append(p)
    ops: list[CommentOp] = []
    for (paragraph_id, start, end), group in groups.items():
        flipped = [f for pid, f in flips if pid == paragraph_id and start <= f.at < end]
        changed = ", ".join(f"{g.item.old} → {g.item.new}" for g in group)
        lines = [
            f"[{'error' if flipped else 'warning'}] refresh: "
            f"{'value' if len(group) == 1 else 'values'} changed {changed}; "
            "check the claim still holds."
        ]
        lines += [f"Significance flip: {f.message}." for f in flipped]
        lines += [
            f"{g.item.old} had been edited since it was bound; it is replaced "
            f"from {g.item.key}."
            for g in group
            if g.item.edited_by_hand
        ]
        ops.append(
            CommentOp(
                paragraph_id=paragraph_id,
                start=start,
                end=end,
                text="\n".join(lines),
            )
        )
    return ops


def _markdown(report: RefreshReport) -> str:
    lines = [
        f"# Refresh report: {Path(report.file).name}",
        "",
        f"- project: `{report.project_id}` ({report.project_dir})",
        f"- refreshed copy: `{report.output}` ({report.comments_written} comment(s) "
        f"written, {len(report.refused)} operation(s) refused)",
        f"- changed: {len(report.changes)}, unchanged: {report.unchanged}, "
        f"errors: {report.errors}",
        f"- bound numbers in the document without a ledger entry (untouched): "
        f"{len(report.unbound_anchors)}",
        f"- time: {report.seconds} s",
        "",
        "## Changed",
        "",
    ]
    lines += [
        f"- paragraph {c.paragraph}: `{c.old}` → `{c.new}` ({c.key})"
        f"{'' if c.applied else ' NOT WRITTEN'}"
        f"{' (edited by hand since it was bound)' if c.edited_by_hand else ''}"
        for c in report.changes
    ] or ["None."]
    lines += ["", "## Significance flips (errors)", ""]
    lines += [f"- paragraph {f.paragraph}: {f.message}" for f in report.flips] or [
        "None."
    ]
    lines += ["", "## Problems", ""]
    lines += [
        f"- {p.severity}: `{p.anchor_id}`: {p.message}" for p in report.problems
    ] or ["None."]
    if report.refused:
        lines += ["", "## Refused by the writer", ""]
        lines += [f"- {reason}" for reason in report.refused]
    return "\n".join(lines) + "\n"


def _handle(params: RefreshParams) -> int:
    try:
        run = run_refresh(params)
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    report = run.report
    print(f"refreshed copy: {run.output}")
    print(f"report: {run.report_md} and {run.report_json}")
    print(
        f"{len(report.changes)} number(s) changed, {report.unchanged} unchanged, "
        f"{report.errors} error(s)"
    )
    return report.exit_code


REFRESH_VERB = Verb(
    name="refresh",
    summary="recompute the bound numbers of a .docx from the result files as "
    "tracked changes (example: refresh paper.docx --results results/)",
    params=RefreshParams,
    handler=_handle,
)

VERBS: tuple[Verb, ...] = (REFRESH_VERB,)
