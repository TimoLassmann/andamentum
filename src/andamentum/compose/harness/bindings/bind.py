"""``bind-numbers``: bind the numbers in a document to the result files that state them.

``andamentum-word bind-numbers FILE.docx --results DIR --model M`` never
modifies FILE. For each paragraph that holds numbers:

1. code lists the *candidates*: result numbers that equal a number of the
   paragraph at the precision it is written with (so ``23.4%`` has the stored
   ``0.234`` and ``23.4`` as candidates, ``23%`` also ``0.231``);
2. the model proposes, for each number the authors report as their own result,
   which candidate it states (an id from that list, or none). It never writes a
   file name or a key, so it cannot invent a pointer;
3. code verifies every proposal: the sentence and the value must occur in the
   paragraph, the number must not already be bound, the candidate must exist,
   and its value must equal the written one within the written precision. Only
   then is a ``Bind`` op made (a hidden content control tagged ``am:<id>``,
   kind ``number``) and a ledger entry written to the project's ``claims.yaml``.

The copy ``<stem>.bound.docx`` carries the controls; the report lists what was
bound and every number that was proposed but not bound, with the reason. Every
other number of an examined paragraph is listed as not bound, found in code, so
the list does not depend on the model naming it (table, figure and section
numbers and bracketed citation numbers are left out). The model is asked once per paragraph,
sequentially; answers are cached in the project by paragraph, candidates and
model. At most ``MAX_PARAGRAPHS`` paragraphs are examined per run.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import sys
import time
from pathlib import Path

from pydantic import BaseModel, Field

from andamentum.core.agents import run_agent_with_usage

from ..checks.numbers import _NUMBER, MAX_PARAGRAPHS, _examined
from ..kernel import (
    Bind,
    DocumentView,
    Paragraph,
    ProjectStore,
    RunStats,
    Transaction,
    Verb,
)
from .common import (
    EXIT_OK,
    EXIT_USAGE,
    UsageError,
    open_document,
    output_path,
)
from ..project import refuse_taken, write_copy, written_files
from .ledger import Ledger, NumberBinding
from .results import Catalogue, ResultEntry, load_catalogue
from .written import WrittenNumber, parse_written, scale_for

AUTHOR = "andamentum bind-numbers"
MAX_CANDIDATES = 40
_NAMESPACE = "bind_numbers"

_INSTRUCTIONS = (
    "You read one paragraph of a research manuscript and a numbered list of "
    "candidate result numbers taken from the authors' result files. List each "
    "numeric claim the authors make about their OWN results (means, "
    "percentages, counts, rates, p-values, effect sizes, confidence limits). "
    "Skip years, dates, citation numbers, figure and table numbers, and "
    "numbers attributed to other work. For each claim give the full sentence "
    "exactly as written, the number exactly as written (keep a % sign, "
    "decimals and a minus), and the id of the candidate that is the SAME "
    "quantity, judging by the file, column and row names. If no candidate is "
    "the same quantity, leave the id empty. Never pick a candidate only "
    "because its value is close."
)


class Proposal(BaseModel):
    sentence: str
    value: str
    candidate: str  # an id from the list, or empty


class Proposals(BaseModel):
    proposals: list[Proposal]


class BindParams(BaseModel):
    file: Path = Field(
        json_schema_extra={"positional": True}, description="the .docx to bind"
    )
    results: Path = Field(description="folder of result files (csv, tsv, json)")
    model: str | None = Field(
        default=None, description="model id that proposes the matches (required)"
    )
    out: Path = Field(description="the bound copy to write (a .docx path)")
    report: Path | None = Field(
        default=None, description="markdown report (default: beside --out)"
    )
    project: Path | None = Field(
        default=None, description="folder that keeps the ledger and logs (default: state/ beside the output)"
    )


class BoundItem(BaseModel):
    anchor_id: str
    paragraph: int
    written: str
    sentence: str
    file: str
    key: str
    value: float


class UnmatchedItem(BaseModel):
    paragraph: int
    written: str
    sentence: str
    reason: str


class BindReport(BaseModel):
    file: str
    output: str
    project_id: str
    project_dir: str
    bound: list[BoundItem]
    unmatched: list[UnmatchedItem]
    refused: list[str]
    notes: list[str]
    model_calls: int
    input_tokens: int
    output_tokens: int
    seconds: float


class BindRun(BaseModel):
    report: BindReport
    report_md: Path
    report_json: Path
    output: Path


def number_tokens(text: str) -> list[tuple[int, int, WrittenNumber]]:
    """Every number in *text* as ``(start, end, written)``; a ``%`` is part of it."""
    found = []
    for match in _NUMBER.finditer(text):
        start, end = match.span()
        while end > start and text[end - 1] in ".,":
            end -= 1
        if text[end : end + 1] == "%":
            end += 1
        written = parse_written(text[start:end])
        if written is not None:
            found.append((start, end, written))
    return found


def candidates_for(text: str, catalogue: Catalogue) -> tuple[list[ResultEntry], int]:
    """Result numbers equal to a number of *text* at its written precision, and
    how many more there were beyond ``MAX_CANDIDATES``."""
    tokens = [written for _, _, written in number_tokens(text)]
    chosen: list[ResultEntry] = []
    for entry in catalogue.entries:
        if any(scale_for(w, entry.value) is not None for w in tokens):
            chosen.append(entry)
    return chosen[:MAX_CANDIDATES], max(0, len(chosen) - MAX_CANDIDATES)


def _find_value(
    text: str, proposal: Proposal, used: set[int]
) -> tuple[int, int] | None:
    """Offsets of the proposal's value inside its sentence, skipping spans already
    taken by an earlier proposal; None when the sentence or the value is absent."""
    sentence_at = text.find(proposal.sentence)
    value = proposal.value.strip()
    if sentence_at < 0 or not value:
        return None
    pattern = re.compile(rf"(?<![\d.,]){re.escape(value)}(?!\d|[.,]\d)")
    for match in pattern.finditer(proposal.sentence):
        start = sentence_at + match.start()
        if start not in used:
            return start, start + len(value)
    return None


def _numeric_span(text: str, where: tuple[int, int]) -> tuple[int, int]:
    """The number inside the located span, found with the scan ``write`` and the
    unlisted-number list use: ``412 mg`` binds ``412``, ``p = .003`` binds ``.003``.
    A span holding no number is returned unchanged (and then fails to parse)."""
    start, end = where
    for t_start, t_end, _ in number_tokens(text[start:end]):
        return start + t_start, start + t_end
    return where


_REFERENCE = re.compile(
    r"(?:\b(?:Tables?|Figures?|Figs?\.?|Sections?|Supplementary\s+\w+)\s+)\d+(?:\s*[-,]\s*\d+)*"
    r"|\[\d+(?:\s*[-,]\s*\d+)*\]"
)


def _sentence_around(text: str, start: int, end: int) -> str:
    """The sentence of *text* holding the span, cut at full stops, ? and ! and paragraph ends."""
    before = max(
        (text.rfind(mark, 0, start) for mark in (". ", "? ", "! ")), default=-1
    )
    ends = [i for i in (text.find(m, end) for m in (". ", "? ", "! ")) if i >= 0]
    return text[before + 2 if before >= 0 else 0 : min(ends) + 1 if ends else len(text)]


def _unlisted(
    paragraph: Paragraph, *, used: set[int], bound_spans: set[tuple[str, int, int]]
) -> list[UnmatchedItem]:
    """Numbers of the paragraph that were neither bound nor proposed and rejected.

    Found in code, so the list does not depend on the model naming every number.
    Table, figure and section numbers and bracketed citation numbers are left out.
    """
    references = [m.span() for m in _REFERENCE.finditer(paragraph.text)]
    found: list[UnmatchedItem] = []
    for start, end, _ in number_tokens(paragraph.text):
        if start in used or any(s <= start < e for s, e in references):
            continue
        if any(
            pid == paragraph.id and a_start < end and start < a_end
            for pid, a_start, a_end in bound_spans
        ):
            continue
        found.append(
            UnmatchedItem(
                paragraph=paragraph.index,
                written=paragraph.text[start:end],
                sentence=_sentence_around(paragraph.text, start, end),
                reason="not proposed by the model; not bound",
            )
        )
    return found


def _prompt(text: str, candidates: list[ResultEntry]) -> str:
    lines = [
        f"c{i}: {c.key}{f' (row {c.context!r})' if c.context else ''} = {c.value:g}"
        for i, c in enumerate(candidates, 1)
    ]
    listed = "\n".join(lines) if lines else "(none)"
    return f"Paragraph:\n{text}\n\nCandidates:\n{listed}"


async def _propose(
    paragraph: Paragraph,
    candidates: list[ResultEntry],
    *,
    model: str,
    store: ProjectStore,
    stats: RunStats,
) -> list[Proposal]:
    message = _prompt(paragraph.text, candidates)
    key = hashlib.sha256(f"{model}\x1f{message}".encode()).hexdigest()
    cached = store.cache_get(_NAMESPACE, key)
    if cached is not None:
        return Proposals.model_validate({"proposals": cached}).proposals
    output, usage = await run_agent_with_usage(
        model,
        instructions=_INSTRUCTIONS,
        output_type=Proposals,
        user_message=message,
    )
    stats.add_usage(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
    store.cache_put(_NAMESPACE, key, [p.model_dump() for p in output.proposals])
    return output.proposals


def _anchor_id(paragraph: Paragraph, start: int, key: str, taken: set[str]) -> str:
    base = hashlib.sha256(f"{paragraph.text}\x1f{start}\x1f{key}".encode()).hexdigest()
    for length in range(8, len(base) + 1):
        candidate = f"num-{base[:length]}"
        if candidate not in taken:
            return candidate
    raise AssertionError("sha256 prefixes exhausted")


async def plan_bindings(
    view: DocumentView,
    catalogue: Catalogue,
    *,
    model: str,
    store: ProjectStore,
    stats: RunStats,
) -> tuple[
    list[Bind], list[NumberBinding], list[BoundItem], list[UnmatchedItem], list[str]
]:
    """Propose with the model, verify with code; the ops, bindings and report rows."""
    paragraphs = _examined(view)
    examined, left = paragraphs[:MAX_PARAGRAPHS], paragraphs[MAX_PARAGRAPHS:]
    taken_ids = {a.id for a in view.anchors}
    bound_spans = {
        (a.paragraph_id, a.start, a.end) for a in view.anchors if a.kind == "number"
    }
    binds: list[Bind] = []
    bindings: list[NumberBinding] = []
    items: list[BoundItem] = []
    unmatched: list[UnmatchedItem] = []
    notes: list[str] = []
    ignored = 0

    for paragraph in examined:
        candidates, beyond = candidates_for(paragraph.text, catalogue)
        if beyond:
            notes.append(
                f"paragraph {paragraph.index}: {beyond} more candidate(s) than the "
                f"limit of {MAX_CANDIDATES}; the model saw the first {MAX_CANDIDATES}."
            )
        proposals = await _propose(
            paragraph, candidates, model=model, store=store, stats=stats
        )
        ids = {f"c{i}": c for i, c in enumerate(candidates, 1)}
        used: set[int] = set()
        for proposal in proposals:
            where = _find_value(paragraph.text, proposal, used)
            if where is None:
                ignored += 1
                continue
            used.add(where[0])
            start, end = _numeric_span(paragraph.text, where)
            used.add(start)
            written_text = paragraph.text[start:end]

            def reject(reason: str) -> None:
                unmatched.append(
                    UnmatchedItem(
                        paragraph=paragraph.index,
                        written=written_text,
                        sentence=proposal.sentence,
                        reason=reason,
                    )
                )

            if any(
                a_paragraph == paragraph.id and a_start < end and start < a_end
                for a_paragraph, a_start, a_end in bound_spans
            ):
                reject("already bound")
                continue
            candidate = ids.get(proposal.candidate.strip())
            if not proposal.candidate.strip():
                reject("no result number is the same quantity")
                continue
            if candidate is None:
                reject(f"{proposal.candidate!r} is not one of the candidates")
                continue
            written = parse_written(written_text)
            scale = None if written is None else scale_for(written, candidate.value)
            if written is None or scale is None:
                reject(
                    f"{candidate.key} = {candidate.value:g} does not equal "
                    f"{written_text!r} at its written precision"
                )
                continue
            anchor_id = _anchor_id(paragraph, start, candidate.key, taken_ids)
            taken_ids.add(anchor_id)
            binds.append(
                Bind(
                    paragraph_id=paragraph.id,
                    start=start,
                    end=end,
                    anchor_id=anchor_id,
                    kind="number",
                )
            )
            bindings.append(
                NumberBinding(
                    anchor_id=anchor_id,
                    sentence=proposal.sentence,
                    file=candidate.file,
                    key=candidate.key,
                    value=candidate.value,
                    context=candidate.context,
                    written=written_text,
                    scale=scale,
                )
            )
            items.append(
                BoundItem(
                    anchor_id=anchor_id,
                    paragraph=paragraph.index,
                    written=written_text,
                    sentence=proposal.sentence,
                    file=candidate.file,
                    key=candidate.key,
                    value=candidate.value,
                )
            )
        unmatched.extend(_unlisted(paragraph, used=used, bound_spans=bound_spans))
    if ignored:
        notes.append(
            f"{ignored} proposed number(s) did not occur in the text and were ignored."
        )
    if left:
        notes.append(
            f"{len(left)} paragraph(s) with numbers were not examined "
            f"(limit {MAX_PARAGRAPHS} per run)."
        )
    return binds, bindings, items, unmatched, notes


async def run_bind(params: BindParams) -> BindRun:
    """Run ``bind-numbers``; the programmatic entry the verb handler wraps."""
    started = time.monotonic()
    if not params.model:
        raise UsageError("bind-numbers needs --model (the model proposes the matches)")
    out = output_path(params.file, params.out)
    refuse_taken(written_files(params.out, params.report), error=UsageError)
    adapter, view, store, project_id = open_document(
        params.file, out=params.out, project=params.project, results=params.results
    )
    catalogue = load_catalogue(params.results)
    if not catalogue.entries:
        raise UsageError(f"no numbers found in the result files under {params.results}")
    stats = RunStats()
    binds, bindings, items, unmatched, notes = await plan_bindings(
        view, catalogue, model=params.model, store=store, stats=stats
    )
    transaction = Transaction(
        ops=list(binds),
        author=AUTHOR,
        reason="bind numbers to result files",
        source="bind-numbers",
    )
    result = write_copy(adapter, view, transaction, out=out, project_id=project_id)
    present = {a.id for a in adapter.open(out).anchors}
    kept = [b for b in bindings if b.anchor_id in present]
    Ledger(store).record(kept)
    report = BindReport(
        file=str(params.file),
        output=str(out),
        project_id=project_id,
        project_dir=str(store.path),
        bound=[i for i in items if i.anchor_id in present],
        unmatched=unmatched,
        refused=list(result.refused),
        notes=notes,
        model_calls=stats.model_calls,
        input_tokens=stats.input_tokens,
        output_tokens=stats.output_tokens,
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
            "source": "bind-numbers",
            "author": AUTHOR,
            "ops": len(binds),
            "applied": result.applied,
            "refused": result.refused,
            "output": str(out),
        }
    )
    store.snapshot_view(view, message=f"bind-numbers: {len(report.bound)} bound")
    return BindRun(
        report=report, report_md=report_md, report_json=report_json, output=out
    )


def _markdown(report: BindReport) -> str:
    lines = [
        f"# Bind-numbers report: {Path(report.file).name}",
        "",
        f"- project: `{report.project_id}` ({report.project_dir})",
        f"- bound copy: `{report.output}`",
        f"- bound: {len(report.bound)}, not bound: {len(report.unmatched)}",
        f"- model calls: {report.model_calls} "
        f"({report.input_tokens} input / {report.output_tokens} output tokens)",
        f"- time: {report.seconds} s",
        "",
        "## Bound",
        "",
    ]
    lines += [
        f"- paragraph {i.paragraph}: `{i.written}` is {i.key} = {i.value:g} (`{i.anchor_id}`)"
        for i in report.bound
    ] or ["None."]
    lines += ["", "## Not bound", ""]
    lines += [
        f"- paragraph {i.paragraph}: `{i.written}`: {i.reason}"
        for i in report.unmatched
    ] or ["None."]
    if report.refused:
        lines += ["", "## Refused by the writer", ""]
        lines += [f"- {reason}" for reason in report.refused]
    if report.notes:
        lines += ["", "## Notes", ""]
        lines += [f"- {note}" for note in report.notes]
    return "\n".join(lines) + "\n"


def _handle(params: BindParams) -> int:
    try:
        run = asyncio.run(run_bind(params))
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    report = run.report
    print(f"bound copy: {run.output}")
    print(f"report: {run.report_md} and {run.report_json}")
    print(f"{len(report.bound)} number(s) bound, {len(report.unmatched)} not bound")
    return EXIT_OK


BIND_VERB = Verb(
    name="bind-numbers",
    summary="bind the numbers in a .docx to the result files that state them "
    "(example: bind-numbers paper.docx --results results/ --model M)",
    params=BindParams,
    handler=_handle,
)

VERBS: tuple[Verb, ...] = (BIND_VERB,)
