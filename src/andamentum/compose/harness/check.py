"""The ``check`` verb: open a ``.docx``, run the checks, write a commented copy and a report.

``andamentum-word check FILE.docx`` never modifies FILE. It

1. opens the document through the docx adapter and the project folder
   (``state/`` beside the output copy, or ``--project DIR``);
2. runs the selected checks one after another (default: every registered check
   whose needs are met by the flags: network checks unless ``--offline``, the
   numbers check only with ``--results`` and ``--model``, the citation pre-gate
   only with ``--decision-model``);
3. writes a COPY (``<stem>.checked.docx``) where each located diagnostic is a
   comment signed "andamentum check"; document-level diagnostics go to the
   report only;
4. writes the report as markdown and JSON, appends events to the project log
   and commits the text view to the project's shadow git.

Exit codes: 0 no errors found, 1 errors found, 2 usage error, 3 crash (raised
to the caller, which prints the traceback). Deterministic checks that need no
network, model or results file are cached in the project by (check id, check
version, document text and ``checks.yaml``); every other check runs each time
(the HTTP answers behind the network checks are cached, and the numbers and
pre-gate checks cache their own model answers).

``run_check`` is the programmatic entry (the benchmark uses it); the verb
handler wraps it with printing and exit codes.
"""

from __future__ import annotations

import inspect
import json
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from andamentum.core.decisions import DecisionModel, DecisionModelUnavailable
from andamentum.references import ReferenceHttp

from .docx.reader import DocxReadError
from .errors import WordUsageError
from .docx.writer import DocxAdapter
from .kernel import (
    CheckContext,
    CommentOp,
    Diagnostic,
    DocumentView,
    ProjectStore,
    Transaction,
    Verb,
    all_checks,
    select,
)
from .kernel.checks import (
    Check,
    RunStats,
    cache_key,
    effective_tier,
    load_cached,
    load_precisions,
    save_cached,
)
from .kernel.models import open_decision_model
from .kernel.store import ANSWER_MAX_AGE, ProjectCache
from .project import project_dir, refuse_taken, write_copy, written_files

# Importing the check modules registers them.
from .checks import (  # noqa: F401  (imported for registration)
    abstract_body,
    citations,
    crossrefs,
    needs_citation,
    numbers,
    statistics,
    structure,
)

AUTHOR = "andamentum check"
EXIT_OK, EXIT_ERRORS, EXIT_USAGE = 0, 1, 2
_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}
_PRECISION_FILE = "precision.json"


class CheckUsageError(WordUsageError):
    """A bad invocation (missing file, unknown check, unmet need): exit code 2."""


class CheckParams(BaseModel):
    file: Path = Field(
        json_schema_extra={"positional": True}, description="the .docx to check"
    )
    results: Path | None = Field(
        default=None, description="folder of result files (csv, tsv, json) for numbers"
    )
    checks: list[str] = Field(
        default_factory=list, description="check ids to run (repeatable or comma list)"
    )
    model: str | None = Field(
        default=None, description="model id for the checks that need a model"
    )
    decision_model: str | None = Field(
        default=None, description="ollama decision model for the citation pre-gate"
    )
    out: Path = Field(description="the commented copy to write (a .docx path)")
    report: Path | None = Field(
        default=None, description="markdown report (default: beside --out)"
    )
    project: Path | None = Field(
        default=None, description="folder that keeps the ledger and logs (default: state/ beside the output)"
    )
    mailto: str | None = Field(
        default=None, description="contact address for Crossref's polite pool"
    )
    offline: bool = Field(default=False, description="make no network calls")


class SkippedCheck(BaseModel):
    id: str
    reason: str


class CheckReport(BaseModel):
    file: str
    project_id: str
    project_dir: str
    output: str
    checks_run: list[str]
    checks_from_cache: list[str]
    skipped: list[SkippedCheck]
    diagnostics: list[Diagnostic]
    counts: dict[str, int]
    comments_written: int
    comments_refused: list[str]
    network_calls: int
    http_cache_hits: int
    http_cache_misses: int
    model_calls: int
    input_tokens: int
    output_tokens: int
    seconds: float

    @property
    def exit_code(self) -> int:
        return EXIT_ERRORS if self.counts.get("error") else EXIT_OK


class CheckRun(BaseModel):
    report: CheckReport
    report_md: Path
    report_json: Path
    output: Path


# ----------------------------------------------------------------- selection


def _available(params: CheckParams) -> set[str]:
    have: set[str] = set()
    if not params.offline:
        have.add("network")
    if params.results is not None:
        have.add("results")
    if params.model:
        have.add("model")
    if params.decision_model:
        have.add("decision_model")
    return have


_NEED_HINTS = {
    "network": "needs the network (--offline is set)",
    "results": "needs --results",
    "model": "needs --model",
    "decision_model": "needs --decision-model",
}


def _unmet(check: Check, have: set[str]) -> str:
    return "; ".join(_NEED_HINTS[n] for n in check.spec.needs if n not in have)


def _split_ids(values: list[str]) -> list[str]:
    return [
        part.strip() for value in values for part in value.split(",") if part.strip()
    ]


def _select(params: CheckParams) -> tuple[list[Check], list[SkippedCheck]]:
    have = _available(params)
    requested = _split_ids(params.checks)
    if requested:
        try:
            chosen = select(ids=requested)
        except KeyError as exc:
            raise CheckUsageError(str(exc.args[0])) from None
        blocked = [c for c in chosen if _unmet(c, have)]
        if blocked:
            raise CheckUsageError(
                "; ".join(f"check {c.spec.id!r} {_unmet(c, have)}" for c in blocked)
            )
        runnable = chosen
    else:
        runnable = select(needs=have)
    chosen_ids = {c.spec.id for c in runnable}
    skipped = [
        SkippedCheck(
            id=c.spec.id,
            reason=_unmet(c, have) or "not selected by --checks",
        )
        for c in all_checks()
        if c.spec.id not in chosen_ids
    ]
    return runnable, skipped


# ------------------------------------------------------------------- the run


def _precisions(store: ProjectStore) -> dict[str, float]:
    return load_precisions(store.path / _PRECISION_FILE)


def _comment_text(diagnostic: Diagnostic) -> str:
    lines = [f"[{diagnostic.severity}] {diagnostic.check}: {diagnostic.message}"]
    lines += [f"Evidence: {item}" for item in diagnostic.evidence[:4]]
    return "\n".join(lines)


def comment_ops(view: DocumentView, diagnostics: list[Diagnostic]) -> list[CommentOp]:
    lengths = {p.id: len(p.text) for p in view.paragraphs}
    ops: list[CommentOp] = []
    for diagnostic in diagnostics:
        if (
            diagnostic.paragraph_id is None
            or diagnostic.start is None
            or diagnostic.end is None
            or diagnostic.paragraph_id not in lengths
        ):
            continue
        size = lengths[diagnostic.paragraph_id]
        start = min(diagnostic.start, size)
        end = min(max(diagnostic.end, start + 1), size)
        if end <= start:
            continue
        ops.append(
            CommentOp(
                paragraph_id=diagnostic.paragraph_id,
                start=start,
                end=end,
                text=_comment_text(diagnostic),
            )
        )
    return ops


def _outputs(params: CheckParams) -> tuple[Path, Path]:
    source = params.file
    out = params.out
    if out.resolve() == source.resolve():
        raise CheckUsageError("the output must not be the input file; check never edits it")
    files = written_files(out, params.report)
    refuse_taken(files, error=CheckUsageError)
    return out, files[1]


async def _run_one(
    check: Check, view: DocumentView, ctx: CheckContext, store: ProjectStore
) -> tuple[list[Diagnostic], bool]:
    """Diagnostics of one check, and whether they came from the project cache."""
    cacheable = not check.spec.needs
    key = ""
    if cacheable:
        # Everything the view carries (styles, heading levels, field data,
        # anchors, comments) and the project config the checks read, not only
        # the text: a fixed field payload under unchanged display text must
        # not be answered from the cache.
        key = cache_key(
            check,
            version=check.version,
            inputs=[
                view.model_dump_json(exclude={"path"}),
                json.dumps(store.config(), sort_keys=True, default=str),
            ],
        )
        cached = load_cached(store, key)
        if cached is not None:
            return cached, True
    outcome = check.run(view, ctx)
    result = await outcome if inspect.isawaitable(outcome) else outcome
    if cacheable:
        save_cached(store, key, result)
    return result, False


async def run_check(
    params: CheckParams,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    url_check: Callable[[str], tuple[bool, str]] | None = None,
    decision_handle: DecisionModel | None = None,
) -> CheckRun:
    """Run the check verb. ``transport``, ``url_check`` and ``decision_handle``
    let the benchmark and tests serve recorded answers; production leaves all
    three at their defaults (``--decision-model`` is opened and checked)."""
    started = time.monotonic()
    if decision_handle is not None and not params.decision_model:
        params = params.model_copy(update={"decision_model": decision_handle.id})
    if not params.file.is_file() or params.file.suffix.lower() != ".docx":
        raise CheckUsageError(f"{params.file} is not a .docx file")
    if params.results is not None and not params.results.is_dir():
        raise CheckUsageError(f"--results {params.results} is not a folder")
    checks, skipped = _select(params)
    out, report_md = _outputs(params)

    adapter = DocxAdapter()
    try:
        view = adapter.open(params.file)
    except DocxReadError as exc:
        raise CheckUsageError(str(exc)) from exc
    store = ProjectStore(project_dir(out, params.project))
    try:
        meta = store.init(params.file, project_id=view.project_id)
    except ValueError as exc:
        raise CheckUsageError(str(exc)) from exc

    try:
        decision, owned = await open_decision_model(
            params.decision_model, injected=decision_handle
        )
    except DecisionModelUnavailable as exc:
        raise CheckUsageError(str(exc)) from exc
    precisions = _precisions(store)
    stats = RunStats()
    models = {
        role: value
        for role, value in (
            ("model", params.model),
            ("decision_model", params.decision_model),
        )
        if value
    }

    async def count_request(_: httpx.Request) -> None:
        stats.network_calls += 1

    http: ReferenceHttp | None = None
    if not params.offline:
        client = httpx.AsyncClient(
            transport=transport,
            timeout=20.0,
            event_hooks={"request": [count_request]},
        )
        cache = ProjectCache(store.http_cache, max_age=ANSWER_MAX_AGE)
        http = (
            ReferenceHttp(
                client=client, cache=cache, mailto=params.mailto, url_check=url_check
            )
            if url_check is not None
            else ReferenceHttp(client=client, cache=cache, mailto=params.mailto)
        )
    ctx = CheckContext(
        store=store,
        offline=params.offline,
        models=models,
        results_dir=params.results,
        http=http,
        decision_model=decision,
        stats=stats,
    )

    diagnostics: list[Diagnostic] = []
    from_cache: list[str] = []
    try:
        for check in checks:
            found, cached = await _run_one(check, view, ctx, store)
            tier = effective_tier(check, precision=precisions.get(check.spec.id))
            diagnostics.extend(
                d if d.tier == tier else d.model_copy(update={"tier": tier})
                for d in found
            )
            if cached:
                from_cache.append(check.spec.id)
    finally:
        if http is not None:
            await http.client.aclose()
        if owned and decision is not None:
            await decision.aclose()

    ordered = _ordered(view, diagnostics)
    transaction = Transaction(
        ops=list(comment_ops(view, ordered)),
        author=AUTHOR,
        reason="check diagnostics as comments",
        source="check",
    )
    result = write_copy(adapter, view, transaction, out=out, project_id=meta.id)
    counts = Counter(d.severity for d in ordered)
    report = CheckReport(
        file=str(params.file),
        project_id=meta.id,
        project_dir=str(store.path),
        output=str(out),
        checks_run=[c.spec.id for c in checks],
        checks_from_cache=from_cache,
        skipped=skipped,
        diagnostics=ordered,
        counts={s: counts.get(s, 0) for s in ("error", "warning", "info")},
        comments_written=result.applied,
        comments_refused=result.refused,
        network_calls=stats.network_calls,
        http_cache_hits=store.http_cache.hits,
        http_cache_misses=store.http_cache.misses,
        model_calls=stats.model_calls,
        input_tokens=stats.input_tokens,
        output_tokens=stats.output_tokens,
        seconds=round(time.monotonic() - started, 2),
    )
    report_json = report_md.with_suffix(".json")
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    report_md.write_text(_markdown(report, view), encoding="utf-8")
    store.append_event(
        {
            "type": "transaction",
            "source": "check",
            "author": AUTHOR,
            "ops": result.applied + len(result.refused),
            "applied": result.applied,
            "refused": result.refused,
            "output": str(out),
        }
    )
    store.append_event(
        {
            "type": "check_run",
            "checks": report.checks_run,
            "from_cache": from_cache,
            "skipped": [s.id for s in skipped],
            "counts": report.counts,
            "network_calls": stats.network_calls,
            "model_calls": stats.model_calls,
            "report": str(report_md),
        }
    )
    store.snapshot_view(
        view,
        message=f"check: {report.counts['error']} error(s), "
        f"{report.counts['warning']} warning(s), {report.counts['info']} info",
    )
    return CheckRun(
        report=report, report_md=report_md, report_json=report_json, output=out
    )


def _ordered(view: DocumentView, diagnostics: list[Diagnostic]) -> list[Diagnostic]:
    index = {p.id: p.index for p in view.paragraphs}
    return sorted(
        diagnostics,
        key=lambda d: (
            d.check,
            _SEVERITY_ORDER[d.severity],
            index.get(d.paragraph_id or "", -1),
            d.start or 0,
        ),
    )


# -------------------------------------------------------------------- report


def _markdown(report: CheckReport, view: DocumentView) -> str:
    texts = {p.id: p for p in view.paragraphs}
    lines = [
        f"# Check report: {Path(report.file).name}",
        "",
        f"- project: `{report.project_id}` ({report.project_dir})",
        f"- commented copy: `{report.output}` "
        f"({report.comments_written} comment(s) written, "
        f"{len(report.comments_refused)} refused)",
        f"- errors: {report.counts['error']}, warnings: {report.counts['warning']}, "
        f"info: {report.counts['info']}",
        f"- checks run: {', '.join(report.checks_run) or 'none'}",
    ]
    if report.checks_from_cache:
        lines.append(
            f"- served from the project cache: {', '.join(report.checks_from_cache)}"
        )
    lines += [
        f"- network calls: {report.network_calls} "
        f"(HTTP cache: {report.http_cache_hits} hit(s), {report.http_cache_misses} miss(es))",
        f"- model calls: {report.model_calls} "
        f"({report.input_tokens} input / {report.output_tokens} output tokens)",
        f"- time: {report.seconds} s",
        "",
    ]
    if report.skipped:
        lines += ["## Skipped checks", ""]
        lines += [f"- `{s.id}`: {s.reason}" for s in report.skipped]
        lines.append("")
    if report.comments_refused:
        lines += ["## Comments not written", ""]
        lines += [f"- {reason}" for reason in report.comments_refused]
        lines.append("")
    by_check: dict[str, list[Diagnostic]] = {}
    for diagnostic in report.diagnostics:
        by_check.setdefault(diagnostic.check, []).append(diagnostic)
    for check_id in report.checks_run:
        found = by_check.get(check_id, [])
        lines += [f"## {check_id} ({len(found)})", ""]
        if not found:
            lines += ["No findings.", ""]
            continue
        for diagnostic in found:
            where = "document"
            paragraph = texts.get(diagnostic.paragraph_id or "")
            if paragraph is not None:
                where = f"paragraph {paragraph.index}"
                if diagnostic.start is not None and diagnostic.end is not None:
                    where += f" [{diagnostic.start}:{diagnostic.end}]"
            lines.append(
                f"- **{diagnostic.severity}** (tier {diagnostic.tier}) {where}: "
                f"{diagnostic.message}"
            )
            lines += [f"  - evidence: {item}" for item in diagnostic.evidence[:4]]
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- the verb


def _handle(params: CheckParams) -> int:
    import asyncio

    try:
        run = asyncio.run(run_check(params))
    except CheckUsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    report = run.report
    print(f"commented copy: {run.output}")
    print(f"report: {run.report_md} and {run.report_json}")
    print(
        f"{report.counts['error']} error(s), {report.counts['warning']} warning(s), "
        f"{report.counts['info']} info; checks run: {', '.join(report.checks_run)}"
    )
    for skipped in report.skipped:
        print(f"skipped {skipped.id}: {skipped.reason}")
    return report.exit_code


CHECK_VERB = Verb(
    name="check",
    summary="check a .docx (citations, cross-references, statistics, numbers); "
    "writes a commented copy and a report. "
    "Example: andamentum-word check collaborator.docx --results results",
    params=CheckParams,
    handler=_handle,
)

VERBS: tuple[Verb, ...] = (CHECK_VERB,)
