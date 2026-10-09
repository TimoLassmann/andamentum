"""Loss attribution: at which stage of a run did the answer get lost?

A failed run is a diagnosis in seconds when the run log is read by code. For a
SimpleQA item the question is where the gold answer string disappears on its
way through the run: search snippets, the fetched page texts, the text given to
the note-taking call, the findings, the reflect answer, the report. For a
LitSearch query the question is the same for each gold work: was it among the
search results, was it chosen, was it read, was it cited, was it ranked in the
top 20. A finder cell is traced from the step lists its output keeps
(``ArmOutput.finder``): was it returned by the searches or by the citation hop,
was it among the candidates shown to the ranking call, did the model choose it,
did it rank in the top 20 (``trace_finder``).

The functions ``trace_simpleqa`` and ``trace_litsearch`` are pure over a
``RunLog`` and the gold, ``trace_finder`` over a finder arm's output; ``collect_losses`` reads the registry, the data and the
logs an experiment run left, and ``Losses.to_markdown`` / ``model_dump_json``
render the result. A SimpleQA item counts as failed when it was not graded
correct. The events and their fields are those of the run log catalogue
(``runlog.py``): ``search_results``, ``choice``, ``fetched``, ``notes_input``,
``finding_noted``, ``reflect``, ``report`` and ``final_ranking``.

Diagnoses, first match in this order (for a failed SimpleQA item):

- ``no_log``: the cell left no run log (a cell whose log was not kept);
- ``search_miss``: the gold is in no search snippet or title and in no fetched text;
- ``not_chosen``: the gold is in a snippet of a page that no ``choice`` picked or reserved;
- ``unreadable``: such a page was chosen, but its text did not give the gold (the
  fetch failed, or the text lacks it);
- ``window``: the gold is in a fetched text and not in the text given to the notes;
- ``notes``: it is in the text given to the notes and in no finding;
- ``reflect``: it is in a finding and not in the reflect answer;
- ``write``: it is in the reflect answer and not in the report;
- ``report_wrong``: it is in the report and the item was still graded incorrect.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, JsonValue

from andamentum.references import Work
from andamentum.references.works import canonical_url

from ..types import LogEvent, RunLog
from .arms import ArmOutput
from .experiment import Experiment
from .experiment_run import DataDirs, Item, cell_log_path, load_items
from .matching import contains_phrase
from .metrics import work_matches
from .registry import RunRecord, registry_path, runs_of
from .tables import table

Stage = Literal[
    "search_snippets", "fetched_text", "notes_input", "findings", "reflect", "report"
]
STAGES: tuple[Stage, ...] = (
    "search_snippets",
    "fetched_text",
    "notes_input",
    "findings",
    "reflect",
    "report",
)
Diagnosis = Literal[
    "no_log",
    "search_miss",
    "not_chosen",
    "unreadable",
    "window",
    "notes",
    "reflect",
    "write",
    "report_wrong",
]
DIAGNOSES: tuple[Diagnosis, ...] = (
    "no_log",
    "search_miss",
    "not_chosen",
    "unreadable",
    "window",
    "notes",
    "reflect",
    "write",
    "report_wrong",
)
LitStage = Literal["search_results", "choice", "read", "cited", "ranked_top_20"]
LIT_STAGES: tuple[LitStage, ...] = (
    "search_results",
    "choice",
    "read",
    "cited",
    "ranked_top_20",
)
# A finder cell: the step lists of ``FinderResult``. ``retrieved`` is searched or
# followed; ``lost_at`` is the first of FINDER_CHAIN a gold work is missing from
# (``chosen`` is shown, not chained: a work the model left out may still rank in
# the top 20 by the code's order).
FinderStage = Literal[
    "searched", "followed", "retrieved", "shown", "chosen", "ranked_top_20"
]
FINDER_STAGES: tuple[FinderStage, ...] = (
    "searched",
    "followed",
    "retrieved",
    "shown",
    "chosen",
    "ranked_top_20",
)
FINDER_CHAIN: tuple[FinderStage, ...] = ("retrieved", "shown", "ranked_top_20")
TOP = 20
FOUND = "found"


# -- reading the log ------------------------------------------------------------------


def _events(log: RunLog, name: str) -> list[LogEvent]:
    return [e for e in log.events if e.event == name]


def _rows(value: JsonValue | None) -> list[dict[str, JsonValue]]:
    """The dict entries of a JSON list (anything else gives none)."""
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _text(value: JsonValue | None) -> str:
    return value if isinstance(value, str) else ""


def _url_key(url: str) -> str:
    return canonical_url(url) or url


def _has(texts: Sequence[str], gold: str) -> bool:
    return any(t and contains_phrase(t, gold) for t in texts)


def _snippet_hit_urls(log: RunLog, gold: str) -> set[str]:
    """The (canonical) urls of the search results whose snippet or title holds *gold*."""
    hits: set[str] = set()
    for event in _events(log, "search_results"):
        for row in _rows(event.data.get("results")):
            if _has([_text(row.get("snippet")), _text(row.get("title"))], gold):
                hits.add(_url_key(_text(row.get("url"))))
    return hits


def _chosen_urls(log: RunLog) -> set[str]:
    """The urls any ``choice`` picked or reserved."""
    chosen: set[str] = set()
    for event in _events(log, "choice"):
        for key in ("picks", "reserves"):
            chosen |= {
                _url_key(_text(row.get("url"))) for row in _rows(event.data.get(key))
            }
    return chosen


# -- SimpleQA -------------------------------------------------------------------------


class SimpleQATrace(BaseModel):
    """Where the gold answer of one run was present, and the diagnosis if the
    item was lost. ``stages`` is empty when there is no log."""

    diagnosis: Diagnosis
    stages: dict[Stage, bool] = Field(default_factory=dict)


def trace_simpleqa(log: RunLog | None, gold: str) -> SimpleQATrace:
    """The stage trace and the diagnosis of one run of a failed SimpleQA item.

    Pure over the run log and the gold answer; matching is
    ``matching.contains_phrase`` (normalised, whole words). For an item that
    was graded correct the diagnosis is not meaningful (the caller does not
    ask for it).
    """
    if log is None or not log.events:
        return SimpleQATrace(diagnosis="no_log")
    snippet_hits = _snippet_hit_urls(log, gold)
    present: dict[Stage, bool] = {
        "search_snippets": bool(snippet_hits),
        "fetched_text": _has(
            [_text(e.data.get("text")) for e in _events(log, "fetched")], gold
        ),
        "notes_input": _has(
            [_text(e.data.get("text")) for e in _events(log, "notes_input")], gold
        ),
        "findings": _has(
            [
                _text(e.data.get(key))
                for e in _events(log, "finding_noted")
                for key in ("claim", "quote")
            ],
            gold,
        ),
        "reflect": _has(
            [_text(e.data.get("answer")) for e in _events(log, "reflect")], gold
        ),
        "report": _has(
            [_text(e.data.get("text")) for e in _events(log, "report")], gold
        ),
    }
    return SimpleQATrace(
        diagnosis=_diagnose(log, snippet_hits, present), stages=present
    )


def _diagnose(
    log: RunLog, snippet_hits: set[str], present: dict[Stage, bool]
) -> Diagnosis:
    if not present["search_snippets"] and not present["fetched_text"]:
        return "search_miss"
    if not present["fetched_text"]:
        if not snippet_hits & _chosen_urls(log):
            return "not_chosen"
        return "unreadable"
    if not present["notes_input"]:
        return "window"
    if not present["findings"]:
        return "notes"
    if not present["reflect"]:
        return "reflect"
    if not present["report"]:
        return "write"
    return "report_wrong"


# -- LitSearch ------------------------------------------------------------------------


class GoldTrace(BaseModel):
    """One gold work of one run and the stages it reached. ``lost_at`` is the
    first stage it is missing from, or ``found``."""

    title: str
    stages: dict[str, bool] = Field(default_factory=dict)
    lost_at: str


def _as_work(row: dict[str, JsonValue]) -> Work | None:
    """A log entry (title, url) as a ``Work``, or None when it is not one."""
    title = _text(row.get("title"))
    url = _text(row.get("url")) or None
    try:
        return Work(title=title, url=url)
    except ValueError:
        return None


def _any_match(rows: Sequence[dict[str, JsonValue]], gold: Work) -> bool:
    for row in rows:
        work = _as_work(row)
        if work is not None and work_matches(work, gold):
            return True
    return False


def trace_litsearch(log: RunLog | None, gold: Sequence[Work]) -> list[GoldTrace]:
    """For each gold work, the first stage of the run it is missing from.

    The stages: it was among the ``search_results``; a ``choice`` picked or
    reserved it; it was read (a ``fetched`` page that gave text, or text given
    to the notes); the final ranking lists it as cited; it ranks within the top
    20. Matching is ``metrics.work_matches``. A work that reached a later stage
    counts as found by the search (a citation expansion adds candidates that no
    search returned). Without a log every gold work is lost at ``no_log``.
    """
    if log is None or not log.events:
        return [GoldTrace(title=g.title, lost_at="no_log") for g in gold]
    searched = [
        row
        for e in _events(log, "search_results")
        for row in _rows(e.data.get("results"))
    ]
    chosen = [
        row
        for e in _events(log, "choice")
        for key in ("picks", "reserves")
        for row in _rows(e.data.get(key))
    ]
    read = [
        row
        for e in _events(log, "fetched")
        if e.data.get("ok") is True
        for row in [e.data]
    ] + [row for e in _events(log, "notes_input") for row in [e.data]]
    ranking = [
        row for e in _events(log, "final_ranking") for row in _rows(e.data.get("works"))
    ]
    cited = [row for row in ranking if row.get("basis") == "cited"]
    top = [
        row
        for row in ranking
        if isinstance(row.get("rank"), int) and int(str(row.get("rank"))) <= TOP
    ]
    traces: list[GoldTrace] = []
    for work in gold:
        present: dict[str, bool] = {
            "search_results": _any_match(searched, work),
            "choice": _any_match(chosen, work),
            "read": _any_match(read, work),
            "cited": _any_match(cited, work),
            "ranked_top_20": _any_match(top, work),
        }
        # a work that was chosen, read or ranked was found, however it got in
        present["search_results"] = present["search_results"] or any(
            present[s] for s in ("choice", "read", "cited", "ranked_top_20")
        )
        lost = next((s for s in LIT_STAGES if not present[s]), FOUND)
        traces.append(GoldTrace(title=work.title, stages=present, lost_at=lost))
    return traces


def trace_finder(output: ArmOutput, gold: Sequence[Work]) -> list[GoldTrace]:
    """For each gold work, the finder steps it reached (``FINDER_STAGES``) and
    the first of ``FINDER_CHAIN`` it is missing from, or ``found``.

    The step lists of ``output.finder`` hold work ids of ``output.works``;
    ``ranked_top_20`` is the first 20 of ``output.works`` (what recall@20
    reads). Matching is ``metrics.work_matches``.
    """
    if output.finder is None:
        raise ValueError("trace_finder needs a finder arm's output")
    steps = output.finder
    by_id = {w.work_id: w for w in output.works if w.work_id}

    def works_of(ids: Sequence[str]) -> list[Work]:
        return [by_id[i] for i in ids if i in by_id]

    lists: dict[FinderStage, list[Work]] = {
        "searched": works_of(steps.searched),
        "followed": works_of(steps.followed),
        "shown": works_of(steps.shown),
        "chosen": works_of(steps.chosen),
        "ranked_top_20": list(output.works[:TOP]),
    }
    traces: list[GoldTrace] = []
    for work in gold:
        present: dict[str, bool] = {
            stage: any(work_matches(w, work) for w in found)
            for stage, found in lists.items()
        }
        present["retrieved"] = present["searched"] or present["followed"]
        ordered = {s: present[s] for s in FINDER_STAGES}
        lost = next((s for s in FINDER_CHAIN if not ordered[s]), FOUND)
        traces.append(GoldTrace(title=work.title, stages=ordered, lost_at=lost))
    return traces


# -- the registry and the logs ---------------------------------------------------------


class SimpleQALoss(BaseModel):
    item_id: str
    repeat: int
    question: str
    gold: str
    diagnosis: Diagnosis
    stages: dict[Stage, bool]


class LitSearchRow(BaseModel):
    item_id: str
    repeat: int
    query: str
    gold: GoldTrace


class ArmLosses(BaseModel):
    arm: str
    simpleqa_items: int = 0
    simpleqa: list[SimpleQALoss] = Field(default_factory=list)
    litsearch_items: int = 0
    litsearch: list[LitSearchRow] = Field(default_factory=list)
    # the stage columns of the LitSearch rows: the engine's, or the finder's
    litsearch_stages: list[str] = Field(default_factory=lambda: list(LIT_STAGES))


class Losses(BaseModel):
    experiment: str
    arms: list[ArmLosses]

    def to_markdown(self, *, only_failed: bool = False) -> str:
        return render_markdown(self, only_failed=only_failed)


def cell_jsonl_path(out_dir: Path, *, arm: str, item_id: str, repeat: int) -> Path:
    """The run log of a cell as lines of JSON (beside its Markdown log)."""
    md = cell_log_path(out_dir, arm=arm, item_id=item_id, repeat=repeat)
    return md.with_name(md.name.removesuffix(".md") + ".jsonl")


def read_cell_log(out_dir: Path, record: RunRecord) -> RunLog | None:
    """The run log a cell left, or None when it left none."""
    path = cell_jsonl_path(
        out_dir, arm=record.arm, item_id=record.item_id, repeat=record.repeat
    )
    if not path.is_file():
        return None
    return RunLog.from_jsonl(path.read_text(encoding="utf-8"))


def _failed(record: RunRecord) -> bool:
    return bool(record.error) or record.metrics.get("correct") != 1.0


def collect_losses(
    exp: Experiment,
    *,
    out_dir: Path,
    dirs: DataDirs,
    arms: Sequence[str] | None = None,
) -> Losses:
    """Read the registry, the data and the run logs of *out_dir* and attribute
    every SimpleQA failure and every LitSearch gold work of the chosen arms.

    *arms* defaults to every arm of the experiment. A set is read only when some chosen arm has cells on it; then its
    data directory is required.
    """
    path = registry_path(out_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist: run the experiment first")
    known = {a.name for a in exp.arms}
    if arms is not None:
        unknown = [a for a in arms if a not in known]
        if unknown:
            raise ValueError(f"not arms of this experiment: {', '.join(unknown)}")
        chosen = list(arms)
    else:
        chosen = [a.name for a in exp.arms]
    runs = runs_of(path, experiment_hash=exp.config_hash())
    items: dict[str, dict[str, Item]] = {}
    for set_spec in exp.sets:
        if any(r.set == set_spec.name and r.arm in chosen for r in runs):
            items[set_spec.name] = {i.item_id: i for i in load_items(set_spec, dirs)}
    result: list[ArmLosses] = []
    for arm in chosen:
        losses = ArmLosses(arm=arm)
        for record in sorted(
            (r for r in runs if r.arm == arm),
            key=lambda r: (r.set, r.item_id, r.repeat),
        ):
            item = items.get(record.set, {}).get(record.item_id)
            if item is None:
                raise ValueError(
                    f"{record.set}/{record.item_id} is in the registry but not in "
                    "the data the experiment file selects"
                )
            log = read_cell_log(out_dir, record)
            if item.simpleqa is not None:
                losses.simpleqa_items += 1
                if not _failed(record):
                    continue
                trace = trace_simpleqa(log, item.simpleqa.answer)
                losses.simpleqa.append(
                    SimpleQALoss(
                        item_id=item.item_id,
                        repeat=record.repeat,
                        question=item.question,
                        gold=item.simpleqa.answer,
                        diagnosis=trace.diagnosis,
                        stages=trace.stages,
                    )
                )
            elif item.litsearch is not None:
                losses.litsearch_items += 1
                output = record.output
                if output is not None and output.finder is not None:
                    traces = trace_finder(output, item.litsearch.gold)
                    losses.litsearch_stages = list(FINDER_STAGES)
                else:
                    traces = trace_litsearch(log, item.litsearch.gold)
                losses.litsearch += [
                    LitSearchRow(
                        item_id=item.item_id,
                        repeat=record.repeat,
                        query=item.question,
                        gold=g,
                    )
                    for g in traces
                ]
        result.append(losses)
    return Losses(experiment=exp.name, arms=result)


# -- rendering --------------------------------------------------------------------------

_MARK = {True: "+", False: "-"}


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def _label(item_id: str, repeat: int) -> str:
    return item_id if repeat == 1 else f"{item_id}#r{repeat}"


def _simpleqa_section(losses: ArmLosses) -> list[str]:
    lines = [
        f"{losses.simpleqa_items} items, {len(losses.simpleqa)} not graded correct.",
        "",
    ]
    counts = Counter(row.diagnosis for row in losses.simpleqa)
    lines += [
        table(
            ["diagnosis", "count"],
            [[d, str(counts[d])] for d in DIAGNOSES if counts[d]]
            or [["(no failed item)", "0"]],
        ),
        "",
    ]
    if losses.simpleqa:
        lines += [
            table(
                ["item", "question", "gold", "diagnosis", *STAGES],
                [
                    [
                        _label(row.item_id, row.repeat),
                        _cell(row.question),
                        _cell(row.gold),
                        row.diagnosis,
                        *[
                            _MARK[row.stages[s]] if s in row.stages else "n/a"
                            for s in STAGES
                        ],
                    ]
                    for row in losses.simpleqa
                ],
            ),
            "",
            "`+`: the gold answer is in that stage's text, `-`: it is not, `n/a`: no log.",
            "",
        ]
    return lines


def _litsearch_section(losses: ArmLosses, *, only_failed: bool) -> list[str]:
    rows = [r for r in losses.litsearch if not only_failed or r.gold.lost_at != FOUND]
    counts = Counter(r.gold.lost_at for r in losses.litsearch)
    stages = losses.litsearch_stages
    order = [*stages, "no_log", FOUND]
    lines = [
        f"{losses.litsearch_items} queries, {len(losses.litsearch)} gold works.",
        "",
        table(
            ["first stage missing", "gold works"],
            [[s, str(counts[s])] for s in order if counts[s]]
            or [["(no gold work)", "0"]],
        ),
        "",
    ]
    if rows:
        lines += [
            table(
                ["query", "gold work", "lost at", *stages],
                [
                    [
                        _label(r.item_id, r.repeat),
                        _cell(r.gold.title),
                        r.gold.lost_at,
                        *[
                            _MARK[r.gold.stages[s]] if s in r.gold.stages else "n/a"
                            for s in stages
                        ],
                    ]
                    for r in rows
                ],
            ),
            "",
        ]
    return lines


def render_markdown(losses: Losses, *, only_failed: bool = False) -> str:
    """The loss tables: per arm, a table of diagnoses and counts and one row per
    failed SimpleQA item; a table of first-missing stages and one row per gold
    work for LitSearch (``only_failed`` leaves out the gold works that were found)."""
    lines = [f"# Losses in {losses.experiment}", ""]
    for arm in losses.arms:
        if arm.simpleqa_items:
            lines += [f"## SimpleQA, arm {arm.arm}", ""]
            lines += _simpleqa_section(arm)
        if arm.litsearch_items:
            lines += [f"## LitSearch, arm {arm.arm}", ""]
            lines += _litsearch_section(arm, only_failed=only_failed)
        if not (arm.simpleqa_items or arm.litsearch_items):
            lines += [
                f"## Arm {arm.arm}",
                "",
                "No cell of this arm in the registry.",
                "",
            ]
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "DIAGNOSES",
    "FINDER_STAGES",
    "STAGES",
    "ArmLosses",
    "GoldTrace",
    "Losses",
    "SimpleQATrace",
    "collect_losses",
    "render_markdown",
    "trace_finder",
    "trace_litsearch",
    "trace_simpleqa",
]
