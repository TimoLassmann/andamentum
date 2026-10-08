"""Running an experiment: every arm on the same items, one cell at a time.

``run_experiment`` walks the items of each set (every arm's cell of the first
item, then of the second, and so on, so the arms meet the same state of the
network and a stopped run leaves paired cells behind), runs each (arm, item,
repeat) cell, and appends a registry line after it. A cell already in the
registry is skipped, so an interrupted run resumes where it stopped.
``concurrency.resolve_concurrency`` allows more than one cell in flight only
when every language model is an Ollama ``-cloud`` model (at most 4). A cell
that raises is recorded with its error and counts against the arm (outcome
metrics are zero), never dropped. Arms with ``solo`` set (clean-timing re-runs
of a few items) run last, one cell at a time whatever the concurrency, so
nothing else of the experiment shares their wall time.

Every cell is live. The web is probed before each cell that searches it (a
web set), and a cell whose search reports failed
searches is not a result. By default the run stops with an ``InfrastructureDown``
error before that cell is recorded, so a resume runs it again once the outage
is over. An arm registered with ``outage: record`` goes on instead: the cell is
recorded as failed (``outage`` set, error text, no metric values at all, so it
is neither a zero nor a pairing partner) and the report counts it.

Each cell's run log is written under ``<out-dir>/logs/<arm>/`` (Markdown and
JSON lines); ``attribution`` reads those to say where a lost answer was lost.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..runlog import write_log
from ..sources.base import SourceConfig
from ..store import ResearchStore
from ..types import Mode, RunLog
from .arms import AnswerArm, ArmOutput, ResearchArm
from .concurrency import resolve_concurrency, run_bounded
from .experiment import ArmSpec, Experiment, SetSpec
from .health import SearxDown, SearxHealth
from .litsearch_local import (
    DEFAULT_SUBSET,
    LitSearchItem,
    load_litsearch_local,
)
from .metrics import DEFAULT_KS, Cost, retrieval_score
from .registry import (
    ArmRecord,
    RunRecord,
    StartRecord,
    append,
    check_same_experiment,
    registry_path,
    runs_of,
)
from .simpleqa import AnswerGrader, SimpleQAItem, load_simpleqa

# the sets scored by recall of gold papers
LITSEARCH_SETS = ("litsearch-web",)

# Metrics where a failed item is a zero, not an absent value: a crashing arm
# must not look better than a working one.
OUTCOME_METRICS: dict[str, tuple[str, ...]] = {
    "litsearch-web": (
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "recall_at_50",
        "found_any",
    ),
    "simpleqa": ("correct", "attempted"),
}


@dataclass(frozen=True)
class Item:
    """One thing an arm is run on, with what its metrics need."""

    set_name: str
    item_id: str
    question: str
    litsearch: LitSearchItem | None = None
    simpleqa: SimpleQAItem | None = None


@dataclass
class DataDirs:
    """Where the downloaded data is; named on the command line."""

    litsearch_local: Path | None = None
    simpleqa: Path | None = None


def load_items(spec: SetSpec, dirs: DataDirs) -> list[Item]:
    """The deterministic items of one set, in a fixed order."""
    if spec.name == "litsearch-web":
        # the LitSearch queries and gold papers, searched on the open web and
        # literature; a gold paper is matched by title (its local id is in no
        # live source)
        if dirs.litsearch_local is None:
            raise ValueError(f"the {spec.name} set needs --litsearch-local-dir")
        local = load_litsearch_local(
            dirs.litsearch_local,
            n=spec.n or DEFAULT_SUBSET,
            seed=spec.seed,
            skip=spec.skip,
        )
        return [
            Item(spec.name, i.query.query_id, i.query.query, litsearch=i)
            for i in local.items
        ]
    if dirs.simpleqa is None:
        raise ValueError("the simpleqa set needs --simpleqa-dir")
    return [
        Item("simpleqa", q.item_id, q.question, simpleqa=q)
        for q in load_simpleqa(
            dirs.simpleqa,
            n=spec.n or DEFAULT_SUBSET,
            seed=spec.seed,
            skip=spec.skip,
        )
    ]


# -- metrics ------------------------------------------------------------------------


def cost_metrics(cost: Cost | None) -> dict[str, float | None]:
    if cost is None:
        return {}
    return {
        "wall_s": cost.wall_s,
        "llm_calls": None if cost.llm_calls is None else float(cost.llm_calls),
        "input_tokens": None if cost.input_tokens is None else float(cost.input_tokens),
        "output_tokens": None
        if cost.output_tokens is None
        else float(cost.output_tokens),
    }


def litsearch_metrics(
    output: ArmOutput, item: LitSearchItem
) -> dict[str, float | None]:
    score = retrieval_score(output.works, item.gold, ks=DEFAULT_KS)
    metrics: dict[str, float | None] = {
        f"recall_at_{k}": score.recall_at[k] for k in DEFAULT_KS
    }
    metrics["found_any"] = 1.0 if any(r is not None for r in score.gold_ranks) else 0.0
    metrics["kept_precision"] = score.kept_precision
    metrics["kept_n"] = float(score.kept_n)
    return metrics


# -- arms ---------------------------------------------------------------------------


class InfrastructureDown(RuntimeError):
    """A cell met a search or source outage; its result would say nothing
    about the system, so the run stops without recording it."""


SEARCH_FAILURE_MARKERS = ("search failed for", "searches failed")


def infrastructure_failure(output: ArmOutput) -> str | None:
    """The first note of *output* that reports failed searches, if any."""
    for note in output.degraded:
        if any(marker in note for marker in SEARCH_FAILURE_MARKERS):
            return note
    return None


@dataclass
class Services:
    """What the arms and checks need that is not in the experiment file."""

    source_config: SourceConfig = field(default_factory=SourceConfig)
    # grades the simpleqa answers (the experiment file names its model)
    grader: AnswerGrader | None = None
    # a shared read-through HTTP cache for the arms
    http_cache: ResearchStore | None = None
    # no API key is read or used (the experiment file says so); recorded in the registry
    keyless: bool = False
    # probes the web search before each cell that uses it (the CLI sets it; None: no probe)
    check_searxng: Callable[[str | None], Awaitable[SearxHealth]] | None = None
    # tests replace this to run stub arms; the default builds the real ones
    answer_arm: Callable[[ArmSpec, Mode], AnswerArm] | None = None


def build_answer_arm(spec: ArmSpec, mode: Mode, services: Services) -> AnswerArm:
    if services.answer_arm is not None:
        return services.answer_arm(spec, mode)
    return ResearchArm(
        mode=mode,
        model=spec.model,
        sources=services.source_config,
        http_cache=services.http_cache,
        name=spec.name,
    )


def set_mode(set_name: str) -> Mode:
    return "scholarly" if set_name in LITSEARCH_SETS else "web"


def mode_is_web(set_name: str) -> bool:
    return set_mode(set_name) == "web"


def arm_record(exp: Experiment, spec: ArmSpec) -> ArmRecord:
    """What identifies an arm in the registry."""
    return ArmRecord(
        name=spec.name,
        system=spec.system,
        model=spec.model,
        repeats=exp.repeats_of(spec),
        solo=spec.solo,
        outage=spec.outage,
    )


# -- the run ------------------------------------------------------------------------


def _error_text(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _zero_outcomes(set_name: str, metrics: dict[str, float | None]) -> None:
    for name in OUTCOME_METRICS.get(set_name, ()):
        metrics[name] = 0.0


async def _answer_cell(
    arm: AnswerArm, item: Item, *, services: Services
) -> tuple[dict[str, float | None], ArmOutput]:
    output = await arm.run(item.question)
    metrics: dict[str, float | None]
    if item.litsearch is not None:
        metrics = litsearch_metrics(output, item.litsearch)
    elif item.simpleqa is not None:
        if services.grader is None:
            raise ValueError("the simpleqa set needs the experiment's grader")
        grade = await services.grader.grade(
            question=item.simpleqa.question,
            gold=item.simpleqa.answer,
            answer=output.answer_text,
        )
        metrics = {
            "correct": 1.0 if grade == "correct" else 0.0,
            "attempted": 0.0 if grade == "not_attempted" else 1.0,
        }
    else:
        raise ValueError(f"{item.set_name}: nothing to measure for {item.item_id}")
    metrics.update(cost_metrics(output.cost))
    return metrics, output


async def run_experiment(
    exp: Experiment,
    *,
    out_dir: Path,
    dirs: DataDirs,
    services: Services,
    progress: Callable[[str], None] = lambda _line: None,
    clock: Callable[[], float] = time.time,
    only_sets: Sequence[str] | None = None,
    only_arms: Sequence[str] | None = None,
    concurrency: int = 1,
) -> list[RunRecord]:
    """Run every cell not yet in the registry; returns the records written now.

    ``only_sets`` restricts this invocation to some of the file's sets (a set
    whose data or infrastructure is not available yet); the file, its hash and
    the registry are unchanged, so a later invocation runs the rest.
    ``only_arms`` does the same for arms (the order stays the file's, solo arms
    last).

    The cells of the non-solo arms run item by item. ``concurrency`` is how many
    cells are in flight at once: 1 unless every language model of the experiment
    (the arms' and the grader's) is an Ollama ``-cloud`` model
    (``concurrency.resolve_concurrency`` refuses anything else). It is recorded
    in the registry's ``start`` line, and a cell's wall time is measured under
    that load.
    """
    models: list[str | None] = [a.model for a in exp.arms]
    if exp.grader is not None:
        models.append(exp.grader)
    concurrency = resolve_concurrency(concurrency, models=models)
    if only_sets is not None:
        unknown = [n for n in only_sets if n not in {s.name for s in exp.sets}]
        if unknown:
            raise ValueError(f"not sets of this experiment: {', '.join(unknown)}")
    chosen = [s for s in exp.sets if only_sets is None or s.name in only_sets]
    if only_arms is not None:
        unknown_arms = [n for n in only_arms if n not in {a.name for a in exp.arms}]
        if unknown_arms:
            raise ValueError(f"not arms of this experiment: {', '.join(unknown_arms)}")
    if exp.keyless and (
        services.source_config.openalex_api_key
        or services.source_config.semantic_scholar_api_key
    ):
        raise ValueError(
            "the experiment is keyless but the source configuration holds an API key"
        )
    path = registry_path(out_dir)
    check_same_experiment(path, experiment=exp.name, experiment_hash=exp.config_hash())
    items = {s.name: load_items(s, dirs) for s in chosen}
    arm_records = [arm_record(exp, spec) for spec in exp.arms]
    by_name = {r.name: r for r in arm_records}
    append(
        path,
        StartRecord(
            experiment=exp.name,
            experiment_hash=exp.config_hash(),
            started_at=clock(),
            arms=arm_records,
            notes=exp.notes,
            concurrency=concurrency,
            keyless=exp.keyless,
        ),
    )
    done = {r.cell for r in runs_of(path, experiment_hash=exp.config_hash())}
    # (item position, set position, arm position) -> the cell's job
    shared: list[tuple[tuple[int, int, int], Callable[[], Awaitable[RunRecord]]]] = []
    solo: list[list[Callable[[], Awaitable[RunRecord]]]] = []
    # clean-timing (solo) arms go last, one cell at a time
    ordered = [a for a in exp.arms if not a.is_solo] + exp.solo_arms
    for arm_position, spec in enumerate(ordered):
        if only_arms is not None and spec.name not in only_arms:
            continue
        info = by_name[spec.name]
        # one arm object per mode
        answer_arms: dict[Mode, AnswerArm] = {}
        jobs: list[Callable[[], Awaitable[RunRecord]]] = []
        for set_position, set_spec in enumerate(chosen):
            if not spec.runs_on(set_spec.name):
                continue
            arm_items = items[set_spec.name]
            if spec.solo is not None:
                arm_items = arm_items[: spec.solo]
            for item_position, item in enumerate(arm_items):
                for repeat in range(1, exp.repeats_of(spec) + 1):
                    cell = (spec.name, item.set_name, item.item_id, repeat)
                    if cell in done:
                        continue
                    mode = set_mode(set_spec.name)
                    if mode not in answer_arms:
                        answer_arms[mode] = build_answer_arm(spec, mode, services)
                    job = functools.partial(
                        _run_cell,
                        exp=exp,
                        spec=spec,
                        info=info,
                        arm=answer_arms[mode],
                        item=item,
                        repeat=repeat,
                        path=path,
                        out_dir=out_dir,
                        services=services,
                        progress=progress,
                        clock=clock,
                    )
                    if spec.is_solo:
                        jobs.append(job)
                    else:
                        key = (item_position, set_position, arm_position)
                        shared.append((key, job))
        if jobs:
            solo.append(jobs)
    shared.sort(key=lambda entry: entry[0])
    written = await run_bounded([job for _key, job in shared], concurrency=concurrency)
    for jobs in solo:
        written += await run_bounded(jobs, concurrency=1)
    return written


LOG_DIR = "logs"


def cell_log_path(out_dir: Path, *, arm: str, item_id: str, repeat: int) -> Path:
    """Where a cell's run log goes: ``<out-dir>/logs/<arm>/<item>-r<repeat>.log.md``
    (the ``.jsonl`` beside it)."""
    return out_dir / LOG_DIR / arm / f"{item_id}-r{repeat}.log.md"


def write_cell_log(log: RunLog, *, out_dir: Path, cell: RunRecord) -> None:
    write_log(
        log,
        md_path=cell_log_path(
            out_dir, arm=cell.arm, item_id=cell.item_id, repeat=cell.repeat
        ),
    )


async def _run_cell(
    *,
    exp: Experiment,
    spec: ArmSpec,
    info: ArmRecord,
    arm: AnswerArm,
    item: Item,
    repeat: int,
    path: Path,
    out_dir: Path,
    services: Services,
    progress: Callable[[str], None],
    clock: Callable[[], float],
) -> RunRecord:
    """One (arm, item, repeat) cell: run it, record it, append it to the registry.

    A cell that raises is recorded with its error; an ``InfrastructureDown`` (a
    failed search) is not recorded and propagates (the other cells in flight are
    cancelled).
    """
    # the engine searches the web only in web mode
    searches_web = mode_is_web(item.set_name)
    progress(f"{spec.name} {item.set_name}/{item.item_id} repeat {repeat}")
    record = RunRecord(
        experiment=exp.name,
        experiment_hash=exp.config_hash(),
        arm=spec.name,
        set=item.set_name,
        item_id=item.item_id,
        repeat=repeat,
        started_at=clock(),
        model=info.model,
    )
    try:
        if searches_web and services.check_searxng is not None:
            await services.check_searxng(services.source_config.searxng_url)
        metrics, output = await _answer_cell(arm, item, services=services)
        if output.log is not None:
            write_cell_log(output.log, out_dir=out_dir, cell=record)
        if note := infrastructure_failure(output):
            raise InfrastructureDown(
                f"{spec.name} {item.set_name}/{item.item_id} "
                f"#{repeat}: {note}. The cell is not recorded; "
                "fix the outage and run the same command again"
            )
        record.output = output
        record.metrics = metrics
    except (InfrastructureDown, SearxDown) as outage:
        if spec.outage != "record":
            raise
        # recorded as failed with no metric values: not a zero, not a partner
        record.error = f"outage: {_error_text(outage)}"
        record.outage = True
        record.metrics = {}
    except Exception as error:  # recorded and counted, never dropped
        partial = getattr(error, "run_log", None)
        if isinstance(partial, RunLog):
            write_cell_log(partial, out_dir=out_dir, cell=record)
        record.error = _error_text(error)
        if spec.outage == "record":
            # a failed cell has no metric values: it drops out of the pairing
            # instead of counting as zero (it stays listed in the failed-runs
            # table, with outage False)
            record.metrics = {}
        else:
            _zero_outcomes(item.set_name, record.metrics)
    append(path, record)
    return record
