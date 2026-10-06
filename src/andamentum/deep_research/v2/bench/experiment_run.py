"""Running an experiment: every arm on the same items, strictly in sequence.

``run_experiment`` walks arm by arm (a local model is loaded once per arm, not
swapped per item), then set, item and repeat, and appends a registry line after
each cell. Cells run one at a time unless ``concurrency`` is raised, which
``concurrency.resolve_concurrency`` allows only when every language model is an
Ollama ``-cloud`` model (at most 4 cells of one arm in flight; a local Ollama
server answers one call at a time, and decision-model requests are serialised
by ``core.decisions`` whatever the value). A cell that raises is recorded with its error and counts against the
arm (outcome metrics are zero), never dropped. A cell already in the registry is
skipped, so an interrupted run resumes where it stopped. Arms with ``solo`` set
(clean-timing re-runs of a few items) run last, one cell at a time whatever the
concurrency, so nothing else of the experiment shares their wall time.

What "frozen" means for a v2 arm: Plan and Search come from the snapshot taken
by ``bench freeze`` (``freeze.py``), the sources' cached responses come from the
snapshot's store, and the run is entirely offline. Every (arm, repeat, item) cell
gets its own fresh *working copy* of that store, deleted after the cell, so no
arm or repeat can see anything an earlier one wrote and the snapshot itself is
never written to. Frozen runs fetch nothing, so their wall time excludes Plan,
Search and page fetches; the cost that includes them is the live arm's.

A live cell (the legacy arm, or a v2 arm that is not frozen) depends on the
network. SearxNG is probed before each live web cell (and before every cell of
the legacy arm, which searches the web for any set), and a live cell whose
search reports failed searches is not a result. By default the run stops with an
``InfrastructureDown`` error before that cell is recorded, so a resume runs it
again once the outage is over. An arm registered with ``outage: record`` goes
on instead: the cell is recorded as failed (``outage`` set, error text, no
metric values at all, so it is neither a zero nor a pairing partner) and the
report counts it.
"""

from __future__ import annotations

import functools
import shutil
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from andamentum.core.decisions import DecisionModel
from andamentum.references.works import canonical_url

from ..config import ResearchConfig, config_hash, preset
from ..frozen import FrozenInputs
from ..sources.base import SourceConfig
from ..store import ResearchStore
from ..types import Mode
from .concurrency import resolve_concurrency, run_bounded
from .arms import ArmOutput, ExperimentArm, LegacyArm, V2Arm
from .experiment import ArmSpec, Experiment, SetSpec
from .freeze import (
    STORE_FILE,
    SnapshotMissing,
    load_frozen,
    plant_into_store,
    planted_frozen,
    snapshot_dir,
)
from .health import SearxDown, SearxHealth
from .judging import ClaimJudge, UrlChecker, check_links
from .litsearch import DEFAULT_SUBSET, LitSearchItem, load_litsearch
from .metrics import (
    DEFAULT_KS,
    Cost,
    claim_support,
    retrieval_score,
)
from .planted import PlantedSource, planted_source
from .registry import (
    ArmRecord,
    RunRecord,
    StartRecord,
    append,
    check_same_experiment,
    registry_path,
    runs_of,
)
from .screening import ScreeningArm, parse_arm
from .stability import GOOD_PLANTED_OUTCOMES, planted_outcome
from .subsets import select_subset
from .synergy import DEFAULT_REVIEWS, ScreeningReview, load_review, sample_records
from .web_set import WEB_QUESTIONS, WebQuestion, answer_correct

WORK_DIR = "work"
NO_SNAPSHOT = "no snapshot"

# Metrics where a failed item is a zero, not an absent value: a crashing arm
# must not look better than a working one.
OUTCOME_METRICS: dict[str, tuple[str, ...]] = {
    "litsearch": (
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "recall_at_50",
        "found_any",
    ),
    "web": ("correct",),
    # adopted_false_claim is not here: a crash is not "did not adopt", and the
    # rule that reads it needs every item to have a value
    "web_planted": ("correct", "good_planted_outcome"),
    "synergy": ("kept_recall", "wss_at_95"),
}


@dataclass(frozen=True)
class Item:
    """One thing an arm is run on, with what its metrics need."""

    set_name: str
    item_id: str
    question: str
    litsearch: LitSearchItem | None = None
    web: WebQuestion | None = None
    planted: PlantedSource | None = None
    review: ScreeningReview | None = None


@dataclass
class DataDirs:
    """Where the downloaded data and the snapshots are; named on the command line."""

    snapshots: Path | None = None
    litsearch: Path | None = None
    synergy: Path | None = None


def _web_questions(spec: SetSpec) -> list[WebQuestion]:
    if spec.n is None or spec.n >= len(WEB_QUESTIONS):
        return list(WEB_QUESTIONS)
    wanted = set(
        select_subset([q.question_id for q in WEB_QUESTIONS], spec.n, seed=spec.seed)
    )
    return [q for q in WEB_QUESTIONS if q.question_id in wanted]


def load_items(spec: SetSpec, dirs: DataDirs) -> list[Item]:
    """The deterministic items of one set, in a fixed order."""
    if spec.name == "litsearch":
        if dirs.litsearch is None:
            raise ValueError("the litsearch set needs --litsearch-dir")
        data = load_litsearch(
            dirs.litsearch, n=spec.n or DEFAULT_SUBSET, seed=spec.seed
        )
        return [
            Item("litsearch", i.query.query_id, i.query.query, litsearch=i)
            for i in data.items
        ]
    if spec.name == "web":
        return [
            Item("web", q.question_id, q.question, web=q) for q in _web_questions(spec)
        ]
    if spec.name == "web_planted":
        return [
            Item(
                "web_planted",
                f"{q.question_id}-planted",
                q.question,
                web=q,
                planted=planted_source(q),
            )
            for q in _web_questions(spec)
        ]
    if dirs.synergy is None:
        raise ValueError("the synergy set needs --synergy-dir")
    items: list[Item] = []
    for key in spec.reviews or DEFAULT_REVIEWS:
        review = load_review(dirs.synergy, key, seed=spec.seed)
        sampled = review.model_copy(
            update={
                "records": sample_records(
                    review.records, spec.records, seed=review.seed
                )
            }
        )
        items.append(Item("synergy", key, review.question, review=sampled))
    return items


# -- metrics ------------------------------------------------------------------------


def cost_metrics(cost: Cost | None) -> dict[str, float | None]:
    if cost is None:
        return {}
    return {
        "wall_s": cost.wall_s,
        "llm_calls": None if cost.llm_calls is None else float(cost.llm_calls),
        "decision_calls": (
            None if cost.decision_calls is None else float(cost.decision_calls)
        ),
        "input_tokens": None if cost.input_tokens is None else float(cost.input_tokens),
        "output_tokens": None
        if cost.output_tokens is None
        else float(cost.output_tokens),
    }


def litsearch_metrics(
    output: ArmOutput, item: LitSearchItem, *, returns_pages: bool
) -> dict[str, float | None]:
    score = retrieval_score(
        output.works, item.gold, ks=DEFAULT_KS, title_contained=returns_pages
    )
    metrics: dict[str, float | None] = {
        f"recall_at_{k}": score.recall_at[k] for k in DEFAULT_KS
    }
    metrics["found_any"] = 1.0 if any(r is not None for r in score.gold_ranks) else 0.0
    metrics["kept_precision"] = score.kept_precision
    metrics["kept_n"] = float(score.kept_n)
    return metrics


def web_metrics(
    output: ArmOutput,
    question: WebQuestion,
    *,
    verdicts: Sequence[bool] | None,
    link_integrity: float | None,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "correct": 1.0 if answer_correct(output.answer_text, question.answer) else 0.0
    }
    if verdicts is not None:
        support = claim_support(verdicts, n_unsourced=output.n_unsourced)
        claims = support.n_judged + support.n_unsourced
        metrics["claim_precision"] = support.precision
        metrics["unsourced_share"] = support.n_unsourced / claims if claims else None
        metrics["n_unsourced"] = float(support.n_unsourced)
    if link_integrity is not None:
        metrics["link_integrity"] = link_integrity
    return metrics


def planted_metrics(
    output: ArmOutput, question: WebQuestion, source: PlantedSource
) -> tuple[dict[str, float | None], str]:
    """The planted-source outcome of one run and its 0/1 indicators."""
    result = output.result
    if result is None:
        raise ValueError("the planted-source measure needs a v2 result")
    correct = answer_correct(output.answer_text, question.answer)
    planted_key = canonical_url(source.url) or source.url
    planted_id = next(
        (
            w.work_id
            for w in result.works
            if (canonical_url(w.url or "") or w.url) == planted_key
        ),
        f"absent:{source.url}",
    )
    outcome = planted_outcome(
        result, planted_work_id=planted_id, answer_correct=correct
    )
    metrics: dict[str, float | None] = {
        "correct": 1.0 if correct else 0.0,
        "adopted_false_claim": 1.0 if outcome == "adopted_false_claim" else 0.0,
        "stated_conflict": 1.0 if outcome == "stated_conflict" else 0.0,
        "ignored_correct": 1.0 if outcome == "ignored_correct" else 0.0,
        "good_planted_outcome": 1.0 if outcome in GOOD_PLANTED_OUTCOMES else 0.0,
    }
    return metrics, outcome


# -- arms ---------------------------------------------------------------------------


class InfrastructureDown(RuntimeError):
    """A live cell met a search or source outage; its result would say nothing
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

    decisions: Mapping[str, DecisionModel] = field(default_factory=dict)
    source_config: SourceConfig = field(default_factory=SourceConfig)
    judge: ClaimJudge | None = None
    judge_label: str | None = None
    url_checker: UrlChecker | None = None
    judge_digest: str | None = None
    # a shared read-through HTTP cache for the live v2 arms (frozen arms are offline)
    http_cache: ResearchStore | None = None
    # probes SearxNG before each live web cell (the CLI sets it; None: no probe)
    check_searxng: Callable[[str | None], Awaitable[SearxHealth]] | None = None
    # tests replace these to run stub arms; the defaults build the real ones
    answer_arm: (
        Callable[[ArmSpec, Mode, DecisionModel | None], ExperimentArm] | None
    ) = None
    screening_arm: Callable[[ArmSpec, DecisionModel | None], ScreeningArm] | None = None

    def decision_for(self, spec: ArmSpec) -> DecisionModel | None:
        if spec.decision_model is None:
            return None
        try:
            return self.decisions[spec.decision_model]
        except KeyError:
            raise ValueError(
                f"arm {spec.name}: the decision model {spec.decision_model!r} was not opened"
            ) from None


def preset_of(spec: ArmSpec) -> ResearchConfig:
    if spec.preset is None:
        raise ValueError(f"arm {spec.name} names no preset")
    try:
        return preset(spec.preset)
    except KeyError as error:
        raise ValueError(f"arm {spec.name}: {error.args[0]}") from None


def build_answer_arm(spec: ArmSpec, mode: Mode, services: Services) -> ExperimentArm:
    decision = services.decision_for(spec)
    if services.answer_arm is not None:
        return services.answer_arm(spec, mode, decision)
    if spec.model is None:
        raise ValueError(f"arm {spec.name} needs a model")
    if spec.system == "legacy":
        if not services.source_config.searxng_url:
            raise ValueError(f"arm {spec.name}: the legacy arm needs --searxng-url")
        return LegacyArm(
            model=spec.model,
            searxng_url=services.source_config.searxng_url,
            decision_model=decision,
        )
    return V2Arm(
        mode=mode,
        model=spec.model,
        decision_model=decision,
        sources=services.source_config,
        config=preset_of(spec),
        name=spec.name,
        http_cache=services.http_cache,
    )


def build_screening_arm(spec: ArmSpec, services: Services) -> ScreeningArm:
    decision = services.decision_for(spec)
    if services.screening_arm is not None:
        return services.screening_arm(spec, decision)
    if spec.screen is None:
        raise ValueError(f"arm {spec.name} names no screen")
    kind, scorer = parse_arm(spec.screen)
    return ScreeningArm(
        kind=kind, scorer=scorer, model=spec.model, decision_model=decision
    )


def set_mode(set_name: str) -> Mode:
    return "scholarly" if set_name == "litsearch" else "web"


def mode_is_web(set_name: str) -> bool:
    return set_mode(set_name) == "web"


def arm_record(exp: Experiment, spec: ArmSpec, services: Services) -> ArmRecord:
    """What identifies an arm in the registry: config hash and model digests."""
    decision = services.decision_for(spec)
    return ArmRecord(
        name=spec.name,
        system=spec.system,
        preset=spec.preset,
        screen=spec.screen,
        config_hash=config_hash(preset_of(spec)) if spec.system == "v2" else None,
        model=spec.model,
        decision_model=decision.id if decision else spec.decision_model,
        decision_digest=decision.digest if decision else None,
        frozen=spec.is_frozen,
        repeats=exp.repeats_of(spec),
        solo=spec.solo,
        outage=spec.outage,
    )


# -- working stores -----------------------------------------------------------------


def working_dir(out_dir: Path, *, arm: str, repeat: int, item: Item) -> Path:
    """One cell's own working store directory."""
    return out_dir / WORK_DIR / arm / f"r{repeat}" / item.set_name / item.item_id


def prepare_working_copy(
    item: Item, *, out_dir: Path, snapshots: Path, arm: str, repeat: int
) -> Path:
    """A fresh copy of the item's snapshot for one cell; returns the store file.

    The snapshot is never written to and no cell shares a copy with another. For
    a planted item the copy is of the plain web snapshot with the false page
    added to the cache.
    """
    target = working_dir(out_dir, arm=arm, repeat=repeat, item=item)
    if target.exists():  # left by a cell that died: never reuse it
        shutil.rmtree(target)
    source_id = item.web.question_id if item.planted and item.web else item.item_id
    source_set = "web" if item.planted else item.set_name
    source = snapshot_dir(snapshots, source_set, source_id)
    if not (source / STORE_FILE).exists():
        raise SnapshotMissing(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    shutil.copytree(source, partial)
    if item.planted is not None:
        plant_into_store(partial / STORE_FILE, item.planted)
    partial.rename(target)
    return target / STORE_FILE


def frozen_for(item: Item, *, snapshots: Path) -> FrozenInputs:
    if item.planted is not None and item.web is not None:
        base = load_frozen(snapshot_dir(snapshots, "web", item.web.question_id))
        return planted_frozen(base, item.planted)
    return load_frozen(snapshot_dir(snapshots, item.set_name, item.item_id))


def missing_snapshots(
    exp: Experiment, items: Mapping[str, Sequence[Item]], snapshots: Path | None
) -> list[str]:
    """What a frozen arm needs and the snapshots directory does not hold."""
    missing: list[str] = []
    for set_name, set_items in items.items():
        if not any(a.is_frozen and a.runs_on(set_name) for a in exp.arms):
            continue
        for item in set_items:
            base_set = "web" if item.planted else set_name
            base_id = (
                item.web.question_id if item.planted and item.web else item.item_id
            )
            if (
                snapshots is None
                or not (
                    snapshot_dir(snapshots, base_set, base_id) / STORE_FILE
                ).exists()
            ):
                missing.append(f"{base_set}/{base_id}")
    return sorted(set(missing))


# -- the run ------------------------------------------------------------------------


def _error_text(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _zero_outcomes(set_name: str, metrics: dict[str, float | None]) -> None:
    for name in OUTCOME_METRICS.get(set_name, ()):
        metrics[name] = 0.0


async def _answer_cell(
    arm: ExperimentArm,
    spec: ArmSpec,
    item: Item,
    *,
    services: Services,
    frozen: FrozenInputs | None,
    store: Path | None,
) -> tuple[dict[str, float | None], ArmOutput, str | None]:
    output = await arm.run(item.question, frozen=frozen, store=store)
    outcome: str | None = None
    if item.set_name == "litsearch" and item.litsearch is not None:
        metrics = litsearch_metrics(
            output, item.litsearch, returns_pages=arm.returns_pages
        )
    elif item.web is not None:
        verdicts: list[bool] | None = None
        if services.judge is not None:
            verdicts = await services.judge.judge(output.claims)
        integrity: float | None = None
        if services.url_checker is not None:
            integrity = (await check_links(output.urls, services.url_checker)).integrity
        if item.planted is not None:
            metrics, outcome = planted_metrics(output, item.web, item.planted)
            metrics.update(
                {
                    k: v
                    for k, v in web_metrics(
                        output, item.web, verdicts=verdicts, link_integrity=integrity
                    ).items()
                    if k not in metrics
                }
            )
        else:
            metrics = web_metrics(
                output, item.web, verdicts=verdicts, link_integrity=integrity
            )
    else:
        raise ValueError(f"{item.set_name}: nothing to measure for {item.item_id}")
    metrics.update(cost_metrics(output.cost))
    return metrics, output, outcome


async def _screening_cell(
    arm: ScreeningArm, item: Item
) -> tuple[dict[str, float | None], RunRecord | None, Any]:
    if item.review is None:
        raise ValueError("a screening arm runs on a synergy review")
    outcome = await arm.run(item.review)
    score = outcome.score
    metrics: dict[str, float | None] = {
        "kept_recall": score.kept_recall,
        "kept_precision": score.kept_precision,
        "work_saved_kept": score.work_saved_kept,
        "wss_at_95": score.wss_at_95,
        "kept_n": float(score.kept_n),
    }
    metrics.update(cost_metrics(outcome.cost))
    return metrics, None, score


async def run_experiment(
    exp: Experiment,
    *,
    out_dir: Path,
    dirs: DataDirs,
    services: Services,
    progress: Callable[[str], None] = lambda _line: None,
    clock: Callable[[], float] = time.time,
    only_sets: Sequence[str] | None = None,
    concurrency: int = 1,
) -> list[RunRecord]:
    """Run every cell not yet in the registry; returns the records written now.

    ``only_sets`` restricts this invocation to some of the file's sets (a set
    whose data or infrastructure is not available yet); the file, its hash and
    the registry are unchanged, so a later invocation runs the rest.

    ``concurrency`` is how many cells of one arm are in flight at once (arms
    still run one after another). It is 1 unless every language model of the
    experiment (the arms' and the judge's) is an Ollama ``-cloud`` model
    (``concurrency.resolve_concurrency`` refuses anything else); the
    decision model's requests are serialised by ``core.decisions`` whatever the
    value. It is recorded in the registry's ``start`` line, and a cell's wall
    time is measured under that load.
    """
    models = [a.model for a in exp.arms]
    if exp.judge is not None:
        models.append(exp.judge.model)
    concurrency = resolve_concurrency(concurrency, models=models)
    if only_sets is not None:
        unknown = [n for n in only_sets if n not in {s.name for s in exp.sets}]
        if unknown:
            raise ValueError(f"not sets of this experiment: {', '.join(unknown)}")
    chosen = [s for s in exp.sets if only_sets is None or s.name in only_sets]
    path = registry_path(out_dir)
    check_same_experiment(path, experiment=exp.name, experiment_hash=exp.config_hash())
    items = {s.name: load_items(s, dirs) for s in chosen}
    missing = missing_snapshots(exp, items, dirs.snapshots)
    if missing:
        raise SnapshotMissing(
            f"no snapshot for {', '.join(missing)}: run `bench freeze` for those sets"
        )
    if exp.needs_judge() and services.judge is None:
        raise ValueError(
            "the experiment's rules read claim support, so its judge "
            f"({exp.judge.label if exp.judge else 'none named'}) must be open"
        )
    arm_records = [arm_record(exp, spec, services) for spec in exp.arms]
    by_name = {r.name: r for r in arm_records}
    append(
        path,
        StartRecord(
            experiment=exp.name,
            experiment_hash=exp.config_hash(),
            started_at=clock(),
            arms=arm_records,
            judge=services.judge_label,
            judge_digest=services.judge_digest,
            notes=exp.notes,
            concurrency=concurrency,
        ),
    )
    done = {r.cell for r in runs_of(path, experiment_hash=exp.config_hash())}
    written: list[RunRecord] = []
    # clean-timing (solo) arms go last, one cell at a time
    ordered = [a for a in exp.arms if not a.is_solo] + exp.solo_arms
    for spec in ordered:
        info = by_name[spec.name]
        # one arm object per mode: a set's mode is part of what an arm is built for
        answer_arms: dict[Mode, ExperimentArm] = {}
        screening_arm: ScreeningArm | None = None
        jobs: list[Callable[[], Awaitable[RunRecord]]] = []
        for set_spec in chosen:
            if not spec.runs_on(set_spec.name):
                continue
            arm_items = items[set_spec.name]
            if spec.solo is not None:
                arm_items = arm_items[: spec.solo]
            for item in arm_items:
                for repeat in range(1, exp.repeats_of(spec) + 1):
                    cell = (spec.name, item.set_name, item.item_id, repeat)
                    if cell in done:
                        continue
                    arm: ExperimentArm | ScreeningArm
                    if spec.system == "screening":
                        screening_arm = screening_arm or build_screening_arm(
                            spec, services
                        )
                        arm = screening_arm
                    else:
                        mode = set_mode(set_spec.name)
                        if mode not in answer_arms:
                            answer_arms[mode] = build_answer_arm(spec, mode, services)
                        arm = answer_arms[mode]
                    jobs.append(
                        functools.partial(
                            _run_cell,
                            exp=exp,
                            spec=spec,
                            info=info,
                            arm=arm,
                            item=item,
                            repeat=repeat,
                            path=path,
                            out_dir=out_dir,
                            dirs=dirs,
                            services=services,
                            progress=progress,
                            clock=clock,
                        )
                    )
        written += await run_bounded(
            jobs, concurrency=1 if spec.is_solo else concurrency
        )
    return written


async def _run_cell(
    *,
    exp: Experiment,
    spec: ArmSpec,
    info: ArmRecord,
    arm: ExperimentArm | ScreeningArm,
    item: Item,
    repeat: int,
    path: Path,
    out_dir: Path,
    dirs: DataDirs,
    services: Services,
    progress: Callable[[str], None],
    clock: Callable[[], float],
) -> RunRecord:
    """One (arm, item, repeat) cell: run it, record it, append it to the registry.

    A cell that raises is recorded with its error; an ``InfrastructureDown`` is
    not recorded and propagates (the other cells in flight are cancelled).
    """
    live = spec.system == "legacy" or (spec.system == "v2" and not spec.is_frozen)
    # the legacy pipeline searches the web whatever the set; v2 only in web mode
    searches_web = spec.system == "legacy" or mode_is_web(item.set_name)
    progress(f"{spec.name} {item.set_name}/{item.item_id} repeat {repeat}")
    record = RunRecord(
        experiment=exp.name,
        experiment_hash=exp.config_hash(),
        arm=spec.name,
        set=item.set_name,
        item_id=item.item_id,
        repeat=repeat,
        started_at=clock(),
        config_hash=info.config_hash,
        model=info.model,
        decision_model=info.decision_model,
        decision_digest=info.decision_digest,
        judge=services.judge_label,
        judge_digest=services.judge_digest,
        frozen=info.frozen,
    )
    store: Path | None = None
    try:
        if live and searches_web and services.check_searxng is not None:
            await services.check_searxng(services.source_config.searxng_url)
        if isinstance(arm, ScreeningArm):
            metrics, _unused, score = await _screening_cell(arm, item)
            record.screening = score
        else:
            frozen: FrozenInputs | None = None
            if spec.is_frozen:
                if dirs.snapshots is None:
                    raise SnapshotMissing(NO_SNAPSHOT)
                frozen = frozen_for(item, snapshots=dirs.snapshots)
                store = prepare_working_copy(
                    item,
                    out_dir=out_dir,
                    snapshots=dirs.snapshots,
                    arm=spec.name,
                    repeat=repeat,
                )
            metrics, output, outcome = await _answer_cell(
                arm,
                spec,
                item,
                services=services,
                frozen=frozen,
                store=store,
            )
            if live and (note := infrastructure_failure(output)):
                raise InfrastructureDown(
                    f"{spec.name} {item.set_name}/{item.item_id} "
                    f"#{repeat}: {note}. The cell is not recorded; "
                    "fix the outage and run the same command again"
                )
            record.output = output
            record.planted_outcome = outcome
        record.metrics = metrics
    except (InfrastructureDown, SearxDown) as outage:
        if spec.outage != "record":
            raise
        # recorded as failed with no metric values: not a zero, not a partner
        record.error = f"outage: {_error_text(outage)}"
        record.outage = True
        record.metrics = {}
    except Exception as error:  # recorded and counted, never dropped
        record.error = _error_text(error)
        if spec.outage == "record":
            # a failed legacy cell has no metric values: it drops out of the
            # pairing instead of counting as zero recall (it stays listed in the
            # failed-runs table, with outage False)
            record.metrics = {}
        else:
            _zero_outcomes(item.set_name, record.metrics)
    finally:
        if store is not None:
            shutil.rmtree(store.parent, ignore_errors=True)
    append(path, record)
    return record
