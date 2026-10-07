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

Inputs profiles (``experiment.InputsSpec``): when the file declares them, a
frozen arm names the profile of each set it runs on (``ArmSpec.inputs``); its
snapshots are ``<snapshots>/<profile>/<set>/<item>/`` and its run-time
``SourceConfig`` is the run's own updated with the profile's ``sources`` (so
its offline text sources and its citation expansion match its freeze). A
snapshot marked ``search_only`` is refused. ``only_arms`` runs some of the
file's arms (always in file order, solo arms last, snapshots checked only for
them); ``Services.decision_cache`` is a store the cells share their decision
answers through (never the solo cells).

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
import json
import shutil
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from andamentum.core.decisions import DecisionModel, DecisionModelUnavailable
from andamentum.references.works import canonical_url

from ..config import ResearchConfig, config_hash, preset
from ..dense import EmbeddingError
from ..frozen import FrozenInputs
from ..sources.base import SourceConfig
from ..store import ResearchStore
from ..runlog import write_log
from ..types import Mode, RunLog
from .concurrency import resolve_concurrency, run_bounded
from .arms import ArmOutput, Bm25Arm, ExperimentArm, LegacyArm, V2Arm
from .experiment import ArmSpec, Experiment, SetSpec
from .freeze import (
    META_FILE,
    STORE_FILE,
    SnapshotMissing,
    SnapshotSearchOnly,
    load_frozen,
    plant_into_store,
    planted_frozen,
    snapshot_dir,
    snapshot_search_only,
)
from .health import SearxDown, SearxHealth
from .judging import ClaimJudge, UrlChecker, check_links
from .litsearch import DEFAULT_SUBSET, LitSearchItem, load_litsearch
from .litsearch_local import load_litsearch_local
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
# the sets scored by recall of gold papers
LITSEARCH_SETS = ("litsearch", "litsearch-local")
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
    "litsearch-local": (
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
    litsearch_local: Path | None = None
    synergy: Path | None = None


def _web_questions(spec: SetSpec) -> list[WebQuestion]:
    if spec.skip == 0 and (spec.n is None or spec.n >= len(WEB_QUESTIONS)):
        return list(WEB_QUESTIONS)
    wanted = set(
        select_subset(
            [q.question_id for q in WEB_QUESTIONS],
            spec.skip + (spec.n if spec.n is not None else len(WEB_QUESTIONS)),
            seed=spec.seed,
        )[spec.skip :]
    )
    return [q for q in WEB_QUESTIONS if q.question_id in wanted]


def load_items(spec: SetSpec, dirs: DataDirs) -> list[Item]:
    """The deterministic items of one set, in a fixed order."""
    if spec.name == "litsearch":
        if dirs.litsearch is None:
            raise ValueError("the litsearch set needs --litsearch-dir")
        data = load_litsearch(
            dirs.litsearch, n=spec.n or DEFAULT_SUBSET, seed=spec.seed, skip=spec.skip
        )
        return [
            Item("litsearch", i.query.query_id, i.query.query, litsearch=i)
            for i in data.items
        ]
    if spec.name == "litsearch-local":
        if dirs.litsearch_local is None:
            raise ValueError("the litsearch-local set needs --litsearch-local-dir")
        local = load_litsearch_local(
            dirs.litsearch_local,
            n=spec.n or DEFAULT_SUBSET,
            seed=spec.seed,
            skip=spec.skip,
        )
        return [
            Item("litsearch-local", i.query.query_id, i.query.query, litsearch=i)
            for i in local.items
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


def failed_search_share(frozen: FrozenInputs) -> float | None:
    """The share of the snapshot's searches that failed, over every source
    (``None`` for a snapshot that recorded no search accounting)."""
    attempted = sum(frozen.searches.values())
    if attempted == 0:
        return None
    return sum(frozen.search_failures.values()) / attempted


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
    # the local embedding model of the v2 arms whose preset ranks by dense
    # similarity (``hybrid``); other arms are not given it
    embedding_model: str | None = None
    # a store the cells' decision caches read through to and write to as well
    # (``research(shared_decisions=)``); a solo (clean-timing) cell never uses it
    decision_cache: ResearchStore | None = None
    # no API key is read or used (the experiment file says so); recorded in the registry
    keyless: bool = False
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


def build_answer_arm(
    spec: ArmSpec,
    mode: Mode,
    services: Services,
    *,
    sources: SourceConfig | None = None,
) -> ExperimentArm:
    decision = services.decision_for(spec)
    if services.answer_arm is not None:
        return services.answer_arm(spec, mode, decision)
    if spec.system == "bm25":
        corpus = (sources or services.source_config).local_corpus
        if corpus is None or spec.top_k is None:
            raise ValueError(
                f"arm {spec.name}: the BM25 reference needs --local-corpus (the corpus folder)"
            )
        return Bm25Arm(corpus_dir=corpus, top_k=spec.top_k, name=spec.name)
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
        sources=sources or services.source_config,
        config=preset_of(spec),
        name=spec.name,
        http_cache=services.http_cache,
        shared_decisions=None if spec.is_solo else services.decision_cache,
        embedding_model=services.embedding_model
        if preset_of(spec).needs_embeddings
        else None,
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
    return "scholarly" if set_name in LITSEARCH_SETS else "web"


def mode_is_web(set_name: str) -> bool:
    return set_mode(set_name) == "web"


def profile_sources(
    exp: Experiment, spec: ArmSpec, set_name: str, services: Services
) -> SourceConfig:
    """The ``SourceConfig`` a frozen arm runs *set_name* with: the run's own,
    updated with the sources of the arm's inputs profile for that set. A live v2
    arm takes the experiment's ``set_sources`` for the set, when it has any (the
    BM25 reference takes the same, so both read the one corpus)."""
    name = exp.profile_of(spec, set_name)
    if name is None:
        own = exp.sources_for(set_name)
        if own is None or spec.is_frozen:
            return services.source_config
        return services.source_config.model_copy(update=own.updates())
    profile = exp.inputs[name]
    if profile.existing:
        return services.source_config
    return services.source_config.model_copy(update=profile.sources.updates())


def snapshot_root(snapshots: Path, profile: str | None) -> Path:
    """Where the snapshots of an inputs profile are (the folder itself without one)."""
    return snapshots if profile is None else snapshots / profile


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
        embedding_model=services.embedding_model
        if spec.system == "v2" and preset_of(spec).needs_embeddings
        else None,
        frozen=spec.is_frozen,
        repeats=exp.repeats_of(spec),
        solo=spec.solo,
        outage=spec.outage,
        inputs=dict(spec.inputs),
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


def _needed_snapshots(
    exp: Experiment,
    items: Mapping[str, Sequence[Item]],
    snapshots: Path | None,
    only_arms: Sequence[str] | None,
) -> list[tuple[str, Path | None]]:
    """The snapshots the frozen arms to run need: a label and the folder (None
    when no snapshots directory was given)."""
    needed: dict[str, Path | None] = {}
    for set_name, set_items in items.items():
        for arm in exp.arms:
            if not arm.is_frozen or not arm.runs_on(set_name):
                continue
            if only_arms is not None and arm.name not in only_arms:
                continue
            profile = exp.profile_of(arm, set_name)
            for item in set_items:
                base_set = "web" if item.planted else set_name
                base_id = (
                    item.web.question_id if item.planted and item.web else item.item_id
                )
                label = f"{profile + '/' if profile else ''}{base_set}/{base_id}"
                needed[label] = (
                    None
                    if snapshots is None
                    else snapshot_dir(
                        snapshot_root(snapshots, profile), base_set, base_id
                    )
                )
    return sorted(needed.items())


def missing_snapshots(
    exp: Experiment,
    items: Mapping[str, Sequence[Item]],
    snapshots: Path | None,
    only_arms: Sequence[str] | None = None,
) -> list[str]:
    """What a frozen arm needs and the snapshots directory does not hold."""
    return [
        label
        for label, folder in _needed_snapshots(exp, items, snapshots, only_arms)
        if folder is None or not (folder / STORE_FILE).exists()
    ]


def search_only_snapshots(
    exp: Experiment,
    items: Mapping[str, Sequence[Item]],
    snapshots: Path | None,
    only_arms: Sequence[str] | None = None,
) -> list[str]:
    """The needed snapshots that are marked ``search_only`` (not to be run on)."""
    return [
        label
        for label, folder in _needed_snapshots(exp, items, snapshots, only_arms)
        if folder is not None and snapshot_search_only(folder)
    ]


class SnapshotMismatch(SnapshotMissing):
    """A snapshot was not taken with the inputs profile the experiment names."""


def mismatched_snapshots(
    exp: Experiment,
    items: Mapping[str, Sequence[Item]],
    snapshots: Path | None,
    only_arms: Sequence[str] | None = None,
) -> list[str]:
    """What is wrong with the needed snapshots of an inputs profile: each entry
    names a snapshot and the check it failed. A snapshot must record the
    profile's name and hash and the experiment's keylessness, and, when the
    profile takes its plan from another, the digest of that profile's snapshot
    of the same item (so every arm reads the same sub-questions and queries).
    Snapshots of ``existing`` profiles are used as they are."""
    problems: list[str] = []
    for label, folder in _needed_snapshots(exp, items, snapshots, only_arms):
        parts = label.split("/")
        if folder is None or len(parts) != 3:
            continue
        profile_name = parts[0]
        profile = exp.inputs.get(profile_name)
        if profile is None or profile.existing:
            continue
        try:
            meta = json.loads((folder / META_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            problems.append(f"{label}: no readable {META_FILE}")
            continue
        recorded = meta.get("inputs") or {}
        if recorded.get("profile") != profile_name:
            problems.append(f"{label}: taken for profile {recorded.get('profile')!r}")
        elif recorded.get("hash") != profile.profile_hash():
            problems.append(f"{label}: taken with another definition of the profile")
        if bool(meta.get("keyless")) != exp.keyless:
            problems.append(f"{label}: keyless is {meta.get('keyless')!r}")
        if profile.plan != "live":
            plan_folder = folder.parents[2] / profile.plan / parts[1] / parts[2]
            try:
                plan_meta = json.loads(
                    (plan_folder / META_FILE).read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                problems.append(f"{label}: its plan snapshot {profile.plan} is missing")
                continue
            used = (meta.get("plan_from") or {}).get("digest")
            if used is None or used != plan_meta.get("digest"):
                problems.append(
                    f"{label}: its queries are not those of the {profile.plan} snapshot"
                )
    return problems


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
    if item.set_name in LITSEARCH_SETS and item.litsearch is not None:
        metrics = litsearch_metrics(
            output, item.litsearch, returns_pages=arm.returns_pages
        )
        if frozen is not None and (share := failed_search_share(frozen)) is not None:
            metrics["inputs_failed_search_share"] = share
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
    if output.result is not None:
        stats = output.result.stats
        metrics["decision_answers"] = float(
            stats.decision_calls + stats.decision_cache_hits
        )
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
    only_arms: Sequence[str] | None = None,
    concurrency: int = 1,
) -> list[RunRecord]:
    """Run every cell not yet in the registry; returns the records written now.

    ``only_sets`` restricts this invocation to some of the file's sets (a set
    whose data or infrastructure is not available yet); the file, its hash and
    the registry are unchanged, so a later invocation runs the rest.
    ``only_arms`` does the same for arms (snapshots are checked only for them;
    the order stays the file's, solo arms last).

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
    if (
        services.answer_arm is None
        and services.source_config.local_corpus is None
        and any(
            s.name == "litsearch-local"
            and any(
                a.runs_on(s.name) and (only_arms is None or a.name in only_arms)
                for a in exp.arms
            )
            for s in chosen
        )
    ):
        raise ValueError(
            "the litsearch-local set needs --local-corpus (the corpus folder its "
            "arms search)"
        )
    if services.answer_arm is None and services.embedding_model is None:
        dense = [
            a.name
            for a in exp.arms
            if a.system == "v2"
            and (only_arms is None or a.name in only_arms)
            and preset_of(a).needs_embeddings
        ]
        if dense:
            raise ValueError(
                f"{', '.join(dense)} rank by dense similarity and need "
                "--embedding-model (a local Ollama embedding model)"
            )
    path = registry_path(out_dir)
    check_same_experiment(path, experiment=exp.name, experiment_hash=exp.config_hash())
    items = {s.name: load_items(s, dirs) for s in chosen}
    missing = missing_snapshots(exp, items, dirs.snapshots, only_arms)
    if missing:
        raise SnapshotMissing(
            f"no snapshot for {', '.join(missing)}: run `bench freeze` for those sets"
        )
    mismatched = mismatched_snapshots(exp, items, dirs.snapshots, only_arms)
    if mismatched:
        raise SnapshotMismatch(
            "snapshots that do not match the experiment: " + "; ".join(mismatched)
        )
    search_only = search_only_snapshots(exp, items, dirs.snapshots, only_arms)
    if search_only:
        raise SnapshotSearchOnly(
            f"{', '.join(search_only)} {'is a' if len(search_only) == 1 else 'are'} "
            "search-only snapshot(s) (no texts, no citation hops, no retraction "
            "check): they compare candidate pools and are not run on"
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
            keyless=exp.keyless,
            inputs={n: i.profile_hash() for n, i in exp.inputs.items()},
        ),
    )
    done = {r.cell for r in runs_of(path, experiment_hash=exp.config_hash())}
    written: list[RunRecord] = []
    # clean-timing (solo) arms go last, one cell at a time
    ordered = [a for a in exp.arms if not a.is_solo] + exp.solo_arms
    for spec in ordered:
        if only_arms is not None and spec.name not in only_arms:
            continue
        info = by_name[spec.name]
        # one arm object per mode and inputs profile: both are part of what an
        # arm is built for (a profile sets the sources of its offline run)
        answer_arms: dict[tuple[Mode, str | None], ExperimentArm] = {}
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
                        built = (mode, exp.profile_of(spec, set_spec.name))
                        if built not in answer_arms:
                            answer_arms[built] = build_answer_arm(
                                spec,
                                mode,
                                services,
                                sources=profile_sources(
                                    exp, spec, set_spec.name, services
                                ),
                            )
                        arm = answer_arms[built]
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

    A cell that raises is recorded with its error; an ``InfrastructureDown`` (a
    failed search, or an embedding or decision model that is missing or
    unreachable) is not recorded and propagates (the other cells in flight are
    cancelled).
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
                root = snapshot_root(
                    dirs.snapshots, exp.profile_of(spec, item.set_name)
                )
                frozen = frozen_for(item, snapshots=root)
                store = prepare_working_copy(
                    item,
                    out_dir=out_dir,
                    snapshots=root,
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
            if output.log is not None:
                write_cell_log(output.log, out_dir=out_dir, cell=record)
            if live and (note := infrastructure_failure(output)):
                raise InfrastructureDown(
                    f"{spec.name} {item.set_name}/{item.item_id} "
                    f"#{repeat}: {note}. The cell is not recorded; "
                    "fix the outage and run the same command again"
                )
            record.output = output
            record.planted_outcome = outcome
        record.metrics = metrics
    except (EmbeddingError, DecisionModelUnavailable) as error:
        # a model that is missing or unreachable is the machine's state, not the
        # arm's result: never a zero; a resume runs the cell once it is fixed
        raise InfrastructureDown(
            f"{spec.name} {item.set_name}/{item.item_id} #{repeat}: "
            f"{_error_text(error)}. The cell is not recorded; fix the model "
            "and run the same command again"
        ) from error
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
