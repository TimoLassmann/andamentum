"""The candidate pool of a snapshot, and what became of each gold paper in a run.

Two read-only analyses over files that already exist (no model, no network):

- **Pools** (``bench pool``, the report's pool section): for the snapshots of
  one root (``<root>/litsearch/<item>/``, or ``<root>/<profile>/...``), which
  gold papers are in the candidate pool and which source found them, how many
  of them are in the top 20 of some query's result list, how large each
  source's share of the pool is, how many queries a source answered with fewer
  than the 20 results asked for, and (against another root) which gold papers
  entered or left the pool.
- **Losses** (the report's loss table): for each arm, every gold paper of every
  run falls in one category, read from the run's result and its screening
  record (``ResearchResult.screening``): ``hit`` (kept, ranked 20 or better),
  ``(c)`` kept but ranked below 20, ``(b)`` screened and dropped, ``(b')`` in
  the pool but never screened, ``(a)`` not in the pool. The categories are
  those of the round-3 diagnosis. A run that failed has no result and is
  counted apart.

A gold paper is matched with ``metrics.work_matches`` (identifiers, else the
normalised title), the way recall is.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from andamentum.references import Work

from ..frozen import FrozenInputs
from ..types import Candidate
from .freeze import FROZEN_FILE, META_FILE, snapshot_dir
from .litsearch import LitSearchItem
from .metrics import work_matches
from .registry import RunRecord
from .tables import NOTHING, fmt, table

RESULTS_ASKED = 20  # the results a query asks of each source; fewer is a short list
TOP = 20  # the rank within which a gold paper counts as found

HIT = "hit"
KEPT_LOW = "(c)"
SCREENED_OUT = "(b)"
NOT_SCREENED = "(b')"
NOT_IN_POOL = "(a)"
KEPT_EXCLUDED = "(x)"
UNRECORDED = "unrecorded"
FAILED = "failed"
LOSS_CATEGORIES: tuple[str, ...] = (
    HIT,
    KEPT_LOW,
    SCREENED_OUT,
    NOT_SCREENED,
    NOT_IN_POOL,
    KEPT_EXCLUDED,
    UNRECORDED,
    FAILED,
)
LOSS_MEANING: dict[str, str] = {
    HIT: f"kept and ranked {TOP} or better",
    KEPT_LOW: f"kept, ranked below {TOP}",
    SCREENED_OUT: "screened and dropped",
    NOT_SCREENED: "in the pool, never screened",
    NOT_IN_POOL: "not in the pool",
    KEPT_EXCLUDED: "kept, then excluded from the result (retracted)",
    UNRECORDED: "the run screened candidates but holds no screening record (it "
    "predates it), so the loss cannot be placed",
    FAILED: "the run failed (no result)",
}


# -- pools --------------------------------------------------------------------------


class SearchMeta(BaseModel):
    """The search accounting a freeze wrote into ``meta.json``."""

    model_config = ConfigDict(extra="ignore")

    searches: dict[str, int] = Field(default_factory=dict)
    search_failures: dict[str, int] = Field(default_factory=dict)
    search_s: float | None = None
    tolerated_failures: list[str] = Field(default_factory=list)


class CacheMeta(BaseModel):
    """What the freeze asked of the network, from ``meta.json``."""

    model_config = ConfigDict(extra="ignore")

    network_requests: dict[str, int] = Field(default_factory=dict)


class SnapshotMeta(BaseModel):
    """The parts of a snapshot's ``meta.json`` the pool tables read; the rest is
    ignored, and a snapshot that predates a field simply lacks it."""

    model_config = ConfigDict(extra="ignore")

    search: SearchMeta = Field(default_factory=SearchMeta)
    http_cache: CacheMeta = Field(default_factory=CacheMeta)


@dataclass(frozen=True)
class PoolItem:
    """One item's snapshot: its gold papers, the frozen inputs, what ``meta.json`` says."""

    item_id: str
    gold: Sequence[Work]
    frozen: FrozenInputs
    meta: SnapshotMeta


def load_pool_items(
    root: Path, items: Sequence[LitSearchItem], *, set_name: str = "litsearch"
) -> tuple[list[PoolItem], list[str]]:
    """The snapshots under ``root/<set>/<item>/`` of *items*, and the ids of the
    items that have none."""
    found: list[PoolItem] = []
    missing: list[str] = []
    for item in items:
        folder = snapshot_dir(root, set_name, item.query.query_id)
        if not (folder / FROZEN_FILE).exists():
            missing.append(item.query.query_id)
            continue
        meta_path = folder / META_FILE
        meta = (
            SnapshotMeta.model_validate(
                json.loads(meta_path.read_text(encoding="utf-8"))
            )
            if meta_path.exists()
            else SnapshotMeta()
        )
        found.append(
            PoolItem(
                item_id=item.query.query_id,
                gold=item.gold,
                frozen=FrozenInputs.load(folder / FROZEN_FILE),
                meta=meta,
            )
        )
    return found, missing


def matching(pool: Sequence[Candidate], gold: Work) -> list[Candidate]:
    """The candidates of *pool* that are *gold* (usually none or one)."""
    return [c for c in pool if work_matches(c.work, gold)]


def search_top(candidate: Candidate, *, top: int = TOP) -> bool:
    """Was the candidate among the first *top* results of some search list (a
    list of one source, not a citation hop)?"""
    return any(
        rank <= top for name, rank in candidate.source_ranks.items() if ":" not in name
    )


def found_by(candidates: Sequence[Candidate]) -> set[str]:
    """Every source that returned any of *candidates*."""
    return {name for c in candidates for name in c.sources}


@dataclass(frozen=True)
class GoldPoolEntry:
    """One gold paper of one item and how the pool holds it."""

    item_id: str
    gold: Work
    in_pool: bool
    in_search_top: bool
    sources: frozenset[str]


def gold_entries(item: PoolItem) -> list[GoldPoolEntry]:
    entries = []
    for gold in item.gold:
        hits = matching(item.frozen.candidates, gold)
        entries.append(
            GoldPoolEntry(
                item_id=item.item_id,
                gold=gold,
                in_pool=bool(hits),
                in_search_top=any(search_top(c) for c in hits),
                sources=frozenset(found_by(hits)),
            )
        )
    return entries


def _share(part: int, whole: int) -> str:
    return f"{part} ({part / whole:.0%})" if whole else NOTHING


def pool_tables(items: Sequence[PoolItem], *, missing: Sequence[str] = ()) -> str:
    """The candidate-pool tables of *items* (markdown)."""
    if not items:
        return "_No snapshot to read._"
    entries = [e for item in items for e in gold_entries(item)]
    n_gold = len(entries)
    in_pool = [e for e in entries if e.in_pool]
    sources = sorted(
        {s for item in items for c in item.frozen.candidates for s in c.sources}
    )
    lines: list[str] = []
    note = (
        f" ({len(missing)} item(s) without a snapshot: {', '.join(missing)})"
        if missing
        else ""
    )
    lines.append(f"{len(items)} items, {n_gold} gold papers{note}.")
    lines.append("")
    lines.append(
        table(
            ["measure", "gold papers"],
            [
                ["in the candidate pool", _share(len(in_pool), n_gold)],
                [
                    f"in the top {TOP} of some query's list",
                    _share(sum(e.in_search_top for e in entries), n_gold),
                ],
                ["not in the pool", _share(n_gold - len(in_pool), n_gold)],
            ],
        )
    )
    lines.append("")
    lines.append("Which source found the gold papers that are in the pool:")
    lines.append("")
    lines.append(
        table(
            ["source", "found by it", "found by it alone"],
            [
                [
                    name,
                    str(sum(name in e.sources for e in in_pool)),
                    str(sum(e.sources == frozenset({name}) for e in in_pool)),
                ]
                for name in sources
            ],
        )
    )
    lines.append("")
    sizes = {
        name: [sum(name in c.sources for c in item.frozen.candidates) for item in items]
        for name in sources
    }
    total_candidates = sum(len(item.frozen.candidates) for item in items)
    lines.append("Pool size per item, and each source's share of all candidates:")
    lines.append("")
    lines.append(
        table(
            [
                "source",
                "mean per item",
                "median per item",
                "max per item",
                "share of the pool",
            ],
            [
                [
                    name,
                    fmt(statistics.mean(values), 1),
                    fmt(statistics.median(values), 1),
                    str(max(values)),
                    _share(sum(values), total_candidates),
                ]
                for name, values in sizes.items()
            ],
        )
    )
    lines.append("")
    lines.append(_short_lists(items, sources))
    return "\n".join(lines)


def _short_lists(items: Sequence[PoolItem], sources: Sequence[str]) -> str:
    """Queries a source answered with fewer than ``RESULTS_ASKED`` results."""
    if not any(item.frozen.result_counts for item in items):
        return (
            f"_Queries with fewer than {RESULTS_ASKED} results per source: n/a "
            "(these snapshots predate the search accounting)._"
        )
    rows = []
    for name in sources:
        counts = [
            n
            for item in items
            for n in item.frozen.result_counts.get(name, {}).values()
        ]
        if not counts:
            continue
        short = [n for n in counts if n < RESULTS_ASKED]
        rows.append(
            [
                name,
                str(len(counts)),
                _share(len(short), len(counts)),
                str(sum(1 for n in counts if n == 0)),
                fmt(statistics.median(counts), 1),
            ]
        )
    return (
        f"Queries answered with fewer than {RESULTS_ASKED} results, per source:\n\n"
        + table(
            ["source", "queries", f"under {RESULTS_ASKED}", "none", "median results"],
            rows,
        )
    )


def pool_comparison(
    this: Sequence[PoolItem],
    against: Sequence[PoolItem],
    *,
    this_label: str,
    against_label: str,
) -> str:
    """Which gold papers are in the pool of *this* and not of *against* (and the
    reverse), and through which source, for the items both hold."""
    base = {item.item_id: item for item in against}
    both = [item for item in this if item.item_id in base]
    if not both:
        return "_No item has a snapshot in both roots._"
    entered: list[tuple[GoldPoolEntry, GoldPoolEntry]] = []
    left: list[tuple[GoldPoolEntry, GoldPoolEntry]] = []
    n_gold = n_before = n_now = 0
    for item in both:
        now_entries = gold_entries(item)
        before_entries = gold_entries(base[item.item_id])
        for now, before in zip(now_entries, before_entries, strict=True):
            n_gold += 1
            n_before += before.in_pool
            n_now += now.in_pool
            if now.in_pool and not before.in_pool:
                entered.append((now, before))
            if before.in_pool and not now.in_pool:
                left.append((now, before))
    by_source: dict[str, int] = defaultdict(int)
    for now, _ in entered:
        for name in now.sources:
            by_source[name] += 1
    lines = [
        f"{len(both)} items with a snapshot in both roots, {n_gold} gold papers: "
        f"in the pool of {against_label} {n_before}, of {this_label} {n_now}; "
        f"{len(entered)} entered and {len(left)} left.",
        "",
    ]
    if entered:
        lines.append(
            f"Entered the pool, by the source that found them in {this_label}: "
            + ", ".join(f"{n} {s}" for s, n in sorted(by_source.items()))
            + "."
        )
        lines.append("")
        lines += [f"- {now.item_id}: {now.gold.title}" for now, _ in entered]
    if left:
        lines.append("")
        lines.append(f"Left the pool ({against_label} held them):")
        lines.append("")
        lines += [f"- {now.item_id}: {now.gold.title}" for now, _ in left]
    return "\n".join(lines)


# -- losses -------------------------------------------------------------------------


def classify_gold(record: RunRecord, gold: Work) -> str:
    """The loss category of one gold paper in one run (see the module docstring)."""
    output = record.output
    result = None if output is None else output.result
    if output is None or result is None or record.error:
        return FAILED
    rank = next(
        (
            position
            for position, work in enumerate(output.works, start=1)
            if work_matches(work, gold)
        ),
        None,
    )
    if rank is not None:
        return HIT if rank <= TOP else KEPT_LOW
    held = [e for e in result.screening if work_matches(e.identity(), gold)]
    if not held:
        stats = result.stats
        screened_something = (
            stats.decision_calls
            or stats.decision_cache_hits
            or stats.step_llm_calls.get("Screen", 0)
        )
        if not result.screening and screened_something:
            return UNRECORDED
        return NOT_IN_POOL
    if any(e.kept for e in held):
        return KEPT_EXCLUDED
    if any(e.p is not None for e in held):
        return SCREENED_OUT
    return NOT_SCREENED


def loss_counts(
    records: Sequence[RunRecord], gold_of: Mapping[str, Sequence[Work]]
) -> tuple[dict[str, int], dict[str, list[str]]]:
    """For the runs of one arm: how many gold papers fall in each category, and
    which (as ``<item>: <title>``)."""
    counts = dict.fromkeys(LOSS_CATEGORIES, 0)
    named: dict[str, list[str]] = {c: [] for c in LOSS_CATEGORIES}
    for record in records:
        for gold in gold_of.get(record.item_id, ()):
            category = classify_gold(record, gold)
            counts[category] += 1
            named[category].append(f"{record.item_id}: {gold.title}")
    return counts, named


def loss_section(
    arms: Sequence[str],
    records: Sequence[RunRecord],
    gold_of: Mapping[str, Sequence[Work]],
) -> str:
    """The loss table of the arms (counts), then the gold papers outside the hit
    category, arm by arm."""
    rows = []
    named_by_arm: dict[str, dict[str, list[str]]] = {}
    for arm in arms:
        mine = [r for r in records if r.arm == arm and r.set == "litsearch"]
        if not mine:
            continue
        counts, named = loss_counts(mine, gold_of)
        named_by_arm[arm] = named
        rows.append(
            [
                arm,
                str(len(mine)),
                *[str(counts[c]) for c in LOSS_CATEGORIES],
                str(sum(counts.values())),
            ]
        )
    if not rows:
        return "_No LitSearch run to read._"
    text = table(["arm", "runs", *LOSS_CATEGORIES, "gold papers"], rows)
    text += (
        "\n\n" + "; ".join(f"`{c}` {LOSS_MEANING[c]}" for c in LOSS_CATEGORIES) + "."
    )
    for arm, named in named_by_arm.items():
        text += f"\n\n**{arm}**, gold papers not kept in the top {TOP}:"
        any_listed = False
        for category in LOSS_CATEGORIES[1:]:
            if named[category]:
                any_listed = True
                text += f"\n\n`{category}` {LOSS_MEANING[category]}:\n\n"
                text += "\n".join(f"- {line}" for line in named[category])
        if not any_listed:
            text += " none."
    return text


__all__ = [
    "FAILED",
    "HIT",
    "KEPT_EXCLUDED",
    "KEPT_LOW",
    "LOSS_CATEGORIES",
    "NOT_IN_POOL",
    "NOT_SCREENED",
    "SCREENED_OUT",
    "UNRECORDED",
    "GoldPoolEntry",
    "PoolItem",
    "classify_gold",
    "gold_entries",
    "load_pool_items",
    "loss_counts",
    "loss_section",
    "pool_comparison",
    "pool_tables",
]
