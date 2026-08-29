"""MULTI-MODEL CONFIDENCE REVISIT (2026-07) — capability ladder + combination.

Design, hypotheses, staging, and decision criteria: see MULTIMODEL_PLAN.md in
this folder. In one sentence: re-test "a single verbalized histogram's entropy
is a sufficient wrong-answer detector" on the newly installed local models —
the gemma4 capability ladder (e2b/12b/26b/31b), gpt-oss:20b, qwen3.6:35b,
muse-glimmer:30b, ornith-1.5:35b —
and, for the first time, test whether *cross-model disagreement* (the
never-exercised ``between_model`` term of ``mixture.variance_decomposition``)
beats any single model's self-reported uncertainty.

Everything model-facing is cached; ``analyze``/``combos``/``status`` are pure
math and never call a model.

USAGE
=====
    # 0. one call per model: latency + schema sanity + projected gather time
    uv run python -m experiments.dirichlet_confidence.multimodel_experiment probe

    # 1. gather (resumable; stage with --models; failures cached as None,
    #    re-attempt them with --retry-missing)
    uv run python -m experiments.dirichlet_confidence.multimodel_experiment gather --n-claims 60

    # 2. cache fill per model (no model calls)
    uv run python -m experiments.dirichlet_confidence.multimodel_experiment status --n-claims 60

    # 3. per-model analysis: single-call entropy vs temperature band (no calls)
    uv run python -m experiments.dirichlet_confidence.multimodel_experiment analyze --n-claims 60

    # 4. combination analysis: between-model disagreement (no calls)
    uv run python -m experiments.dirichlet_confidence.multimodel_experiment combos --n-claims 60
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.dirichlet_confidence import metrics  # noqa: E402
from experiments.dirichlet_confidence.mixture import (  # noqa: E402
    MixtureOfDirichlets,
    repair_row,
)
from experiments.dirichlet_confidence.scifact import (  # noqa: E402
    Example,
    balanced_subset,
    label_counts,
    load_scifact,
)
from experiments.dirichlet_confidence.validate_local import (  # noqa: E402
    elicit_engineered,
)

console = Console()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default set: the strongest model from each family — chosen for max accuracy
# and to make cross-family consensus the headline. The wider set (gemma4
# e2b/12b/26b for the capability-ladder question) is still reachable via
# --models; nothing is hard-coded to a fixed panel size.
MODELS = [
    "ollama:gemma4:31b-nvfp4",      # strongest gemma
    "ollama:gpt-oss:20b",           # cross-family anchor — June's best (AUROC 0.876)
    "ollama:qwen3.6:35b-mlx",       # cross-family
    "ollama:qwen3.8:27b-mlx",       # same family as qwen3.6, newer + smaller
    "ollama:muse-glimmer:30b-mlx",  # cross-family, least one-hot of the set (45%)
    "ollama:ornith-1.5:35b",        # cross-family, added 2026-08-20; no prior probe
]

# Combo-key stem per model. Two qwen generations are in the panel, so the stem
# carries the version — a bare "qwen" would silently collide and one of the two
# would overwrite the other's row.
COMBO_STEM = {
    "ollama:gemma4:31b-nvfp4": "gemma31",
    "ollama:gpt-oss:20b": "gpt-oss",
    "ollama:qwen3.6:35b-mlx": "qwen36",
    "ollama:qwen3.8:27b-mlx": "qwen38",
    "ollama:muse-glimmer:30b-mlx": "muse",
    "ollama:ornith-1.5:35b": "ornith",
}

# The full panel, under a size-independent key so adding a model doesn't
# invalidate every reader. Readers should use FULL_PANEL, never a literal.
FULL_PANEL = "all-models"


def _build_combos(models: list[str]) -> dict[str, list[str]]:
    """Every pair, plus the full panel.

    Derived rather than hand-listed: the pair count is quadratic in the roster
    (5 models = 10 pairs), so a hand-maintained literal drifts out of sync with
    MODELS the moment someone adds a model and misses a row. Scoring is post-hoc
    from the SAME cache, so defining every pair costs zero extra inference.
    """
    missing = [m for m in models if m not in COMBO_STEM]
    if missing:
        raise KeyError(
            f"No COMBO_STEM entry for {missing}. Add one short, unique stem per "
            f"model in MODELS — it names that model's rows in the combos table."
        )
    combos = {
        f"{COMBO_STEM[a]}+{COMBO_STEM[b]}": [a, b]
        for a, b in itertools.combinations(models, 2)
    }
    combos[FULL_PANEL] = list(models)
    return combos


COMBOS: dict[str, list[str]] = _build_combos(MODELS)

# K = how many times each model re-answers at TEMP_DRAWS, to measure its own
# self-consistency. Default 3 (was 5): a std over 3 draws is noisier but the
# call budget matters more. K can be lowered further, even to 0 (panel-only —
# just the single temp-0 call per model, the cheapest viable run).
DEFAULT_K = 3
TEMP_SINGLE = 0.0        # the deterministic single-call variant (always gathered)
TEMP_DRAWS = 0.7         # matches the original band experiment's temperature arm
SCIFACT_PATH = "experiments/dirichlet_confidence/data/scifact"

CACHE = Path("experiments/dirichlet_confidence/cache/multimodel.jsonl")
RESULTS = Path("experiments/dirichlet_confidence/results/multimodel.json")
RESULTS_COMBOS = Path("experiments/dirichlet_confidence/results/multimodel_combos.json")

_SEP = "\x1f"


def _variants(k: int) -> list[str]:
    """The per-(model, claim) call list: one deterministic 'single' + k temp draws."""
    return ["single"] + [f"temp{j}" for j in range(k)]


def _short(model: str) -> str:
    return model.split(":", 1)[-1]


# ---------------------------------------------------------------------------
# Cache: keyed by (model, claim_id, variant). A cached None records a failed
# call so gather doesn't hammer a broken model; --retry-missing re-attempts.
# ---------------------------------------------------------------------------


class Cache:
    def __init__(self, path: Path = CACHE) -> None:
        self.path = Path(path)
        self._rows: dict[str, list[float] | None] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self._rows[self._key(r["model"], r["claim_id"], r["variant"])] = r["row"]

    @staticmethod
    def _key(model: str, claim_id: str, variant: str) -> str:
        return f"{model}{_SEP}{claim_id}{_SEP}{variant}"

    def has(self, model: str, claim_id: str, variant: str) -> bool:
        return self._key(model, claim_id, variant) in self._rows

    def get(self, model: str, claim_id: str, variant: str) -> list[float] | None:
        return self._rows.get(self._key(model, claim_id, variant))

    def temp_rows(self, model: str, claim_id: str) -> list[list[float] | None]:
        """All cached 'temp*' rows for (model, claim), in variant order — so
        analysis reads whatever K was actually gathered without needing to know
        it. Failed calls (None) are included; callers filter them."""
        prefix = f"{model}{_SEP}{claim_id}{_SEP}temp"
        keys = sorted(
            (k for k in self._rows if k.startswith(prefix)),
            key=lambda k: int(k.rsplit("temp", 1)[-1]),
        )
        return [self._rows[k] for k in keys]

    def n_temp(self, model: str, claim_id: str) -> int:
        prefix = f"{model}{_SEP}{claim_id}{_SEP}temp"
        return sum(1 for k in self._rows if k.startswith(prefix))

    def put(self, model: str, claim_id: str, variant: str, row: list[float] | None) -> None:
        # Append-only JSONL; on reload, the last write for a key wins.
        self._rows[self._key(model, claim_id, variant)] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(
                json.dumps(
                    {"model": model, "claim_id": claim_id, "variant": variant, "row": row}
                )
                + "\n"
            )


# ---------------------------------------------------------------------------
# Math helpers (same definitions as validate_local)
# ---------------------------------------------------------------------------


def _normed(row: list[float] | None) -> np.ndarray | None:
    """Normalise a raw 0-100 histogram to a probability vector (None if invalid)."""
    if row is None:
        return None
    arr = np.asarray(row, dtype=float)
    if not np.all(np.isfinite(arr)) or arr.min() < 0:
        return None
    total = arr.sum()
    if total <= 0:
        return None
    return arr / total


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy, normalised to [0, 1] by log(C). Higher = less sure."""
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def _boot_auroc(
    score: np.ndarray, correct: np.ndarray, seed: int = 42, n: int = 2000
) -> dict[str, float]:
    """Point + 95% bootstrap CI for error-detection AUROC."""
    rng = np.random.default_rng(seed)
    point = metrics.error_detection_auroc(score, correct)
    boots = []
    m = len(score)
    for _ in range(n):
        idx = rng.integers(0, m, m)
        val = metrics.error_detection_auroc(score[idx], correct[idx])
        if not np.isnan(val):
            boots.append(val)
    if boots:
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    else:
        lo, hi = float("nan"), float("nan")
    return {"point": float(point), "lo": lo, "hi": hi}


def _calibration(pools: np.ndarray, labels: np.ndarray) -> dict:
    """Calibration of a pooled/panel prediction — the block a training signal
    needs. ``conf`` is the predicted class's probability (the soft-label value);
    a well-calibrated judge has low ECE, and its reliability curve tracks the
    diagonal. Brier is the proper-scoring-rule summary.
    """
    conf = pools.max(axis=1)
    correct = pools.argmax(axis=1) == labels
    return {
        "ece": metrics.expected_calibration_error(conf, correct),
        "brier": metrics.brier_score(pools, labels),
        "mean_confidence": float(conf.mean()),
        "reliability_curve": metrics.reliability_curve(conf, correct),
    }


def _fmt_auroc(a: dict[str, float] | None) -> str:
    if a is None or np.isnan(a["point"]):
        return "—"
    mark = "[green]*[/green]" if a["lo"] > 0.5 else " "
    return f"{a['point']:.3f} [{a['lo']:.2f},{a['hi']:.2f}]{mark}"


# ---------------------------------------------------------------------------
# Gather
# ---------------------------------------------------------------------------


def _examples(n: int) -> list[Example]:
    exs = balanced_subset(load_scifact(SCIFACT_PATH), n)
    console.print(f"[dim]Loaded {len(exs)} claims  ·  labels {label_counts(exs)}[/dim]")
    return exs


def _temp_for(variant: str) -> float:
    return TEMP_SINGLE if variant == "single" else TEMP_DRAWS


async def _gather(
    examples: list[Example], cache: Cache, models: list[str], k: int, retry_missing: bool
) -> None:
    variants = _variants(k)
    total = len(examples) * len(models) * len(variants)
    done = 0
    for model in models:  # model-outer: each model loads into Ollama once
        t_model = time.perf_counter()
        n_calls = 0
        for ex in examples:
            for variant in variants:
                done += 1
                if cache.has(model, ex.claim_id, variant):
                    if not (retry_missing and cache.get(model, ex.claim_id, variant) is None):
                        continue
                row = await elicit_engineered(
                    ex.claim, ex.evidence, model, _temp_for(variant)
                )
                cache.put(model, ex.claim_id, variant, row)
                n_calls += 1
                rate = (time.perf_counter() - t_model) / n_calls
                console.print(
                    f"[dim]{done}/{total}[/dim] {_short(model)} {ex.claim_id} {variant} "
                    f"-> {row}  [dim]({rate:.1f}s/call avg)[/dim]"
                )
        if n_calls:
            console.print(
                f"[bold]{_short(model)}[/bold]: {n_calls} calls in "
                f"{time.perf_counter() - t_model:.0f}s"
            )


async def _probe(models: list[str], k: int) -> None:
    """One call per model: latency, schema sanity, projected gather time."""
    exs = balanced_subset(load_scifact(SCIFACT_PATH), 3)
    ex = exs[0]
    calls_each = len(_variants(k))
    console.print(
        Panel.fit(
            "Latency probe — 1 engineered-schema call per model\n"
            "(first call includes one-time model load; steady-state is faster)",
            style="bold",
        )
    )
    per_model: dict[str, float] = {}
    for model in models:
        t0 = time.perf_counter()
        row = await elicit_engineered(ex.claim, ex.evidence, model, TEMP_SINGLE)
        dt = time.perf_counter() - t0
        per_model[model] = dt
        ok = "[green]ok[/green]" if row is not None else "[red]FAILED[/red]"
        console.print(f"{_short(model):22s} {dt:6.1f}s  {ok}  -> {row}")
    for n in (60, 150):
        est = sum(per_model.values()) * n * calls_each
        console.print(
            f"[dim]projected gather at n={n} ({len(models)} models, "
            f"{calls_each} calls each (K={k}), cold-load rates — an overestimate: "
            f"~{est / 3600:.1f}h; steady-state is a fraction of this)[/dim]"
        )


def _status(examples: list[Example], cache: Cache, models: list[str], k: int) -> None:
    variants = _variants(k)
    t = Table(title=f"Cache fill — {len(examples)} claims × {len(variants)} variants (K={k})")
    t.add_column("model")
    t.add_column("cached", justify="right")
    t.add_column("failed (None)", justify="right")
    t.add_column("remaining", justify="right")
    for model in models:
        cached = failed = 0
        for ex in examples:
            for variant in variants:
                if cache.has(model, ex.claim_id, variant):
                    cached += 1
                    if cache.get(model, ex.claim_id, variant) is None:
                        failed += 1
        t.add_row(
            _short(model),
            str(cached),
            str(failed),
            str(len(examples) * len(variants) - cached),
        )
    console.print(t)


# ---------------------------------------------------------------------------
# Per-model analysis: S1 single-call entropy, S2 temperature band, S3 pooled
# entropy. Each score is graded against its own predictor's errors.
# ---------------------------------------------------------------------------


def _analyze_model(model: str, examples: list[Example], cache: Cache) -> dict | None:
    s_probs, s_labels = [], []          # single-call arm
    t_pools, t_bands, t_labels = [], [], []  # temperature arm
    n_missing_single = n_missing_temp = 0

    for ex in examples:
        p = _normed(cache.get(model, ex.claim_id, "single"))
        if p is None:
            n_missing_single += 1
        else:
            s_probs.append(p)
            s_labels.append(ex.label)

        rows = [_normed(r) for r in cache.temp_rows(model, ex.claim_id)]
        rows = [r for r in rows if r is not None]
        if len(rows) < 2:  # a band needs at least two draws
            n_missing_temp += 1
            continue
        arr = np.vstack(rows)
        pooled = arr.mean(axis=0)
        pred = int(pooled.argmax())
        t_pools.append(pooled)
        t_bands.append(float(arr[:, pred].std()))
        t_labels.append(ex.label)

    if not s_probs and not t_pools:
        return None

    out: dict = {"model": model, "n_missing_single": n_missing_single,
                 "n_missing_temp": n_missing_temp}

    if s_probs:
        probs = np.vstack(s_probs)
        labels = np.asarray(s_labels)
        conf = probs.max(axis=1)
        entropies = np.apply_along_axis(_entropy, 1, probs)
        correct = probs.argmax(axis=1) == labels
        out["single"] = {
            "n": int(len(labels)),
            "accuracy": metrics.accuracy(probs, labels),
            "n_wrong": int((~correct).sum()),
            "ece": metrics.expected_calibration_error(conf, correct),
            "brier": metrics.brier_score(probs, labels),
            "auroc_entropy": _boot_auroc(entropies, correct),
            "mean_entropy": float(entropies.mean()),
            "degeneracy_rate": float((conf >= 0.95).mean()),
            "risk_coverage": metrics.risk_coverage(entropies, correct),
            # Full calibration block so a single model can be plotted on the same
            # reliability / ECE figure as the panels (its own histogram is its
            # "final answer" — total doubt = solo doubt = its entropy).
            "calibration": _calibration(probs, labels),
        }

    if t_pools:
        pools = np.vstack(t_pools)
        labels = np.asarray(t_labels)
        bands = np.asarray(t_bands)
        pooled_entropies = np.apply_along_axis(_entropy, 1, pools)
        correct = pools.argmax(axis=1) == labels
        out["temperature"] = {
            "n": int(len(labels)),
            "accuracy": metrics.accuracy(pools, labels),
            "n_wrong": int((~correct).sum()),
            "auroc_band": _boot_auroc(bands, correct),
            "auroc_pooled_entropy": _boot_auroc(pooled_entropies, correct),
            "mean_band": float(bands.mean()),
        }

    return out


def _render_analyze(out: dict) -> None:
    t = Table(
        title=f"Per-model — n={out['config']['n_claims']} balanced SciFact claims. "
        "* = AUROC CI excludes 0.5 (usable signal)"
    )
    t.add_column("model")
    t.add_column("acc\n(1-call)", justify="right")
    t.add_column("degen\n(one-hot)", justify="right")
    t.add_column("S1 entropy AUROC\n(1 call, temp 0)", justify="right")
    t.add_column("acc\n(K-pool)", justify="right")
    t.add_column("S2 band AUROC\n(K=5, temp 0.7)", justify="right")
    t.add_column("S3 pooled-entropy\nAUROC", justify="right")
    for model, r in out["results"].items():
        if r is None:
            t.add_row(_short(model), *["[red]no rows[/red]"] * 6)
            continue
        s, tp = r.get("single"), r.get("temperature")
        t.add_row(
            _short(model),
            f"{s['accuracy']:.2f}" if s else "—",
            f"{s['degeneracy_rate']:.0%}" if s else "—",
            _fmt_auroc(s["auroc_entropy"]) if s else "—",
            f"{tp['accuracy']:.2f}" if tp else "—",
            _fmt_auroc(tp["auroc_band"]) if tp else "—",
            _fmt_auroc(tp["auroc_pooled_entropy"]) if tp else "—",
        )
    console.print(t)
    console.print(
        "[dim]H1: S1 AUROC should rise up the gemma ladder as degeneracy falls.  "
        "H2: S1 ≈ S2/S3 within CIs on usable models.[/dim]"
    )


# ---------------------------------------------------------------------------
# Combination analysis: the between-model signal.
# ---------------------------------------------------------------------------


def _analyze_combo(
    name: str, members: list[str], examples: list[Example], cache: Cache
) -> dict:
    # Singles arm: M models x 1 temp-0 call each.
    preds_ok, band_between, mean_ent, pooled_ent, labels = [], [], [], [], []
    votes_for_pred = []  # how many members' top pick matches the pooled label
    n_skipped_singles = 0
    # Mixture arm: M x K temp rows through MixtureOfDirichlets.
    mix_ok, mix_between, mix_within, mix_labels = [], [], [], []
    n_skipped_mix = 0

    for ex in examples:
        # --- singles arm ---
        rows = [_normed(cache.get(m, ex.claim_id, "single")) for m in members]
        valid = [r for r in rows if r is not None]
        if len(valid) < len(members):
            n_skipped_singles += 1
        else:
            arr = np.vstack(valid)  # (M, C)
            pooled = arr.mean(axis=0)
            pred = int(pooled.argmax())
            preds_ok.append(pooled)
            band_between.append(float(arr[:, pred].std()))
            mean_ent.append(float(np.mean([_entropy(r) for r in valid])))
            pooled_ent.append(_entropy(pooled))
            votes_for_pred.append(int((arr.argmax(axis=1) == pred).sum()))
            labels.append(ex.label)

        # --- mixture arm ---
        means, model_ids = [], []
        for m in members:
            for raw in cache.temp_rows(m, ex.claim_id):
                if raw is None:
                    continue
                repaired = repair_row(np.asarray(raw, dtype=float))
                if repaired is not None:
                    means.append(repaired)
                    model_ids.append(m)
        # Require every member to contribute at least 2 rows, else the
        # between/within split is degenerate for that claim.
        counts = {m: model_ids.count(m) for m in members}
        if any(counts[m] < 2 for m in members):
            n_skipped_mix += 1
            continue
        mix = MixtureOfDirichlets(means=np.vstack(means), model_ids=np.asarray(model_ids))
        decomp = mix.variance_decomposition()
        pred = int(decomp["pooled_mean"].argmax())
        mix_ok.append(decomp["pooled_mean"])
        mix_between.append(float(np.sqrt(decomp["between_model"][pred])))
        mix_within.append(float(np.sqrt(decomp["within_model"][pred])))
        mix_labels.append(ex.label)

    out: dict = {"name": name, "members": members}

    if preds_ok:
        pools = np.vstack(preds_ok)
        lab = np.asarray(labels)
        correct = pools.argmax(axis=1) == lab
        bands = np.asarray(band_between)
        # The information-theoretic split of the pooled histogram's total doubt:
        #   total (pooled entropy) = solo (mean per-model entropy) + disagreement.
        # "disagreement" is the mutual information / generalised Jensen-Shannon
        # divergence between the members — non-negative, and unlike the std-band
        # it uses the whole shape of every histogram, not just the winning class.
        total_h = np.asarray(pooled_ent)
        solo_h = np.asarray(mean_ent)
        disagreement = np.clip(total_h - solo_h, 0.0, None)
        # Vote gate: the discrete, no-threshold signal — did the members' top
        # picks agree? Reports accuracy on the unanimous vs split buckets and
        # what share of all errors the (small) split bucket contains.
        votes = np.asarray(votes_for_pred)
        unanimous = votes == len(members)
        n_wrong = int((~correct).sum())
        wrong_in_split = int((~correct & ~unanimous).sum())
        vote_gate = {
            "n_unanimous": int(unanimous.sum()),
            "frac_unanimous": float(unanimous.mean()),
            "acc_unanimous": float(correct[unanimous].mean()) if unanimous.any() else None,
            "n_split": int((~unanimous).sum()),
            "frac_split": float((~unanimous).mean()),
            "acc_split": float(correct[~unanimous].mean()) if (~unanimous).any() else None,
            "share_of_errors_in_split": (wrong_in_split / n_wrong) if n_wrong else None,
        }
        out["singles_arm"] = {
            "n": int(len(lab)),
            "n_skipped": n_skipped_singles,
            "accuracy": metrics.accuracy(pools, lab),
            "n_wrong": n_wrong,
            "auroc_total_entropy": _boot_auroc(total_h, correct),
            "auroc_solo_entropy": _boot_auroc(solo_h, correct),
            "auroc_disagreement_mi": _boot_auroc(disagreement, correct),
            "auroc_between_band": _boot_auroc(bands, correct),
            "vote_gate": vote_gate,
            "risk_coverage": metrics.risk_coverage(bands, correct),
            "calibration": _calibration(pools, lab),
        }

    if mix_ok:
        pools = np.vstack(mix_ok)
        lab = np.asarray(mix_labels)
        correct = pools.argmax(axis=1) == lab
        out["mixture_arm"] = {
            "n": int(len(lab)),
            "n_skipped": n_skipped_mix,
            "accuracy": metrics.accuracy(pools, lab),
            "n_wrong": int((~correct).sum()),
            "auroc_between_model": _boot_auroc(np.asarray(mix_between), correct),
            "auroc_within_model": _boot_auroc(np.asarray(mix_within), correct),
            "calibration": _calibration(pools, lab),
        }

    return out


def _render_combos(out: dict, single_results: dict | None) -> None:
    t = Table(
        title=f"Combinations — n={out['config']['n_claims']} claims. "
        "* = AUROC CI excludes 0.5"
    )
    t.add_column("combo")
    t.add_column("acc", justify="right")
    t.add_column("ECE\n(↓)", justify="right")
    t.add_column("total-doubt\nAUROC", justify="right")
    t.add_column("solo-doubt\nAUROC", justify="right")
    t.add_column("disagreement\nAUROC", justify="right")
    t.add_column("unanimous:\nacc / share", justify="right")
    for name, r in out["results"].items():
        s = r.get("singles_arm")
        if not s:
            t.add_row(name, *["—"] * 6)
            continue
        vg = s["vote_gate"]
        unan = (f"{vg['acc_unanimous']:.2f} / {vg['frac_unanimous']:.0%}"
                if vg["acc_unanimous"] is not None else "—")
        t.add_row(
            name,
            f"{s['accuracy']:.2f}",
            f"{s['calibration']['ece']:.3f}",
            _fmt_auroc(s["auroc_total_entropy"]),
            _fmt_auroc(s["auroc_solo_entropy"]),
            _fmt_auroc(s["auroc_disagreement_mi"]),
            unan,
        )
    console.print(t)
    if single_results:
        best = None
        for model, r in single_results.items():
            if r and r.get("single"):
                a = r["single"]["auroc_entropy"]["point"]
                if best is None or a > best[1]:
                    best = (model, a)
        if best:
            console.print(
                f"[dim]best single-model entropy for comparison: "
                f"{_short(best[0])} at {best[1]:.3f} (own errors, own claims)[/dim]"
            )
    console.print(
        "[dim]total doubt = solo doubt (avg per-model entropy) + disagreement (mutual info). "
        "unanimous = all members' top pick agree.[/dim]"
    )


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def _run_analyze(n: int, models: list[str]) -> dict:
    cache = Cache()
    exs = _examples(n)
    out = {
        "config": {
            "n_claims": n,
            "models": models,
            "temp_single": TEMP_SINGLE,
            "temp_draws": TEMP_DRAWS,
            "elicitation": "engineered (reasoning-first schema, validate_local)",
        },
        "results": {m: _analyze_model(m, exs, cache) for m in models},
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    _render_analyze(out)
    console.print(f"[dim]results -> {RESULTS}[/dim]")
    return out


def _run_combos(n: int, combos: dict[str, list[str]]) -> dict:
    cache = Cache()
    exs = _examples(n)
    out = {
        "config": {"n_claims": n, "combos": combos},
        "results": {
            name: _analyze_combo(name, members, exs, cache)
            for name, members in combos.items()
        },
    }
    RESULTS_COMBOS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_COMBOS.write_text(json.dumps(out, indent=2))
    single = None
    if RESULTS.exists():
        single = json.loads(RESULTS.read_text()).get("results")
    _render_combos(out, single)
    console.print(f"[dim]results -> {RESULTS_COMBOS}[/dim]")
    return out


def _parse_combos(specs: list[str] | None) -> dict[str, list[str]]:
    """--combo 'name=modelA,modelB,...' (repeatable); default: COMBOS."""
    if not specs:
        return COMBOS
    out: dict[str, list[str]] = {}
    for spec in specs:
        name, _, rest = spec.partition("=")
        members = [m.strip() for m in rest.split(",") if m.strip()]
        if not name or len(members) < 2:
            raise SystemExit(f"bad --combo spec (need name=m1,m2,...): {spec!r}")
        out[name] = members
    return out


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    pp = sub.add_parser("probe")
    pp.add_argument("--models", nargs="+", default=None)
    pp.add_argument("--k", type=int, default=DEFAULT_K,
                    help=f"temp draws per model for the projection (default {DEFAULT_K})")
    for name in ("gather", "status", "analyze"):
        sp = sub.add_parser(name)
        sp.add_argument("--n-claims", type=int, default=60)
        sp.add_argument("--models", nargs="+", default=None,
                        help="override model list (default: MODELS)")
        if name in ("gather", "status"):
            sp.add_argument("--k", type=int, default=DEFAULT_K,
                            help=f"temp draws per (model, claim); 0 = panel-only "
                                 f"(single call each). default {DEFAULT_K}")
        if name == "gather":
            sp.add_argument("--retry-missing", action="store_true",
                            help="re-attempt calls that previously failed (cached None)")
    cp = sub.add_parser("combos")
    cp.add_argument("--n-claims", type=int, default=60)
    cp.add_argument("--combo", action="append", default=None, metavar="NAME=M1,M2,...",
                    help="override the default combinations (repeatable)")
    args = p.parse_args(argv)

    models = getattr(args, "models", None) or MODELS

    if args.mode == "probe":
        asyncio.run(_probe(models, args.k))
    elif args.mode == "gather":
        asyncio.run(
            _gather(_examples(args.n_claims), Cache(), models, args.k, args.retry_missing)
        )
    elif args.mode == "status":
        _status(_examples(args.n_claims), Cache(), models, args.k)
    elif args.mode == "analyze":
        _run_analyze(args.n_claims, models)
    elif args.mode == "combos":
        _run_combos(args.n_claims, _parse_combos(args.combo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
