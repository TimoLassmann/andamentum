"""Compare the disagreement BAND against standard abstention scores, with CIs.

WHAT THIS ANSWERS
=================
"Is our band a better error-detector than the established sampling-based
uncertainty scores — self-consistency and semantic entropy?" This is the
make-or-break baseline experiment: if the band does not beat (or at least match,
more cheaply) these, the contribution is the framing/decomposition, not the
score itself.

THE SCORES (all are "alarm scores": higher = more likely WRONG)
===============================================================
For each claim we have K elicited histograms (the cached phrasing/temperature
arms). The point prediction is held FIXED (pooled-mean argmax — the same one the
band uses) so we only ever compare the *uncertainty score*, never the
aggregation.

  • band               — std across the K histograms of the winning class's
                         probability (between-component spread). Our method.
  • self_consistency   — 1 - (max vote fraction over the K argmax labels).
                         Wang et al.; abstain when the K votes split.
  • semantic_entropy   — entropy of the K-vote label distribution. On a CLOSED
                         3-class task semantic entropy (Kuhn/Gal/Farquhar)
                         degenerates to this, because each class is its own
                         semantic cluster — there is no free-text to cluster.
                         SciFact is therefore a LOWER BOUND venue for semantic
                         entropy; its real strength needs a generation task.
  • predictive_entropy — entropy of the MEAN of the K histograms (soft version
                         that uses the elicited probabilities, not just argmax).

THE STATISTICS
==============
  • Per-score error-detection AUROC with a percentile bootstrap CI over claims.
  • PAIRED bootstrap of ΔAUROC (band - baseline) on identical resampled claims —
    the correct test for "is the band better?". If the Δ CI excludes 0 it is.

NO API CALLS — pure re-analysis of cache/band_experiment*.jsonl.

    uv run python -m experiments.dirichlet_confidence.compare_baselines
    uv run python -m experiments.dirichlet_confidence.compare_baselines --out results/baselines.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.dirichlet_confidence import metrics  # noqa: E402
from experiments.dirichlet_confidence.band_experiment import (  # noqa: E402
    BandCache,
    _claim_band,
)
from experiments.dirichlet_confidence.scifact import (  # noqa: E402
    balanced_subset,
    load_scifact,
)

console = Console()

SCIFACT = "experiments/dirichlet_confidence/data/scifact"
N_CLAIMS = 150
K = 5
ALPHA0 = 0.5
SEED = 42
N_BOOT = 2000
EPS = 1e-12

MODELS = [
    ("openai:gpt-5.4-nano", "experiments/dirichlet_confidence/cache/band_experiment.jsonl"),
    ("openai:gpt-5.4-mini", "experiments/dirichlet_confidence/cache/band_experiment_mini.jsonl"),
    ("openai:gpt-4o-mini", "experiments/dirichlet_confidence/cache/band_experiment_4omini.jsonl"),
]
ARMS = ["phrasing", "temperature"]
SCORES = ["band", "self_consistency", "semantic_entropy", "predictive_entropy"]


def _normed(row: list[float] | None) -> np.ndarray | None:
    if row is None:
        return None
    v = np.clip(np.asarray(row, dtype=float), 0.0, None)
    total = v.sum()
    return v / total if total > 0 else None


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0)
    return float(-(p * np.log(p)).sum())


def _per_claim_scores(arm: str, examples, cache: BandCache):
    """Return {score_name: np.ndarray}, plus the shared correctness vector."""
    rows_by_score: dict[str, list[float]] = {s: [] for s in SCORES}
    correct: list[bool] = []
    for ex in examples:
        raw = [cache.get(arm, ex.claim_id, v) for v in range(K)]
        good = [r for r in raw if r is not None]
        if len(good) < 2:
            continue
        res = _claim_band(good, ALPHA0)  # shared prediction (pooled argmax) + band
        if res is None:
            continue
        pred, band = res

        normed = [_normed(r) for r in good]
        normed = [n for n in normed if n is not None]
        if len(normed) < 2:
            continue
        n_classes = len(normed[0])

        votes = np.array([int(np.argmax(n)) for n in normed])
        counts = np.bincount(votes, minlength=n_classes).astype(float)
        vote_frac = counts / counts.sum()
        mean_dist = np.mean(np.stack(normed), axis=0)

        rows_by_score["band"].append(band)
        rows_by_score["self_consistency"].append(1.0 - float(vote_frac.max()))
        rows_by_score["semantic_entropy"].append(_entropy(vote_frac))
        rows_by_score["predictive_entropy"].append(_entropy(mean_dist))
        correct.append(pred == ex.label)

    return ({s: np.array(v) for s, v in rows_by_score.items()}, np.array(correct, dtype=bool))


def _auroc(score: np.ndarray, correct: np.ndarray) -> float:
    return metrics.error_detection_auroc(score, correct)


def _boot_ci(score: np.ndarray, correct: np.ndarray, rng) -> tuple[float, float, float]:
    """Point AUROC + percentile bootstrap 95% CI over claims."""
    point = _auroc(score, correct)
    n = len(correct)
    vals = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        a = _auroc(score[idx], correct[idx])
        if not np.isnan(a):
            vals.append(a)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return point, float(lo), float(hi)


def _paired_delta(band: np.ndarray, other: np.ndarray, correct: np.ndarray, rng):
    """Paired bootstrap of ΔAUROC = band - other on identical resampled claims."""
    point = _auroc(band, correct) - _auroc(other, correct)
    n = len(correct)
    deltas = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        ab, ao = _auroc(band[idx], correct[idx]), _auroc(other[idx], correct[idx])
        if not (np.isnan(ab) or np.isnan(ao)):
            deltas.append(ab - ao)
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5]) if len(deltas) else (np.nan, np.nan)
    frac_pos = float((deltas > 0).mean()) if len(deltas) else float("nan")
    return point, float(lo), float(hi), frac_pos


def run(args: argparse.Namespace) -> dict:
    examples = balanced_subset(load_scifact(SCIFACT), N_CLAIMS, seed=SEED)
    out: dict = {"config": {"n_claims": N_CLAIMS, "k": K, "alpha0": ALPHA0,
                            "n_boot": N_BOOT, "seed": SEED}, "results": {}}

    for model_id, cache_path in MODELS:
        cache = BandCache(_REPO_ROOT / cache_path)
        out["results"][model_id] = {}
        for arm in ARMS:
            scores, correct = _per_claim_scores(arm, examples, cache)
            if len(correct) == 0 or correct.all() or (~correct).all():
                continue
            entry = {"n": int(len(correct)), "accuracy": float(correct.mean()),
                     "n_wrong": int((~correct).sum()), "auroc": {}, "vs_band": {}}
            for s in SCORES:
                rng_s = np.random.default_rng(SEED)
                pt, lo, hi = _boot_ci(scores[s], correct, rng_s)
                entry["auroc"][s] = {"point": pt, "lo": lo, "hi": hi}
            for s in SCORES:
                if s == "band":
                    continue
                rng_d = np.random.default_rng(SEED + 1)
                d, dlo, dhi, fp = _paired_delta(scores["band"], scores[s], correct, rng_d)
                entry["vs_band"][s] = {"delta": d, "lo": dlo, "hi": dhi, "frac_band_wins": fp}
            out["results"][model_id][arm] = entry

    _render(out)
    if args.out:
        p = _REPO_ROOT / "experiments" / "dirichlet_confidence" / args.out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2, default=float))
        console.print(f"\nWritten to [bold]{p}[/bold]")
    return out


def _render(out: dict) -> None:
    for model_id, arms in out["results"].items():
        for arm, e in arms.items():
            t = Table(title=f"{model_id.split(':')[-1]}  ·  {arm} arm   "
                            f"(n={e['n']}, accuracy={e['accuracy']:.0%}, "
                            f"wrong={e['n_wrong']})")
            t.add_column("Abstention score")
            t.add_column("Error-detect AUROC  [95% CI]", justify="right")
            for s in SCORES:
                a = e["auroc"][s]
                mark = "  ← ours" if s == "band" else ""
                t.add_row(s, f"{a['point']:.3f}  [{a['lo']:.3f}, {a['hi']:.3f}]{mark}")
            console.print(); console.print(t)

            d = Table(title="Is the band better?  ΔAUROC = band − baseline (paired bootstrap)")
            d.add_column("Baseline")
            d.add_column("ΔAUROC  [95% CI]", justify="right")
            d.add_column("P(band wins)", justify="right")
            d.add_column("Verdict", justify="left")
            for s, v in e["vs_band"].items():
                excludes_zero = (v["lo"] > 0) or (v["hi"] < 0)
                verdict = ("band better" if v["lo"] > 0 else
                           "baseline better" if v["hi"] < 0 else "no difference")
                d.add_row(s, f"{v['delta']:+.3f}  [{v['lo']:+.3f}, {v['hi']:+.3f}]",
                          f"{v['frac_band_wins']:.0%}",
                          f"[bold]{verdict}[/bold]" if excludes_zero else verdict)
            console.print(d)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=None, help="write JSON results to this path (relative to module dir)")
    run(p.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
