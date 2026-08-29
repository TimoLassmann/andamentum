"""ANALYZE — the free half of the springboard. NO API calls.

Reads the frozen elicitation cache and scores every arm. Because it never calls
a model, you can run it as often as you like while developing the method; the
numbers are fully reproducible from the cache.

THE EXPERIMENT
==============

One question: does aggregating a small model's confidence across SEVERAL
PHRASINGS of the same question give better-calibrated confidence than asking it
once? Model and evidence are held fixed; only the reasoning-over-elicitations
changes.

Arms (all judged on the same claims, the same way):

  Baselines (each model asked once, phrasing 0) — fixed reference points:
    • gpt-5.4        — frontier reference
    • gpt-5.4-mini   — mid reference
    • gpt-5.4-nano   — the small model, asked once (the bar to beat)

  Our method (gpt-5.4-nano, K phrasings, the same question worded K ways):
    • phrasing-mean  — average the K repaired histograms (point estimate)
    • Dirichlet      — mixture-of-Dirichlets with fitted s; adds the Monte-Carlo
                       uncertainty band on top of the same predictive mean

The method's point estimate and the Dirichlet share a predictive mean, so they
tie on Accuracy / Brier / ECE / AUC by construction; the Dirichlet's extra value
shows only in the sampled-NLL and the variance band.

METRICS (all computed on the predictive mean — independent of the fitted s, so
no train/test leakage for the headline numbers):

  • Accuracy  — how often the top class is correct.
  • Brier     — squared error of the whole probability vector (lower better).
  • ECE       — calibration error: does a stated 70% mean 70%? (lower better).
  • AUC       — on the SUPPORT-vs-CONTRADICT subset, how well the confidence
                ranks correct-vs-incorrect direction (0.5 = chance, 1.0 = perfect).

VARIANCE DECOMPOSITION (method only) — splits the method's uncertainty into:
  • within_model  — how much the K PHRASINGS disagreed (phrasing brittleness).
  • within_component — Monte-Carlo Dirichlet noise (set by s).
(With a single model there is no between-model term.)

RUN
===
    uv run python -m experiments.dirichlet_confidence.analyze
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.dirichlet_confidence import metrics  # noqa: E402
from experiments.dirichlet_confidence.cache_io import (  # noqa: E402
    CACHE_PATH,
    ElicitationCache,
)
from experiments.dirichlet_confidence.mixture import (  # noqa: E402
    build_mixture,
    fit_concentration_s,
    repair_row,
)
from experiments.dirichlet_confidence.scifact import (  # noqa: E402
    Example,
    balanced_subset,
    label_counts,
    load_scifact,
)

console = Console()

BASELINE_MODELS = ["openai:gpt-5.4", "openai:gpt-5.4-mini", "openai:gpt-5.4-nano"]
METHOD_MODEL = "openai:gpt-5.4-nano"
CLASSES = ["SUPPORT", "CONTRADICT", "NOINFO"]


def _short(model: str) -> str:
    return model.split(":", 1)[-1]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _score(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Accuracy / Brier / ECE / AUC for a stack of predictive distributions.

    AUC is computed on the SUPPORT-vs-CONTRADICT subset only (NEI excluded), to
    match the standard SciFact calibration framing: positive = SUPPORT, score =
    P(SUPPORT) − P(CONTRADICT).
    """
    preds = probs.argmax(axis=1)
    out = {
        "accuracy": metrics.accuracy(probs, labels),
        "brier": metrics.brier_score(probs, labels),
        "ece": metrics.expected_calibration_error(probs.max(axis=1), preds == labels),
        "n": int(len(labels)),
    }
    supcon = labels < 2  # 0=SUPPORT, 1=CONTRADICT
    if supcon.sum() > 0:
        scores = probs[supcon, 0] - probs[supcon, 1]  # P(SUP) - P(CON)
        positives = labels[supcon] == 0
        out["auc"] = metrics.roc_auc(scores, positives)
        out["n_auc"] = int(supcon.sum())
    else:
        out["auc"] = float("nan")
        out["n_auc"] = 0
    return out


def baseline_probs(model: str, examples: list[Example], cache: ElicitationCache, alpha0: float):
    """Predictive distributions for a baseline model (one ask, phrasing 0)."""
    probs, labels = [], []
    for ex in examples:
        row = cache.get("baseline", ex.claim_id, model, 0)
        if row is None:
            row = cache.get_legacy(ex.claim_id, model, "0")  # reuse old cache
        if row is None:
            continue
        p = repair_row(np.array(row), alpha0=alpha0)
        if p is None:
            continue
        probs.append(p)
        labels.append(ex.label)
    return np.array(probs), np.array(labels)


def method_mixtures(
    examples: list[Example], cache: ElicitationCache, model: str, n_phrasings: int, alpha0: float
) -> dict:
    """One mixture-of-Dirichlets per claim, built from the K phrasing rows."""
    out = {}
    for ex in examples:
        rows = [cache.get("method", ex.claim_id, model, k) for k in range(n_phrasings)]
        good = [r for r in rows if r is not None]
        out[ex.claim_id] = build_mixture(
            [{"model": model, "phrasing_id": i, "row": r} for i, r in enumerate(good)],
            alpha0=alpha0,
        )
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _cell(m: dict, key: str, fmt: str) -> str:
    if not m or key not in m:
        return "—"
    v = m[key]
    if isinstance(v, float) and np.isnan(v):
        return "n/a"
    return format(v, fmt)


def arms_table(arms: list[tuple[str, dict]]) -> Table:
    t = Table(title="All arms · scored on the same claims")
    t.add_column("Arm")
    t.add_column("Accuracy", justify="right")
    t.add_column("Brier ↓", justify="right")
    t.add_column("ECE ↓", justify="right")
    t.add_column("AUC ↑", justify="right")
    t.add_column("Claims", justify="right")
    for name, m in arms:
        t.add_row(
            name,
            _cell(m, "accuracy", ".0%"),
            _cell(m, "brier", ".3f"),
            _cell(m, "ece", ".3f"),
            _cell(m, "auc", ".3f"),
            str(m.get("n", 0)),
        )
    return t


def variance_table(vd: dict, s_fit: float) -> Table:
    t = Table(title="Where the method's uncertainty comes from (summed over classes)")
    t.add_column("Source")
    t.add_column("Value", justify="right")
    t.add_column("Meaning")
    t.add_row(
        "within_model",
        f"{vd['within_model']:.4f}",
        "how much the 5 phrasings disagreed (phrasing brittleness)",
    )
    t.add_row(
        "within_component",
        f"{vd['within_component']:.4f}",
        f"Monte-Carlo Dirichlet noise (set by fitted s={s_fit:.0f})",
    )
    t.add_row(
        "between_model",
        f"{vd['between_model']:.4f}",
        "0 — single model, no model-family disagreement here",
    )
    return t


def band_quality_table(bq: dict) -> Table:
    t = Table(title="Is the uncertainty band itself any good? (does it flag errors?)")
    t.add_column("Measure")
    t.add_column("Value", justify="right")
    t.add_column("Meaning")
    auroc = bq["error_detection_auroc"]
    auroc_str = "n/a" if np.isnan(auroc) else f"{auroc:.3f}"
    t.add_row(
        "error-detection AUROC",
        auroc_str,
        "how well a bigger band predicts a wrong answer (0.5=noise, 1.0=perfect)",
    )
    t.add_row(
        "mean band when RIGHT",
        f"{bq['mean_band_when_right']:.4f}",
        "should be small",
    )
    t.add_row(
        "mean band when WRONG",
        f"{bq['mean_band_when_wrong']:.4f}",
        "should be larger than 'when RIGHT' if the band is informative",
    )
    return t


def risk_coverage_table(rc: list[dict]) -> Table:
    t = Table(title="Risk-coverage: abstain on the highest-band claims first")
    t.add_column("Keep (coverage)", justify="right")
    t.add_column("Claims kept", justify="right")
    t.add_column("Accuracy on kept", justify="right")
    for r in rc:
        t.add_row(f"{r['coverage']:.0%}", str(r["n"]), f"{r['accuracy']:.0%}")
    return t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def analyze(args: argparse.Namespace) -> dict:
    examples = load_scifact(args.scifact_path)
    examples = balanced_subset(examples, args.n_claims, seed=args.seed)
    console.print(
        Panel(
            f"Claims: {len(examples)}  ({label_counts(examples)})\n"
            f"Baselines: {', '.join(_short(m) for m in args.baseline_models)} "
            f"(each asked once)\n"
            f"Method: {_short(args.method_model)} across {args.n_phrasings} phrasings "
            f"(same question, reworded)\n"
            f"Cache: {args.cache}  ·  this analysis makes NO API calls",
            title="What is being analysed",
        )
    )

    cache = ElicitationCache(Path(args.cache))
    # Make the legacy baseline rows visible under the new schema if present.
    migrated = cache.migrate_legacy_baselines(args.baseline_models + [args.method_model])
    if migrated:
        console.print(f"[dim]Reused {migrated} baseline row(s) from the previous cache.[/dim]")

    # --- baselines ---
    arms: list[tuple[str, dict]] = []
    coverage_warnings: list[str] = []
    for m in args.baseline_models:
        probs, labels = baseline_probs(m, examples, cache, args.alpha0)
        score = _score(probs, labels) if len(probs) else {"n": 0}
        label = f"{_short(m)} (asked once)"
        if m == args.method_model:
            label = f"{_short(m)} (asked once) ← the bar to beat"
        arms.append((label, score))
        if score.get("n", 0) < len(examples):
            coverage_warnings.append(
                f"{_short(m)}: only {score.get('n', 0)}/{len(examples)} claims cached"
            )

    # --- method: phrasing-mean and Dirichlet ---
    mixes = method_mixtures(
        examples, cache, args.method_model, args.n_phrasings, args.alpha0
    )
    built = [(ex, mixes[ex.claim_id]) for ex in examples if mixes[ex.claim_id] is not None]
    if len(built) < len(examples):
        coverage_warnings.append(
            f"method: only {len(built)}/{len(examples)} claims had phrasing rows"
        )

    point = dirichlet = {"n": 0}
    vd = {"within_component": 0.0, "within_model": 0.0, "between_model": 0.0}
    s_fit = float("nan")
    if built:
        probs = np.array([mix.pooled_mean() for _, mix in built])
        labels = np.array([ex.label for ex, _ in built])
        point = _score(probs, labels)
        dirichlet = dict(point)  # same predictive mean

        # Fit s on a dev split purely for the Dirichlet's band / sampled-NLL.
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(built))
        n_dev = max(1, int(len(built) * args.dev_frac))
        dev_pairs = [(built[i][1], built[i][0].label) for i in idx[:n_dev]]
        s_fit = fit_concentration_s(dev_pairs) if dev_pairs else 50.0
        all_pairs = [(mix, ex.label) for ex, mix in built]
        point["sampled_nll"] = metrics.mean_sampled_nll(all_pairs, 1e6, args.mc_samples, rng)
        dirichlet["sampled_nll"] = metrics.mean_sampled_nll(all_pairs, s_fit, args.mc_samples, rng)

        acc = {k: [] for k in vd}
        # Per-claim band (predicted-class std under the mixture) + correctness,
        # for grading the UNCERTAINTY ESTIMATE itself (does the band flag errors?).
        bands: list[float] = []
        correct: list[bool] = []
        for (ex, mix) in built:
            mix_s = mix.with_s(s_fit)
            d = mix_s.variance_decomposition()
            for k in vd:
                acc[k].append(float(d[k].sum()))
            pooled = mix_s.pooled_mean()
            ci = int(np.argmax(pooled))
            bands.append(float(np.sqrt(d["total"][ci])))
            correct.append(ci == ex.label)
        vd = {k: float(np.mean(v)) for k, v in acc.items()}

        band_arr = np.array(bands)
        correct_arr = np.array(correct)
        band_quality = {
            "error_detection_auroc": metrics.error_detection_auroc(band_arr, correct_arr),
            "risk_coverage": metrics.risk_coverage(band_arr, correct_arr),
            "mean_band_when_right": float(band_arr[correct_arr].mean())
            if correct_arr.any()
            else float("nan"),
            "mean_band_when_wrong": float(band_arr[~correct_arr].mean())
            if (~correct_arr).any()
            else float("nan"),
            "n": int(len(band_arr)),
        }
        dirichlet["band_quality"] = band_quality

    arms.append((f"{_short(args.method_model)} × {args.n_phrasings} phrasings (mean)", point))
    arms.append(
        (f"{_short(args.method_model)} × {args.n_phrasings} phrasings (Dirichlet, s={s_fit:.0f})", dirichlet)
    )

    # --- render ---
    console.print()
    console.print(arms_table(arms))
    console.print()
    if built:
        console.print(variance_table(vd, s_fit))
        console.print()
        bq = dirichlet.get("band_quality")
        if bq and bq["n"] > 0:
            console.print(band_quality_table(bq))
            console.print()
            console.print(risk_coverage_table(bq["risk_coverage"]))
            console.print()
    if coverage_warnings:
        console.print(
            Panel(
                "\n".join(coverage_warnings)
                + "\n\nRun `gather` to fill the gaps (it only fetches what's missing).",
                title="[yellow]Coverage gaps[/yellow]",
            )
        )
    nano = _short(args.method_model)
    console.print(
        Panel(
            "The real goal here is not a better point answer — it is a GOOD\n"
            "UNCERTAINTY ESTIMATE: a band that is small when the model is right\n"
            "and large when it is wrong.\n\n"
            "• Arms table (Accuracy/Brier/ECE/AUC) grades the point prediction.\n"
            "  The phrasing arms can look slightly worse here — that's fine; it's\n"
            "  not what we're optimising.\n"
            "• 'Is the uncertainty band any good?' is the headline: error-detection\n"
            "  AUROC near 0.5 means the band is noise; toward 1.0 means it reliably\n"
            "  flags the wrong answers. 'band when WRONG' should exceed 'when RIGHT'.\n"
            "• Risk-coverage: if accuracy on the kept claims rises as we drop the\n"
            "  highest-band ones, the band is telling us which answers to trust.\n\n"
            f"At {len(examples)} claims these are indicative, not conclusive.",
            title="How to read this",
        )
    )

    return {
        "arms": {name: m for name, m in arms},
        "variance": vd,
        "s_fit": s_fit,
        "config": {
            "n_claims": len(examples),
            "baseline_models": args.baseline_models,
            "method_model": args.method_model,
            "n_phrasings": args.n_phrasings,
            "alpha0": args.alpha0,
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scifact-path", default="experiments/dirichlet_confidence/data/scifact")
    p.add_argument("--n-claims", type=int, default=30)
    p.add_argument("--baseline-models", nargs="+", default=BASELINE_MODELS)
    p.add_argument("--method-model", default=METHOD_MODEL)
    p.add_argument("--n-phrasings", type=int, default=5)
    p.add_argument("--alpha0", type=float, default=0.5)
    p.add_argument("--dev-frac", type=float, default=0.3)
    p.add_argument("--mc-samples", type=int, default=2000)
    p.add_argument("--cache", default=str(CACHE_PATH))
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    results = analyze(args)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        console.print(f"\nResults written to [bold]{out}[/bold]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
