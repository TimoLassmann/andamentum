"""Render the band-validation results as three plain-language figures.

WHAT THIS PLOTS
===============
The experiment asked a language model the SAME yes/no-ish question (does this
abstract SUPPORT / CONTRADICT / give NO INFO about this claim?) five times,
varying only the wording ("phrasing" arm) or the sampling temperature
("temperature" arm). Each of the five answers is a full probability split over
the three options. The *band* is how much those five answers disagreed about
the option the model finally picked — a small band means "the model said the
same thing however I asked it", a big band means "the model was easy to push
around". The question is whether a big band reliably flags the cases where the
model's final answer was actually WRONG (gold labels come from SciFact).

This script makes NO API calls. It reuses the experiment's own cached model
answers (cache/band_experiment*.jsonl) and its own band maths
(band_experiment._claim_band), so every number here matches the published
results JSON exactly. Figures land in experiments/dirichlet_confidence/figures/.

    uv run python -m experiments.dirichlet_confidence.make_figures
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from andamentum.figures import despine, get_palette, savefig, setup_style
from andamentum.figures.plots import grouped_boxplot
from experiments.dirichlet_confidence.band_experiment import BandCache, _claim_band
from experiments.dirichlet_confidence.compare_baselines import _entropy, _normed
from experiments.dirichlet_confidence.scifact import balanced_subset, load_scifact

# ---------------------------------------------------------------------------
# What to plot: the three models that were run, in capability order.
# ---------------------------------------------------------------------------
HERE = Path("experiments/dirichlet_confidence")
SCIFACT = HERE / "data" / "scifact"
OUTDIR = HERE / "figures"
N_CLAIMS = 150
K = 5
ALPHA0 = 0.5
SEED = 42

MODELS = [
    {"label": "gpt-5.4-nano\n(smallest)", "cache": "cache/band_experiment.jsonl",
     "summary": "results/band150.json"},
    {"label": "gpt-5.4-mini\n(mid-size)", "cache": "cache/band_experiment_mini.jsonl",
     "summary": "results/band150_mini.json"},
    {"label": "gpt-4o-mini\n(mid-size)", "cache": "cache/band_experiment_4omini.jsonl",
     "summary": "results/band150_4omini.json"},
]
ARMS = ["phrasing", "temperature"]
ARM_LABEL = {
    "phrasing": "Re-word the question\n(5 wordings)",
    "temperature": "Re-roll the answer\n(5 samples)",
}


def _load_summary(rel: str) -> dict:
    return json.loads((_REPO_ROOT / "experiments" / "dirichlet_confidence" / rel).read_text())


def _per_claim_bands(cache_path: str) -> dict[str, dict[str, list[float]]]:
    """Recompute the per-claim band for every claim, split by right/wrong.

    Returns {arm: {"right": [...bands...], "wrong": [...bands...]}}.
    Uses the experiment's own _claim_band so the means match the results JSON.
    """
    examples = balanced_subset(load_scifact(str(SCIFACT)), N_CLAIMS, seed=SEED)
    cache = BandCache(_REPO_ROOT / "experiments" / "dirichlet_confidence" / cache_path)
    out: dict[str, dict[str, list[float]]] = {a: {"right": [], "wrong": []} for a in ARMS}
    for arm in ARMS:
        for ex in examples:
            rows = [cache.get(arm, ex.claim_id, v) for v in range(K)]
            if all(r is None for r in rows):
                continue
            res = _claim_band(rows, ALPHA0)
            if res is None:
                continue
            predicted_class, band = res
            bucket = "right" if predicted_class == ex.label else "wrong"
            out[arm][bucket].append(band)
    return out


# ---------------------------------------------------------------------------
# Figure 1 — the scorecard: can the band detect wrong answers at all?
# ---------------------------------------------------------------------------
def figure_scorecard(summaries: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = get_palette("npg", 2)
    n_models = len(summaries)
    width = 0.36
    x = np.arange(n_models)

    for j, arm in enumerate(ARMS):
        vals = [s["arms"][arm]["error_detection_auroc"] for s in summaries]
        pos = x + (j - 0.5) * width
        bars = ax.bar(pos, vals, width=width * 0.92, color=colors[j],
                      label=ARM_LABEL[arm].replace("\n", " "), zorder=3)
        for p, v in zip(bars, vals):
            ax.text(p.get_x() + p.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)

    # Reference lines: chance and a rule-of-thumb "usable" floor.
    ax.axhspan(0.6, 0.85, color="#2E7D32", alpha=0.06, zorder=0)
    ax.axhline(0.5, ls="--", lw=1.1, color="#555555", zorder=2)
    ax.axhline(0.6, ls=":", lw=1.0, color="#2E7D32", zorder=2)
    ax.text(n_models - 0.5, 0.505, "0.50  =  coin-flip — the band tells you nothing",
            ha="right", va="bottom", fontsize=7.5, color="#555555")
    ax.text(n_models - 0.5, 0.608, "0.60  =  rule-of-thumb floor for a usable warning light",
            ha="right", va="bottom", fontsize=7.5, color="#2E7D32")

    ax.set_xticks(x)
    ax.set_xticklabels([s["config"]["model"].split(":")[-1] for s in summaries])
    ax.set_ylim(0.45, 0.85)
    ax.set_ylabel("Error-detection score (AUROC)\nhigher = bigger band more reliably means wrong answer")
    ax.set_xlabel("Model")
    ax.legend(title="How we created the disagreement", loc="upper left", frameon=False, fontsize=8)
    despine(ax)

    fig.suptitle("Does a model's self-disagreement flag its wrong answers?",
                 fontsize=13, fontweight="bold", y=0.99)
    ax.set_title("Ask the same question 5 ways; measure how much the answers disagree (the “band”).\n"
                 "Then test: does a bigger band rank the wrong answers above the right ones?",
                 fontsize=9, color="#444444", pad=10)
    fig.text(0.5, -0.06,
             "How to read this: each bar is one model + one way of generating disagreement, scored on 150 SciFact "
             "claims.\nThe score (AUROC) is the chance that a randomly chosen WRONG answer had a bigger band than a "
             "randomly chosen RIGHT one.\nTakeaway: the band is a useful warning light for the two mid-size models "
             "(bars clear the green floor), but not\nfor the smallest model (near the coin-flip line) — small "
             "models can be confidently wrong, so they do not disagree with themselves.",
             ha="center", va="top", fontsize=7.6, color="#333333", wrap=True)

    fig.subplots_adjust(bottom=0.26, top=0.84)
    out = savefig(fig, OUTDIR / "fig1_band_error_detection_scorecard.png", dpi=200)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2 — the mechanism: wrong answers carry a bigger band.
# ---------------------------------------------------------------------------
def figure_mechanism(bands_by_model: list[tuple[str, dict]]) -> None:
    n = len(bands_by_model)
    fig, axes = plt.subplots(1, n, figsize=(11.5, 4.4), sharey=True)
    colors = get_palette("npg", 4)
    wrong_c, right_c = colors[0], colors[3]  # warm/red for WRONG (intuitive)

    for ax, (model_id, per) in zip(axes, bands_by_model):
        data = {
            "Model was RIGHT": {ARM_LABEL[a]: per[a]["right"] for a in ARMS},
            "Model was WRONG": {ARM_LABEL[a]: per[a]["wrong"] for a in ARMS},
        }
        cats = [ARM_LABEL[a] for a in ARMS]
        grouped_boxplot(ax, data, cats, ["Model was RIGHT", "Model was WRONG"],
                        colors=[right_c, wrong_c], alpha=0.75)
        # Jittered points so the reader sees the actual claims, not just a box.
        groups = ["Model was RIGHT", "Model was WRONG"]
        width = 0.8 / len(groups)
        for i, a in enumerate(ARMS):
            for jg, g in enumerate(groups):
                pts = per[a]["right" if g.endswith("RIGHT") else "wrong"]
                if not pts:
                    continue
                pos = i + (jg - len(groups) / 2 + 0.5) * width
                jit = (np.arange(len(pts)) % 7 - 3) / 3.0 * width * 0.22
                ax.scatter(np.full(len(pts), pos) + jit, pts, s=6,
                           color=(right_c if jg == 0 else wrong_c),
                           edgecolor="white", linewidth=0.2, alpha=0.55, zorder=4)
        ax.set_title(model_id.split(":")[-1], fontsize=10, fontweight="bold")
        despine(ax)
        ax.set_xlabel("")

    axes[0].set_ylabel("Disagreement band\n(spread across the 5 re-asks; 0 = identical answers)")
    # One shared legend.
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=right_c, alpha=0.75, label="Model's answer was RIGHT"),
               Patch(facecolor=wrong_c, alpha=0.75, label="Model's answer was WRONG")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.945))

    fig.suptitle("Why it works: wrong answers come with a bigger disagreement band",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.text(0.5, -0.04,
             "How to read this: every dot is one of 150 claims; its height is how much the model's 5 re-asks "
             "disagreed (the band).\nClaims are split by whether the model's final answer matched the gold label. "
             "Within each panel the left pair re-words the question,\nthe right pair re-rolls the same prompt at "
             "higher temperature. Takeaway: the red (WRONG) boxes sit higher than the blue\n(RIGHT) boxes — "
             "the model wobbles more when it is about to be wrong. The gap is clearest for the two mid-size models.",
             ha="center", va="top", fontsize=7.6, color="#333333", wrap=True)

    fig.subplots_adjust(bottom=0.24, top=0.82, wspace=0.08)
    out = savefig(fig, OUTDIR / "fig2_band_separation_right_vs_wrong.png", dpi=200)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure 3 — the payoff: discard the most-uncertain answers, keep the rest.
# ---------------------------------------------------------------------------
def _fine_risk_coverage(per_arm: dict[str, list[float]], n_points: int = 20):
    """Recompute the risk-coverage curve at fine granularity.

    Rebuilds the (band, correct) pairs from the right/wrong band lists, sorts by
    band ascending (most self-consistent first), and reports accuracy on the
    retained fraction at ``n_points`` coverage levels from 100% down to 15%.
    Same logic as metrics.risk_coverage, just more points.
    """
    bands = np.array(per_arm["right"] + per_arm["wrong"], dtype=float)
    correct = np.array([True] * len(per_arm["right"]) + [False] * len(per_arm["wrong"]))
    order = bands.argsort()  # smallest band (most confident) first
    sorted_correct = correct[order]
    n = len(bands)
    cov, acc = [], []
    for frac in np.linspace(1.0, 0.15, n_points):
        k = max(1, int(round(n * frac)))
        cov.append(k / n * 100)
        acc.append(sorted_correct[:k].mean() * 100)
    return np.array(cov), np.array(acc)


def figure_risk_coverage(bands_by_model: list[tuple[str, dict]]) -> None:
    n = len(bands_by_model)
    fig, axes = plt.subplots(1, n, figsize=(11.5, 4.2), sharey=True)
    colors = get_palette("npg", 2)

    for ax, (model_id, per) in zip(axes, bands_by_model):
        base = None
        for j, arm in enumerate(ARMS):
            cov, acc = _fine_risk_coverage(per[arm])
            ax.plot(cov, acc, marker="o", ms=3.5, lw=1.3, color=colors[j],
                    label=ARM_LABEL[arm].replace("\n", " "), zorder=3)
            if arm == "phrasing":
                base = acc[0]
        ax.axhline(base, ls="--", lw=1.0, color="#888888", zorder=1)
        ax.text(99, base + 0.6, "accuracy if you keep everything", ha="right",
                va="bottom", fontsize=7, color="#777777")
        ax.set_title(model_id.split(":")[-1], fontsize=10, fontweight="bold")
        ax.set_xlabel("% of claims kept\n(highest-band claims discarded first)")
        ax.set_xlim(105, 15)  # decreasing: move right = keep only the most self-consistent
        despine(ax)

    axes[0].set_ylabel("Accuracy on the claims you keep (%)")
    axes[-1].legend(title="How disagreement was created", loc="lower left",
                    frameon=False, fontsize=7.5)

    fig.suptitle("The payoff: set aside the answers the model disagreed with itself on",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.text(0.5, -0.05,
             "How to read this: start at the right-hand end (keep 100% of answers) and walk left, throwing away the "
             "claims with the\nbiggest disagreement band first. The line climbs when discarding high-band claims "
             "removes mostly wrong answers — i.e.\nthe band correctly pointed at the errors. Takeaway: for the "
             "mid-size models, keeping only the 40% the model was most\nconsistent about lifts accuracy from "
             "~68–70% to ~83–93%. For the smallest model the curve is flat — filtering buys little.",
             ha="center", va="top", fontsize=7.6, color="#333333", wrap=True)

    fig.subplots_adjust(bottom=0.30, top=0.86, wspace=0.08)
    out = savefig(fig, OUTDIR / "fig3_risk_coverage.png", dpi=200)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure 4 — payoff curve, one line per abstention SCORE (not per arm).
# ---------------------------------------------------------------------------
# Which uncertainty signal best concentrates the errors? Same fixed prediction
# (pooled-mean argmax) for every score, so all lines share the 100%-coverage
# start and diverge only by how well the score ranks the wrong answers. Shown on
# the phrasing arm (its variant-0 is the clean deterministic single ask).
# Confidence-based scores are drawn solid; disagreement-based scores dashed, to
# make the headline visible: confidence beats disagreement.
SCORE_LINES = [
    ("self_consistency", "self-consistency / semantic entropy (votes, 5 calls)", "-"),
    ("entropy_1call", "verbalized entropy (1 call)", "-"),
    ("predictive_entropy", "predictive entropy (5 calls)", "-"),
    ("band", "disagreement band (5 calls) — ours", "--"),
    ("mutual_information", "mutual information / BALD (5 calls)", "--"),
]


def _score_table(arm: str, examples, cache: BandCache):
    """Per-claim {score: array} + shared correctness, all from the same K rows."""
    cols: dict[str, list[float]] = {s: [] for s, _, _ in SCORE_LINES}
    correct: list[bool] = []
    for ex in examples:
        raw = [cache.get(arm, ex.claim_id, v) for v in range(5)]
        good = [r for r in raw if r is not None]
        if len(good) < 2 or raw[0] is None:
            continue
        res = _claim_band(good, 0.5)
        if res is None:
            continue
        pred, band = res
        normed = [n for n in (_normed(r) for r in good) if n is not None]
        one = _normed(raw[0])
        if one is None or len(normed) < 2:
            continue
        votes = np.array([int(np.argmax(n)) for n in normed])
        vote_frac = np.bincount(votes, minlength=len(normed[0])).astype(float)
        vote_frac /= vote_frac.sum()
        per_call = [_entropy(n) for n in normed]
        pred_ent = _entropy(np.mean(np.stack(normed), axis=0))
        cols["self_consistency"].append(1.0 - float(vote_frac.max()))
        cols["entropy_1call"].append(_entropy(one))
        cols["predictive_entropy"].append(pred_ent)
        cols["band"].append(band)
        cols["mutual_information"].append(pred_ent - float(np.mean(per_call)))
        correct.append(pred == ex.label)
    return {s: np.array(v) for s, v in cols.items()}, np.array(correct, dtype=bool)


def _rc_from_score(score: np.ndarray, correct: np.ndarray, n_points: int = 20):
    order = score.argsort()  # keep most-confident (lowest score) first
    sorted_correct = correct[order]
    n = len(score)
    cov, acc = [], []
    for frac in np.linspace(1.0, 0.15, n_points):
        k = max(1, int(round(n * frac)))
        cov.append(k / n * 100)
        acc.append(sorted_correct[:k].mean() * 100)
    return np.array(cov), np.array(acc)


def figure_payoff_by_score(examples) -> None:
    n = len(MODELS)
    fig, axes = plt.subplots(1, n, figsize=(12.5, 4.6), sharey=True)
    colors = get_palette("npg", len(SCORE_LINES))

    for ax, m in zip(axes, MODELS):
        cache = BandCache(_REPO_ROOT / "experiments" / "dirichlet_confidence" / m["cache"])
        scores, correct = _score_table("phrasing", examples, cache)
        base = correct.mean() * 100
        for (s, _label, ls), c in zip(SCORE_LINES, colors):
            cov, acc = _rc_from_score(scores[s], correct)
            ax.plot(cov, acc, marker="o", ms=3, lw=1.5, ls=ls, color=c, zorder=3)
        ax.axhline(base, ls=":", lw=1.0, color="#888888", zorder=1)
        ax.text(99, base + 0.6, "keep everything", ha="right", va="bottom",
                fontsize=7, color="#777777")
        ax.set_title(m["label"].split("\n")[0], fontsize=10, fontweight="bold")
        ax.set_xlabel("% of claims kept\n(most-uncertain discarded first)")
        ax.set_xlim(105, 12)
        despine(ax)

    axes[0].set_ylabel("Accuracy on the claims you keep (%)")
    handles = [plt.Line2D([], [], color=c, ls=ls, marker="o", ms=4, lw=1.5, label=lab)
               for (_s, lab, ls), c in zip(SCORE_LINES, colors)]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.965))

    fig.suptitle("Which uncertainty signal best flags wrong answers? (payoff curves by score)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.text(0.5, -0.05,
             "How to read this: every line is one way of scoring how uncertain the model is; the prediction itself is "
             "identical across lines, so\nall start at the same point and a line that climbs FASTER as you discard "
             "high-uncertainty claims is a better error-detector. Solid =\nbased on the model's stated confidence; "
             "dashed = based on disagreement between repeated asks. Takeaway: the confidence-based\nlines (incl. a "
             "single call) rise highest; the disagreement-based band and mutual-information lines lag — repeating "
             "the question\nand measuring disagreement does not beat simply reading the model's verbalized confidence "
             "once. Shown on the phrasing arm.",
             ha="center", va="top", fontsize=7.6, color="#333333", wrap=True)

    fig.subplots_adjust(bottom=0.30, top=0.80, wspace=0.08)
    out = savefig(fig, OUTDIR / "fig4_payoff_by_score.png", dpi=200)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure 5 — residual error composition: what kind of error survives the gate?
# ---------------------------------------------------------------------------
# An "answer-flip" claim = the top answer CHANGES across the 5 wordings (label
# disagreement). A "stable" claim = all 5 wordings agree on the top answer.
# Among the claims a confidence gate KEEPS, we split the remaining errors into
# flip-errors and stable-errors, under two gates:
#   • 1-call gate  — keep lowest single-histogram entropy. Blind to flips (each
#                    individual call looks confident).
#   • 5-call gate  — keep lowest predictive entropy (mean of 5). A flip raises
#                    predictive entropy, so flips SHOULD be filtered out.
# Claim under test: flip-errors persist under the 1-call gate but vanish under
# the 5-call gate — i.e. the extra calls specifically remove the confident-flip
# residual a single call cannot see.


def _residual_table(arm: str, examples, cache: BandCache):
    correct, flipper, gate_1call, gate_pred = [], [], [], []
    for ex in examples:
        raw = [cache.get(arm, ex.claim_id, v) for v in range(5)]
        good = [r for r in raw if r is not None]
        if len(good) < 2 or raw[0] is None:
            continue
        res = _claim_band(good, 0.5)
        if res is None:
            continue
        pred, _band = res
        normed = [n for n in (_normed(r) for r in good) if n is not None]
        one = _normed(raw[0])
        if one is None or len(normed) < 2:
            continue
        labels = [int(np.argmax(n)) for n in normed]
        correct.append(pred == ex.label)
        flipper.append(len(set(labels)) > 1)
        gate_1call.append(_entropy(one))
        gate_pred.append(_entropy(np.mean(np.stack(normed), axis=0)))
    return (np.array(correct, dtype=bool), np.array(flipper, dtype=bool),
            np.array(gate_1call), np.array(gate_pred))


def _residual_rates(gate: np.ndarray, correct: np.ndarray, flipper: np.ndarray,
                    n_points: int = 18):
    """Per coverage: error rate among kept claims, split flip vs stable."""
    order = gate.argsort()  # keep most-confident (lowest gate) first
    wrong = ~correct
    n = len(gate)
    cov, flip_err, stable_err = [], [], []
    for frac in np.linspace(1.0, 0.2, n_points):
        k = max(1, int(round(n * frac)))
        kept = order[:k]
        cov.append(k / n * 100)
        flip_err.append(100 * float((wrong[kept] & flipper[kept]).sum()) / k)
        stable_err.append(100 * float((wrong[kept] & ~flipper[kept]).sum()) / k)
    return np.array(cov), np.array(flip_err), np.array(stable_err)


# Arm-specific wording so the temperature arm reads correctly (it is a CONTROL:
# resampling flips the answer less than rewording does, so the effect should be
# weaker — confirming the residual is driven by *paraphrase* instability).
ARM_WORDS = {
    "phrasing": {"verb": "reword the question", "across": "the 5 wordings",
                 "kind": "paraphrase"},
    "temperature": {"verb": "resample at higher temperature",
                    "across": "the 5 resamples", "kind": "sampling"},
}


def figure_residual_composition(examples, arm: str = "phrasing") -> dict:
    n = len(MODELS)
    fig, axes = plt.subplots(1, n, figsize=(12.5, 4.6), sharey=True)
    colors = get_palette("npg", 4)
    flip_c, stable_c = colors[0], colors[3]  # red = flip, blue = stable
    w = ARM_WORDS[arm]
    headline: dict = {}

    print(f"\nResidual composition ({arm} arm) — flip-error rate among kept, at 50% coverage:")
    for ax, m in zip(axes, MODELS):
        cache = BandCache(_REPO_ROOT / "experiments" / "dirichlet_confidence" / m["cache"])
        correct, flipper, g1, gp = _residual_table(arm, examples, cache)
        flip_overall = 100 * flipper.mean()

        cov, fe1, se1 = _residual_rates(g1, correct, flipper)
        _cov, fep, sep = _residual_rates(gp, correct, flipper)
        ax.plot(cov, fe1, marker="o", ms=3.5, lw=1.8, ls="-", color=flip_c, zorder=4)
        ax.plot(cov, fep, marker="o", ms=3, lw=1.5, ls="--", color=flip_c, zorder=4)
        ax.plot(cov, se1, marker="s", ms=3, lw=1.5, ls="-", color=stable_c, zorder=3)
        ax.plot(_cov, sep, marker="s", ms=2.5, lw=1.2, ls="--", color=stable_c, zorder=3)
        ax.set_title(f"{m['label'].split(chr(10))[0]}\n({flip_overall:.0f}% of claims flip)",
                     fontsize=9.5, fontweight="bold")
        ax.set_xlabel("% of claims kept\n(most-confident kept; gate tightens →)")
        ax.set_xlim(105, 15)
        despine(ax)

        # headline number at ~50% coverage
        i50 = int(np.argmin(np.abs(cov - 50)))
        model_name = m["label"].split(chr(10))[0]
        headline[model_name] = {"flip_1call": float(fe1[i50]), "flip_5call": float(fep[i50]),
                                "stable_1call": float(se1[i50]), "stable_5call": float(sep[i50]),
                                "flip_overall": float(flip_overall)}
        print(f"  {model_name:>13}: flip-errors  1-call gate "
              f"{fe1[i50]:4.1f}%  vs  5-call gate {fep[i50]:4.1f}%   "
              f"(stable-errors {se1[i50]:4.1f}% → {sep[i50]:4.1f}%)")

    axes[0].set_ylabel("Error rate among kept claims (%)")
    handles = [
        plt.Line2D([], [], color=flip_c, ls="-", marker="o", lw=1.8,
                   label="answer-flip errors · 1-call gate (blind to flips)"),
        plt.Line2D([], [], color=flip_c, ls="--", marker="o", lw=1.5,
                   label="answer-flip errors · 5-call gate (should catch flips)"),
        plt.Line2D([], [], color=stable_c, ls="-", marker="s", lw=1.5,
                   label="stable errors · 1-call gate"),
        plt.Line2D([], [], color=stable_c, ls="--", marker="s", lw=1.2,
                   label="stable errors · 5-call gate"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.965))

    fig.suptitle(f"What kind of error survives a confidence gate? ({w['kind']} arm — "
                 f"{'main' if arm == 'phrasing' else 'control'})",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.text(0.5, -0.05,
             f"How to read this: an 'answer-flip' claim is one whose top answer CHANGES when you {w['verb']}; "
             "a 'stable' claim\nkeeps the same answer. Among the claims a confidence gate keeps, we split the leftover "
             "ERRORS into flip (red) and stable (blue),\nunder a 1-call gate (solid) and a 5-call predictive-entropy "
             "gate (dashed). As the gate tightens (move right) stable errors fall —\nhonest uncertainty is filtered. "
             "The question is the RED lines: if solid stays high while dashed drops, the confident-flip errors are "
             f"invisible to a single call but removed by 5 calls. "
             f"Flips here come from {w['kind']} variation ({w['across']}). Pilot, small n.",
             ha="center", va="top", fontsize=7.6, color="#333333", wrap=True)

    fig.subplots_adjust(bottom=0.30, top=0.80, wspace=0.08)
    out = savefig(fig, OUTDIR / f"fig5_residual_{arm}.png", dpi=200)
    print(f"  wrote {out}")
    return headline


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    summaries = [_load_summary(m["summary"]) for m in MODELS]
    print("Recomputing per-claim bands from cache (no API calls)...")
    bands = [(s["config"]["model"], _per_claim_bands(m["cache"]))
             for m, s in zip(MODELS, summaries)]
    print("Rendering figures...")
    figure_scorecard(summaries)
    figure_mechanism(bands)
    figure_risk_coverage(bands)
    examples = balanced_subset(load_scifact(str(SCIFACT)), N_CLAIMS, seed=SEED)
    figure_payoff_by_score(examples)
    h_phr = figure_residual_composition(examples, "phrasing")
    h_tmp = figure_residual_composition(examples, "temperature")
    print("\nControl comparison — flip-errors left by a 1-call gate at 50% coverage:")
    print(f"  {'model':>13}  {'phrasing (reword)':>18}  {'temperature (resample)':>22}")
    for name in h_phr:
        print(f"  {name:>13}  {h_phr[name]['flip_1call']:>16.1f}%  "
              f"{h_tmp[name]['flip_1call']:>20.1f}%")
    print(f"\nAll figures in {OUTDIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
