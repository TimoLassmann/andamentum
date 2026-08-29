"""ABLATION: does asking 5 times beat asking ONCE for the abstention signal?

THE QUESTION
============
The baseline run showed predictive_entropy (entropy of the MEAN of the 5
histograms) is the strongest error-detector — better than the band, better than
the vote-based scores. But predictive_entropy still consumes 5 model calls. If a
SINGLE verbalized-confidence call carries the same signal, the entire multi-ask
machinery (band, self-consistency, semantic entropy) is unnecessary for
abstention on this task, and the honest paper shrinks accordingly.

So we put, on the SAME cached answers and the SAME held-fixed prediction
(pooled-mean argmax over the 5), these scores head to head:

  1-CALL
  • entropy_1call       — entropy of ONE histogram (variant 0). The "you don't
                          need to re-ask" baseline.

  5-CALL (decomposed)
  • mean_within_entropy — average of the 5 single-call entropies (the typical
                          per-call uncertainty; ~what 1 call sees, averaged).
  • predictive_entropy  — entropy of the MEAN of the 5 (= mean_within + the
                          spread between calls). The previous winner.
  • mutual_information  — predictive_entropy − mean_within_entropy. The pure
                          BETWEEN-call disagreement term (BALD epistemic
                          uncertainty). The principled "disagreement" score.
  • band                — our Dirichlet between-component spread.

THE TESTS THAT MATTER (paired bootstrap ΔAUROC on identical resampled claims):
  • predictive_entropy − entropy_1call : does 5-call averaging beat 1 call?
  • mutual_information  − entropy_1call : does pure disagreement add over 1 call?
  • band                − predictive_entropy : is the band worse than plain
                          5-call entropy? (confirm the baseline finding)

NO API CALLS — pure re-analysis of cache/band_experiment*.jsonl.

    uv run python -m experiments.dirichlet_confidence.ablation_1v5
    uv run python -m experiments.dirichlet_confidence.ablation_1v5 --out results/ablation_1v5.json
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

from experiments.dirichlet_confidence.band_experiment import (  # noqa: E402
    BandCache,
    _claim_band,
)
from experiments.dirichlet_confidence.compare_baselines import (  # noqa: E402
    _boot_ci,
    _entropy,
    _normed,
    _paired_delta,
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

MODELS = [
    ("openai:gpt-5.4-nano", "experiments/dirichlet_confidence/cache/band_experiment.jsonl"),
    ("openai:gpt-5.4-mini", "experiments/dirichlet_confidence/cache/band_experiment_mini.jsonl"),
    ("openai:gpt-4o-mini", "experiments/dirichlet_confidence/cache/band_experiment_4omini.jsonl"),
]
ARMS = ["phrasing", "temperature"]

# Display order: 1-call first, then the 5-call family.
SCORES = ["entropy_1call", "mean_within_entropy", "predictive_entropy",
          "mutual_information", "band"]
SCORE_CALLS = {"entropy_1call": "1 call", "mean_within_entropy": "5 calls",
               "predictive_entropy": "5 calls", "mutual_information": "5 calls",
               "band": "5 calls"}

# The head-to-head comparisons that decide whether multi-sampling earns its keep.
DELTAS = [
    ("predictive_entropy", "entropy_1call", "does averaging 5 beat 1 call?"),
    ("mutual_information", "entropy_1call", "does pure disagreement add over 1 call?"),
    ("predictive_entropy", "band", "is plain 5-call entropy ≥ the band?"),
]


def _per_claim_scores(arm: str, examples, cache: BandCache):
    out: dict[str, list[float]] = {s: [] for s in SCORES}
    correct: list[bool] = []
    for ex in examples:
        raw = [cache.get(arm, ex.claim_id, v) for v in range(K)]
        good = [r for r in raw if r is not None]
        if len(good) < 2 or raw[0] is None:  # need 5-call spread AND a 1-call value
            continue
        res = _claim_band(good, ALPHA0)  # shared prediction + band
        if res is None:
            continue
        pred, band = res

        normed = [n for n in (_normed(r) for r in good) if n is not None]
        one = _normed(raw[0])
        if one is None or len(normed) < 2:
            continue

        per_call_entropies = [_entropy(n) for n in normed]
        mean_within = float(np.mean(per_call_entropies))
        pred_entropy = _entropy(np.mean(np.stack(normed), axis=0))

        out["entropy_1call"].append(_entropy(one))
        out["mean_within_entropy"].append(mean_within)
        out["predictive_entropy"].append(pred_entropy)
        out["mutual_information"].append(pred_entropy - mean_within)  # ≥ 0 (Jensen)
        out["band"].append(band)
        correct.append(pred == ex.label)

    return ({s: np.array(v) for s, v in out.items()}, np.array(correct, dtype=bool))


def run(args: argparse.Namespace) -> dict:
    examples = balanced_subset(load_scifact(SCIFACT), N_CLAIMS, seed=SEED)
    out: dict = {"config": {"n_claims": N_CLAIMS, "k": K, "alpha0": ALPHA0, "seed": SEED},
                 "results": {}}

    for model_id, cache_path in MODELS:
        cache = BandCache(_REPO_ROOT / cache_path)
        out["results"][model_id] = {}
        for arm in ARMS:
            scores, correct = _per_claim_scores(arm, examples, cache)
            if len(correct) == 0 or correct.all() or (~correct).all():
                continue
            entry = {"n": int(len(correct)), "n_wrong": int((~correct).sum()),
                     "auroc": {}, "deltas": {}}
            for s in SCORES:
                rng = np.random.default_rng(SEED)
                pt, lo, hi = _boot_ci(scores[s], correct, rng)
                entry["auroc"][s] = {"point": pt, "lo": lo, "hi": hi}
            for a, b, _q in DELTAS:
                rng = np.random.default_rng(SEED + 1)
                d, dlo, dhi, fp = _paired_delta(scores[a], scores[b], correct, rng)
                entry["deltas"][f"{a}__minus__{b}"] = {
                    "delta": d, "lo": dlo, "hi": dhi, "frac_first_wins": fp}
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
                            f"(n={e['n']}, wrong={e['n_wrong']})")
            t.add_column("Score")
            t.add_column("Calls", justify="center")
            t.add_column("Error-detect AUROC  [95% CI]", justify="right")
            for s in SCORES:
                a = e["auroc"][s]
                t.add_row(s, SCORE_CALLS[s],
                          f"{a['point']:.3f}  [{a['lo']:.3f}, {a['hi']:.3f}]")
            console.print(); console.print(t)

            d = Table(title="Does multi-sampling earn its keep? (paired bootstrap ΔAUROC)")
            d.add_column("Comparison")
            d.add_column("Question")
            d.add_column("ΔAUROC  [95% CI]", justify="right")
            d.add_column("Verdict")
            for a, b, q in DELTAS:
                v = e["deltas"][f"{a}__minus__{b}"]
                excludes_zero = (v["lo"] > 0) or (v["hi"] < 0)
                if v["lo"] > 0:
                    verdict = f"{a.split('_')[0]} better"
                elif v["hi"] < 0:
                    verdict = f"{b.split('_')[0]} better"
                else:
                    verdict = "no difference"
                d.add_row(f"{a} − {b}", q,
                          f"{v['delta']:+.3f}  [{v['lo']:+.3f}, {v['hi']:+.3f}]",
                          f"[bold]{verdict}[/bold]" if excludes_zero else verdict)
            console.print(d)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=None, help="write JSON results (relative to module dir)")
    run(p.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
