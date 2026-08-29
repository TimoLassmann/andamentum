"""How much do the 5 histograms actually vary across calls? (cache re-analysis)

If the 5 calls return near-identical histograms there is no disagreement signal
to extract — which would explain why the band / mutual-information added nothing.
The two arms should differ sharply:
  • phrasing arm    — 5 different wordings at TEMPERATURE 0 (deterministic decode;
                      variation comes ONLY from rewording).
  • temperature arm — 1 fixed wording, 5 samples at TEMPERATURE 0.7 (variation
                      comes from sampling randomness).

Per model × arm we report, over the 5 histograms of each claim:
  • TV distance — mean pairwise total-variation distance, 0.5·Σ|p−q|. Reads as
    "what fraction of the 100 points moves between two calls" (0 = identical,
    1 = totally different). The headline number.
  • per-class SD — average standard deviation of a class's probability across
    the 5 calls (the raw material the band is built from).
  • label agreement — % of claims where all 5 calls pick the SAME top answer.

NO API CALLS.
    uv run python -m experiments.dirichlet_confidence.call_variability
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.dirichlet_confidence.band_experiment import BandCache  # noqa: E402
from experiments.dirichlet_confidence.compare_baselines import _normed  # noqa: E402
from experiments.dirichlet_confidence.scifact import (  # noqa: E402
    balanced_subset,
    load_scifact,
)

console = Console()
SCIFACT = "experiments/dirichlet_confidence/data/scifact"
N_CLAIMS, K = 150, 5
MODELS = [
    ("gpt-5.4-nano", "experiments/dirichlet_confidence/cache/band_experiment.jsonl"),
    ("gpt-5.4-mini", "experiments/dirichlet_confidence/cache/band_experiment_mini.jsonl"),
    ("gpt-4o-mini", "experiments/dirichlet_confidence/cache/band_experiment_4omini.jsonl"),
]
ARMS = ["phrasing", "temperature"]


def _tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def _stats(arm: str, examples, cache: BandCache):
    tvs, sds, agree, n_distinct = [], [], [], []
    for ex in examples:
        raw = [cache.get(arm, ex.claim_id, v) for v in range(K)]
        hist = [n for n in (_normed(r) for r in raw) if n is not None]
        if len(hist) < 2:
            continue
        H = np.stack(hist)
        tvs.append(np.mean([_tv(H[i], H[j]) for i, j in combinations(range(len(H)), 2)]))
        sds.append(float(H.std(axis=0).mean()))
        labels = H.argmax(axis=1)
        agree.append(len(set(labels.tolist())) == 1)
        n_distinct.append(len(set(labels.tolist())))
    return {
        "n": len(tvs),
        "tv": float(np.mean(tvs)),
        "sd": float(np.mean(sds)),
        "agree_pct": 100 * float(np.mean(agree)),
        "mean_distinct_labels": float(np.mean(n_distinct)),
    }


def _example(arm: str, cache: BandCache, claim_id: str):
    rows = [cache.get(arm, claim_id, v) for v in range(K)]
    return [r for r in rows if r is not None]


def main() -> int:
    examples = balanced_subset(load_scifact(SCIFACT), N_CLAIMS, seed=42)
    t = Table(title="How much do the 5 histograms vary across calls?")
    t.add_column("Model")
    t.add_column("Arm")
    t.add_column("TV dist\n(0=identical)", justify="right")
    t.add_column("per-class SD", justify="right")
    t.add_column("all 5 agree\non top answer", justify="right")
    t.add_column("avg # distinct\ntop answers", justify="right")
    for model_id, cache_path in MODELS:
        cache = BandCache(_REPO_ROOT / cache_path)
        for arm in ARMS:
            s = _stats(arm, examples, cache)
            t.add_row(model_id, arm, f"{s['tv']:.3f}", f"{s['sd']:.3f}",
                      f"{s['agree_pct']:.0f}%", f"{s['mean_distinct_labels']:.2f}")
        t.add_section()
    console.print(t)

    # Two concrete claims for mini (where rephrasing caused the most flipping):
    # the MOST stable and the MOST variable, phrasing arm, so the range is visible.
    cache = BandCache(_REPO_ROOT / MODELS[1][1])

    def _spread(claim_id: str) -> float:
        hist = [n for n in (_normed(cache.get("phrasing", claim_id, v))
                            for v in range(K)) if n is not None]
        if len(hist) < 2:
            return -1.0
        H = np.stack(hist)
        return float(np.mean([_tv(H[i], H[j])
                              for i, j in combinations(range(len(H)), 2)]))

    scored = sorted(((s, ex) for ex in examples if (s := _spread(ex.claim_id)) >= 0),
                    key=lambda t: t[0])
    picks = [("most STABLE", scored[0][1]), ("most VARIABLE", scored[-1][1])]
    console.print(f"\n[bold]Examples — {MODELS[1][0]}, phrasing arm[/bold] "
                  "(raw 0-100 histograms: [SUPPORT, CONTRADICT, NO-INFO])")
    for tag, ex in picks:
        console.print(f"\n  {tag} claim {ex.claim_id} (gold={ex.label_name}):")
        for i, row in enumerate(_example("phrasing", cache, ex.claim_id)):
            console.print(f"    wording {i}: {[round(x) for x in row]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
