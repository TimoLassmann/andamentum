"""Small LIVE benchmark for Tier 0 + Tier 1 on the worktree's epistemic code.

Runs the REAL `epistemic_judge_evidence` agent (production prompt) on a handful
of SciFact claims via a live local model, then pushes each judgment through the
REAL `compute_posterior` (Tier 1) to confirm both tiers work end-to-end on live
data and to get a feel for the effect.

Deliberately small (default 12 claims). Reads SciFact data by absolute path so it
does NOT import the main-repo experiments package — the worktree's `andamentum`
stays the only one on the path.

    uv run python tier_live_benchmark.py
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import numpy as np

from andamentum.epistemic.runner import DefaultAgentRunner
from andamentum.epistemic.judge import judge_evidence
from andamentum.epistemic.entities import Evidence
from andamentum.epistemic.confidence import _evidence_counting_vote


def _counting_posterior(dist: list[float] | None, verdict: str) -> float:
    """Single-evidence counting posterior via the REAL Tier 1 vote function."""
    ev = Evidence(
        objective_id="x",
        source_type="web_search",
        source_ref="x",
        support_judgment=verdict,
        judgment_distribution=dist,
    )
    s, c = _evidence_counting_vote(ev)
    return 1.0 / (1.0 + math.exp(-(s - c)))

SCIFACT_DIR = Path(
    "/Users/timo/code/andamentum/experiments/dirichlet_confidence/data/scifact"
)
MODEL = "ollama:gemma4:12b-mxfp8"
N = 12
LABEL_ID = {"supports": 0, "contradicts": 1, "no_bearing": 2}
GOLD_NAME = {0: "SUPPORT", 1: "CONTRADICT", 2: "NOINFO"}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def load_scifact_balanced(n: int) -> list[dict]:
    """Minimal SciFact loader → balanced [{claim, evidence, gold}] (gold: 0/1/2)."""
    corpus = {
        str(r["doc_id"]): " ".join(r.get("abstract", []))
        for r in _read_jsonl(SCIFACT_DIR / "corpus.jsonl")
    }
    examples: list[dict] = []
    for row in _read_jsonl(SCIFACT_DIR / "claims_dev.jsonl"):
        ev_map = row.get("evidence") or {}
        if ev_map:
            for doc_id, entries in ev_map.items():
                ab = corpus.get(str(doc_id))
                if not ab or not entries:
                    continue
                lab = {"SUPPORT": 0, "CONTRADICT": 1}[str(entries[0]["label"]).upper()]
                examples.append({"claim": row["claim"], "evidence": ab, "gold": lab})
        else:
            for doc_id in (row.get("cited_doc_ids") or [])[:1]:
                ab = corpus.get(str(doc_id))
                if ab:
                    examples.append({"claim": row["claim"], "evidence": ab, "gold": 2})
    rng = np.random.default_rng(42)
    by: dict[int, list[dict]] = {0: [], 1: [], 2: []}
    for ex in examples:
        by[ex["gold"]].append(ex)
    per = n // 3
    picked: list[dict] = []
    for lab in (0, 1, 2):
        pool = by[lab]
        idx = rng.permutation(len(pool))[:per]
        picked.extend(pool[i] for i in idx)
    rng.shuffle(picked)
    return picked


def roc_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """ROC-AUC via rank-sum; NaN if a class is empty."""
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


async def main() -> None:
    examples = load_scifact_balanced(N)
    runner = DefaultAgentRunner(model=MODEL)

    rows: list[dict] = []
    print(f"Running {len(examples)} live judgments on {MODEL} ...\n")
    for ex in examples:
        j = await judge_evidence(
            claim_statement=ex["claim"],
            claim_scope="",
            evidence_content=ex["evidence"],
            evidence_source="scifact: abstract",
            runner=runner,
        )
        pred = LABEL_ID[j.verdict]
        # Tier 1: soft (with distribution) vs hard (no distribution), both via
        # the REAL _evidence_counting_vote production function.
        cp_soft = _counting_posterior(j.distribution, j.verdict)
        cp_hard = _counting_posterior(None, j.verdict)

        ok = "✓" if pred == ex["gold"] else "✗"
        rows.append(
            {
                "gold": ex["gold"],
                "pred": pred,
                "correct": pred == ex["gold"],
                "dist": j.distribution,
                "entropy": j.entropy,
                "one_hot": j.is_one_hot,
                "cp_soft": cp_soft,
                "cp_hard": cp_hard,
            }
        )
        print(
            f"{ok} gold={GOLD_NAME[ex['gold']]:<10} pred={j.verdict:<11} "
            f"dist={[round(x, 2) for x in j.distribution]} H={j.entropy:.2f} "
            f"cp_soft={cp_soft:.3f} cp_hard={cp_hard:.3f}"
        )

    correct = np.array([r["correct"] for r in rows])
    entropy = np.array([r["entropy"] for r in rows])
    one_hot = np.array([r["one_hot"] for r in rows])
    cp_soft = np.array([r["cp_soft"] for r in rows])
    cp_hard = np.array([r["cp_hard"] for r in rows])

    print("\n" + "=" * 60)
    print(f"Tier 0 (real judge agent, live {MODEL}, n={len(rows)})")
    print(f"  verdict accuracy      : {correct.mean():.3f}  ({correct.sum()}/{len(rows)})")
    print(f"  degeneracy (one-hot)  : {one_hot.mean():.0%}")
    print(f"  mean entropy          : {entropy.mean():.3f}")
    auroc = roc_auc(entropy, ~correct)
    print(f"  error-detection AUROC : {auroc:.3f}  (entropy flags wrong verdicts)")
    print("\nTier 1 (real compute_posterior counting, soft vs hard)")
    print(f"  mean |cp-0.5| soft    : {np.abs(cp_soft - 0.5).mean():.3f}")
    print(f"  mean |cp-0.5| hard    : {np.abs(cp_hard - 0.5).mean():.3f}")
    print("  (soft ≤ hard ⇒ graded judgments are less overconfident than votes)")


if __name__ == "__main__":
    asyncio.run(main())
