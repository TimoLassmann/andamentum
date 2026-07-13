"""One-off: curate ~20 short, readable, label-balanced illustrative examples
from the two source experiments and bake them into ``examples.json``.

- ``compare`` examples come from JudgeBench (``experiments/pairwise_judge``):
  a question + two candidate answers + an objective "which is better" gold
  label. These drive :func:`andamentum.llm_judge.judge_compare`.
- ``score`` examples come from SciFact (``experiments/dirichlet_confidence``):
  a claim + an evidence passage + a SUPPORT / CONTRADICT / NOINFO gold label.
  These drive :func:`andamentum.llm_judge.judge_score` under a single
  "factual accuracy" criterion, with the gold label mapped to an expected
  verdict (SUPPORT -> meets, CONTRADICT -> fails, NOINFO -> partial).

This reads the (untracked) experiment data dirs directly and writes a
self-contained ``examples.json`` next to it, so the benchmark itself needs
neither the experiment dirs nor a network connection. Run once:

    uv run python benchmarks/llm_judge/extract_examples.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
_EXPERIMENTS = _REPO / "experiments"

# Reuse the experiments' own loaders rather than re-parsing their formats.
sys.path.insert(0, str(_EXPERIMENTS / "pairwise_judge"))
sys.path.insert(0, str(_EXPERIMENTS / "dirichlet_confidence"))

import judgebench  # noqa: E402
import scifact  # noqa: E402

N_COMPARE = 10
N_SCORE = 10
# Keep examples readable on a console: cap the total character budget so we
# pick short, self-contained items rather than 3k-char reasoning chains.
COMPARE_MAX_CHARS = 2600
SCORE_MAX_CHARS = 1100

_SCIFACT_EXPECTED = {"SUPPORT": "meets", "CONTRADICT": "fails", "NOINFO": "partial"}


def _pick_compare() -> list[dict]:
    examples = judgebench.load_judgebench()
    short = [
        e
        for e in examples
        if len(e.question) + len(e.response_a) + len(e.response_b) <= COMPARE_MAX_CHARS
    ]
    # Balance across gold (A better / B better), stable order.
    by_gold: dict[int, list] = {0: [], 1: []}
    for e in sorted(short, key=lambda e: e.pair_id):
        by_gold[e.gold].append(e)
    picked: list[dict] = []
    for i in range(N_COMPARE):
        pool = by_gold[i % 2]
        if not pool:
            continue
        e = pool.pop(0)
        picked.append(
            {
                "id": e.pair_id,
                "question": e.question.strip(),
                "response_a": e.response_a.strip(),
                "response_b": e.response_b.strip(),
                "gold": "a" if e.gold == 0 else "b",
                "source": e.source,
            }
        )
    return picked


def _pick_score() -> list[dict]:
    examples = scifact.load_scifact(
        _EXPERIMENTS / "dirichlet_confidence" / "data" / "scifact"
    )
    short = [e for e in examples if len(e.claim) + len(e.evidence) <= SCORE_MAX_CHARS]
    by_label: dict[int, list] = {0: [], 1: [], 2: []}
    for e in sorted(short, key=lambda e: e.claim_id):
        by_label[e.label].append(e)
    # 4 SUPPORT, 4 CONTRADICT, 2 NOINFO.
    quota = {0: 4, 1: 4, 2: 2}
    picked: list[dict] = []
    for label, want in quota.items():
        for e in by_label[label][:want]:
            picked.append(
                {
                    "id": e.claim_id,
                    "claim": e.claim.strip(),
                    "evidence": e.evidence.strip(),
                    "gold_label": e.label_name,
                    "expected_verdict": _SCIFACT_EXPECTED[e.label_name],
                }
            )
    return picked[:N_SCORE]


def main() -> None:
    compare = _pick_compare()
    score = _pick_score()
    out = {
        "provenance": {
            "compare": "JudgeBench (experiments/pairwise_judge) — objective which-is-better labels",
            "score": "SciFact (experiments/dirichlet_confidence) — claim/evidence SUPPORT/CONTRADICT/NOINFO",
        },
        "compare": compare,
        "score": score,
    }
    dest = _HERE / "examples.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(compare)} compare + {len(score)} score examples -> {dest}")


if __name__ == "__main__":
    main()
