# What the exposed fields buy — through the real module

`field_benchmark.py` drives the actual `judge_score` / `judge_compare` entry
points (not raw elicitation like `run.py`) and compares the discrete verdict
the module reports against the continuous field it now also exposes.

**Models:** `gpt-5.4-nano`, `gpt-5.4-mini`. **Data:** balanced SciFact score
(n=60, 20/class) + JudgeBench compare (n=50, 25/class) from `data_powered.json`.
All calls cached to `results/field_cache.jsonl` — re-run recomputes offline.

## Score — `expected_score` vs argmax `overall`

| model | argmax acc (3-way) | AUROC(expected_score; SUPPORT vs CONTRADICT) | risk-cov @60% | @40% |
|---|---:|---:|---:|---:|
| gpt-5.4-nano | 80% | **0.995** | 100% | 100% |
| gpt-5.4-mini | 90% | **0.995** | 100% | 100% |

The standout. The **same elicited numbers** that argmax to an 80–90%-accurate
3-way label separate SUPPORT from CONTRADICT almost perfectly as a *ranking*
signal (AUROC 0.995 on both models). And the field is a usable **threshold
knob**: rank the binary claims by `|expected_score − 0.5|`, drop the least
confident 40%, and directional accuracy on what's kept goes to **100%**. The
argmax label cannot express any of this — it collapses a confident pass and a
near-miss to the same word.

## Compare — `expected_preference` vs argmax `winner`

| model | argmax acc | tie-rate | E-pref acc | delta |
|---|---:|---:|---:|---:|
| gpt-5.4-nano | 66% | 2% | 68% | **+2%** |
| gpt-5.4-mini | 66% | 2% | 66% | +0% |

Marginal on *accuracy* here — the module already ties only 2% of the time, so
converting ties into decisions can only move a point or two (nano +2, mini +0;
inside the noise at n=50). The value of `expected_preference` on compare is
therefore **not** an accuracy jump — it's the graded margin (a 0.875 win vs a
0.51 near-coin-flip read the same 'a' from argmax) for ranking / thresholding /
routing, and a lean to inspect when the panel `winner` is a conservative 'tie'.

## Read

- **`expected_score` earns its place** — a near-perfect, ground-truth-free
  ranking signal and a real risk-coverage dial, at zero extra model cost, off
  numbers the module was already throwing away.
- **`expected_preference` is a margin, not an accuracy boost** — worth exposing
  (free, and it de-collapses ties and near-ties) but don't sell it as making
  the judge more correct at picking a winner.

## Caveats

One dataset per task, n=60/50, single-run (no repeated-draw averaging), one
model family. AUROC and accuracy are different metrics — not compared across
the two tables. Compare used the six default criteria folded into the prompt;
score used a single factual-support criterion. Consistent with the raw-signal
finding in `RESULTS.md` (there: expected AUROC ≈ 0.98).
