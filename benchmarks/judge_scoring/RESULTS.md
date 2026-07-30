# Results — argmax vs continuous judge scoring

**Models:** `gpt-5.4-nano`, `gpt-5.4-mini`. (Ollama cloud dropped — it returns no
logprobs; see README Test 0.)
**Data:** 100 JudgeBench compare pairs (balanced 50 a / 50 b) and 150 SciFact
claims (balanced 50 SUPPORT / 50 CONTRADICT / 50 NOINFO).

## Headline

**The paper's logit-expectation method loses to the belief-point distribution
`llm_judge` already elicits — on both models, both tasks, and it still loses
after being given a fair shot at reasoning first.**

The genuinely free win is unrelated to logits: **stop collapsing the elicited
distribution to an argmax and expose its expectation.** That costs zero extra
calls and is a far better ranking signal than the discrete label.

## Score task — AUROC separating SUPPORT from CONTRADICT (n=150)

AUROC is the apples-to-apples metric: both continuous methods are scored the same
way. (The argmax row is 3-way classification accuracy — a different metric, shown
only to anchor what the module does today.)

| Method | nano | mini |
|---|---|---|
| *argmax verdict (accuracy, what ships today)* | *76%* | *83%* |
| **Verbalized expectation** (already elicited — free) | **0.984** | **0.983** |
| Logit-expectation, reason-then-score, G=5 | 0.825 | 0.954 |
| Logit-expectation, score-only, G=2 | 0.824 | 0.900 |
| Logit-expectation, score-only, G=5 | 0.800 | 0.926 |
| Logit-expectation, score-only, G=9 | 0.778 | 0.916 |

## Compare task — accuracy (n=100, gold has no ties)

| Method | nano | mini |
|---|---|---|
| argmax (tie counted wrong) | 60% *(tie-rate 8%)* | 75% *(tie-rate 2%)* |
| **Verbalized expectation** (never abstains) | **62%** | **77%** |
| Logit-expectation, score-only | 53% *(1 abstain)* | 71% |
| Logit-expectation, reason-then-answer | 43% *(**30 abstains**)* | 65% |

## What we learned

**1. The logit route is worse, not better.** On score, the verbalized expectation
(0.984 / 0.983) beats the best logit configuration (0.825 / 0.954) on both models.
On compare, every logit variant is *below* the plain argmax baseline. The paper's
central claim does not reproduce on these models for these tasks.

**2. Reasoning-first was a real confound, and correcting it changed the numbers
but not the verdict.** The first logit run asked for a bare digit; the verbalized
route reasons *before* it writes numbers, so the original comparison partly
measured reasoning-vs-none. Letting the logit route reason first improved it
(score AUROC 0.800 → 0.825 nano, 0.926 → 0.954 mini) — real, but nowhere near
enough to close the gap.

**3. Granularity buys nothing.** No peak. On nano finer granularity is
monotonically *worse* (G=2 0.824 → G=9 0.778). A pilot at N=10 appeared to show a
G=5 peak of 0.938 for mini; the powered run erased it. That pilot would have been
a wrong recommendation had we shipped it.

**4. Continuous > argmax — this is the finding worth acting on.** The expectation
beats the argmax on compare for both models (+2 points), purely by converting ties
into decisions. More importantly, the *same elicited numbers* rank SUPPORT above
CONTRADICT with AUROC ≈ **0.98** — a strong, usable ranking signal that the module
currently throws away when it reduces to one of three labels.

**5. The logit route is also operationally fragile.** It needs the model to emit a
score token in an exact position. `nano` failed that format on **30 of 100**
reasoned compare calls (no parseable answer token). The verbalized route uses
structured output and cannot fail this way. Two harness bugs along the way make the
same point: `SCORE` tokenizes as `'S'`+`'CORE'` (so naive marker matching misses
it), and the model's top-5 logprobs contained `'②'`, which `str.isdigit()` accepts
but `int()` rejects.

## Recommendation

- **Do not adopt logit-expectation.** It is worse, it only works on backends that
  expose logprobs (not Ollama cloud), it requires bypassing the module's structured
  output, and it is format-fragile.
- **Do expose the continuous expectation** alongside the existing verdict — e.g. an
  `expected_score` on `ScoreResult` and an `expected_preference` on `CompareResult`,
  derived from the distribution already in hand. Zero extra model calls. Use the
  label for a decision, the continuous value for ranking/routing/thresholding.

## Limitations

One dataset per task; n=100/150; a single factual-accuracy criterion on score;
two models from one family. Test 3 (repeated evaluation, K draws averaged) was
**not** run. AUROC vs accuracy are different metrics and are not compared across
rows. These results justify the recommendation above; they do not settle whether
logit-expectation helps on other model families.
