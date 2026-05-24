# Implementation Spec: Multi-Phrasing Verbalized Distributions for Calibrated Agent Confidence

## Purpose

This document specifies a general, domain-agnostic method for obtaining better-calibrated,
more honest confidence estimates from LLM agents that must output a probability distribution over
a fixed set of categories.

The setting: an agent is asked a classification-style question whose answer is one of `C` mutually
exclusive categories, and instead of a single answer we want a full probability distribution over
those categories, along with a meaningful estimate of how *uncertain* that distribution is. Those
distributions are then consumed by a downstream Monte Carlo simulation that propagates the
uncertainty through whatever scoring function the system computes.

The method is additive: it sits between "call the agent" and "run the downstream simulation," and
does not assume anything about what the categories mean or how the downstream score is computed.

## Background / Motivation

A common pattern is to have an agent emit a single verbalized categorical distribution: a row of
integers, one per category, constrained to sum to 100 (e.g. for a 4-, 5-, 6-, or 7-way
classification). These integers are normalized to a probability distribution and a downstream
process samples from them.

Two known problems with single-prompt verbalized distributions:

1. **Prompt sensitivity.** The reliability of verbalized confidence/probability depends heavily
   on the exact wording of the prompt. Asking the "same" question different ways yields
   different distributions. A single fixed phrasing bakes in whatever bias that phrasing carries.

2. **Mode collapse / overconfidence.** Aligned models tend to over-concentrate probability mass
   on one or two categories, understating genuine uncertainty. This makes any downstream variance
   estimate too small (falsely confident).

### Core idea

For each agent query, ask the model under **K different but semantically equivalent phrasings** of
the same question. Each phrasing returns its own categorical distribution. Pool the K
distributions — or, better, carry all K forward into the Monte Carlo step. The spread *across*
phrasings is itself a measurement of second-order uncertainty (uncertainty about the distribution)
and should widen the downstream variance to a more honest level.

This is grounded in three threads in the recent literature; see References.

## Key references and the math they provide

### 1. Verbalized probability distributions + the re-softmax problem (the "Amazon paper")

**Wang, Szarvas, Balazs, Danchenko, Ernst. "Calibrating Verbalized Probabilities for Large
Language Models." arXiv:2410.06707.**

This paper generalizes single-score verbalized confidence to a *full categorical distribution
over classes* — exactly the output format used here. The critical contribution for us is a
calibration warning:

- The integers an LLM emits already **look like normalized probabilities** (they sum to 100).
- Standard post-hoc calibration tools (e.g. temperature scaling) are mathematically defined to
  operate on **unnormalized logits** `z`, via `p = softmax(z / T)`, **not** on already-normalized
  probabilities.
- Applying temperature scaling directly to verbalized probabilities applies softmax a second time
  ("re-softmax"), which is incorrect and distorts the distribution.
- **Fix — the invert-softmax trick:** recover an approximate logit from the verbalized
  probability before calibrating. Treat the verbalized probability `p_i` as if it were a softmax
  output and invert it: `z_i = log(p_i)` (up to an arbitrary additive constant, which softmax is
  invariant to). Then apply temperature scaling on those recovered logits:
  `p_calibrated = softmax(log(p) / T)`. Fit `T` on a held-out labeled set.

**Implication:** if/when calibration is added, do NOT temperature-scale the raw 100-point rows.
First convert to pseudo-logits via `log(p)`, scale, then re-softmax. See `calibrate_distribution()`
below.

### 2. Verbalized Sampling (the "ask for a distribution" framing helps)

**"Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity." arXiv (2025);
see also VS-Standard / VS-CoT / VS-Multi.**

Findings relevant to us:

- Prompting for a *distribution of options with probabilities* (rather than a single answer)
  counteracts alignment-induced mode collapse and recovers a richer distribution latent in the
  model. Emitting a distribution at all is therefore already beneficial.
- **However, format still matters:** they compared asking for "probability" vs "percentage" vs
  "confidence" and found all helped but one was empirically best; they standardized on it. So
  distribution-framing reduces but does not eliminate prompt sensitivity — which is precisely why
  averaging over multiple phrasings (this spec) is worthwhile.

### 3. Imprecise probabilities (why cross-phrasing spread is meaningful)

**"Verbalizing LLM's Higher-order Uncertainty via Imprecise Probabilities." arXiv:2603.10396.**

Distinguishes first-order uncertainty (over answers) from second-order uncertainty (uncertainty
about the distribution itself). Our cross-phrasing spread is an empirical estimate of that
second-order uncertainty. This justifies carrying all K phrasing-distributions into the Monte
Carlo step rather than collapsing them too early.

## Functional requirements

### FR1 — Multiple phrasings per query

For each agent/question type, maintain `K` (default `K = 5`) prompt templates that ask the same
question in semantically equivalent but lexically different ways. They must:

- Preserve the **exact same category set, order, and labels**.
- Preserve the output constraints: integers 0–100, summing to exactly 100, plus whatever other
  columns the output schema requires (justification, supporting evidence/IDs, etc.).
- Vary only surface form: e.g. "distribute 100 points across…", "give the probability (%) that…",
  "how confident are you, as percentages, that…", "rate the likelihood of each category…", plus a
  chain-of-thought variant ("reason step by step, then output the table").

Store templates in a config file (e.g. `phrasings.yaml`) keyed by question type, so they can be
edited without touching code. Each template is parameterized by the same input variables the
current prompts use.

### FR2 — Parsing and validation per response

For each of the `K` responses per query:

1. Parse the structured output into a vector of integers `v` of length `C` (number of categories).
2. **Repair the sum constraint** (LLMs frequently emit sums of 99/101): if `sum(v) != 100` but
   `sum(v) > 0`, renormalize to probabilities `p = v / sum(v)`. If `sum(v) == 0` or the row failed
   to parse, mark that phrasing as **failed** and exclude it (cap retries at 2; do not retry
   indefinitely).
3. Clip negatives to 0 before normalizing.
4. Result: up to `K` probability vectors `p^(1) ... p^(K)`, each length `C`, each summing to 1.

Keep the per-phrasing distributions; do not average yet (see FR4 for why both options exist).

### FR3 — Temperature 0 caveat

If agents run at `temperature = 0` for reproducibility, the *same* prompt returns the *same*
output, so diversity must come from the **different phrasings**, not from resampling one phrasing.
Keep `temperature = 0`. The K phrasings are the only source of variation at the elicitation stage;
the Monte Carlo step remains the only place stochastic sampling occurs. (If you later want
within-phrasing diversity too, raise temperature and sample each phrasing `m` times — out of scope
here, and it changes reproducibility guarantees.)

### FR4 — Pooling strategy (make it configurable)

Provide two modes; default to `mixture`.

- **`mean` (simple pool):** average the K vectors elementwise:
  `p_pooled = (1/K) * sum_k p^(k)`. Feed `p_pooled` to Monte Carlo. Simple, but collapses
  second-order uncertainty into the first-order distribution.

- **`mixture` (preferred — preserves second-order uncertainty):** treat the K phrasings as a
  uniform mixture. In the Monte Carlo loop, for each iteration, **first pick one of the K phrasings
  uniformly at random, then sample a category from that phrasing's distribution.** This is equal in
  expectation to `mean`, but it **propagates the disagreement between phrasings into the Monte
  Carlo variance**, which is the whole point (honest confidence).

```
# mixture sampling inside the MC loop, per query:
k = uniform_int(0, K-1)         # choose a phrasing
category = categorical(p[k])    # sample from that phrasing's distribution
```

### FR5 — Optional calibration hook (off by default)

Expose `calibrate_distribution(p, T)` implementing the invert-softmax trick from the Amazon paper,
so a temperature `T` fitted on labeled data can be applied correctly later:

```
def calibrate_distribution(p, T):
    # p: normalized probability vector (sums to 1); floor zeros first
    p = clip(p, eps, 1.0)
    z = log(p)                  # invert-softmax: recover pseudo-logits
    return softmax(z / T)       # re-normalize after scaling
```

Apply per-phrasing **before** pooling. Default `T = 1.0` (no-op). `T` must be fit against
ground-truth labels — calibration without labels is not possible and must not be faked.

### FR6 — Reporting / diagnostics

For each query, log:

- The K individual distributions.
- A cross-phrasing disagreement metric, e.g. mean pairwise total-variation distance, or the
  entropy of the pooled distribution minus the mean entropy of individual distributions. High
  disagreement = the answer is phrasing-sensitive and should be flagged for human review.
- Which phrasings failed parsing/validation.

This metric is itself a useful output: it tells reviewers which downstream scores rest on shaky,
phrasing-dependent agent judgments.

## What must NOT change

- The category sets, their order, and any mapping from category to downstream score.
- The downstream scoring function and the Monte Carlo aggregation outputs (expected value,
  variance, sample distribution), the sample count `N`, and the random seed. Only the *source* of
  the sampled categories changes (FR4 mixture mode).
- Any agents that produce discrete classifications rather than distributions — those are out of
  scope.

## Suggested implementation order

1. Add `phrasings.yaml` with K=5 templates per question type (preserve all output constraints).
2. Add an agent-runner that loops over phrasings, calls the model at T=0, parses, validates, and
   repairs each row (FR2), returning a list of K distributions.
3. Add `pool(distributions, mode)` (FR4) and wire `mixture` into the Monte Carlo sampling step.
4. Add `calibrate_distribution()` as a no-op-by-default hook (FR5).
5. Add diagnostics logging (FR6).
6. Regression test: with K=1 and mode=`mean`, output must be identical to the current behavior.
   This guarantees the change is a strict superset of what exists today.

## Validation plan (recommended, not blocking)

None of this guarantees *calibration* — it reduces phrasing brittleness and produces more honest
variance, but whether a stated 70% means 70% still must be checked against labels. If a labeled
set is available:

- Fit temperature `T` per question type via the invert-softmax route (FR5).
- Compare Expected Calibration Error (ECE) and reliability diagrams: single-prompt vs.
  multi-phrasing-mean vs. multi-phrasing-mixture, with and without calibration.
- Confirm the multi-phrasing variance is better correlated with actual agent error than the
  single-prompt variance.

## References

1. Wang, Szarvas, Balazs, Danchenko, Ernst. *Calibrating Verbalized Probabilities for Large
   Language Models.* arXiv:2410.06707. — full categorical distribution elicitation; re-softmax
   problem and invert-softmax fix.
2. *Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity.* arXiv (2025). —
   distribution-framing recovers diversity; prompt format still matters.
3. *Verbalizing LLM's Higher-order Uncertainty via Imprecise Probabilities.* arXiv:2603.10396. —
   first- vs second-order uncertainty; justifies preserving cross-phrasing spread.
4. Yang et al. *On Verbalized Confidence Scores for LLMs.* arXiv:2412.14737. — baseline evidence
   that verbalized confidence reliability depends strongly on prompt design and model capacity.
