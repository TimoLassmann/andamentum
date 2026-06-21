# Tier 0 design — verbalized confidence at the evidence-judgment layer

Goal (validated by `docs/confidence_analysis.md`): **stop discarding the verbalized distribution** at the system's highest-frequency LLM judgment (`epistemic_judge_evidence`). Capture the 3-way belief distribution, derive a per-judgment confidence/entropy/one-hot signal, and store it — **without changing any decision the system makes today**. Heavier consumption (posterior rewiring) is explicitly deferred (Tier 1, benchmark-gated).

## Principle: strictly additive

`Evidence.support_judgment` (the categorical verdict) remains the load-bearing field every downstream site reads (gates, posterior counting, audit, reports). Tier 0 does not touch those. It only:
1. changes *how* the judge produces the verdict (graded distribution → argmax), and
2. *adds* the distribution + derived signals to `Evidence`.

The derived verdict is identical in meaning to today's, so behaviour is unchanged unless/until a future tier chooses to read the new signal.

## Changes

1. **`thresholds.py`** — add `JUDGMENT_ONE_HOT_THRESHOLD = 0.95` (the validation's degeneracy cutoff: max class prob ≥ 0.95 ⇒ entropy uninformative for that call).

2. **`judgment_signal.py`** (new leaf module, no epistemic deps) — the canonical 3 classes and pure math reused by both the schema and the entity:
   - `JUDGMENT_CLASSES = ("supports", "contradicts", "no_bearing")`
   - `normalize_distribution(...)`, `distribution_confidence`, `distribution_entropy` (normalised Shannon, [0,1]), `distribution_is_one_hot`, `argmax_verdict`.

3. **`agents/output_models.py::EvidenceJudgmentOutput`** — the validated recipe:
   - keep scope decomposition (`claim_scope_summary`, `evidence_scope_summary`, `in_scope`) — this *is* the reasoning-first structure;
   - move `reasoning` **before** the numbers;
   - replace the single `verdict` choice with `belief_supports` / `belief_contradicts` / `belief_no_bearing` (0–100, with field descriptions + sum-to-100 contract);
   - `verdict` becomes a derived `@property` = `argmax_verdict(distribution)` — single source of truth, no LLM/derived divergence;
   - add `distribution`, `confidence`, `entropy`, `is_one_hot` properties;
   - `model_validator` keeps the original semantic invariant, now over the derived verdict: `in_scope=False ⇒ argmax=no_bearing`; `in_scope=True ⇒ argmax∈{supports,contradicts}`. (Direction uncertainty when in-scope is expressed by splitting supports/contradicts, *not* by piling onto no_bearing.)

4. **`entities/evidence.py`** — add `judgment_distribution: list[float] | None` (normalised, ordered by `JUDGMENT_CLASSES`). `judgment_confidence` / `judgment_entropy` / `judgment_one_hot` are **properties** derived from it (single stored field, everything else computed). Wire into `_extra_metadata` / `_from_metadata` for round-trip persistence.

5. **`judge.py`** — add `apply_judgment(evidence, judgment)`: the single place that maps a judgment onto an `Evidence` (verdict + reasoning + distribution). Replaces the duplicated 2-line assignment at the 4 LLM-judge sites (`claims`, `seed_claim`, `multi_seed_claim`, graph `nodes`). The adversarial path in `verification.py` builds evidence from a different agent with no distribution → `judgment_distribution` stays `None` (one-hot property returns `None` = "unknown").

6. **`agents/judge.py`** — update `JUDGE_EVIDENCE_PROMPT` Step 4 + Output: ask for the belief distribution with the calibration-preserving nudge (reserve 0/100 for unambiguous evidence; express in-scope direction doubt by splitting supports/contradicts; reserve no_bearing for out-of-scope/irrelevant).

7. **Tests** — update `test_judge_invariants.py` to the distribution schema; add `test_judgment_signal.py` (math + properties + serialization round-trip).

## Prompt genericity (post-live-run decision)

A 12-claim live run on `gemma4:12b` showed 92% verdict accuracy and the one error
being the highest-entropy call (entropy flagged it; Tier 1 softened it) — but 83%
one-hot output. We deliberately do **not** chase a lower one-hot rate:

- **Degeneracy is not the objective.** What matters is error-detection /
  calibration, which already works (validation ECE 0.039; the live error was the
  least-confident call). High one-hot is mostly "confident *and correct*" on easy
  items — appropriate, not a defect. The system already records `judgment_one_hot`
  as a meta-signal for the cases where entropy is uninformative.
- **No domain-anchoring examples.** The epistemic judge must stay domain-generic;
  worked examples seeded with belief numbers (90/95-style) both anchor toward
  extremes *and* bias toward their domain. So the worked examples now teach the
  scope decomposition only (no belief numbers), and calibration is a
  domain-neutral **betting-odds rubric** in the instructions — no examples.
- **No post-hoc fix exists anyway.** Temperature-scaling the distribution cannot
  rescue a true one-hot (`0` stays `0`), so there is no numeric post-processing
  lever — another reason to accept it rather than engineer against it.

Net: the prompt was made *more* generic than the first Tier 0 cut, and prompt
work stops here.

## Explicitly NOT in Tier 0
- No change to `compute_posterior` / counting (still `1 + log(corroboration)`). That's Tier 1, behind a benchmark.
- No multi-call / paraphrase tripwire (Tier 2).
- No per-model prompt branching — one uniform recipe (validated on gemma + gpt-oss).
