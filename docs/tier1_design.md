# Tier 1 design — verbalized confidence informs the posterior

Goal: make the Tier 0 signal *do* something. The evidence-claim judge now emits a
belief distribution (Tier 0); Tier 1 lets that confidence shape the **counting
posterior** instead of a hard one-vote-per-item count — the direct analogue of
the experiment's load-bearing lesson ("keep the soft probabilities; vote-counting
is the weakest method").

## The change (single, bounded site)

`confidence.py` — new `_evidence_counting_vote(e) -> (supporting, contradicting)`,
used by **both** counting loops (the diagnostic-only oscillation branch and the
main counting). With a captured distribution `[p_s, p_c, p_n]` and cluster weight
`w = 1 + log(cluster_size)`, the item contributes `w·p_s` to supporting and
`w·p_c` to contradicting (no_bearing mass contributes to neither). A near-tie
nets ~0; a confident judgment nets ~±w.

## Backward-compatibility is a proven property, not a hope

- **One-hot distribution** (`[1,0,0]`) → contributes exactly `w` to one side ⇒
  identical to the old hard vote. One-hot is the small-model degeneracy mode, so
  pre-Tier-0 / degenerate runs are unaffected.
- **No distribution** (`None` — adversarial counter-evidence, legacy data) →
  falls back to the hard `support_judgment` vote ⇒ identical to old.

Every existing `test_posterior.py` assertion (which uses one-hot / no-distribution
evidence) still passes unchanged. New `test_posterior_confidence_weighting.py`
proves both the limiting case and the softening on graded input.

## Scope — what Tier 1 does and does NOT touch

- **Does** soft-weight the counting posterior: the always-reported diagnostic
  (`counting_posterior`, `supporting_count`, `contradicting_count`) and the
  headline in the **counting-fallback** branches (rule-aware UNION; no integrated
  claims). The cycle-cap / retrieval-failed penalties stack on top as before.
- **Does NOT** touch the dominant **integration-driven** headline. That path uses
  per-claim `integrated_confidence` from the IBE chain — a *different*,
  already-model-reported signal. Threading evidence-judgment confidence into IBE
  would risk double-counting (the IBE chain already saw the evidence) and is a
  deeper, separate decision — deliberately not done here.
- **Does NOT** touch promotion gates (`gates.py` counts discrete supporting
  *sources*, not the weighted posterior) — bounded blast radius.

## Validation status

- Deterministic: full suite green (2273 passed), pyright 0 errors in src, ruff clean.
- Backward-compat proven by construction + tests.
- **Open**: the *quality uplift* of softening on real graded data (vs. the proven
  safety) should be measured with a representative epistemic benchmark run
  (live model over a SciFact-style set) before relying on the softened counting
  for headline fallback decisions. Safe to ship as-is (it can only equal or
  refine the old behaviour); the benchmark tells us how much it helps.
