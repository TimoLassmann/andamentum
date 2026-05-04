# Confidence aggregation — honest treatment of cycle-capped claims

**Goal.** Stop discarding the signal a cycle-capped claim already carries
(integrated assessment, evidence count) when computing the headline
posterior. Surface "verdict produced under cycle-capped inquiry" as
provenance in the explanation, not by zeroing the number to 0.5.

**Trigger.** SciFact 5-case smoke run. Cases 54 and 957 were reported
as failures with posterior=0.500. The DBs say:

* Case 54 (gold=CON): claim reached SUPPORTED with
  `integrated_assessment="contradicts"` at conf=0.727; adversarial
  balance 0.365; evidence 11 contra / 3 sup; synthesis writer wrote
  *"No. Evidence ... contradicts a pro-fibrotic effect."* — and then
  `compute_posterior` returned 0.500 because `cycle_capped=True`.
* Case 957 (gold=SUP): cycle-capped at HYPOTHESIS, no integrated
  assessment, but evidence 23 supports / 5 contradicts. Same outcome:
  signal exists, posterior throws it away.

The synthesis text gets it right. The numerical posterior gets it
wrong — and the SciFact harness reads the number, not the prose.

---

## The principled distinction

The current design conflates two epistemic situations:

* **Verdict instability** (Peirce — really suspend judgment): "I keep
  changing my mind about whether X is true."
* **Inquiry inexhaustibility** (just deeper questioning): "I keep
  finding more questions to ask, but my answer to the original
  question hasn't moved."

Cycle 54 is pure inquiry inexhaustibility — IBE certified the verdict
before the cap fired; what scrutiny kept asking about was scope and
mechanism, not whether the headline holds.

Per Peirce, no inquiry truly terminates — verdicts are always
provisional. "Provisional" ≠ "0.5". It means "best current answer,
welcomes refutation."

---

## Three-way rule (replaces the unconditional "all-capped → 0.5")

For claims with `cycle_capped=True`:

1. **Has `integrated_assessment`** → use it. The IBE chain certified the
   verdict before the cap fired; the cap is about residual questioning,
   not verdict instability. Apply a small confidence penalty (the
   inquiry didn't converge) but include in aggregation. **Cases like 54
   land here.**

2. **No `integrated_assessment` but evidence count is one-sided** →
   use the counting fallback with reduced weight. The claim never
   reached IBE but the evidence pool has clear directional signal.
   **Cases like 957 land here.**

3. **No `integrated_assessment` and evidence count is roughly
   balanced** → THEN 0.5 is the honest answer (`terminal_state=
   "oscillation_detected"`). This is the case the original design
   was trying to catch; the rewrite preserves it.

The threshold for "one-sided" in case 2 is the existing log-odds
calculation; we just stop nulling it because of the cap.

---

## Scan: where this pattern exists in the codebase

The scan distinguished **inquiry-layer** filtering (correct — don't
run more LLM work on a non-converging claim) from **output-layer**
filtering (the bug — signal already acquired is being thrown away).

### Same-bug sites — fix in this plan

| Site | What it does | Fix |
|---|---|---|
| `confidence.py:240-275` (`compute_posterior`) | All-capped → posterior=0.5; partial-cap → drop capped from `active_claims` before aggregation. | Replace with the three-way rule. Capped claims with `integrated_assessment` contribute their verdict; capped claims without contribute their evidence count; only the genuinely-balanced-and-no-verdict case yields `oscillation_detected`. |
| `combination.py:122-129` (`combine_claim_verdicts`) | Multi-claim aggregation skips capped claims (`continue`), feeds None into `claim_posteriors`. | Same three-way rule, applied per-claim before the AND/OR/WEIGHTED_AND combiner. |

### Defensible sites — not touched

| Site | Why it stays |
|---|---|
| `nodes.py: PromoteToSupported, RunVerification, IBE chain entries` | Inquiry-layer. Running IBE / verification on a capped claim feeds a verdict-fabrication loop. The cap is exactly the right place to stop more LLM work. |
| `nodes.py: CheckSynthesisDemand._maybe_loop_back` | Inquiry-layer. Don't loop a CSD demand back into Scrutinize on a capped claim — that's the load-bearing safety. |
| `stages.py: _check_*` | Stage-exit invariants are progress checks; capped claims correctly excluded from "active work". |
| `synthesis.py: 108-126` | Already correct — filters by `abandoned` only, NOT by `cycle_capped`. The writer sees capped claims and their verdicts (proven by case 54's correct prose). |
| `gates.py` invalidated-evidence filters | Different flag — `invalidated` means TMS judged the content untrustworthy. Discarding it is correct. |

The scan turned up no third site with the same bug.

---

## Phases

Each phase is independently committable, with a local test gate.

### Phase 1 — Three-way rule for compute_posterior

- [ ] Replace the all-capped short-circuit (lines 240-266) with the
      three-way rule. Capped+IA → contribute the verdict (with a
      configurable `cycle_cap_confidence_penalty`, default 0.7);
      capped-no-IA → contribute via the counting path; capped+balanced
      → `oscillation_detected` as before.
- [ ] Drop the partial-cap filter (lines 271-275). Capped claims now
      flow through aggregation under the three-way rule.
- [ ] Update the `explanation` string to say "verdict produced under
      cycle-capped inquiry (cap=N rounds)" so provenance is loud.
- [ ] Tests: 6-8 unit tests covering each branch; assertion against the
      5 SciFact DBs (offline, no LLM) showing 54 → ~0.10 and 957 → ~0.95.
- [ ] **Acceptance:** new tests pass; existing posterior tests pass
      (any that pinned the old 0.5 behaviour need updating, with the
      change documented).

### Phase 2 — Same rule for combine_claim_verdicts

- [ ] Apply the three-way rule per-claim before the AND/OR/WEIGHTED_AND
      combiner. Capped claims now contribute a posterior to
      `claim_posteriors` instead of None (when they have signal).
- [ ] Keep `n_capped` count in the diagnostic for traceability.
- [ ] Tests: extend the existing combine tests with cycle_capped
      claims that have integrated_assessment.
- [ ] **Acceptance:** new tests pass; existing combine tests still
      pass.

### Phase 3 — Closeout

- [ ] CLAUDE.md: update the Snapshot/Artefact section + the
      `compute_posterior` description with the three-way rule.
- [ ] Memory: write a feedback memory naming the principle —
      *"capped at the inquiry layer ≠ no signal at the output layer."*
- [ ] Re-read all 5 SciFact DBs through the patched `compute_posterior`
      offline; record the new posteriors next to the old ones.

---

## Risks

| Risk | Mitigation |
|---|---|
| Existing tests pinned the old 0.5 behaviour | Phase 1 explicitly sweeps for them. Most likely: tests of `oscillation_detected` need to be re-targeted to the genuinely-balanced case (still possible to construct). |
| Confidence penalty value is a tuning knob | Start at 0.7 (per-IBE confidence × 0.7 when capped). Document it as a constant; don't make it an API parameter. |
| Multi-claim cases where all claims are capped with the SAME verdict shouldn't average to 0.5 | The three-way rule handles this naturally — each capped claim contributes its own verdict, the combiner aggregates them. Test pinned. |
| External code reading `terminal_state="oscillation_detected"` | Still emitted, but only in the genuine case (3). Strictly tighter, no breaking change for callers checking for this state. |

---

## Acceptance criteria for the whole effort

1. The 5-case SciFact smoke run, re-evaluated through the patched
   `compute_posterior`:
   * Case 54: posterior ≤ 0.20 (was 0.500). Correct CON.
   * Case 957: posterior ≥ 0.80 (was 0.500). Correct SUP.
   * Cases 141, 439, 847: unchanged.
2. All existing tests pass; new tests cover the three branches.
3. The `explanation` string on a cycle-capped result names the cap
   rounds and the verdict provenance.
4. `terminal_state="oscillation_detected"` still fires, but only when
   no claim has either an integrated assessment or a one-sided
   evidence pool.
5. Pyright + ruff clean.

---

## What this plan does NOT cover

- The cap value itself (3 rounds) — separate question, not part of
  this fix.
- Tuning the confidence penalty empirically — keep it at the
  conservative default; revisit after a larger benchmark.
- Speed / pre-judge filter / provider rank diversity — separate
  runway items.

---

*Written 2026-05-04 to fix the cycle_capped → 0.5 over-aggression
surfaced by the SciFact 5-case smoke run.*
