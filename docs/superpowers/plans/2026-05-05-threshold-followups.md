# Threshold harmonization — three follow-up commits

> Companion plan to `2026-05-05-threshold-harmonization.md` (sibling
> commit `f65845c`). Closes the three remaining gaps that a careful
> reviewer would flag.

**Goal.** Finish the threshold-harmonization story so a manuscript
section can claim, without footnotes, that every decision-relevant
threshold lives in `epistemic/thresholds.py` with named theoretical
basis.

**Discipline that makes this safe.** Each of the three commits is
designed as a **pure refactor** — values stay identical to current
production. The point is consolidating where they live and what they're
called, not what they're set to. Tuning, if any, is a separate
decision after the threshold story is clean.

---

## Pre-flight: what the audit found

Before I drafted this plan I traced every potentially-risky downstream
path. The findings:

### `AdversarialEvidence.verdict` (the string from `determine_verdict`) is **read-only for display**

Three call sites in production code:
- `cli_handlers.py:1522` — prints the verdict string in a CLI panel.
- `operations/verification.py:337` — embeds the string in a log message.
- `operations/synthesis.py:651` — adds the string to a report description.

**No graph node, gate, or routing decision branches on the verdict
string.** Decisions branch on the *balance value* directly, which
already goes through `ADVERSARIAL_REFUTED_THRESHOLD` and
`ADVERSARIAL_SURVIVED_THRESHOLD` in `gates.py`, `nodes.py`, etc. (the
sibling commit). So changing the bands in `adversarial_balance.py`
changes only narrative text, not claim trajectories.

### `convergence_verdict` IS read for routing

Two consumer sites:
- `nodes.py:1539` — fast-path-to-IBE check: requires at least one
  active SUPPORTED claim to have `convergence_verdict == "CONVERGENT"`,
  AND all of them to have a terminal verdict. When both conditions
  hold, the graph skips `ResolveUncertainties` and goes straight to
  IBE.
- `nodes.py:1493` — comment about SKIP'd convergence (no decision).

The verdict comes from `convergence_detector._determine_verdict`,
which produces "CONVERGENT" only when `strength >= 0.7`. If we
lower this threshold, more claims get "CONVERGENT" → the fast-path
fires more → fewer `ResolveUncertainties` calls. If we raise it,
the opposite. Either direction is bounded by the cycle caps.

**Implication for this plan.** Commit 3 will keep `strength >= 0.7`
at literally 0.7 — wire it to a named constant `CONVERGENCE_STRONG_THRESHOLD = 0.7`,
do not change the value. Same for the other convergence thresholds.

---

## Cycling concerns — the user's specific worry

The user's question: *"will it cause breaking changes? downstream
consequences that bring back cycling behaviours?"*

The graph has five bounded loops. None of these commits touches any
of them:

| Loop | Cap | Touched by these commits? |
|---|---|---|
| Investigation cycle | `PEIRCE_CYCLE_CAP=3` | No |
| Scrutiny↔Resolve | `PEIRCE_CYCLE_CAP=3` | No |
| ResolveUncertainties self-loop | `PEIRCE_CYCLE_CAP=3` | No |
| CheckSynthesisDemand loop-back | `cycle_capped` + `PEIRCE_CYCLE_CAP` | No |
| Writer-validator | `MAX_VALIDATION_ROUNDS=3` | No |

Per-claim cap enforcement (the load-bearing safety) is unchanged.
**The graph's cycling behaviour is provably unaffected by all three
commits.**

The only at-risk loop-adjacent behaviour is the IBE fast-path firing
rate (Commit 3). Worst case if values change: more `ResolveUncertainties`
runs OR more fast-path skips. Both terminate within the existing
caps. No new cycling can be introduced because the caps still apply.

---

## Commit 1 — wire `_verdict_label` to `POSTERIOR_DIRECTIONAL_BREAKPOINT`

**Risk: zero.** The constant value is already 0.66 (matches the
current inline literal). One-line refactor.

### Diff

```python
# graph/combination.py
+ from ..thresholds import POSTERIOR_DIRECTIONAL_BREAKPOINT

def _verdict_label(p: float) -> str:
-    if p > 0.66:
+    if p > POSTERIOR_DIRECTIONAL_BREAKPOINT:
         return "supports"
-    if p < 0.34:
+    if p < 1.0 - POSTERIOR_DIRECTIONAL_BREAKPOINT:
         return "contradicts"
     return "insufficient"
```

### What downstream changes
Nothing. Bare 0.66 ≡ `POSTERIOR_DIRECTIONAL_BREAKPOINT` (= 0.66).

### Verification
- All 989 tests must still pass (any test pinning the value 0.66 still
  passes because the value is 0.66).
- Drift-detection on `combination.py` may need a hash update.
- SciFact offline check should be byte-identical.

### Acceptance
- Bare `0.66` and `0.34` removed from `combination._verdict_label`.
- Test suite green. SciFact verified identical.

---

## Commit 2 — reconcile `adversarial_balance.py` 5-band interpretation

**Risk: low.** The functions involved (`interpret_balance`,
`determine_verdict`) feed only display/narrative paths. No
routing, no gates, no claim trajectories.

### Two options

**Option A (chosen — safest for manuscript):** Rewrite the bands to
align with the principled three-band system, but **derive** the
intermediate bands from the canonical thresholds rather than reinvent
them. Output strings change slightly (some balance values get a
different label) but no decisions change.

**Option B (rejected):** Keep the bands as-is and rename the function
to `interpret_balance_for_narrative` to make clear it's not a
decision threshold. Cosmetic only; doesn't actually fix the
"two parallel banding systems" problem the manuscript-defensibility
audit flagged.

### Option A diff sketch

```python
# adversarial_balance.py
+ from .thresholds import (
+     ADVERSARIAL_REFUTED_THRESHOLD,
+     ADVERSARIAL_SURVIVED_THRESHOLD,
+     ADVERSARIAL_SUSPICIOUS_THRESHOLD,
+ )

def interpret_balance(balance: float) -> str:
    """Three principled bands plus a SUSPICIOUS extreme.

    Derived from the canonical thresholds in epistemic.thresholds.
    """
    if balance > ADVERSARIAL_SUSPICIOUS_THRESHOLD:
        return "Suspiciously uncontested..."
    if balance >= ADVERSARIAL_SURVIVED_THRESHOLD:
        return "Survived adversarial challenge..."
    if balance < ADVERSARIAL_REFUTED_THRESHOLD:
        return "Refuted by adversarial evidence..."
    return "Contested — significant counterevidence..."
```

`determine_verdict` similarly. The verdict labels collapse from
five (SUPPORTED/CONTESTED/CHALLENGED/REFUTED + suspicious extreme)
to a smaller set aligned with the principled bands.

### What downstream changes
- **Reports and CLI panels**: text describing adversarial outcomes
  shifts slightly. Balance = 0.5 still says "contested". Balance =
  0.35 used to say "weakly supported" / "challenged"; now says
  "contested" (above REFUTED 0.3, below SURVIVED 0.7). Balance =
  0.25 used to say "challenged"; now says "refuted" (matches what
  the gate already does at 0.25).
- **No graph routing**: confirmed by audit — no decision branches on
  the verdict string.
- **No tests should break** structurally; tests pinning specific
  text strings will need updates.

### Risks I considered and ruled out

| Risk | Mitigation |
|---|---|
| `determine_verdict`'s `(verdict, recommendation, confidence)` tuple is used somewhere I missed | Searched all `.py` files; only consumers are the three display sites listed above. Triple-check by adding a test that asserts no node/gate reads `AdversarialEvidence.verdict` for a decision. |
| Tests pin specific verdict strings (e.g. "SUPPORTED" at balance 0.75) | Surface them via the test run; update test assertions to the new strings. No change in behaviour, only assertion-text. |

### Acceptance
- 5-band logic in `interpret_balance` and `determine_verdict`
  rewritten to use the canonical thresholds.
- 989+ tests still pass; any pinned-text assertions updated.
- SciFact offline check: posteriors byte-identical (no decision
  changes).
- Manuscript story: one banding system, not two.

---

## Commit 3 — canonical convergence thresholds

**Risk: low IF values are kept identical.** Convergence verdict
feeds the IBE fast-path; preserving the existing 0.7 strength
threshold preserves the existing fast-path firing rate exactly.

### What gets named

In `epistemic/thresholds.py`, add a new section:

```python
# ── Convergence (Reichenbach common-cause / Mill's methods) ────────
#
# Multi-domain convergence: a claim that holds across genuinely
# independent evidence pools is more credible than the same claim
# holding in one pool. ``convergence_detector._determine_verdict``
# maps a strength score to one of {NO_EVIDENCE, SINGLE_DOMAIN,
# PARTIAL, CONVERGENT}. The CONVERGENT label is load-bearing: it
# triggers the fast-path-to-IBE in graph/nodes.py:RunVerification.

CONVERGENCE_STRONG_THRESHOLD: float = 0.7
"""Strength score at or above which the convergence verdict is
CONVERGENT (load-bearing for the IBE fast-path). Read by:
convergence_detector._determine_verdict."""

CONVERGENCE_INTRA_DIVERSITY_THRESHOLD: float = 0.5
"""Minimum intra-cluster diversity ratio for the diversity check
in independence assessment. Read by:
convergence_detector._compute_independence_checks."""

CONVERGENCE_INTER_DOMAIN_DISTANCE_LOW: float = 0.3
"""Below this average inter-domain distance, the independence
score is weakened (clusters are too similar to count as truly
independent domains). Read by:
convergence_detector._compute_independence_score."""
```

### Diff sketch in `convergence_detector.py`

```python
+ from .thresholds import (
+     CONVERGENCE_STRONG_THRESHOLD,
+     CONVERGENCE_INTRA_DIVERSITY_THRESHOLD,
+     CONVERGENCE_INTER_DOMAIN_DISTANCE_LOW,
+ )

def _determine_verdict(...):
    ...
-    if strength >= 0.7:
+    if strength >= CONVERGENCE_STRONG_THRESHOLD:
         return "CONVERGENT"
    return "PARTIAL"
```

Plus two more migrations at the diversity and inter-domain-distance
sites.

### What downstream changes
**Nothing observably**, because:
- `CONVERGENCE_STRONG_THRESHOLD = 0.7` matches the inline `0.7` exactly.
- `CONVERGENCE_INTRA_DIVERSITY_THRESHOLD = 0.5` matches `0.5` exactly.
- `CONVERGENCE_INTER_DOMAIN_DISTANCE_LOW = 0.3` matches `0.3` exactly.

The IBE fast-path firing rate is therefore byte-identical.

### Risks considered

| Risk | Mitigation |
|---|---|
| Convergence verdict changes for some claim → fast-path firing changes → ResolveUncertainties is skipped/added → claim trajectory differs | Keep all three values exactly equal to current inline literals. Verify on SciFact: posteriors and convergence_verdict counts must be identical. |
| Other inline numerics in `convergence_detector.py` that I haven't named (blend coefficients `0.4 / 0.4 / 0.2` etc.) | These are scoring weights, not decision thresholds. Leave them inline; document why in the file (they sum to 1.0 for a designed weighted average; not Reichenbach-grounded). |
| Domain distance bands (`0.2 / 0.35 / 0.5 / 0.7`) in `domain_distance.py` are still scattered | Out of scope for this commit. Add a TODO note in `thresholds.py` if a future pass migrates them. |

### Acceptance
- Three new constants in `thresholds.py` with Reichenbach motivation.
- Three sites in `convergence_detector.py` migrated.
- 989+ tests still pass.
- SciFact offline check: posteriors AND `convergence_verdict` per claim
  byte-identical to pre-commit baseline.

---

## Verification protocol — applied to each commit

1. **Pre-commit snapshot**: re-run `compute_posterior` offline on the
   v15 + v16 SciFact DBs. Record posterior, mode, terminal_state.
2. **Per-commit changes**: only the migrations described in that
   commit. No coincidental tweaks.
3. **Post-commit snapshot**: re-run the same offline check. Diff
   against pre-commit. Expected diff: empty.
4. **Run full test suite**: `uv run pytest src/andamentum/epistemic/`.
   Expected: 989 pass (Commit 1, Commit 3) or 989 + maybe 1-2 text-
   assertion updates (Commit 2).
5. **Run ruff check + format**: clean.
6. **Run pyright on touched files**: 0 new errors.

If any of (1) → (3) shows a non-empty diff, do not commit. Investigate.

---

## Risk register — across the three commits

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| New cycling behaviour | None | — | Graph cap structure unchanged |
| Claim trajectory changes through graph | None for Commits 1+3; very low for Commit 2 | — | Decisions branch on values, not strings; values unchanged |
| IBE fast-path firing rate shifts | None if Commit 3 keeps values identical | High if it shifts | Strict no-value-change discipline; verify via SciFact diff |
| Test text-assertion failures | Low for Commits 1+3; medium for Commit 2 | Low | Update assertions; no semantic change |
| Drift-detection hash on a load-bearing function | Low | Low | Standard procedure: update the hash and document why |

---

## Order of operations

Ship in this order, with the SciFact diff verification between each:

1. **Commit 1** (verdict label) — five-line change. Validates the
   workflow; demonstrates the SciFact offline check.
2. **Commit 3** (convergence thresholds) — three constants, three
   migration sites. Strict no-value-change.
3. **Commit 2** (adversarial bands) — last because it's the only
   one that intentionally changes output text. Save for after the
   workflow is proven on the easier two.

If 1 and 3 produce a clean diff, that's strong evidence the workflow
is sound. Commit 2 then ships with confidence in the verification
protocol.

---

## What this plan does NOT do

- **Does not move `MAX_VALIDATION_ROUNDS`, `_EMPTY_EXTRACTION_THRESHOLD`,
  `MAX_ADVERSARIAL_TEMPLATES`, `LLM_PANEL_CAP`** etc. into
  `thresholds.py`. These are operational caps, not theoretical
  commitments. A "Operational caps" comment block in `thresholds.py`
  pointing at where they live would be a small companion change but
  isn't required for manuscript-defensibility.
- **Does not migrate `DEDUP_SIMILARITY_THRESHOLD`** to thresholds.py.
  It's a similarity-clustering parameter, not a Popper/Lakatos/Peirce
  commitment. The audit can mention it in the manuscript section as
  "operational similarity threshold" rather than coercing it into the
  theoretical taxonomy.
- **Does not address quality-score thresholds** (`< 0.10` invalidate,
  `0.05` floor in `operations/evidence.py`). Operational; out of scope.
- **Does not address domain distance bands** (`0.2/0.35/0.5/0.7`).
  Possibly a future pass; flag as TODO.

If any of these become important during or after the three commits,
scope a follow-up plan rather than expanding this one.

---

## Acceptance criteria for the whole effort

After all three commits land:

1. `combination._verdict_label` reads `POSTERIOR_DIRECTIONAL_BREAKPOINT`
   from `thresholds.py`.
2. `adversarial_balance.interpret_balance` and `determine_verdict`
   produce labels derived from the canonical three-band system; no
   parallel 5-band system in the file.
3. `convergence_detector._determine_verdict` and the independence-
   check helpers read named convergence thresholds from
   `thresholds.py`.
4. SciFact v15 + v16 offline posteriors byte-identical to pre-effort
   baseline.
5. 989+ epistemic tests pass.
6. Pyright + ruff clean.
7. The "What's still scattered" findings list from the audit shrinks
   from 6 items to 3 (operational caps, similarity, quality scores —
   all flagged as deliberately-not-in-thresholds.py with documentation
   in the canonical module).

---

*Written 2026-05-05 to scope the three follow-up commits cleanly, with
explicit risk analysis to address the user's "will it break things?"
concern. The discipline is no-value-change; the wins are
manuscript-defensibility on three fronts the audit flagged as still
loose.*
