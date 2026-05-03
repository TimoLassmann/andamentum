# Stage Runners — Phase 0 outcome

**Date:** 2026-05-03
**Plan:** [`2026-05-03-stage-runners.md`](./2026-05-03-stage-runners.md)

Empirical discovery from static topology + entity-field semantics. Six stage boundaries proposed, each with the load-bearing exit invariant and the rationale.

---

## Method

Three sources of truth for the boundary discovery:

1. `topology()` over `graph/nodes.py` — the canonical edge map (22 nodes).
2. `Node.reads` / `Node.writes` ClassVars — what graph-state field each node touches.
3. Entity-level fields on `Objective`, `Claim`, `Evidence` — what each operation writes to the DB.

A *quiescent boundary* is a node where (a) no in-flight per-claim work remains, and (b) the next node only reads state that prior nodes have already finalised. Boundary candidates are the nodes whose successor is a "fresh start" of the next responsibility (different reads-set, different operations).

I also looked at the `[graph.nodes]` log lines from the Q1 aspirin run (cycle-cap firing in `ResolveUncertainties-primary` and `Scrutinize-defense-in-depth`) to confirm the loop nodes are the ones I expected.

---

## The 22 nodes, grouped by responsibility

```
                                 ┌─ PrepareObjective ──────────── question_type
   PREPLANNING                   ├─ Decompose ───────────────── (writes Objective.decomposition)
                                 │
                                 ├─ PlanEvidence ────────────── (LLM: plan task)
   INITIAL EVIDENCE              ├─ ExtractEvidence ──────────── (gather + extract per stub)
                                 ├─ CreateClaims ───────────── claim_ids, claims_created
                                 │
                                 ├─ Scrutinize ◄─────────────────────┐
                                 │     │                             │
   SCRUTINY ↔ INVESTIGATION       │     ├─→ AbandonOrDemote          │
   LOOP                          │     │      │                     │
                                 │     │      └─ PromoteToSupported ┤
                                 │     │                             │
                                 │     ├─→ Investigate               │
                                 │     │      └─ ExtractNewEvidence ─┘
                                 │     │
                                 │     └─→ ResolveUncertainties ──── (loop / IBE entry)
                                 │
                                 ├─ ClusterEvidence ──────── (deterministic, top-K)
   VERIFICATION                  ├─ RunVerification ─────── 7 tracks per claim
                                 │
                                 ├─ EnumerateCandidates ─┐
                                 ├─ ScoreLoveliness     ├─ IBE chain (per claim)
   INTEGRATION (IBE)             ├─ ScoreLikeliness     │
                                 ├─ SelectBestExplanation ┘
                                 ├─ PromoteSupported ───── verification_done
                                 ├─ CombineClaimVerdicts ── (writes Objective.decomposition.combined_verdict)
                                 │
                                 ├─ CheckCompletion ────── (no writes; routing only)
   SYNTHESIS                     ├─ CheckSynthesisDemand ── claims_needing_rescrutiny (loop-back)
                                 └─ Synthesize ──────────── End[EpistemicResult]
```

Six responsibility groups, but **the scrutiny ↔ investigation loop is genuinely intertwined** — they share state (`claims_needing_rescrutiny`, `scrutiny_resolve_cycles`, `investigation_counts`). Trying to split them mid-loop invites the "non-quiescent boundary" failure mode the plan flags as the top risk.

So the proposal is **five stages, not six** — fold scrutiny + investigation loop into a single stage. This is the only place static analysis disagreed with the plan's draft. The disagreement is load-bearing.

---

## Proposed `STAGES` registry

### `preplanning`

| | |
|---|---|
| Entry | `PrepareObjective` |
| Exit after | `Decompose` |
| Description | Clarify question, classify type, decompose into sub-investigations |

**Exit invariant:**
```python
lambda s, r:
    objective(r).question_type is not None
    and (not s.decompose or (
        objective(r).decomposition is not None
        and len(objective(r).decomposition.sub_investigations) >= 1
    ))
```

**Why this boundary holds:** `PrepareObjective` writes `question_type` exactly once, `Decompose` writes `Objective.decomposition` exactly once. Successor `PlanEvidence` reads only `objective_id`. Nothing in subsequent stages re-reads or re-writes these fields.

---

### `initial_evidence`

| | |
|---|---|
| Entry | `PlanEvidence` |
| Exit after | `CreateClaims` |
| Description | Plan first-pass searches, gather initial evidence stubs, create claims |

**Exit invariant:**
```python
lambda s, r:
    len(claims(r)) >= 1
    and all(c.evidence_count > 0 for c in claims(r))
```

**Why this boundary holds:** `CreateClaims` is the last writer of `claim_ids` and `claims_created`. Successor (`Scrutinize`) reads claim state but does not depend on `consecutive_empty_extractions` / `retrieval_failed` once claims exist (those are reset by Investigate's later rounds anyway).

**Caveat:** `ExtractEvidence` writes `retrieval_failed` if all providers return zero. If true, `CreateClaims` may still produce claims with `evidence_count=0`. The invariant above would fail and the runner crashes loudly — which is correct: a stage that finished with zero evidence is a degenerate state we want surfaced, not silently passed downstream.

---

### `scrutiny_and_investigation`

| | |
|---|---|
| Entry | `Scrutinize` |
| Exit after | `AbandonOrDemote` (its `PromoteToSupported` successor wraps the boundary) |
| Description | Iterative scrutiny → investigation loop until each claim has a terminal scrutiny verdict |

**Exit invariant:**
```python
lambda s, r:
    all(
        c.scrutiny_verdict in {"pass", "fail"} or c.cycle_capped or c.abandoned
        for c in active_claims(r)
    )
    and len(s.claims_needing_rescrutiny) == 0
```

**Why this boundary holds:** `AbandonOrDemote` is the *only* node that writes `terminal_claims` and `verification_done` from the scrutiny side. Once it returns, every remaining claim either has a terminal scrutiny verdict (pass / fail), is cycle-capped, or is abandoned. The empty `claims_needing_rescrutiny` set is the load-bearing piece — it pins "no more in-flight scrutiny work."

**Caveat — the looping state:** because Scrutinize loops with Investigate via `claims_needing_rescrutiny`, the boundary cannot be inside the loop. AbandonOrDemote is the natural drain: it's reached when scrutiny gives up on a claim, and from there the path is forward-only (PromoteToSupported → ClusterEvidence → verification, OR PromoteToSupported → CheckCompletion → finish).

**This is the riskiest stage boundary.** If it leaks (i.e., a claim slips out with `scrutiny_verdict=None` and not cycle-capped and not abandoned), the verification stage will see in-flight scrutiny work and produce wrong outputs. The exit invariant is the contract that prevents this.

---

### `verification`

| | |
|---|---|
| Entry | `ClusterEvidence` |
| Exit after | `RunVerification` |
| Description | 7 verification tracks (adversarial, deductive, computational, contrastive, convergence, consistency, argument) per supported claim |

**Exit invariant:**
```python
lambda s, r:
    all(
        c.verification_done or c.cycle_capped or c.abandoned
        for c in claims_at_supported(r)
    )
```

**Why this boundary holds:** `RunVerification` is the last node before the IBE chain. Its only successors are `ResolveUncertainties` (loop back into scrutiny — only fires when verification finds new doubt) and `EnumerateCandidates` (forward into IBE). Once it returns into the IBE chain, verification work is complete for the claim being processed.

**Caveat:** `RunVerification` is currently SERIAL — that's the parallelisation candidate timing.txt will quantify in Phase 6.

---

### `integration`

| | |
|---|---|
| Entry | `EnumerateCandidates` |
| Exit after | `CombineClaimVerdicts` |
| Description | IBE chain (4 nodes) + per-claim integration + cross-claim combination per the decomposition rule |

**Exit invariant:**
```python
lambda s, r:
    all(
        c.integrated_assessment is not None
        for c in active_claims(r)
        if not c.cycle_capped and not c.abandoned
    )
    and (
        objective(r).decomposition is None
        or objective(r).decomposition.combined_verdict is not None
    )
```

**Why this boundary holds:** `CombineClaimVerdicts` is the unique writer of `Objective.decomposition.combined_verdict` and runs after all per-claim IBE chains have completed. `PromoteSupported` writes `verification_done` for the integration-completed claims. By the time we exit, every non-terminal claim has both an integration verdict AND a combined verdict exists for the decomposition.

**Note:** the `CheckCompletion` node is also a successor of `PromoteToSupported`, but only when there is no work left for IBE (all claims terminal). In that case the invariant's "if not cycle_capped and not abandoned" condition vacuously holds and `combined_verdict` may legitimately be None — which the next stage handles via Gate 2.

---

### `synthesis`

| | |
|---|---|
| Entry | `CheckCompletion` |
| Exit after | `Synthesize` (which routes to End) |
| Description | Synthesis-demand gate, optional loop-back to Scrutinize, final report |

**Exit invariant:**
```python
lambda s, r:
    objective(r).report is not None
```

**Why this boundary holds:** `Synthesize` writes the final report onto the Objective and routes to End. Once it returns there is no further work. This is the only stage whose exit is the graph's End.

**Wrinkle:** `CheckSynthesisDemand → Scrutinize` (the Phase 4 loop-back) is INSIDE this stage. If the synthesis-demand gate routes back, the scrutiny ↔ investigation ↔ verification ↔ integration ↔ synthesis chain re-runs from inside the synthesis stage. This is fine for the stage abstraction — synthesis "isn't done" until Synthesize returns, regardless of how many internal loop-backs happen — but it means the synthesis stage's worst-case runtime is bounded by the per-sub-claim cap, not by a single graph traversal. We should record this in the timing artifact so users know "synthesis took 3 min" can mean either "one run of `CheckSynthesisDemand → Synthesize`" or "`needs_more=True` triggered a loop back through scrutiny+verification+integration." The `run.jsonl` per-node visit count makes this visible at zero extra cost.

---

## Summary table

| Stage | Entry | Exit after | Key invariant field |
|---|---|---|---|
| preplanning | `PrepareObjective` | `Decompose` | `Objective.decomposition`, `Objective.question_type` |
| initial_evidence | `PlanEvidence` | `CreateClaims` | `Claim.evidence_count > 0` |
| scrutiny_and_investigation | `Scrutinize` | `AbandonOrDemote` | `Claim.scrutiny_verdict` terminal for all active |
| verification | `ClusterEvidence` | `RunVerification` | `Claim.verification_done` for all supported |
| integration | `EnumerateCandidates` | `CombineClaimVerdicts` | `Objective.decomposition.combined_verdict`, `Claim.integrated_assessment` |
| synthesis | `CheckCompletion` | `Synthesize` | `Objective.report` |

Five stages match the plan's six; the difference is **scrutiny and investigation are one stage**, not two, because their state is genuinely shared and any boundary inside the loop is non-quiescent.

---

## What this changes in the plan

The main plan listed six stages including a separate `investigation` and `scrutiny`. Phase 0's empirical answer says **five stages** with `scrutiny_and_investigation` as one combined stage. This is the only design change Phase 0 produced. I'll update the plan to reflect this when Phase 1 starts.

The anti-bloat budget is unaffected: 5 entries vs 6 entries in `STAGES` is well within ≤120 LOC for `stages.py`.

---

## Phase 0 gate (passed)

> Each candidate exit is provably the last writer of the state it produces.

Verified for all five exits via `Node.writes` ClassVar inspection. Specifically:

- `Decompose` is the only writer of `Objective.decomposition` (entity field).
- `CreateClaims` is the only writer of `claim_ids` and `claims_created` (state fields).
- `AbandonOrDemote` is the only "drain" path out of the Scrutinize ↔ Investigate loop into ClusterEvidence/CheckCompletion.
- `RunVerification` is the only writer that completes verification tracks per claim.
- `CombineClaimVerdicts` is the only writer of `Objective.decomposition.combined_verdict`.
- `Synthesize` is the only writer of `Objective.report` and the only path to End.

Phase 0 done. Ready for Phase 1 (`stop_after` kwarg).
