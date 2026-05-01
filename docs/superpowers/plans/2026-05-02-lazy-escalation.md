# Lazy Escalation — Pull-Based Inquiry

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **CRITICAL:** Each phase has a benchmark gate AND an explicit secondary-effect check. Past sessions broke things by changing one thing without auditing what depended on it; this plan front-loads that audit.

**Goal:** Replace eager-broad-search with demand-driven escalation. Easy questions resolve in round 1 with one provider per sub-claim; hard questions naturally escalate to more providers / more rounds, driven by specific gaps from satisfaction checks. Same philosophical principles (Peirce, Lakatos, Lipton, AGM, Kahneman); architecture that *expresses* them rather than contradicts them.

**Constraint:** Reuse existing pieces. No rewrite. Per-sub-claim budgets stay (no global "give up" cap). LLM judgment for the actual "satisfied?" check; deterministic gates first as cheap pre-filters. Ollama-compatible (one extra meta-LLM per layer-transition is acceptable).

**Tech stack:** Python 3.13, asyncio, pydantic-graph, pydantic, the existing `andamentum.epistemic` package. No new dependencies.

**Non-goals:** Rebuilding the graph topology. Changing the IBE chain. Changing per-evidence judging. Changing TMS. Changing the Decomposition step (`DecomposeQuestionOperation` still produces N sub-claims).

---

## The principle in one sentence

**Each layer does the minimum work to either resolve its question OR produce a specific demand for the next layer.**

The graph orchestrates by routing demands. A layer that emits a `Demand(needs_more=False)` is satisfied; the graph terminates that layer's work. A layer that emits `Demand(needs_more=True, justification="...", target_hint="...")` triggers a downstream pull on its specific gap.

---

## The `Demand` object

A single uniform Pydantic model used at every layer. Three fields, all flat — small Ollama models can fill it reliably:

```python
class Demand(BaseModel):
    needs_more: bool = Field(
        description="False when the layer is satisfied; True when more work is needed."
    )
    justification: str = Field(
        description=(
            "Freeform reason. When needs_more=False, explains what evidence "
            "settled the question. When needs_more=True, names the specific "
            "gap that's still missing."
        )
    )
    target_hint: str = Field(
        default="",
        description=(
            "Optional freeform hint about where to look for the gap "
            "(e.g. 'try a clinical-trial registry', 'mechanistic literature', "
            "'a different population subgroup'). Empty when the generator "
            "can't suggest a target."
        )
    )
```

Lives in `andamentum.epistemic.demand` (new module). Consumed and produced at every layer-transition.

**Why three fields and not two**: `target_hint` saves a downstream interpretation LLM call when the upstream layer can already see what's needed. It's optional — generators that can't suggest leave it blank. The cost of the field is zero; the savings on the downstream interpretation can be substantial.

**Why uniform across layers**: small LLMs prefer one schema everywhere. Observability is much better with a uniform `Demand` chain that can be logged, diffed, and rendered. The *content* differs across layers; the *shape* doesn't need to.

---

## The deterministic-then-LLM pattern

Applied uniformly at every satisfaction check:

1. **Cheap deterministic gates first** (no LLM call):
   - "Did we get any evidence at all?" → if no, demand=more, return immediately.
   - "Did all required tracks complete structurally?" → if no, demand=more.
   - "Did the cycle cap fire?" → if yes, demand=satisfied (with caveat in justification), return.
   - "Are all evidence quality scores below 0.05?" → if yes, demand=more, return.

2. **LLM judgment second** (only when deterministic gates pass):
   - "Given this evidence, am I satisfied that the demand is met?"
   - Returns `Demand(needs_more=...)` directly.

The cheap gates produce the demand object themselves — no LLM needed for the obvious cases. The LLM only fires for the genuinely ambiguous cases, which is where its judgment is most valuable. Most invocations short-circuit at the deterministic gates.

---

## Architecture: layer-by-layer integration points

This is the load-bearing section. Each layer's changes are listed with integration points and **what depends on each integration point** so we can audit secondary effects up front.

### Layer P (Plan): one provider per sub-claim in round 1

**Today:** `PlanTaskOperation` (`operations/preplanning.py:~280`) iterates `sub_investigations × providers` and creates a query stub for each combination. With 4 sub-claims × 6 providers = 24 stubs in round 1.

**Change:** In round 1, formulate queries for **one provider per sub-claim**. Provider selection is an LLM call: "given this sub-claim and these unused providers, which is most likely to give a high-density answer?"

**Integration points:**
- `PlanTaskOperation.execute` — replace the all-providers loop with a per-sub LLM-pick.
- New state field: `providers_used_per_sub: dict[str, set[str]]` on `EpistemicGraphState` (where the key is sub_investigation_id, value is set of provider names already queried).
- Reuses: existing `epistemic_select_provider` agent (already in `agents/`). Currently fires per-(sub, provider) for a YES/NO; we'd repurpose it to rank providers and pick the top one.

**What depends on this:**
- Evidence count in round 1 drops from ~24 stubs to ~4 stubs.
- `MultiSeedClaimOperation`'s per-sub evidence pool is smaller initially.
- Scrutiny on those smaller pools is more likely to emit "needs_resolution" → demand triggers next round.
- Adversarial search runs on per-claim basis — unaffected.

### Layer I (Investigate): consume demand, pick next-best unused provider

**Today:** `InvestigateClaimOperation` (`operations/investigation.py`) creates new query stubs targeting evidence gaps. The provider for each new stub is whatever the agent picks (typically `web_search`).

**Change:** Investigate consumes the upstream `Demand` (from scrutiny). Picks the next-most-promising **unused** provider for this sub-claim (LLM call). Generates queries against that provider, addressing the demand's justification.

**Integration points:**
- `InvestigateClaimOperation.execute` — accept demand as input, branch on `demand.target_hint` if present.
- Reads `state.providers_used_per_sub[sub_id]` to know which providers are unused.
- Writes to it after stubs are created.

**What depends on this:**
- `ExtractEvidenceOperation` — new stubs flow through it normally; no change needed.
- The judge step in `ExtractNewEvidence` — already filters by `depends_on_claim_id`; no change.
- Quality scoring — runs per evidence; per-(sub, provider) escalation doesn't affect it.

### Layer S (Scrutinize): emit demand on `needs_resolution`

**Today:** `ScrutiniseClaimOperation` returns a verdict (`pass`, `needs_resolution`, `fail`). The verdict is consumed by the graph node `Scrutinize` which routes accordingly.

**Change:** When verdict is `needs_resolution`, scrutiny ALSO emits a `Demand` describing what's missing. The graph routes the demand to `Investigate` (which already happens) but `Investigate` now consumes the demand explicitly.

**Integration points:**
- `ScrutiniseClaimOperation` — return type extended to include optional `Demand`.
- The scrutiny agent's prompt — slight tweak to also produce justification + target_hint when verdict is `needs_resolution`.
- The `epistemic_scrutinise_claim` agent's `output_model` — gains optional `demand_justification` and `demand_target_hint` fields.

**What depends on this:**
- The existing scrutiny output is consumed by graph nodes via field reads. Adding new optional fields is safe — no consumer breaks.
- The scrutiny operation's drift-detection checksum (`test_drift_detection.py`) will need an update.

### Layer V (RunVerification): keep adversarial mandatory, others demand-driven

**Today:** `RunVerification` runs ALL configured tracks for every SUPPORTED claim.

**Change:** Adversarial search stays mandatory (preserves Lakatos: every SUPPORTED claim survives a refutation attempt). The other tracks (convergence, deductive, computational, argument, contrastive, consistency) become demand-driven.

A satisfaction check after adversarial:
- Did adversarial succeed (claim survived)? → demand-check from upstream synthesis layer.
- If upstream demand is satisfied → skip remaining tracks for this claim.
- If upstream demand is NOT satisfied → run the next track that addresses the gap.

**Integration points:**
- `RunVerification.run` — split into mandatory adversarial phase + demand-driven other-tracks phase.
- Deterministic gate first: "did all tracks complete?" If yes, no LLM check needed; satisfied.
- Each non-mandatory track gets a satisfaction check; LLM-asks "would this track address the current demand?"

**What depends on this:**
- The TMS sweep at the end of `RunVerification` — runs once after all tracks; structural ordering preserved.
- Cycle cap (`SCRUTINY_RESOLVE_CYCLE_CAP`) — counts re-scrutiny passes. Lazy verification doesn't affect this counter.

**This is the highest-risk layer** (Phase 2b's parallelization broke here). Two specific risks:

1. **Skipping convergence track** could cost the convergence-driven termination at line 1471-1480 of `nodes.py` ("if all_terminal and any_positive: return EnumerateCandidates()"). If convergence is demand-driven and doesn't fire, that fast-path doesn't engage either, and IBE doesn't enter from that route.

   **Mitigation:** keep convergence track as mandatory too (it's cheap and gives a fast-path to IBE).

2. **Skipping argument-analysis** could leave claim's `analyzed_arguments` field empty, which downstream synthesis might depend on.

   **Mitigation:** audit downstream readers of each track's output before demand-driving it. Some tracks may need to stay mandatory for downstream invariants.

**Conservative starting position**: only `deductive` and `computational` are demand-driven in Phase 1; the others stay mandatory. Expand later if benchmark shows they're worth gating.

### Layer SY (Synthesize): emit demand if not confident; loop back

**Today:** `SynthesizeReportOperation` runs once at the end. Always synthesizes whatever is there.

**Change:** Before synthesis, run a satisfaction check. If unsatisfied, emit `Demand` and route back to `Investigate` (or `PlanEvidence` if a sub-claim's plan needs more providers).

**Integration points:**
- New graph node `CheckSynthesisDemand` between `CombineClaimVerdicts` and `Synthesize`.
- Returns `Synthesize` (satisfied) or routes to a loop-back node.
- Per-sub-claim budget: if the unsatisfied sub-claim has hit `SCRUTINY_RESOLVE_CYCLE_CAP`, accept best-available and synthesize anyway (no infinite loop).

**What depends on this:**
- Graph reachability tests — new node + new edges. Need to update `test_topology.py` allowlist.
- The contract metadata for the new node (Phase Move-3 P6 contracts) — needs reads/writes/operations declared.
- The `combined_verdict` on the Decomposition is already populated by `CombineClaimVerdicts`. Synthesis demand-check reads it.

**This layer's risk:** infinite loop if budget logic is wrong. **Mitigation:** the per-sub-claim cap is the load-bearing safety. Demand routing to a sub-claim that's at cap → accept current state, synthesize. Test this explicitly.

### Layer C (CheckCompletion): unchanged

`CheckCompletion` already does exactly the right thing — it asks "are there non-abandoned claims?" and routes to Synthesize or End. Lazy escalation doesn't change this.

---

## Secondary-effect audit (per layer)

This section is the safeguard against the "fix one thing, break three" pattern. For each change above, what existing behavior could it disturb?

### From Layer P (one-provider-round-1)

- **Multi-seed-claim per-sub pool size.** Smaller pool means scrutiny's "enough evidence?" judgment fires on a thinner basis. *Risk*: scrutiny gives `needs_resolution` more often → more rounds. *Audit*: benchmark shows whether round-1 satisfaction rate stays acceptable. If <30% of sub-claims resolve in round 1, the eager savings are eaten by extra rounds.
- **Sub-claim-level filtering of evidence.** Existing per-`sub_investigation_id` filtering in `MultiSeedClaim` and `ExtractNewEvidence` continues to work — the field is set by the planner regardless of how many providers are used.
- **Cost asymmetry across sub-claims.** Different sub-claims will resolve at different round counts. *Risk*: combined posterior under AND combination is the min — if one slow sub-claim resolves at round 3 with low confidence, it bounds the answer regardless of how confidently the fast ones resolved. *Audit*: same as today (the `combine_claim_verdicts` semantics don't change).

### From Layer I (investigate consumes demand, picks new provider)

- **Provider tracking in state.** New mutable field `providers_used_per_sub`. *Risk*: forgetting to update it in one of the two write sites (PlanTask round 1, Investigate rounds 2+) leaves stale data. *Audit*: explicit reachability test that asserts all providers are covered after N rounds for an unsatisfied sub-claim.
- **Evidence stubs from investigation get a different provider mix.** Today they're mostly `web_search`. After the change, provider type matches the sub-claim's escalation order. *Risk*: downstream consumers that special-case `source_type == "web_search"` may behave differently. *Audit*: grep for `source_type ==` to find consumers.

### From Layer S (scrutiny emits demand)

- **Scrutiny's output model schema.** New optional fields. *Risk*: serialization round-trip (DocumentStore metadata) — the new fields need to round-trip correctly. *Audit*: pin a regression test that round-trips a scrutiny output through the DB.
- **Drift-detection checksums.** Will need updating. Standard procedure (test failure tells us new hash).

### From Layer V (verification: adversarial mandatory, others demand-driven)

- **Convergence-driven termination at `nodes.py:1471-1480`.** This fast-path requires all SUPPORTED claims to have terminal `convergence_verdict`. If we make convergence demand-driven and don't run it, the fast-path silently doesn't fire — claims fall through to ResolveUncertainties → potentially the IBE-skip bug we already fixed. *Mitigation*: keep `convergence` mandatory for any claim that hasn't been refute-promoted.
- **`adversarial_balance` field on Claim.** Set by `AdversarialSearchOperation`; read by stage gates and posterior calculation. If adversarial stays mandatory (per the recommendation), this is unaffected.
- **TMS sweep timing.** Runs after the track loop. If only some tracks fired, less evidence was created, less to invalidate. Should be strictly safer than before.
- **Track activation (PRIMARY/SECONDARY/SKIP).** The routing profile already decides which tracks fire for which question types. Demand-driving on top of routing-profile activation creates a 2-level gate: *first* the profile, *then* demand. *Risk*: getting the precedence wrong (e.g. demand could try to fire a SKIP'd track). *Mitigation*: demand-driving is restricted to tracks the profile has activated; SKIP'd tracks stay SKIP'd.

### From Layer SY (synthesis demand → loop back)

- **The graph topology changes.** New edges, new node. Topology test (`test_topology.py`) needs updating. *Audit*: write a test that drives a "synthesis-demands-more, then satisfied on second pass" scenario and asserts the path resolves cleanly.
- **The cycle-cap fall-through.** When a sub-claim hits cap and synthesis demands more, we accept current state. *Risk*: the "current state" may have stranded claims (SUPPORTED, integrated_assessment=None). *Mitigation*: the existing `no_stranded_claims` invariant still holds at Synthesize; the demand-loop only fires when claims are NOT stranded but the answer is still uncertain.
- **Combined verdict freshness.** `CombineClaimVerdicts` runs before the synthesis demand check. If demand fires a loop-back, the combined verdict needs to be re-computed after the new evidence is integrated. *Mitigation*: Combine is idempotent (recomputes from current claim state); just need to ensure it's re-run on the loop-back.
- **Post-loop infinite-loop risk.** Synthesis emits demand → routes to investigate → eventually back to synthesis. If satisfaction never improves, we cycle. *Mitigation*: the per-sub-claim cap is global to all loops affecting that sub-claim. Any sub-claim at cap counts as "satisfied" from synthesis's perspective.

### Cross-layer effects

- **Demand object plumbing.** Every layer transition that's now demand-driven needs the demand to flow through. *Audit*: grep for any operation that produces an output the next layer reads — does the demand travel with that output?
- **Test surface.** Roughly 12-15 new tests for the demand-driven paths; ~5 existing tests will need updating for new graph edges or operation signatures.
- **Topology contracts (P6 from Move-3).** Each new node needs `reads` / `writes` / `operations` / `post_invariants` declared. The `test_node_contracts.py` ensures this is enforced.
- **Operation profile output.** The CLI prints an operation profile after each run. With lazy escalation, some operations fire less often — the profile will look different. Expected; no fix needed.

---

## Phases (independently committable, each with benchmark gate)

The phases are sequenced so each is a contained change with a clear test signal. Importantly: **after each phase, run the benchmark and verify shape**, not just unit tests. Past sessions failed by trusting unit tests alone.

### Phase 0 — Demand object + cheap deterministic gates

- [ ] Create `andamentum.epistemic.demand` module with the `Demand` Pydantic model.
- [ ] Add helper `Demand.satisfied()` constructor and `Demand.needs(justification, target_hint="")` constructor for ergonomics.
- [ ] Add tests for the model (Pydantic validation, `model_dump`/`model_validate` round-trip).
- [ ] **Acceptance:** module imports clean; pyright + ruff clean; ~5 tests pass.

This phase doesn't change any behavior. It's the foundation.

### Phase 1 — Synthesis demand check (terminal gate first)

This phase adds the satisfaction check at the END of the pipeline, before Synthesize. It's the demand-PRODUCING side of the loop. Initially it doesn't loop back — just logs the demand. This proves the satisfaction-checking mechanism works without introducing the cycle risk.

- [ ] Create `CheckSynthesisDemand` graph node that runs between `CombineClaimVerdicts` and `Synthesize`.
- [ ] Implement deterministic-then-LLM satisfaction check.
- [ ] In Phase 1, the node ONLY logs the demand and continues to Synthesize regardless. No loop-back yet.
- [ ] Tests: 6-8 unit tests covering satisfied/unsatisfied scenarios, deterministic-gate hits, LLM-judgment hits.
- [ ] Update `test_topology.py` allowlist for the new node.
- [ ] **Acceptance:** benchmark produces same posterior shape as the d280573 baseline; topology test passes; new tests pass.

If this phase shows the satisfaction-LLM is wildly wrong about "is this answer good?", we stop and recalibrate before any loop. **This is the safety phase**.

### Phase 2 — One provider per sub-claim in round 1

- [ ] Add `providers_used_per_sub` field to `EpistemicGraphState`.
- [ ] Modify `PlanTaskOperation`: for round 1 (`providers_used_per_sub[sub_id]` is empty), pick ONE provider per sub-claim via LLM rank.
- [ ] When an existing `epistemic_select_provider` agent is suitable, reuse it. Otherwise add `epistemic_rank_providers` (one new agent).
- [ ] Tests: 4-6 tests covering round-1 narrowing, state-field update, LLM-pick determinism.
- [ ] **Acceptance:** benchmark resolves at least 1 sub-claim in round 1 (confirms narrow path works); no quality regression vs baseline.

### Phase 3 — Investigate consumes demand, picks next provider

- [ ] Modify `ScrutiniseClaimOperation` to emit `Demand` (extend output model).
- [ ] Modify `InvestigateClaimOperation` to consume `Demand`, pick next unused provider per `state.providers_used_per_sub[sub_id]`.
- [ ] Update Scrutiny agent prompt for the new output fields.
- [ ] Update drift-detection checksum.
- [ ] Tests: 6-8 tests covering provider escalation, demand round-trip, exhaustion behavior.
- [ ] **Acceptance:** benchmark shows different providers used in different rounds; baseline shape preserved.

### Phase 4 — Activate the synthesis-demand loop-back

- [ ] Modify `CheckSynthesisDemand` (from Phase 1): when demand=more AND any sub-claim has remaining budget, route back to `Investigate` (or `PlanEvidence` if a fresh provider is needed).
- [ ] Per-sub-claim cap enforcement: if all sub-claims with demands are at cap, accept and synthesize.
- [ ] Tests: 4-6 tests for the loop-back, cap enforcement, infinite-loop prevention.
- [ ] **Acceptance:** benchmark shows fewer total rounds for easy questions; same total rounds for hard questions; no infinite loops.

### Phase 5 — Optional: Verification track demand-driving

- [ ] In `RunVerification`, make `deductive` and `computational` tracks demand-driven (the safest two — keep `adversarial`, `convergence`, `argument`, `contrastive`, `consistency` mandatory for the conservative starting position).
- [ ] Satisfaction check after adversarial fires.
- [ ] Tests: scenarios where deductive/computational don't fire; verify no IBE-skip regression.
- [ ] **Acceptance:** benchmark shows reduced verification cost on confident claims; quality preserved.

This phase is OPTIONAL. Phases 0-4 are sufficient for the cost reduction. Phase 5 adds modest savings but has higher regression risk (the convergence-driven IBE fast-path is fragile, as we already discovered).

### Phase 6 — Closeout

- [ ] Update `CLAUDE.md` with a P7 principle: "Lazy escalation: each layer emits a `Demand` describing what's missing; the graph routes demand to the layer that can satisfy it minimally."
- [ ] Update memory with the secondary-effect lessons from this work.
- [ ] Final benchmark runs at multiple difficulty levels (3-4 different questions) to characterize the actual cost-vs-difficulty curve.

---

## Open decisions

These are deliberate gaps the executing session should resolve:

### 1. The provider-rank LLM call frequency

We add ONE LLM call per sub-claim per round to pick a provider. With 4 sub-claims and up to 3 rounds, that's up to 12 extra LLM calls per run. This is small compared to the savings, but it's not free.

**Recommendation:** measure on the benchmark; if sub-claims rarely escalate past round 1, the cost is just 4 extra calls per run.

### 2. Whether the synthesis-demand LLM call is too soft

The satisfaction LLM at synthesis is asking "is this answer good enough?" That's a high-stakes judgment with no objective ground truth.

**Recommendation:** prompt the LLM tightly. Use a flat schema like `Demand` with a clear `needs_more` boolean. Validate empirically — does the LLM say "satisfied" on questions where the verdict was clearly weak?

### 3. The verification phase activation order

If we go to Phase 5 (demand-driving deductive/computational), the order matters. Adversarial first → satisfaction check → others on demand. But if adversarial finds counter-evidence, the demand might be different from before.

**Recommendation:** start with all-mandatory verification (Phases 0-4 only). Defer Phase 5 until we have multi-question benchmark data.

### 4. Where the Demand object lives

`andamentum.epistemic.demand` is one option. Another is `andamentum.epistemic.entities.demand` if it's an entity-like artifact. Or `graph/demand.py` if it's graph-scoped.

**Recommendation:** `andamentum.epistemic.demand` (top-level module). It's used across operations, graph nodes, and entities — not specific to any subsystem.

### 5. Logging vs persisting the Demand chain

For observability, we'd ideally persist the demand chain to the database (alongside the operation log). Today the operation log is in-memory plus an execution-step trace.

**Recommendation:** Phase 1 logs demands (`logger.info`); persistence as a follow-up if the trace proves valuable.

---

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Synthesis-LLM "satisfied" judgment is miscalibrated | Medium | High | Phase 1 (logging-only) catches this before Phase 4 cycles on it. Tightly-prompted, flat schema. |
| Round-1 narrowing produces too-thin evidence pool | Medium | Medium | Per-sub-claim cap means we recover via investigation. Benchmark gate after Phase 2 catches it. |
| Provider-rank LLM picks badly (consistently the wrong provider) | Low | Medium | Per-sub-claim escalation means the wrong choice resolves on round 2. Less bad than no choice. |
| Synthesis-demand loop-back creates infinite loop | Low | High | Per-sub-claim cap is the load-bearing guard. Test explicitly. |
| Convergence-driven IBE fast-path stops firing | Low | High | Keep convergence mandatory in Phase 5; add reachability test for the fast-path. |
| `state.providers_used_per_sub` gets stale | Low | Medium | Single source of truth in state; one read site, two write sites. Pin with reachability test. |
| Track demand-driving (Phase 5) regresses adversarial/Lakatos coverage | Medium | High | Keep adversarial mandatory; only deductive/computational become demand-driven in Phase 5. |
| Demand-object schema drift across small LLMs | Medium | Medium | 3-field flat schema; validate with an Ollama-prefixed test. |

---

## What this plan does NOT cover

- **Cross-question caching.** Same question asked twice — separate plan.
- **Per-provider quality reputation.** Domain-y; out of scope.
- **Reducing inquiry rounds globally.** No global budget; per-sub-claim caps stay.
- **Changes to IBE chain or TMS.** Both work fine as-is.
- **Concurrency / parallelization.** Phase 2 of the prior plan covered this; it's separate from lazy escalation. Lazy escalation reduces *call count*; concurrency reduces *wall-clock*.

If any of these become necessary mid-implementation, write a follow-up plan.

---

## Acceptance criteria for the whole effort

When all required phases (0-4) complete, the following must be true:

1. The benchmark on `"Does intermittent fasting reduce all-cause mortality?"` with `--decompose` produces a posterior + verdict consistent with the `d280573` baseline shape: `n_no_verdict==0`, IBE fires, posterior valid, headline-prose alignment.
2. **Total LLM call count drops by >30%** for "easy enough" questions — measured by replaying the benchmark on a question whose first-provider hit yields strong evidence.
3. **No quality regression** on the harder benchmark — operation count similar to today; verdict still coherent.
4. The `Demand` object is visible in the operation log / execution trace, providing observability into why each layer decided to escalate or stop.
5. All existing tests (1810+) pass.
6. `topology()` reports the new graph topology cleanly; `test_topology.py` asserts the new edges.
7. Pyright + ruff clean.
8. The system "feels" different in observable ways: easy questions resolve faster with fewer rounds; hard questions still get the depth they need.
