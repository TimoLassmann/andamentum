# K8 Bug #1 — provider-tournament for research-mode round 1

> Companion plan to the threshold harmonization work
> (commits `f65845c`..`c91c13b`). Closes the long-deferred K8 Bug #1
> from the 2026-05-04 freeze sheet, but only when research mode is
> the active workload — verify mode is unaffected.

**Goal.** Replace the per-sub-claim ranker call (which collapses to
the same provider for every sub-claim because each call evaluates
its sub-claim independently) with an **iterative tournament at the
objective level**. The tournament picks K=2 providers for the whole
objective; every sub-claim then queries both providers in round 1.

**The structural win this enables.** With ≥2 providers per
sub-claim, the cross-domain convergence detector
(``convergence_detector.py``) can fire **per claim**, not just
across claims. Reichenbach's common-cause principle: agreement
across genuinely independent sources is worth more than agreement
within one source. K=2 is the minimum that enables this; K=3 just
makes it stronger.

**Constraint.** Domain-agnostic; works for small local models;
small additional cost in round 1 only (lazy-escalation in round 2+
unchanged).

---

## Final shape

For research-mode (decomposition) round 1:

```
1. DECOMPOSE_QUESTION   → produces N sub-claims
2. RANK_PROVIDERS_TOURNAMENT (NEW):
     - Call epistemic_rank_providers with the parent question + all candidates → pick provider P1
     - Remove P1 from candidates
     - Call epistemic_rank_providers with parent question + remaining → pick P2
     (K = RESEARCH_MODE_PROVIDER_K, default 2)
3. PLAN_EVIDENCE — for each (sub-claim, provider) pair:
     - Call epistemic_formulate_query with sub-claim + provider's query_guidance
     - Create one Evidence stub tagged with sub_investigation_id and source_type=provider
4. EXTRACT + JUDGE per stub (unchanged)
```

**Verify mode**: unchanged. The tournament fires only when there's
a decomposition with ≥2 sub-investigations.

**Round 2+**: lazy-escalation behaviour unchanged. Investigation
operations pick next-unused providers per
``state.providers_used_per_sub``; the round-1 tournament leaves
``providers_used_per_sub`` populated with K providers per sub-claim,
so round 2 picks #3, round 3 picks #4, etc.

---

## What changes

### Code (~80 LOC across 2 files)

| File | Change |
|---|---|
| `operations/preplanning.py` | Add ``_run_provider_tournament(question, candidates, k)`` helper that calls the ranker K times with shrinking candidate list. In ``PlanTaskOperation.execute``, replace the per-sub-claim ranker call with one tournament call before the sub-claim loop. Inside the loop, iterate over the K picked providers and create one Evidence stub per (sub-claim, provider) pair. |
| `operations/preplanning.py` | Add ``RESEARCH_MODE_PROVIDER_K = 2`` constant near the top of the file with an inline comment explaining the trade-off (round-1 cost × K, enables per-sub-claim convergence). Operational, not theoretical — kept here rather than in `thresholds.py`. |

### Tests (~30 LOC in 1-2 files)

- New test: tournament picks K different providers for an objective with diverse-shape sub-claims.
- New test: tournament with `K > len(candidates)` clips to all candidates.
- Existing tests pinning "round 1 creates N stubs for N sub-claims" need to update to "K × N stubs."

### Drift detection / docs

- The PlanTask body changes — drift hash on `Plan task / multi-seed branch` (if it exists) may need updating.
- CLAUDE.md's "lazy escalation" P7 description gets a one-line note about K providers per sub-claim in round 1 of research mode.

---

## Phases

### Phase 1 — Tournament helper (no behaviour change yet)

- [ ] Add ``_run_provider_tournament(self, *, question: str, candidates: list[str], k: int) -> list[str]`` to ``PlanTaskOperation`` (or as a module-level function in preplanning.py if it doesn't need self).
- [ ] Helper calls ``epistemic_rank_providers`` K times, removing each pick from the candidate list before the next call. Returns the picked providers in order.
- [ ] Defensive: if the ranker picks an unknown provider (LLM hallucination), fall back to the first remaining candidate. Same fallback as the existing per-sub-claim ranker.
- [ ] Edge case: if K > len(candidates), return all candidates in tournament order (no infinite loop).
- [ ] Tests: 4-5 unit tests covering happy path, K > N candidates, ranker returns unknown provider, K=1 (degrades to existing behaviour), empty candidate list (raises).
- [ ] **Acceptance**: helper exists, tested, but **not yet wired** into PlanTask. Test suite runs at 995 (no behaviour change yet).

This phase is the safety net. The helper is a pure function; we can verify it works before changing PlanTask.

### Phase 2 — Wire tournament into PlanTask

- [ ] In ``PlanTaskOperation.execute``, the multi-seed-claim branch (where `sub_investigations` is set):
  - Replace the per-sub-claim ranker call with one tournament call before the sub-claim loop.
  - Inside the sub-claim loop, iterate over the K picked providers; create one Evidence stub per (sub-claim, provider) pair.
  - Update the plan-message string to reflect "K providers per sub-claim".
- [ ] The `else` branch (no decomposition / open-research path): unchanged.
- [ ] **Acceptance**: research-mode unit tests verify K × N stubs created with the right provider distribution. Verify-mode tests unchanged. SciFact offline check unchanged (verify mode never enters this code path).

### Phase 3 — Update existing test fixtures

- [ ] `test_multi_seed_claim.py` and any test pinning "round 1 = N stubs for N sub-claims" → update to K×N.
- [ ] `test_lazy_escalation.py` (if exists) — round-1 stub count changes; round-2 escalation behaviour unchanged.
- [ ] Any drift-detection hash update.

### Phase 4 — Closeout

- [ ] CLAUDE.md P7 note: "research mode round 1 picks K=2 providers per objective via iterative tournament; every sub-claim queries both. Round 2+ escalates to next-best unused via the existing lazy-escalation mechanism."
- [ ] Memory entry: explain the tournament-vs-per-sub-claim choice (per-objective is cheaper on the ranker side and harder to collapse; per-sub-claim K=2 was rejected because it 8× the ranker calls without enough additional benefit).
- [ ] If any structural manuscript figure references "1 provider per sub-claim", update.

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| 2× round-1 cost (formulator + provider hits + judge calls × 2) | Certain | Medium — wall-clock per case ~1.3-2× | Documented; the convergence-per-sub-claim benefit is the trade-off justification. K=2 is the minimum K that enables convergence. Can monitor on real benchmark and decide on K=3 later if needed. |
| Retrieval-health threshold (``_EMPTY_EXTRACTION_THRESHOLD = 3``) might fire differently with 2× extractions per sub-claim | Low | Low | The threshold counts *consecutive* empties; 2× the extractions doesn't change the consecutive-empty pattern unless one provider systematically returns empty (which is the issue the threshold catches). Monitor on benchmark; if it becomes a problem, the threshold logic can be made provider-aware. |
| Two providers might have heavily overlapping content (e.g. PubMed + EuropePMC index much of the same literature) | Medium | Low | The cross-provider deduplication sweep (``dedupe_evidence_by_source_ref`` in graph/nodes.py:ExtractNewEvidence) catches this — duplicate items get invalidated, judge calls don't fire on duplicates. So real cost overhead may be less than 2× when providers overlap. |
| LLM-hallucinated provider name from the ranker | Low | Low | Existing fallback (use first remaining candidate) catches this in both the helper and the existing per-sub-claim path. |
| Small model fails to pick K different providers | Low | Medium | Tournament structure — the LLM only has to pick one provider per call; on the second call the picked-on-first is removed. The LLM cannot pick the same one twice by construction. |
| ``providers_used_per_sub`` state-field gets twice as much per round | Certain | Low | This is what we want — round 2 escalation has fewer un-used providers, but the existing 10-provider catalogue gives plenty of headroom (round 2 picks #3, round 3 picks #4 ...). |
| Existing tests pin "1 provider per sub-claim in round 1" | Certain | Low | Tests are updated in Phase 3. |

---

## Acceptance criteria

When all four phases complete:

1. ``PlanTaskOperation`` in research mode creates K×N Evidence stubs in round 1 (was N).
2. Each (sub-claim, provider) pair has its own stub tagged with the right `sub_investigation_id` and `source_type`.
3. Lazy-escalation round 2+ correctly excludes the K providers used in round 1 when picking next-best per sub-claim.
4. The cross-domain convergence detector now has ≥2 distinct providers per sub-claim, enabling per-claim convergence checks.
5. Verify mode unchanged — no SciFact regression.
6. Test suite green; new tests cover tournament happy path, edge cases, fallback.
7. Wall-clock cost on a research-mode benchmark increases ~1.3-2× (data-dependent on dedup hit rate).

---

## What this plan does NOT do

- **Per-sub-claim K=2 (rather than per-objective K=2)**: rejected as 4× more ranker calls without enough extra benefit. If a benchmark shows that some sub-claims really need different providers from their siblings (and the per-objective K=2 isn't covering them), revisit then.
- **Adaptive K**: K is a constant. We don't ask the LLM how many providers a question needs. Future work if K=2 vs K=3 matters empirically.
- **Embedding-based deterministic selection**: the LLM-based ranker stays the primary path. An embedding pre-filter (cosine similarity between sub-claim and provider description) would be an optimization, not part of this plan.
- **Convergence detector tuning**: the detector is unchanged; this plan just feeds it the inputs it needs to fire.
- **Verify-mode changes**: irrelevant; verify mode has no decomposition.

---

## How to verify after shipping

1. **Synthetic test (Phase 1)**: feed the tournament a 4-provider candidate list with K=2 → assert it picks 2 different providers.
2. **Unit test (Phase 2)**: run PlanTask on a fixture with 3 sub-claims and 4 candidate providers → assert 6 (3×2) stubs created with the right provider mix.
3. **Offline benchmark check**: run a research-mode question (e.g. the metformin/HbA1c probe DB if available) post-fix; assert that round-1 produced K different providers across sub-claims.
4. **Cost telemetry**: log `[plan_task] obj=... round=1 stubs_created=N×K` so a future analysis can quantify the actual cost overhead.

---

*Written 2026-05-05 to scope the K8 Bug #1 fix. The bug has been
deferred since 2026-05-04 because verify-mode SciFact runs don't
exercise it; this plan is the right move to ship before any
research-mode benchmark, which would surface the collapse on real
data.*
