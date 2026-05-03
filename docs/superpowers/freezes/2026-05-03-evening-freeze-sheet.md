# Freeze sheet — 2026-05-03 (evening session)

This document is the close-of-session record for the second 2026-05-03
session. The first session that day produced
[`2026-05-03-freeze-sheet.md`](./2026-05-03-freeze-sheet.md), which
remains in force as a separate frozen set. This sheet stacks on top of
it; everything in the prior freeze remains frozen unless explicitly
called out.

The methodology, again: declare a *frozen set* (off-limits, even when
tempting) and a *target set* (where the next session works). Bugs in
the frozen set become entry conditions for future sessions, not
in-flight fixes.

---

## Session arc

The session opened against the previous freeze sheet's K-list (K1–K5)
and addressed all five plus two more (K6, K7) surfaced empirically.
Every fix landed mechanically: pyright clean, ruff clean, every test
green, every behavioural assertion held under fake runners. The
session closed with a re-run of the metformin/HbA1c benchmark that
**still produced "Insufficient evidence to answer"** — not because of
any of the K-fixes, but because of an upstream blocker the user
explicitly declared off-limits for the session (provider/search
quality).

The session's load-bearing finding is methodological: **seven
mechanically-correct fixes shifted the system from "fabricates
verdicts" to "honestly says it doesn't know," but did not shift it to
"answers correctly on questions it should be able to answer."** The
remaining gap is not in the parts of the system the K-fixes touched.

---

## Frozen set — DO NOT edit in the next session

### A. K-items closed this session (commits `8091ac9`..`fa0a1e7`)

Seven phases shipped:

| K | Commit | What it added |
|---|---|---|
| K1 | `8091ac9` | `_check_synthesis` invariant reads `obj.artefact_id` (the 1-hop signal) instead of the non-existent `obj.report` |
| K2 | `3bd9d3e` | CLAUDE.md documents the Objective→Snapshot→Artefact indirection + the typed `artefact_type` distinction (summary vs insufficient) |
| K3 | `0bb60bf` | **Maximal B**: new `SynthesizeInsufficient` terminal node parallel to `Synthesize`. `CheckSynthesisDemand._maybe_loop_back` routes to it when no eligible claims remain. Encodes Peirce/Lipton/AGM fallibilism as a graph terminal, not a writer-prompt option |
| K4 (step 1) | `e75a5e1` | Per-round timing instrumentation in `_writer_validator_loop` (no behaviour change). Steps 2+3 deferred — never observed in production this session because we never reached `Synthesize` |
| K5 | `4edecf0` | `DegenerateQuestionError` + `_validate_research_question` at graph entry. Refuses obvious garbage (`Q`, single tokens, <10 chars). Fires only on the fresh-objective branch so resume modes still work |
| K6 | `d8f9785` | **Maximal B extended**: `CheckCompletion` routes `retrieval_failed`, `no_claims`, and all-abandoned exits through `SynthesizeInsufficient` instead of returning End directly. Every "system can't conclude" path now produces a structurally honest artefact |
| K7 | `fa0a1e7` | `AdversarialSearchOperation` no longer hard-codes `support_judgment="contradicts"`. Every quality-passing adversarial-found item is judged by the impartial `epistemic_judge_evidence` agent. Adversarial provenance preserved in the reasoning text |

**Test counts:** 974 epistemic tests pass (+19 net new); 1886+ project
tests pass; pyright +1 net error of an already-present variance
pattern (no new error category); ruff clean; drift-detection checksums
updated for the K4 instrumented section.

### B. Empirically verified

Three probes ran against this freeze:

| Probe | Setup | Result |
|---|---|---|
| **B2** | Hand-built K3 shape (no `combined_verdict`, all claims terminal) → `stage synthesis` | Pre-fix: 90.81s + fabricated "**No.** Aspirin doesn't prevent...". Post-fix: 0.11s + typed "Insufficient evidence to answer." artefact. K3 mechanism verified |
| **B3** | Real aspirin end-to-end | Surfaced K6 (the `retrieval_failed` bypass). 0 artefacts pre-K6 |
| **B3b** | Replay B3 DB with `state.retrieval_failed=True` after K6 | 1 artefact stamped `type="insufficient"`, body surfaces structural counts (3 claims, 44 evidence, 1 cycle-capped, 2 with no integration verdict) + retrieval-failure reason text. K6 mechanism verified |
| **B4** | Real metformin/HbA1c end-to-end | Surfaced K7 (adversarial mislabeling — CD012906 stored as `contradicts`). Verdict: insufficient |
| **B5** | Real metformin re-run after K7 | Adversarial-found items now correctly judged across all three labels (verified). Verdict: still insufficient (because the regular-evidence-flow distribution is unchanged). 2 of 4 claims got integration verdicts (B4: 1 of 3) — incremental progress, but not on the headline |

---

## Known broken things in the frozen set — do not fix in flight

| ID | Where | What's broken |
|---|---|---|
| **K8** | Provider/search-quality layer (specifically the interaction with `--decompose` + lazy-escalation Phase 2 round-1 narrowing) | The load-bearing blocker. On both probe B4 and B5, ~96% of evidence returned by the regular-flow providers (cochrane, europepmc, openalex, pubmed) is judged `no_bearing` by the impartial judge — i.e., off-topic. The Cochrane API returns metformin reviews for PCOS, GDM, endometrial hyperplasia, etc. — but not CD012906, the canonical metformin/T2DM/HbA1c review. Web_search via the adversarial path eventually finds CD012906; the regular flow doesn't. **User flagged this as off-limits for the session — search code "hadn't changed and used to give fantastic results"; the regression is more likely upstream of the provider code itself.** Hypothesis worth checking next session: lazy-escalation Phase 2 (one-provider-per-sub-claim in round 1) is the plausible suspect since the system used to query all providers in parallel. If correct, reverting that phase or adjusting its provider-rank prompt is the targeted fix |
| **K4 step 2+3** | `_writer_validator_loop` (instrumented but unobserved) | We added timing instrumentation but the writer-validator loop never fired in production this session — every probe routed to `SynthesizeInsufficient` (which has no LLM call). The "90s synthesis call" cost remains uncharacterised. Step 2 is "run a probe that reaches `Synthesize`"; step 3 is "fix what the timing data shows." Both stay deferred until K8 is unblocked enough that the system actually produces a directional verdict |
| **K9** | TMS-demote-then-stuck mid-cycle | When a claim's adversarial balance lands just below the 0.7 Popper threshold (probe B5 main metformin claim: 0.66; B3 aspirin claim A: 0.69), the gate refuses to re-promote. The behaviour is philosophically correct per Popper / Lipton / AGM — refusing to claim something whose evidence has been refuted — but the claim then sits at HYPOTHESIS until the cycle cap fires, with no first-class "contested" or "evidence-equivocal" terminal to express the outcome. The K3+K6 fixes catch the *system-level* "can't conclude" paths; they don't address the *claim-level* equivocal terminal. Long-term fix: a typed `Claim.terminal_state = "contested"` that combiner can include in the combined verdict |
| **Open question (carried)** | Lazy-escalation positive loop-back | Still not observed firing on a real run. Probe B5 had 2 claims with `cycle_capped=False, abandoned=False, integ=None` — eligible for loop-back — but the run terminated via `retrieval_failed` before the loop-back could engage. The mechanism remains unvalidated empirically |

K8 is the load-bearing item. The other three are real but downstream
of K8: even with K9 fixed, if the supporting evidence pool is 96%
off-topic, the adversarial balance won't change. K4 step 2+3 can't be
characterised until the system actually reaches the writer-validator,
which requires K8.

---

## Findings worth remembering

From the seven mechanical fixes plus the empirical probes:

1. **Mechanical correctness ≠ answer-quality movement.** Seven fixes
   landed cleanly. Test suite size grew by ~20. Every fix's intended
   contract is verified. None of the seven moved the headline verdict
   on metformin from "insufficient" to "supports". The deciding bottleneck
   was always upstream of where the fixes lived.

2. **The system is now epistemically honest about not knowing.** Pre-
   session, the system fabricated "**No.** Aspirin doesn't prevent..."
   on a no-evidence Q1 run. Post-session, every "can't conclude" path
   produces a typed `Artefact(artefact_type="insufficient")` whose body
   surfaces structural counts and the gate's diagnosis. This is the
   load-bearing Peircean property the architecture is meant to encode.
   It's just not, alone, what makes a publishable system.

3. **The adversarial agent's "contradicts" hard-code (K7) was a real
   bug, not calibration.** It pre-dated the lazy-escalation work; it
   only became load-bearing when supporting-evidence counts dropped.
   Two layers of latent bug + one upstream change made the headline
   degrade. The freeze-sheet methodology caught it on probe B4.

4. **The TMS demote-then-stuck pattern is correctly designed and
   documented** — see commit `e86e834` (Phil 2026-05-01) and
   `87d4998` (2026-04-19) for the explicit Popper-corroboration
   architecture. The user-visible failure mode is a missing terminal,
   not a wrong refusal. K9 above is the structurally-honest fix; the
   refusal itself is the right move.

5. **Frozen-set methodology held under load.** Today's session did
   *not* drift into K8 mid-flight even though the empirical evidence
   would have justified it. Per the user's call, the off-limits item
   stayed off-limits. The cost was that we did not move the headline.
   The benefit was that we ended the session knowing exactly where the
   blocker is, with a clean test fixture (probe B5's saved DB) for the
   next session to inspect without re-running.

---

## Open question for the next session

**Is K8 the right thing to attack first, and if so, where exactly do
we cut?**

Three plausible entry points, in increasing scope:

* **a)** Inspect probe B5's saved DB and look at what queries the
  Cochrane / EuropePMC / OpenAlex providers actually issued. If the
  queries are too broad ("metformin" alone vs "metformin HbA1c type 2
  diabetes"), the fix lives in the query-formulation step (likely the
  per-sub-claim provider-rank agent or its prompt). Smallest blast
  radius.

* **b)** Revert lazy-escalation Phase 2 (commit `1b5d9b9` — "one
  provider per sub-claim in round 1") and re-run B5. If the verdict
  flips to "supports", the lazy-escalation narrowing is the
  regression. Targeted, reversible, immediately empirically testable.

* **c)** Step back and rethink the evidence-gathering layer entirely
  per the older `project_evidence_architecture_rethink` memory (from
  2026-04-21). Highest risk, longest arc, but the user's earlier
  judgement was that this is the right level to think at.

The session that picks K8 should pick exactly one of these and stay
in scope.

---

## How the next session opens

The first action is reading **both** freeze sheets:

1. [`2026-05-03-freeze-sheet.md`](./2026-05-03-freeze-sheet.md) (still
   in force; covers lazy-escalation + stage-runner + the original K1–K5)
2. This sheet (covers today's K1–K7 + K8/K9 surfaced)

The session declares:

1. **Target** — most likely K8 (one of a/b/c above), or the open
   lazy-escalation positive loop-back validation, or something else
   entirely.
2. **Frozen set** — explicitly names what's off-limits. K1–K7 are
   shipped and verified; touching them needs explicit reason.
3. **Observation mechanism** — for the work, what reads state without
   going through the code under test. Probe B5's saved DB
   (`probe_b5_metformin`) is a ready-made fixture for K8 work.

If a finding contradicts what's in either freeze sheet, that's a
separate findings file, not an in-flight edit.

---

## Saved DBs available as fixtures

| DB name | Question | Shape | Use for |
|---|---|---|---|
| `probe_b3_aspirin` | "Does daily low-dose aspirin prevent first myocardial infarction in healthy adults aged 50-70?" | retrieval_failed terminal; 0 artefacts pre-K6 (now should be 1 if re-run) | Reproducing the retrieval_failed path |
| `probe_b4_metformin` | "Does metformin reduce HbA1c in adults with type 2 diabetes?" (pre-K7) | 1 artefact `insufficient`; main claim 1 supports / 6 contradicts (mostly mislabelled) / 55 no_bearing | Comparing pre-K7 adversarial mislabelling to post-K7 |
| `probe_b5_metformin` | (same metformin question, post-K7) | 1 artefact `insufficient`; main claim 1 supports / 1 contradicts / 73 no_bearing; 2 of 4 claims with integration verdicts; adversarial balance 0.66 | **Primary fixture for K8 work.** The supporting count of 1/76 on a textbook-positive question is the empirical signal of the provider/search-quality regression |

---

*Written 2026-05-03 (evening) to close the K1–K7 work and freeze K8
(provider/search quality) as the entry condition for the next session.*
