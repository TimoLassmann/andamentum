# Stage probes — diagnostic dossier

**Date:** 2026-05-03
**Plan:** [`../superpowers/plans/2026-05-03-stage-runners.md`](../superpowers/plans/2026-05-03-stage-runners.md)
**Goal:** use the stage runner machinery to answer four questions:
(a) does the system work, (b) is there runaway computation, (c) what
architectural issues exist, (d) does the code align with the stated
philosophical principles?

This is a *diagnostic* document. Findings here become input to future
plans, not in-flight code changes.

---

## Probe B1 — synthesis stage on the saved Q1 aspirin DB

**Command:**
```bash
andamentum-epistemic stage synthesis \
    --from-db q1_aspirin \
    --db-dir /tmp/lazy_escalation_db \
    --model openai:gpt-5.4-nano \
    --output-dir /tmp/probes/B1
```

**Pre-state:** Q1 DB had 3 sub-claims, all terminal (A cycle-capped, B
abandoned, C cycle-capped+abandoned), 100 evidence items, no
`combined_verdict`. From yesterday: the natural pipeline run never
reached `CheckSynthesisDemand` — open question was whether the
Phase 4 loop-back is reachable at all.

**Wall-clock:** 90.81 s.

### Findings

#### F1 — `[synthesis_demand]` log lines fire (✅ goal a)

Both gate-fired log lines visible in stdout:

```
[synthesis_demand] needs_more=True | No combined verdict produced
  (every claim was abandoned, cycle-capped, or had no integration
  verdict). Without aggregated per-claim posteriors the headline
  answer is the no-data fallback.

[synthesis_demand] needs_more=True but all non-abandoned claims have
  hit per-claim cap; synthesizing anyway. (Existing safety: per-claim
  cap is the loop-bound; no global give-up budget.)
```

This **answers yesterday's open question.** The Phase 4 machinery
*is* reachable; it routed correctly when invoked. The reason it never
fired in the natural Q1 run is that the natural run terminated at
the cycle-cap before reaching `CheckSynthesisDemand`. The stage
runner bypasses that termination and forces synthesis to run on the
saved state.

The safety belt (cap-driven termination when no eligible claims)
fires correctly: log says "synthesizing anyway." Phase 4's load-
bearing safety is **provably exercised**.

#### F2 — synthesis stage's exit invariant is wrong (✅ goal c)

Run produced `Error: Stage 'synthesis' exited at Synthesize but its
invariant is unsatisfied`. My Phase 4 invariant was
`getattr(obj, "report", None) is not None`. **`Objective` has no
`report` attribute.** The check is silently default-False, so the
invariant fails for every successful synthesis run.

**Action:** the invariant needs to walk Objective → Snapshot →
Artefact (see F3). Not fixing in this probe — recording for follow-up.

#### F3 — the report lives 2 hops from the Objective (✅ goal c, architectural)

The Synthesize node:
1. Calls `FreezeSnapshotOperation` → creates a `Snapshot` entity with
   `claim_ids` / `evidence_ids` / `uncertainty_ids` manifest.
2. Calls `SynthesizeReportOperation` → produces an `Artefact` (type
   `summary`) with the report content as `Artefact.content` (2385
   chars in this run).
3. Sets `Objective.snapshot_id = snapshot.entity_id` and
   `Snapshot.artefact_id = artefact.entity_id`.

So the chain is **Objective → Snapshot → Artefact → content**. The
final answer is two indirections deep. A reader of the code looking
for "where's the report" would not find it on the Objective.

**Architectural verdict:** the design is *defensible* — Snapshot is
immutable and preserves history, Artefact lets multiple report
formats coexist (summary, full, etc.). But **the indirection is
undocumented in CLAUDE.md** and the natural intuition is wrong. This
is a documentation + naming issue, not a structural one.

#### F4 — Synthesize takes ~90s (~99% of synthesis stage time) (✅ goal b)

`timing.txt`:
```
Total: 90.81s (3 node visits)
  Synthesize: 90.79s
  CheckSynthesisDemand: 0.01s
  CheckCompletion: 0.01s
```

Synthesis is **a single 90s LLM call** on `openai:gpt-5.4-nano` to
produce a 2385-char report. The deterministic gates are 0.01s each.
The LLM IS the entire cost.

**Implications:**
- For the dev loop, even synthesis-only iteration costs 90s. A
  cheaper model would help. The synthesis report agent should
  be a candidate for a "fast" model setting.
- "Runaway" is not the right word here — the cost is ONE call, not a
  loop. But it's an expensive single call.

#### F5 — the synthesized verdict is "No" despite no combined_verdict (⚠️ goal d, philosophical)

Artefact content begins:
> # Aspirin for Primary Prevention of First Cardiovascular Events in
> Adults Under 70
>
> **Verdict:** No. Th...

The system produced a confident "No" verdict from a state where
**every claim was abandoned or cycle-capped and no integration
verdict existed.** The synthesis prompt was given the per-claim
state (all in terminal-not-passed) and inferred a "No" answer.

**Philosophical alignment check (Lipton — IBE):**
- **Aligned:** the report writer is doing inference *to* a best
  explanation. It pattern-matches "claims didn't pass scrutiny" → "the
  effect probably isn't real" → "No."
- **Divergent:** the *grounds* for "No" are the **failure of
  scrutiny**, not the **presence of contradicting evidence**. In
  Lipton's terms, this is *absence of supporting* explanation, not
  *presence of refuting* one. They are different epistemic states
  and conflating them is a real bug.

**Philosophical alignment check (Peirce — fallibilism):**
- The system has no "we don't know" mode. It always produces a
  verdict. Q1's actual epistemic state is closer to "the evidence
  base is fragmented and our scrutiny rounds didn't converge" — but
  the report flattens it to a confident negative. **This is the
  no-data-fallback issue surfacing as a confident answer.**

**Action for follow-up plan:** the synthesis writer needs an explicit
"insufficient" mode and should refuse to flatten "no integration
verdict" into a directional answer. Not a fix here; a finding.

### B1 verdict per goal

| Goal | Finding |
|---|---|
| (a) does the system work? | Mechanism: yes. F1 confirms Phase 4 reachable + safety fires. F5 shows downstream report-writer collapses epistemic states. |
| (b) runaway computation? | One stage = one 90s LLM call. Not runaway, but expensive. The dev loop benefits if synthesis can use a cheaper model. |
| (c) architectural issues? | F2: invariant wrong. F3: report is 2-hop indirection (defensible but undocumented). |
| (d) philosophical alignment? | F5: synthesis collapses "no-data" → confident "No". Anti-Peircean (no fallibilism mode), Lipton-divergent (mistakes absence-of-support for refutation). |

### Open follow-up items from B1 (not done in this probe)

1. Fix synthesis-stage invariant to check Snapshot → Artefact chain.
2. Document the Objective → Snapshot → Artefact indirection in CLAUDE.md.
3. Make the synthesis writer surface "insufficient" / "no-data" verdicts honestly rather than collapsing to directional answers.
4. Investigate making synthesis use a cheaper model than the rest of the pipeline (90s for a 2.4kB report is too expensive for iteration).
