# Freeze sheet — 2026-05-03

This document is the close-of-session record for 2026-05-02 / 2026-05-03.
Its job is to be the **first thing read** at the start of the next
session, so the next session can stand on this work without re-litigating
it.

The methodology is described in conversation: each session declares
a *frozen set* (off-limits, even when tempting) and a *target set*
(where work happens). Bugs in the frozen set become entry conditions
for future sessions, not in-flight fixes. Observation primitives that
read the frozen set must NOT pass through new code (raw SQL, jq, grep,
direct entity inspection only).

---

## Frozen set — DO NOT edit in the next session

### A. Lazy-escalation pipeline (commits `a436661`..`c112612`, plus fix `37dd07a`)

Six phases shipped:

| Phase | Commit | What it added |
|---|---|---|
| 0 | `9bf2d22` | `Demand` object — flat 3-field Pydantic model (`needs_more` / `justification` / `target_hint`) at `andamentum/epistemic/demand.py` |
| 1 | `5515b6e` (+ `20ad1f4`) | `CheckSynthesisDemand` node with deterministic gates (open-research, no-combined-verdict, stranded-claims, decisive posterior). WARNING-level `[synthesis_demand]` log line on every gate emission |
| 2 | `1b5d9b9` | One provider per sub-claim in round 1 of investigation (`epistemic_rank_providers` agent + `RankProvidersOutput`) |
| 3 | `0862589` (+ `37dd07a`) | Round 2+ provider escalation in `InvestigateClaimOperation` — picks next unused provider via the ranker. `37dd07a` is the *separate* pre-existing bug fix that made the agent's pydantic-model query items reach the gatherer correctly |
| 4 | `ba9975b` | Synthesis-demand loop-back to Scrutinize when eligible claims exist; per-sub-claim `SCRUTINY_RESOLVE_CYCLE_CAP` is the load-bearing safety |
| 6 | `c112612` | P7 principle in `CLAUDE.md`; memory entries for "lazy escalation" + "efficiency knobs regress quality" |

(Phase 5 — verification track demand-driving — was deferred per plan and remains deferred.)

**Verified empirically (probe B1, 2026-05-03):** The `[synthesis_demand]` log lines do fire when `CheckSynthesisDemand` is reached. The cap-driven safety belt fires correctly when no eligible claims remain.

**Not verified empirically:** the loop-back's *positive* path (route to Scrutinize when eligible claims exist) — would require a DB where some claims are not at terminal state. None of our saved DBs have that property.

### B. Stage runners (commits `6687187`..`433836d`)

Five phases shipped:

| Phase | Commit | What it added |
|---|---|---|
| 0 | `6687187` | Plan + empirical boundary discovery doc. Five stages, not the plan's draft six (scrutiny + investigation share state, can't be split) |
| 1 | `6e6718a` | `stop_after: type[Node]` kwarg on `run_epistemic_graph` |
| 2 | `a9703e8` | `start_at: type[Node]` kwarg on `run_epistemic_graph` |
| 3 | `f5db0a3` | `output_dir: Path` kwarg → emits `run.jsonl`, `diff.json`, `timing.txt` |
| 4 | `27c126e` | `stages.py` registry (5 stages, exit invariants, `StageInvariantError`) |
| 5–6 | `433836d` | `andamentum-epistemic stage <name>` and `inspect <db>` CLI subcommands; chain-resume test |

**LOC accounting:** 920 lines added across 4 files vs. the plan's ≤300 budget. About 3× over. Documented in the commit and the close-of-session conversation.

---

## Known broken things in the frozen set — do not fix in flight

These are real bugs. They get fixed in a future session whose explicit target is "fix X" — not as a side effect of other work.

| ID | Where | What's broken |
|---|---|---|
| **K1** | `graph/stages.py:_check_synthesis` | Invariant `getattr(obj, "report", None) is not None` checks an attribute that does not exist on `Objective`. Fails for every successful synthesis run. The actual report lives at `Objective.snapshot_id → Snapshot.artefact_id → Artefact.content` (2-hop indirection — see K2) |
| **K2** | `entities/objective.py` + `entities/snapshot.py` + `entities/artefact.py` | The Objective→Snapshot→Artefact indirection is undocumented in `CLAUDE.md`. Anyone reading code looking for "where's the report" will not find it on the Objective. Defensible design (Snapshot is immutable history, Artefact lets multiple report formats coexist) but reader-hostile |
| **K3** | `operations/synthesis/` (the `SynthesizeReportOperation`) | The synthesis writer agent collapses *failure-of-scrutiny* into *negation-of-claim*. Q1's saved DB had no integration verdicts and all claims terminal; the writer produced a confident "**No.** The aspirin doesn't prevent..." verdict. Anti-Peircean (no fallibilism mode), Lipton-divergent (mistakes absence-of-support for refutation). The writer needs an explicit "insufficient" verdict path |
| **K4** | `operations/synthesis/` | A single 90s LLM call on `openai:gpt-5.4-nano` to produce a 2.4kB report. Not "runaway" but expensive enough that the dev loop pays it every iteration. Candidate for using a cheaper / faster model only at this site |
| **K5** | `cli.py:_stage` | The CLI accepts `--question Q` (one literal letter) and silently produces a degenerate UNION decomposition over 4 sub-investigations of nothing meaningful. Should refuse / warn loudly on garbage input |

K1 was a self-inflicted bug from this session (in code I wrote). K2–K5 are properties of the long-standing system surfaced by this session's probes.

---

## Findings worth remembering

From probe B1 (synthesis stage on saved Q1 aspirin DB, 2026-05-03):

- **Phase 4's loop-back machinery is reachable**, but the cycle-cap terminates the natural pipeline upstream of `CheckSynthesisDemand` for terminal-claim runs. Stage runner forces the call. (Validation of own code; included for completeness.)
- **The synthesis stage spends 99.97% of its time in one LLM call.** The deterministic gates are 0.01s each. Time data is in `/tmp/probes/B1/timing.txt`.
- **The system has no "we don't know" mode** at the synthesis writer. It always produces a directional verdict, even from a no-data state.

---

## Open question (not assigned to a session)

Does the synthesis-demand loop-back's *positive* path actually fire on a real run with eligible claims? We have machinery to test it; we don't have a saved DB with the right shape. To test: build a DB where one or more claims have an integration verdict but the combined posterior lands in the ambiguous middle (~0.5–0.7), then run `stage synthesis --from-db that-db`.

---

## How the next session opens

The **first** action of the next session is reading this file. The session declares:

1. **Target** — one of: K1, K3, the open question, or something else entirely.
2. **Frozen set** — explicitly names what's off-limits.
3. **Observation mechanism** — for the work, what reads state without going through the code under test.

If a finding in this session contradicts what's in this freeze sheet, that's a separate findings file, not an in-flight edit to this one.

---

*Written 2026-05-03 to close the lazy-escalation + stage-runner work.*
