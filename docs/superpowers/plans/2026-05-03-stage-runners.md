# Stage Runners — restoring the develop → test → analyse cycle

**Status:** plan
**Date:** 2026-05-03
**Author:** session continuation, Lassmann + Claude

---

## The principle in one sentence

The graph is its own test harness: expose named stage boundaries on the existing pipeline so a developer can run, save, resume, and inspect any prefix or suffix of the graph against a real DB, with zero new abstractions.

---

## What problem this solves

Today: a real-data run takes 11–20 min. There is one entry point (`run_epistemic_graph`) and one exit (`End[...]`). To verify a change to scrutiny, you re-run the whole pipeline. Iteration is impossible; the dead-code Phase 4 loop-back is invisible until a 20-minute run.

Goal: a developer changing scrutiny code re-runs only scrutiny on a frozen pre-scrutiny DB in <30 s, and an `inspect` command prints what changed. The same code path runs in production; the only difference is where it starts and stops.

---

## What this is NOT

- Not a new test framework. Tests live in `pytest`, that doesn't change.
- Not a parallel-execution layer. Sequential as today.
- Not a synthetic-evidence layer. The DB IS the fixture; real evidence is fine.
- Not a new entity, table, or graph node. **Zero new abstractions.**
- Not a replay/cache layer. Out of scope; revisit only if stage runs are still too slow after this.

---

## Anti-bloat budget (load-bearing)

Hard ceilings. If we exceed any, the plan is wrong, not the budget:

- Total new code: **≤ 300 lines including tests**
- New entities: **0**
- New DB tables: **0**
- New node classes: **0**
- New Python files: **1** (`stages.py`)
- New CLI subcommands: **2** (`stage`, `inspect`)

The whole point of this work is to NOT add a maze of new code on top of an already complex graph.

---

## The mental model

A *stage* is a NAME for "run from graph node X until graph node Y completes." Nothing more.

```
 question ──[preplanning]──> Objective+decomposition
                              │
                              ├──[investigation]──> evidence stubs filled
                              │
                              ├──[scrutiny]─────> claim verdicts
                              │
                              ├──[verification]─> tracks complete
                              │
                              ├──[integration]──> integrated_assessment
                              │
                              └──[synthesis]───> answer + report
```

Each stage's input is the DB state at its entry; its output is the DB state at its exit. The DB is the only currency. No serialised checkpoints, no diff blobs, no resumption tokens.

---

## Mechanism (the entire change)

**One file, two kwargs, three artifacts, two CLI commands.** That is the contract.

### `src/andamentum/epistemic/graph/stages.py` (new, ≤120 LOC)

```python
@dataclass(frozen=True)
class StageDef:
    name: str
    entry: type[Node]               # node to start at
    exit_after: type[Node]          # graph terminates after this node returns
    exit_invariant: Callable[[State, Repo], bool]   # MUST hold at exit_after
    description: str

STAGES: dict[str, StageDef] = {
    "preplanning":   StageDef(...),
    "investigation": StageDef(...),
    "scrutiny":      StageDef(...),
    "verification":  StageDef(...),
    "integration":   StageDef(...),
    "synthesis":     StageDef(...),
}
```

Six entries, total. Discovered empirically (Phase 0) — not declared upfront.

### Two new kwargs on `run_epistemic_graph`

```python
async def run_epistemic_graph(
    ...,
    start_at: type[Node] | None = None,   # skip to this node, expect DB satisfied
    stop_after: type[Node] | None = None, # short-circuit End after this node
) -> EpistemicResult: ...
```

Implementation: a 10-line wrapper around the existing graph constructor that picks the entry node and registers a "after this node, route to End" interceptor. No mutation of any node class.

### Three observability artifacts (per run)

Written next to the DB. **Plain files. Greppable. Diffable. Committable as test fixtures.**

| File | One line per | Used for |
|---|---|---|
| `run.jsonl` | node visit: `{ts, node, op, ms, llm_calls}` | Where did time go? |
| `diff.json` | end-of-run state delta | What changed? |
| `timing.txt` | total + per-node + top-K LLM calls | Profile-at-a-glance |

These come from a single log handler attached to the graph runner, not from instrumenting individual nodes. One place to add it, one place to remove it.

### Two CLI subcommands

```bash
# Run a single stage
andamentum-epistemic stage preplanning   --question Q       --db /tmp/run1.db
andamentum-epistemic stage scrutiny      --from-db /tmp/run1.db
andamentum-epistemic stage synthesis     --from-db /tmp/run1.db

# Print structured state of a saved DB
andamentum-epistemic inspect /tmp/run1.db
# -> Objective: ...
# -> 3 claims: A (supported, posterior=0.61), B (abandoned), ...
# -> 100 evidence (87 with content >200ch)
# -> Last stage exited: investigation (cycle-cap fired on B)
```

`stage` is `run_epistemic_graph(start_at=stage.entry, stop_after=stage.exit_after)`. Five lines.

`inspect` is `EpistemicRepository.summary()` formatted to stdout. No new query patterns.

---

## Exit invariants — the safety belt

The user's "unforeseen edge cases" concern is the real risk. Antidote: every stage has ONE invariant, checked when the exit node returns. If false, the run **crashes loudly**:

```python
"preplanning": StageDef(
    exit_after=DecomposeQuestion,
    exit_invariant=lambda s, r: (
        s.objective.decomposition is not None
        and len(s.objective.decomposition.sub_investigations) > 0
    ),
    ...
),
"integration": StageDef(
    exit_after=Integrate,
    exit_invariant=lambda s, r: all(
        c.integrated_assessment is not None
        for c in active_claims(r)
        if not c.cycle_capped and not c.abandoned
    ),
    ...
),
```

A failing invariant means the stage boundary isn't quiescent — i.e., there's still work in-flight. Then we either move the boundary or fix the bug. Invariants are the contract between stages.

---

## Phases

Each commits independently, each with a verifiable test or smoke-run as gate.

### Phase 0 — Empirical boundary discovery (no code, ~30 min)

Run the current pipeline once with `run.jsonl` instrumentation. Walk the trace. Mark candidate stage boundaries at quiescent points. Sanity-check that no node in a stage's interior writes state past the exit. **Output:** the contents of `STAGES` dict above, plus written notes on each boundary's invariant.

Gate: each candidate exit is provably the last writer of the state it produces.

### Phase 1 — `stop_after` (≤30 LOC)

Add the kwarg. Wrap End-routing. Add ONE test: run with `stop_after=DecomposeQuestion`; assert `Objective.decomposition is not None` and total node visits ≤ N.

Gate: existing tests pass; new test passes.

### Phase 2 — `start_at` (≤30 LOC)

Symmetric to Phase 1. Add ONE test: stop at preplanning, save DB, start fresh runner from `PrepareInvestigation` against that DB; produce the same Evidence as a one-shot run on the same seed.

Gate: round-trip equivalence test passes.

### Phase 3 — Observability artifacts (≤80 LOC)

One handler in `graph/runner.py`. Three files emitted next to the DB. Add ONE test: run preplanning, assert `run.jsonl` lines == nodes visited, assert `diff.json` mentions the new Objective.

Gate: artifacts produced; CI smoke-runs them and diffs against fixtures.

### Phase 4 — `stages.py` registry + invariants (≤60 LOC)

Add the six `StageDef` entries with their exit invariants. Wire invariants into the runner: after `stop_after` returns, check the invariant. Crash if false.

Gate: each stage's invariant tested with a hand-crafted state-violating fixture; failure must crash with a clear message.

### Phase 5 — CLI (≤80 LOC including help text)

Two new subcommands. Thin wrappers. ONE integration test using a tiny stub question to exercise stage chain end-to-end.

Gate: `stage preplanning ... && stage investigation --from-db ... && ...` produces equivalent answer to single `ask` invocation.

### Phase 6 — Use it for the lazy-escalation validation we owe

Run preplanning + investigation + scrutiny + verification + integration on a real question, save DB at each step. Then run **only synthesis** on a DB where `combined_verdict` is None. Watch `[synthesis_demand]` fire. Watch the loop-back route to Scrutinize. Total time: <60 s, not 20 min.

This phase is the proof the whole plan paid off. If we can't quickly verify Phase 4 of lazy-escalation here, the design failed.

Gate: Phase 4 lazy-escalation hot-paths empirically validated in <2 min wall-clock total across all stages.

---

## Open decisions

### 1. Where exactly do stage boundaries land?

Phase 0 is explicit empirical discovery. The candidate set is obvious (preplanning / investigation / scrutiny / verification / integration / synthesis), but the precise *node* for each boundary needs the trace data.

### 2. Do we expose `stages.py` from Python or only via CLI?

**CLI only.** A Python API would invite people to import it from random places, accreting coupling. CLI is one entry point; the boundary stays clear. If someone wants to script, they call the CLI from a shell.

### 3. What if a node in a stage's interior writes state read by a later stage?

That node is in the wrong stage. The invariant catches this — if state X is needed by stage Y but stage X-1 didn't write it, Y's entry-side invariant (which we get from stage X's exit invariant) will fail loudly.

### 4. Looping stages (investigation ↔ scrutiny)?

Each stage's `exit_after` must be a node that terminates the loop, not one inside it. For investigation that's the node where we route to scrutiny; for scrutiny that's where we route to AbandonOrDemote/PromoteToSupported. Phase 0 confirms this empirically.

### 5. Replay/caching of LLM calls?

**Out of scope.** With stage runners, the dev loop becomes "iterate on stage X against frozen pre-X DB." If stage X is fast (< 2 min), we don't need replay. If it's still too slow after this lands, that's a separate plan and we'll know exactly which stage is the bottleneck because of `timing.txt`.

---

## Acceptance criteria for the whole effort

1. Re-running just `synthesis` on a saved DB completes in <30 s with `[synthesis_demand]` log line visible.
2. Re-running just `scrutiny` on a saved DB completes in <60 s.
3. Total new code ≤ 300 LOC. Total new files: 1 + 2 CLI handlers.
4. `inspect <DB>` produces a structured state report I can paste into a PR description.
5. The lazy-escalation Phase 4 loop-back behavior is empirically demonstrated in a CI test that runs in <2 min total wall-clock.
6. Pyright clean, ruff clean, all 1848+ existing tests still pass.

---

## What this plan does NOT cover

- Parallelizing evidence extraction (separate plan; informed by `timing.txt` data this work produces)
- LLM response replay (only if stage runtimes still block iteration after this lands)
- Refactoring node classes (the graph is unchanged; this work is purely additive at the runner level)
- Persisting demand chains (out of scope per lazy-escalation plan; same answer here)

If any of these become necessary mid-implementation, it's a sign the stage boundaries aren't tight enough — fix the boundary, don't expand the scope.

---

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Stage boundary chosen at non-quiescent point | Medium | High | Phase 0 empirical discovery + Phase 4 invariants |
| `stages.py` accretes more than 6 entries over time | Medium | Medium | Anti-bloat budget; PR review |
| Observability artifacts diverge from real run | Low | Medium | Single log handler, no per-node instrumentation |
| `start_at`/`stop_after` interact with existing `quick`/`decompose` flags | Low | Medium | Phase 1+2 tests cover the matrix |
| Stages encourage skipping the full pipeline in CI | Medium | Medium | One CI job runs full `ask` end-to-end at lower frequency |

---

## Why this is publishable architecture

Three properties this design preserves:

1. **One graph.** The pipeline structure visible in a paper diagram is exactly what runs. Stages are *labels* on subsets of the graph, not a parallel control flow.
2. **One state.** The DB IS the run. Stages don't carry side state. Inspection is "read the DB"; no opaque artifacts.
3. **One observability surface.** `run.jsonl` + `diff.json` + `timing.txt` is what a reader of the paper would expect to see — a structured trace, a state delta, and a profile. These artifacts ARE the empirical reproducibility material.

If we cannot publish a system that produces three plain text files per run alongside the DB, then we've over-engineered something that should have been one log handler.
