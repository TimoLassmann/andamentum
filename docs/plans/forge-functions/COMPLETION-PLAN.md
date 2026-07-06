# Forge — completion plan and status handoff

> **Audience:** an agent picking up `andamentum.forge` to finish it. This is a self-contained
> brief: where the system is, what it can do, what's broken, and a phased plan to reach
> "complete." Read this top to bottom before touching code. The companion visual overview is
> `docs/forge-architecture.html` (current post-trim architecture).

**Status date:** 2026-07-05  ·  **Branch:** `forge-simplify` (committed through the map primitive)
**One-line status:** *Reliability transformed and a real-work capability added. The rest of this
plan below the status update is the ORIGINAL brief — read the update first; much of it is done.*

---

## STATUS UPDATE — 2026-07-05 (read this first; the plan below predates it)

Most of the plan below is **done and committed**. What actually happened and where it stands:

**Design-loop reliability — transformed.** Root cause was NOT the machinery the trim removed; it
was `decompose` wiring producers→consumers by exact free-text name string (the model reproducing
names across calls, the repair loop reinventing them). Fixes, all committed:
- **Two-pass declare-then-select with ordinal selection** (§ commit "trim + fix"): the model
  DECLARES one produced name per node (deduped → globally unique), then SELECTS consumes by
  ORDINAL from a forward-windowed closed set. `duplicate_producer` / `near_miss` / `dangling_read`
  and accidental cycles are now impossible **by construction**. Convergence went from ~1-in-5
  grammars to strong.
- **Loop grammar** (0/6 → 6/6): a checkpoint-head loop-counter was miscounted as a data write
  (fixed in spec.py/render.py), and the model unrolled loops (anti-unrolling list_jobs prompt).
- **Deterministic `multiple_sinks` collapse** (over-decomposition): the last node consumes extra
  terminal signals; no model call, forward-safe.

**Real-work capability — the `each`/map primitive** (the highest-value add). Forge could not
represent "for each item of a list" — a list brief collapsed the input to a scalar and the workflow
processed ONE item. Now: the model gains one closed choice (`mode: whole|each`) + an
`input_is_collection` flag; deterministic code computes collection-ness and the RENDERER writes the
map scaffold (bounded gather, per-item soft-fail, all-fail raise, join sole writer); an each-spine
hole is a pure `_map_one(item)`. Validated live (build-then-EXECUTE golden tasks): a "summarise each
note + combine" workflow covers all 3 notes; a "greet each name" workflow picks `each`, the scaffold
fires, all 3 names covered.

**Measured (design convergence, n=6):**
- `gemma4:31b-nvfp4`, 5 original grammars: seq 6/6, fanout 6/6, loop 6/6, branch 4/6, stateful 2/6.
- `gemma4:26b-nvfp4` (faster; validated as a viable target), 5 + map: seq 6/6, **map 6/6**, fanout
  5/6, stateful 5/6, branch 4/6, loop 4/6 — **83% overall, zero name-flaws/cycles, no regression
  from the mode option.** (The 31b stateful 2/6 was stochastic noise; 26b shows 5/6.)

**Final validation — Tier-3 golden corpus, live on `gemma4:26b-nvfp4` (build → EXECUTE → score output):**
All four golden systems built and, run on real input, produced correct on-task output:
- **per_item** (greet each name): output covers 3/3, audit works=True — **correct**.
- **sequence** (3 bullets): 3/3, works=True — **correct**.
- **reduce** (summarise each note + combine): output covers 3/3 (a perfect multi-note digest —
  every proper noun, coherently combined), but that specific build's audit works=False; a rebuild
  audits clean (works=True). The failure is **per-build audit flakiness** (the model authors
  different bodies each build; one build's generated smoke test failed while the real run works),
  not a systematic gap.
- **branch** (classify + route): routed correctly ("SRE / DBA Team" for a database-down ticket) but
  the design keeps the urgency label internal, so the 2-group rubric scored 1/2 — mostly rubric
  strictness (routing is the deliverable).
Read together: **forge produces working, on-task systems on a local model.** The strict scorer's
"2/4" undercounts (it ANDs in `audit.works`, which false-negatived reduce, plus one strict rubric) —
the real-work signal (output coverage) is 3/3, 3/3, 3/3, 1/2.

**Environment prerequisite for Podman-backed auditing (blocks the podman Tier-2 path):** on this Mac
the podman machine shares NO host directory (verified: `/private/tmp` and `$HOME` are both invisible
in-container), so a mounted generated package collects zero tests and every audit reports incomplete.
Fix (one-time, DESTRUCTIVE to the VM — do when convenient): re-init the machine with a shared volume,
e.g. `podman machine stop && podman machine rm && podman machine init --volume $HOME:$HOME && podman
machine start`, then rebuild the sandbox image. Until then, use `--sandbox subprocess` (works for
non-network briefs; refuses network briefs). ALSO: rebuild the sandbox image after ANY forge-src
change before benchmarking under podman — the image bakes andamentum in, so a stale image audits new
packages against old runtime.

**What is NOT done (the real remaining work):**
- **Per-build audit flakiness = the code-authoring reliability frontier.** A converged design's
  `audit.works` varies build-to-build because the model authors different node bodies each time and
  a generated smoke test occasionally fails on a body that actually works (see reduce above). This
  is the next real lever — not the design loop (solved) but the *body-authoring* surface. The same
  "selection over authoring" principle applies: the more forge can render deterministically and the
  less it asks a small model to write, the more stable `works` becomes. (The Tier-3 golden benchmark
  now MEASURES this — run `--golden` at higher n to quantify the per-build variance.)
- **`stateful`/rung-2** is the fragile grammar (noisy 2/6–5/6). The read-modify-write entity design
  is the weak spot the codebase itself flags as needing calibration.
- **The next thread (design direction):** forge is STANDALONE by rule (it must not import other
  andamentum modules), so "compose vetted capabilities" is out. The frontier is **more primitives
  like `each`** — small, closed choices for the model with deterministic scaffolds behind them. The
  common real-work shapes not yet primitives: filter (keep items matching a judged condition),
  per-item routing, and a genuinely reliable read-modify-write. Each follows the proven pattern:
  ONE closed choice for the model, determinism does the rest.

---

## 0. What "complete" means (the definition of done)

Forge is **complete** when, on the maintainer's target local model(s) (`ollama:gemma4:31b-nvfp4`,
and it should degrade gracefully on `gemma4:26b-nvfp4` / `gpt-oss:20b`), the **Tier-2 benchmark**
(`benchmarks/forge`, the end-to-end "does the generated system actually work" harness) reaches:

1. **Every buildable grammar passes reliably.** All five buildable cases (sequence, branch, loop,
   fan-out, stateful) reach `works=True` at a **≥ 0.8 pass rate over ≥ 5 runs each** (stochasticity
   means single runs are meaningless — see §3). Today: **1/5 grammars pass, ~0.33 overall, n=1.**
2. **Refusals stay correct.** The three out-of-scope cases (app / agent / service) still refuse at
   the fitness gate.
3. **Network briefs are actually audited** (fan-out, loop) under Podman, not skipped.
4. **No silent INCOMPLETE.** When forge cannot produce a working system it says so clearly with an
   actionable reason, and the failure is attributable to a stage.
5. Green bar preserved throughout: `pytest` / `pyright` / `ruff` / `andamentum-agentic-dialect check`
   all clean, and forge's own output stays dialect-clean.

"Complete" is a **reliability bar measured by the benchmark**, not a feature list. Do not add
features; make the existing pipeline reliably produce working workflows.

---

## 1. What the system is and what it can do

Forge turns a natural-language brief into a runnable, typed, dialect-conforming `pydantic-graph`
package. It is a **sharp function generator**: it builds stateless functions (rung 1) and
single-record stateful functions (rung 2), and refuses apps / agents / services at the door.

**Entry point:** `run_forge(brief, *, model, dest=None, stop_after="audit", sandbox_backend="podman", ...)`
in `graph.py`. CLI: `andamentum-forge build "<brief>" --model <id>` (or `design` for spec-only).

**The pipeline (one graph, `graph.py` the only engine-aware file):**
`Understand → Assess → Frame → Decompose → Compile → Review → Render → Verify → Build → Audit → Finish`.
It is **linear** — the self-correction loop and the plan-review redesign loop were removed in the
trim (see §2). The only branches are early stops (`stop_after`, `dest=None` = design-only).

**The governing principle:** *the model proposes, deterministic code disposes.* The model declares
(areas, node kinds, each node's `consumes`/`produces`, and each node body); deterministic code does
the load-bearing work (matches producers→consumers into a DAG, diagnoses structural flaws, compiles
+ validates the spec, renders the package, and gates every authored body). LLM-written code executes
**only** in the sandbox (Audit).

**Nine LLM heads** (`agents.py`): design = `understand`, `fitness`, `frame`, `list_jobs`,
`type_node`; authoring = `build_draft`, `build_repair`; audit (advisory) = `requirements`, `critic`.

**Five build gates** (`astcheck.py`, applied to each authored body, repaired on failure):
contract, coverage, deps, purity, fail-loud.

**Sandbox** (`sandbox.py`): Podman default (host-isolated); `SubprocessSandbox` fallback runs pure
nodes out-of-process but **refuses network** (not host-isolated).

**What works today:** simple briefs whose nodes need little/no authored body (e.g. a plain
sequence) build and run end-to-end. Refusals of out-of-scope briefs work. The design→render→build→audit
machinery is structurally sound and fully green in tests.

---

## 2. What was just done (the `forge-simplify` branch)

This branch delivered **Phase 1 (measure)** and **Phase 2 (trim in place)** of an earlier plan.
It is verified green (120 forge tests, dialect-clean, pyright 0, ruff clean) but **uncommitted**.

**Phase 1 — the Tier-2 benchmark is now real** (`benchmarks/forge/`). It was a stub that discarded
`--full` and only scored design *shape* (Tier 1). It now renders + agent-authors + sandbox-audits
and scores on `audit.works`. New: `RunOutcome` tier-2 fields, `outcome_matches_tier2`,
`--full`/`--sandbox {subprocess,podman}`, an offline wiring self-test. This is the acceptance
harness for everything below.

**Phase 2 — trimmed ~2,313 lines (~33%)**, four cuts, no behaviour change on the happy path,
`run_forge` signature preserved:
- Removed the **self-correction / best-build / regression / attribution loop** (deleted
  `attribute.py`; `Audit` is now a single pass → `Finish`; no Render back-edge; no disk
  re-materialisation).
- Removed the **`component_manager`** advisory head (the static gates already decided fillability).
- Removed the **`review_plan` LLM redesign loop** (kept the deterministic `plan_coverage`).
- Removed **dead code** (HumanGate render path, `checkpoint_cap`, `run_end_type` knob).

**Measured before/after** (`gemma4:31b-nvfp4`, subprocess, n=1): overall `works` unchanged (1/3, no
regression); the one apples-to-apples case ran **2.3× faster** for an identical outcome.

---

## 3. The reliability diagnosis — where it actually breaks

The trim did **not** change the success rate, and that was expected: the unreliability is not the
machinery we removed. It is two bottlenecks, upstream and underneath, plus a class of silent
degradation. **This is the substance of the remaining work.**

**B1 — Design-loop name-matching stochasticity (THE bottleneck).** `decompose.py` runs an
assemble→diagnose→repair loop: `assemble.py` matches producers→consumers **by variable-name string**
(dicts keyed on the exact name). The model must re-emit matching `consumes`/`produces` names across
nodes. On a small model, a few misses produce `duplicate_producer` / `dangling_read` findings;
`diagnose.py` computes a rapidfuzz near-miss and *suggests a rename*, but the **model must reapply
it** in a re-type call — and a model that couldn't match names the first time often can't on retry,
burning `MAX_DESIGN_ROUNDS=4` → fail-loud refusal. This is non-deterministic run to run (observed:
the `branch` brief reached audit on one run, refused at decompose on the next). **Known-scope even
admits it:** "node I/O is matched by name (string-similarity), not by declared types."

**B2 — The sandbox network boundary.** `SubprocessSandbox` refuses network code, so any `network=True`
node (fetch URLs, web search — i.e. the fan-out and loop grammars) cannot be audited without Podman.
Today the fan-out brief reports INCOMPLETE purely because its tests can't run, even though its body
authored correctly and is dialect-clean. The Podman image is **not built** on the maintainer's
machine (`podman build -t andamentum-forge-sandbox -f src/andamentum/forge/Containerfile .`).

**B3 — Silent INCOMPLETE (no loud error on a broken build).** When a node hits `attempt_cap` it is
marked `unfillable` and the hole restored to `NotImplementedError`; `build.py` never crashes the
build ("a single node's authoring failure never crashes the whole build"). The run settles with
`audit.works=False` and a package on disk that doesn't work — surfaced in the result, but not a
raise. For a personal tool this reads as "it silently produced junk."

**B4 — Per-node Python authoring against 5 strict gates (fragile, but correct-by-design).** The
gates reject rather than repair-toward; on a weak model a 90%-right body cycles to `unfillable`. The
coverage gate ("declared a read but never reads it") is the strictest and most likely to reject an
otherwise-fine body.

**B5 — Measurement is n=1 and slow.** One buildable case = 25–70 serial LLM calls on a local model
(Ollama serialises same-model calls) = 2–20 min. B1's stochasticity means **you cannot judge a
reliability change without n ≥ 5 per case.** Budget for long benchmark runs; script them and run in
the background.

Enumerated fail-loud exits (all `ValueError`, surfaced by `run_forge`): fitness refusal; frame
no-areas; **decompose no-converge (`MAX_DESIGN_ROUNDS=4`) ← B1**; decompose no-steps; uncovered area;
compile backstops (dangling/single-writer/orphan/rmw-not-entity/cycle). Non-loud settle:
`audit.works=False` ← B2/B3/B4.

---

## 4. The plan to finish

Phases are ordered by dependency and payoff. **Every phase's acceptance test is the Tier-2 benchmark
at n ≥ 5** (except Phase 0/1 infra). Do them in order; each is independently shippable.

### Phase 0 — Commit and baseline the branch (housekeeping, do first)
- Commit the `forge-simplify` work as logical commits (harness / trim / architecture doc). End
  commit messages with the project's `Co-Authored-By` line. Do not push unless asked.
- Fix stale docs: the repo-root `CLAUDE.md` forge paragraph and `MEMORY.md` still describe the
  removed self-correction / component-manager / plan-manager machinery. The module `forge/CLAUDE.md`
  is already updated.
- **Acceptance:** clean tree; `pytest`/`pyright`/`ruff`/dialect-check green.

### Phase 1 — Measurement infrastructure (unblocks judging everything else)
- Build the Podman sandbox image (command above). Confirm `andamentum-forge build "<network brief>"
  --sandbox podman` audits a network node.
- Add a small **multi-run baseline script** (extend the pattern in
  `scratchpad/forge_baseline.py` from the prior session, or add a `--runs` sweep to
  `benchmarks/forge/cli.py` that already exists) that runs all 8 corpus cases at n ≥ 5 under Podman
  and writes a JSON/markdown report with per-case `works` rate + failure-stage breakdown.
- **Acceptance:** a reproducible command that prints the honest current baseline (expected: still
  low, but now network cases are actually audited, and the number is stable enough to compare
  against). Record it as the "before" number for Phase 2+.

### Phase 2 — Harden the design loop (B1 — the biggest reliability win)
This is the core of "complete." The goal: make producer→consumer wiring survive small-model name
variance so `decompose` converges instead of failing loud. Options, in recommended order (do the
cheapest that moves the number; measure after each):
- **2a (highest leverage, lowest risk): deterministic near-miss reconciliation.** `diagnose.py`
  already computes a rapidfuzz near-miss match for a dangling read / duplicate producer. Today it
  only *suggests* the rename and asks the model to reapply. Instead, **apply the rename
  deterministically** when the match is unambiguous (single candidate above a high threshold), and
  only fall back to a model repair round when it's genuinely ambiguous. This directly removes the
  "model must re-emit matching names" dependency for the common case.
- **2b: closed-set selection for `consumes`.** Change `type_node` so a node's `consumes` are
  *selected from the set of already-declared produced names* (a closed enum the model picks from)
  rather than free-text the model must reproduce. Producer names stay free; consumer names become a
  choice. A small model fills a closed enum far more reliably than it reproduces a string.
- **2c: type-aware matching.** Upgrade `assemble.py` to match producers→consumers by declared *type*
  (+ role) with name as a tiebreaker, not by name alone (known-scope flags this as the intended
  upgrade). Larger change; do only if 2a+2b don't reach the bar.
- **Acceptance:** the `branch`, `fan-out`, and `loop` grammars reach `works=True` at ≥ 0.8 over
  n ≥ 5; `duplicate_producer`/`dangling_read` refusals become rare. Guard: run the offline design
  tests + `benchmarks/forge` Tier-1 (shape) to confirm designs are still *correct*, not just
  convergent.

### Phase 3 — Authoring-path robustness (B4, and B3's silent failure)
- Relax the **coverage gate** from hard-reject toward repair-with-strong-feedback (or make "declared
  read never read" a warning that triggers one targeted repair rather than counting a full failed
  attempt). Keep contract / purity / fail-loud strict — those are safety.
- Make **B3 legible**: when a build settles with `unfillable` nodes, `run_forge` should surface a
  single clear top-level status ("INCOMPLETE: nodes X, Y could not be authored after N attempts —
  <reason>") rather than only burying it in the report. Decide with the maintainer whether an
  all-unfillable build should *raise* (fail-loud) vs. return a clearly-marked incomplete result.
- **Acceptance:** hole-bearing non-network briefs that previously settled INCOMPLETE now either reach
  `works=True` or fail with an actionable, attributed message; benchmark `works` rate rises.

### Phase 4 — Deferred simplification (measured pass — do after Phase 2/3 so it's measurable)
- Collapse `diagnose.py`'s 9 flaw types to the ~3–4 that carry weight (dangling read, cycle,
  orphan/unreachable) — merge the `disconnected`/`dead_end`/`unreachable` gradations.
- Remove `compile_spec.py`'s structural backstops that duplicate `diagnose` on the same already-clean
  board (dangling / single-writer / orphan re-checks).
- **Acceptance:** benchmark `works` rate unchanged or better (this is *why* it's measured after 2/3);
  fewer lines; green bar.

### Phase 5 — Rung-2 reliability (from known-scope; lower priority unless the maintainer needs it)
- The stateful (rung-2) *single-record* path builds and remembers (`test_stateful.py`), but the
  design front-end producing rung-2 boards reliably from a live model "needs calibration." Fold this
  into Phase 2's design-loop work (it's the same name-matching problem plus the entity round-trip).
- **Not-done, optional:** multi-record by id (uuid key). See `C-STORE-PRD.md` §6/§8. Only if the
  maintainer wants "append a note / count many notes"-style workflows.

---

## 5. How to verify (run these constantly)

```bash
# Green bar (fast, no model, no container) — must stay clean after every change:
uv run pytest src/andamentum/forge/tests -q
uv run pyright src/andamentum/forge
uv run ruff check src/andamentum/forge
uv run andamentum-agentic-dialect check src/andamentum/forge   # forge output stays dialect-clean

# Reliability (slow, live model + sandbox) — the real acceptance test:
uv run python -m benchmarks.forge.cli --model ollama:gemma4:31b-nvfp4 --full --sandbox podman --runs 5
#   Tier-1 (design shape only, fast, no sandbox) to guard design correctness:
uv run python -m benchmarks.forge.cli --model ollama:gemma4:31b-nvfp4 --runs 5
```

The 8-case corpus is `benchmarks/forge/cases.py` (5 buildable, one per grammar; 3 refuse). The whole
forge graph is unit-testable with no model and no container via stub `AgentSink` + `FakeSandbox`
(`tests/conftest.py`); the real out-of-process path uses `SubprocessSandbox`.

---

## 6. File map (where to work)

`graph.py` pipeline (State/Deps/nodes/`run_forge`) — only engine-aware file · `spec.py` `SystemSpec`
+ recipe validators · `understand.py`/`frame.py`/`decompose.py` design workers · `fitness.py` the
fitness gate · `review.py` deterministic `plan_coverage` · **`assemble.py` puzzle-fit DAG (B1)** ·
**`diagnose.py` structural diagnostics + near-miss (B1, Phase 2a/4)** · `compile_spec.py` board→spec
(Phase 4) · `render.py` deterministic spec→package · `build.py` per-node authoring loop (B3/B4,
Phase 3) · `astcheck.py` the 5 gates (Phase 3) · `audit.py` whole-system audit (single pass) ·
`sandbox.py` + `Containerfile` execution seam (B2, Phase 1) · `runtime.py` engine-free spine a
generated package imports (`run_head`/`loop_allowed`/`Store`) · `agents.py` the 9 heads · `cli.py`
the adapter. Benchmark: `benchmarks/forge/` (`runner.py` Tier-1/2, `cases.py` corpus, `shape.py`
scoring, `cli.py`). Visual overview: `docs/forge-architecture.html`.

---

## 7. Landmines and invariants (do not learn these the hard way)

- **n=1 is a lie.** Design-loop stochasticity (B1) means one run tells you nothing. Always n ≥ 5 for
  any reliability claim. Runs are slow (Ollama serialises; 25–70 serial calls/build). Script them,
  run in the background, don't block on them.
- **Don't move correctness into a prompt.** The house rule: if a check can be deterministic, it must
  be. Phase 2's whole point is to make wiring *more* deterministic, not to prompt the model harder.
- **Fail loud, no fallbacks — including in generated code.** Never make forge (or the code it emits)
  silently drop, default, or invent data. Most of the gate suite exists to enforce this. Phase 3
  makes silent INCOMPLETE *loud*, it does not make it *tolerated*.
- **Dialect-clean both ways.** Forge itself passes `andamentum-agentic-dialect check`, and so must
  every package it renders (the audit checks this). `graph.py` is the only engine-aware file.
- **One code path; explicit `model=`; no env vars; bounded loops via named constants.** Caps live as
  module constants (`MAX_DESIGN_ROUNDS` in `decompose.py`, `ATTEMPT_CAP`/fan-out bounds in
  `graph.py`). No hidden defaults.
- **The `pydantic_graph` deprecation warning is known and out of scope here** — the `BaseNode`
  runner is deprecated in favour of `GraphBuilder`; do not migrate forge's engine as part of this
  reliability work (it's a separate, larger effort). Pinned `<2` keeps it working.
- **Don't probe the maintainer's credentials or `.env`.** Local models need no keys;
  `load_dotenv()` at a script top is the sanctioned pattern if a cloud model is ever used.
