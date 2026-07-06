# Forge — the Tier-2 reliability push

> **Audience:** the maintainer + an agent executing the push. This is the *action plan* to move
> `andamentum.forge` from "produces working systems on a good day" to *clears the Tier-2 bar*. It
> supersedes the forward-looking half of `COMPLETION-PLAN.md` (whose §3 diagnosis is partly stale:
> B1, the design-loop name-matching bottleneck, is **solved** by the two-pass declare-then-select
> with ordinal selection). Read the STATUS UPDATE at the top of `COMPLETION-PLAN.md` first; this
> doc is the plan *from there forward*.
>
> **Status date:** 2026-07-06 · **Governing rule:** *the model proposes, deterministic code
> disposes.* Every fix below prefers a deterministic check over a harder prompt.

---

## 0. The bar (definition of done — unchanged)

On the maintainer's local targets (`ollama:gemma4:31b-nvfp4` primary; degrade gracefully on
`gemma4:26b-nvfp4` / `gpt-oss:20b`), the **Tier-2 benchmark** (`benchmarks/forge/`, build →
render → author → sandbox-audit → score on `audit.works`) reaches:

1. **Every buildable grammar `works=True` at ≥0.8 over ≥5 runs.** (sequence, branch, loop, fan-out,
   stateful, and now map/`each`.)
2. **Refusals stay correct** — the 3 out-of-scope briefs (app / agent / service) still refuse at the
   fitness gate.
3. **Network briefs are actually audited** under Podman (fan-out, loop), not skipped.
4. **No silent INCOMPLETE** — a build that cannot produce a working system says so loudly, attributed
   to a stage.
5. **Green bar throughout** — `pytest` / `pyright` / `ruff` / `andamentum-agentic-dialect check`
   clean, and forge's own output stays dialect-clean.

"Complete" is a *reliability number measured by the benchmark*, not a feature list.

---

## 1. Current baseline (last measured, n=6, `gemma4:26b-nvfp4`)

| Grammar   | Brief shape                                   | Rate      | At 0.8? |
|-----------|-----------------------------------------------|-----------|---------|
| sequence  | summarise → 3 bullets                         | 6/6       | ✅ |
| map/`each`| summarise each item + combine                 | 6/6       | ✅ |
| fan-out   | per-item over a list                          | 5/6       | ✅ (edge) |
| stateful  | reading-list + message → updated list         | 5/6 (**2/6 on 31b**) | ❌ noisy |
| branch    | classify urgency → route to team              | 4/6       | ❌ |
| loop      | research until evidence sufficient            | 4/6       | ❌ |

**The three grammars below the bar are `stateful`, `branch`, `loop`.** `stateful` is *the* fragile
one (the read-modify-write entity round-trip). These numbers are n=6 and single-model — **the first
deliverable is an honest n≥5 baseline across both models under a working Podman** (WS-A), because
per the landmine "n=1 is a lie" we cannot judge any fix without it.

---

## 2. The two levers (what actually moves the number)

The unreliability is no longer the design-loop *name matching* (solved). It is two distinct surfaces:

- **Lever 1 — Design convergence (does the board become a valid DAG?).** Small models make
  structurally-doomed *choices* the repair loop cannot fix, because repair can only re-select
  `consumes`. The proven remedy is a **deterministic pre-repair pass** that detects the doomed choice
  and fixes it *before* the model repair round. Three already ship — `multiple_sinks` collapse,
  `demote_orphan_entities` (the stateful terminal-answer-as-entity wedge), and
  `each`-without-collection demotion. The fragile grammars need *more of these*, each targeting a
  specific captured failure board.

- **Lever 2 — Body authoring (does a converged design's audit pass build-to-build?).** A converged
  design's `audit.works` varies run to run because the model authors different node bodies each build
  and a generated smoke test occasionally fails on a body that *actually works* (the fan-out/reduce
  case: perfect output, `works=False`). This is the newer frontier. The same "selection over
  authoring" principle applies — render more, ask the small model to write less — plus gate
  relaxation and loud-INCOMPLETE.

Everything below is one of these two levers, plus the measurement infra that lets us see them.

---

## 3. Workstreams

Ordered by dependency and payoff. **Every WS's acceptance test is the Tier-2 benchmark at n≥5.**

### WS-A — Measurement infra (unblocks judging everything else) — DO FIRST

- **A1. Fix the Podman audit path.** The image is built; the blocker is the VM shares no host
  directory, so `sandbox.py`'s `-v {pkg}:{pkg}:ro` mounts an empty dir → zero tests collected →
  every network audit reports incomplete. Decision pending (see §5) between: (a) **copy-in transport**
  — rewrite `sandbox.py` to stream the package into the container via `tar`-over-stdin, needing *no*
  host mount (portable, non-destructive, strictly better isolation since the host fs is never
  exposed); (b) destructive VM re-init with default `/Users`+`/private` volumes; (c) a dedicated
  second forge machine. **Acceptance:** `andamentum-forge build "<network brief>" --sandbox podman`
  audits a `network=True` node and collects >0 tests.
- **A2. Multi-run baseline harness.** Extend `benchmarks/forge/cli.py` with a `--runs N` sweep over
  all 8 corpus cases that writes a JSON + markdown report: per-case `works` rate + failure-stage
  breakdown (which of fitness / frame / decompose-no-converge / audit-fail). Background it (25–70
  serial Ollama calls/build). **Acceptance:** one reproducible command prints the honest current
  per-grammar rate on 31b and 26b, with network cases actually audited. Record as the "before".

### WS-B — Deterministic design hardening (Lever 1; the fragile grammars) — HIGHEST PAYOFF

The method for each fragile grammar: **capture the failing board** (the benchmark already can dump
the spec/board on a decompose refusal), **classify the doomed choice**, add a **deterministic
pre-repair pass** in `diagnose.py`/`assemble.py` that runs before the model repair round, and prove
it with an **offline test** that reproduces the captured board and shows the pass fixes it *while
protected shapes stay untouched* (the pattern `demote_orphan_entities` already follows).

- **B1. `stateful` (rung-2) — the read-modify-write round-trip.** `demote_orphan_entities` fixed the
  terminal-answer-as-entity wedge; the residual failures are the *entity round-trip* itself (the
  producer that must read AND write the same entity). Hypotheses to capture-and-harden:
  - the model emits an entity nothing self-consumes *and* no downstream reads, but it *is* meant to
    be durable (mislabelled the other way from the orphan case) — needs a promote/keep signal.
  - the read and the write land on differently-named variables for the same logical record — a
    type/role-aware reconciliation (`assemble.py` matching by declared type + role, not name).
  - **Consider a `stateful`-specific closed choice:** one flag "this node updates a stored record"
    that the renderer turns into the read-modify-write scaffold deterministically (same shape as
    `each`: one closed choice for the model, scaffold behind it). This is the single highest-value
    item — it converts the fragile grammar into a rendered primitive.
- **B2. `branch` — classify + route.** Two things: (i) the design keeps the routed label *internal*
  so the rubric under-scores it — decide whether that's a real gap or rubric strictness (inspect the
  captured branch board; if routing is genuinely the deliverable, tighten the *rubric*, not the
  system). (ii) any residual decompose wedge on the branch shape gets its own deterministic pass.
- **B3. `loop` — bounded research loop.** 0/6 → 6/6 on 31b after the checkpoint-head fix, but noisy
  (4/6) on 26b. Capture the 26b failures: likely the loop-counter/checkpoint binding under a weaker
  model, or the network-audit gap (fixed by WS-A). Re-measure after A1 before adding code.
- **B4. Type-aware matching (only if B1–B3 residuals demand it).** Upgrade `assemble.py` to match
  producers→consumers by declared *type* (+role) with name as tiebreaker — the known-scope's intended
  upgrade. Larger change; do only if the cheaper per-grammar passes don't reach 0.8.

**Acceptance:** `stateful`, `branch`, `loop` each reach `works=True` ≥0.8 over n≥5;
`duplicate_producer` / `dangling_read` / `orphan_output` refusals become rare. **Guard:** the offline
design tests + Tier-1 (shape) benchmark must confirm designs are still *correct*, not merely
convergent (a demotion that makes a wrong board is worse than a loud refusal).

### WS-C — Body-authoring robustness (Lever 2; B3/B4 from the old plan)

- **C1. Relax the coverage gate.** In `astcheck.py`, "declared a read but never reads it" is the
  strictest gate and cycles 90%-right bodies to `unfillable`. Turn it into a *warning that triggers
  one targeted repair with strong feedback* rather than a counted full-attempt failure. **Keep
  contract / purity / fail-loud strict — those are safety, not style.**
- **C2. Make silent INCOMPLETE loud.** When a build settles with `unfillable` nodes, `run_forge`
  surfaces a single top-level status: `INCOMPLETE: nodes X, Y could not be authored after N attempts
  — <reason>`. Decide with the maintainer whether an all-unfillable build should *raise* vs. return a
  clearly-marked incomplete result. (No silent junk — the house rule.)
- **C3. Selection-over-authoring for bodies (stretch).** Identify body shapes forge can render
  deterministically instead of asking the model to write (e.g. the map join, the loop checkpoint, the
  rmw read/write) so `works` stops depending on per-build authoring variance. This is the durable fix
  for the fan-out/reduce flakiness.

**Acceptance:** hole-bearing non-network briefs that previously settled INCOMPLETE now either reach
`works=True` or fail with an actionable attributed message; the per-build variance on a *converged*
design (measured by re-running the same brief n≥5) drops.

### WS-C' — Layering cleanup: lazy `forge/__init__.py` (found during the podman fix)

A generated *runtime* package imports `andamentum.forge.runtime`, which runs
`andamentum/forge/__init__.py` — today an **eager** re-export that pulls the whole
meta-system **builder** (audit → agents → pydantic-ai; diagnose / schemas → rapidfuzz) plus
the runtime. So the sandbox image must carry builder-only deps (`pydantic-ai`, `rapidfuzz`)
purely to let the generated package *import*. Making `__init__` lazy would let a generated
artifact pull only `runtime → core` and shrink the image.

**Caveat (why it wasn't done inline):** a naive PEP 562 `__getattr__` lazy `__init__` breaks
on the three export names that **shadow submodules** (`compile_spec`, `render`, `graph`) —
once the builder imports the submodule, it binds `forge.<name>` to the *module*, so
`from andamentum.forge import compile_spec` returns a module and `__getattr__` is never
consulted (observed: 26 test failures, `'module' object is not callable`). A correct lazy
scheme must resolve those collisions (rename the internal submodules, or a
`__getattr__` that also rebinds/overwrites the shadowed attribute). Not a blocker — the
image now hand-picks the deps — so this is a **measured simplification**, not urgent.
**Acceptance:** `import andamentum.forge.runtime` pulls no builder module; the image drops
`pydantic-ai`/`rapidfuzz` if nothing else needs them; green bar + full forge suite unchanged.

### WS-D — Deferred simplification (measured; do AFTER B/C so it's measurable)

- Collapse `diagnose.py`'s 9 flaw types to the ~3–4 that carry weight (dangling read, cycle,
  orphan/unreachable); merge the `disconnected`/`dead_end`/`unreachable` gradations.
- Remove `compile_spec.py` structural backstops that duplicate `diagnose` on an already-clean board.
- **Acceptance:** benchmark `works` unchanged or better (that's *why* it's measured after B/C); fewer
  lines; green bar.

---

## 4. Sequencing & milestones

1. **M0 — Baseline (WS-A).** Podman path works; n≥5 report on 31b+26b committed as the "before".
   *Gate: we can measure.*
2. **M1 — Stateful primitive (WS-B1).** The rmw closed-choice + scaffold; stateful ≥0.8. *Biggest
   single win.*
3. **M2 — Branch + loop (WS-B2/B3).** Both ≥0.8; refusals rare. *All grammars converge.*
4. **M3 — Authoring stability (WS-C).** Coverage-gate relaxation + loud INCOMPLETE; converged-design
   variance down. *`works` stops flapping.*
5. **M4 — Simplify (WS-D) + full corpus green at n≥5.** *The bar.*

Each milestone is independently shippable and gated by the benchmark. Do not advance on n=1.

## 5. Immediate decision (blocking WS-A1)

The Podman fix has three paths with very different blast radius — see the question posed alongside
this plan. Recommendation: **copy-in transport** (rewrite `sandbox.py` to `tar` the package into the
container over stdin). It needs no VM change, works on any podman host, and is strictly *more*
isolated (the host filesystem is never mounted). The documented re-init is faster but destroys a
data-rich VM (bioinformatics containers, the `searxng` image `deep_research` uses, 24 named volumes).

## 6. Landmines (do not relearn the hard way)

- **n=1 is a lie.** Every reliability claim needs n≥5. Runs are slow (Ollama serialises; 25–70 serial
  calls/build). Script them; background them; don't block.
- **Don't move correctness into a prompt.** If a check can be deterministic, it must be. WS-B's whole
  point is *more* determinism, not harder prompting.
- **A demotion that makes a *wrong* board is worse than a loud refusal.** Every deterministic pass in
  WS-B must be guarded by the Tier-1 shape test — protected shapes stay untouched.
- **Fail loud, no fallbacks — including in generated code.** WS-C2 makes silent INCOMPLETE *loud*, not
  *tolerated*.
- **Dialect-clean both ways.** Forge passes `andamentum-agentic-dialect check`, and so must every
  package it renders. `graph.py` is the only engine-aware file.
- **Rebuild the sandbox image after ANY forge-src change before benchmarking under Podman** — the
  image bakes andamentum in; a stale image audits new packages against old runtime.
