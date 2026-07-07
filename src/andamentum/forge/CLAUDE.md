# CLAUDE.md — `andamentum.forge`

Operating notes for working **in** this module. forge is the *meta-system*: it turns a
natural-language brief into a runnable agentic system. It is built **in** the agentic
dialect, and every system it produces is **held to** the dialect — so treat it as a pair
with `andamentum.agentic_dialect` (the dialect *defines and verifies* the house style;
forge *builds systems in it*).

## What forge is

`run_forge(brief, *, model, dest=None, ...)` (in `graph.py`) is the single entry point. A
brief becomes a typed, recipe-validated `SystemSpec` (Input / Entities / State / Nodes /
Agents — `spec.py`), which is rendered into a runnable package, whose business-logic holes
are then authored by agents and verified in a sandbox.

forge is itself a dialect-conforming `pydantic-graph` pipeline (run
`andamentum-agentic-dialect check src/andamentum/forge` — it must stay clean). The whole
thing is one graph; `graph.py` is the **only** engine-aware file.

## The pipeline (one graph, four authoring stages)

```
Understand → Assess → Frame → Decompose → Compile → Review → Render → Verify → Build → Audit → Finish → End
   └ design ┘ fitness    └ design ┘          det     gate      det      det    agents   sandbox + agents
              gate (L9)
```

The **front fitness gate** enforces forge's scope (a *sharp function generator*; see
`docs/plans/forge-functions/`): forge builds **functions** — rung 1 (stateless) and,
once the store lands, rung 2 (stateful) — and refuses apps / agents / services at the door.

- **Assess** (`fitness.py`, the `fitness` head — dialect law **L9**): right after Understand,
  judges the brief's *shape* (does an external driver own the control loop?) — never its
  vocabulary. `assess_fitness` → a flat `Fitness{realizable_as_function, rung, reason,
  suggested_reshape}`; `is_buildable` keys on the single `rung` axis against
  `BUILDABLE_RUNGS` (today `{"function"}`; add `"stateful_function"` when the rung-2 store
  lands — the one flip point). A non-buildable rung **fails loud** with the concrete reshape
  (`refusal_message`), never a silent pass. The scenario corpus
  (`tests/scenario_corpus.py`) is the acceptance test.

The plan is grounded by a deterministic coverage check before any code is written:

- **Review** (`review.py`): `plan_coverage` requires every framed area to own ≥1 step, else
  a blocking `UNCOVERED_AREA` finding is raised inside `decompose` before it returns. The
  `Review` node itself is then just the render/finish gate on `stop_after` — no LLM call, no
  back-edge. `graph.py` stays the only engine-aware file.

- **Design** (`understand` / `frame` / `decompose`): the model declares the system in a
  **two-pass** scheme that makes wiring correct BY CONSTRUCTION (no character-for-character
  name reproduction across calls):
  - Stage 1 (`list_jobs`): the ordered node board — areas, steps, node ids.
  - Stage 2a DECLARE (`declare_node`): each node declares only its kind and ONE produced
    name (+ produces_kind/control/network); deterministic code canonicalises and **dedupes**
    the produced names, so the produced set is globally UNIQUE by construction.
  - Stage 2b SELECT (`select_consumes`): each node picks its inputs as ORDINALS into the
    closed, numbered list `input` + every produced name (`build_option_names`); deterministic
    `resolve_consumes` maps ordinals → real names. A consume can never reference a name no
    step produces; an out-of-range ordinal is dropped and recorded in `notes` (never a phantom).
  - Then `decompose` runs a bounded **assemble → diagnose → repair** loop:
    - `assemble.py` matches producers→consumers into a typed `DataGraph` (puzzle-fit,
      now total). Fan-in, fan-out, and back-edges fall out of the matching.
    - `diagnose.py` (pure, `rapidfuzz`) gathers every genuine structural problem with a
      concrete fix: orphan output, multiple/zero sinks, unreachable, dead-end, disconnected,
      unintended cycle. (`duplicate_producer` / `near_miss` / `dangling_read` are now
      impossible on the primary path — the unique-produces + index-selected-consumes
      construction rules them out; their checks remain as defensive backstops.)
    - the repair loop re-runs ONLY pass 2b (`select_consumes`) for each flagged node with the
      finding+suggestion as feedback, bounded by `MAX_DESIGN_ROUNDS`. Producer names are
      FROZEN after 2a, so a repair can never reinvent a produce — the name-matching thrash is
      gone. It converges, or **fails loud** with the full report at the cap.
- **Compile** (`compile_spec.py`, deterministic): assembles the validated `SystemSpec`.
  Canonicalises data names, promotes judgment-over-text spine nodes to heads (symmetric to
  the network/consequential→spine demotion), and keeps single-writer / dangling / orphan
  checks as fail-loud backstops.
- **Render** (`render.py`, deterministic, no LLM): the Assembly law — code, not an LLM,
  writes the package (models, prompts, Deps, graph wiring, single-successor heads, a smoke
  test, and a `__main__.py` CLI launcher so the system runs as `python -m <name> "<text>"
  --model <id>` without a hand-written driver). Spine bodies, routing, and gate decisions
  are left as `NotImplementedError` holes. Heads receive a **labelled** user prompt
  (`Label:\n<value>` per declared read), never a JSON dump.
- **Build** (`build.py`): per hole, the draft/repair agents author the node body, gated
  in-process and fed back on failure (see the gate suite below). No LLM-written code runs
  here.
- **Audit** (`audit.py`): where generated code finally executes — in the sandbox. Runs the
  built system's own tests + an end-to-end smoke, the dialect's `check_code` over the
  package, and a requirements + adversarial-critic head. `works` = holes filled + tests pass
  + dialect-clean.

## The build gate suite — suggestions become guarantees

The governing pattern: a build-prompt suggestion is turned into a deterministic check that
verifies the model obeyed it. Model proposes, gate disposes. The gates on every authored
node body (`astcheck.py`, fed back into the bounded repair loop):

- **contract** — reads/writes only *declared* `ctx.state` fields; returns only *declared*
  successors; no dynamic (`getattr`/`setattr`) or bulk (`model_dump`) state access.
- **coverage** — actually **reads every declared input** and **sets every declared output**
  (catches a node that drops its input or never produces its output — faking).
- **deps** — touches only `ctx.deps` attributes the rendered `Deps` actually provides
  (the allowed set is read off the generated `deps.py`, so gate and renderer can't drift).
  Catches the small-model wiring bug where a body invents a handle the system never
  declared (`ctx.deps.repo_url`) — or two nodes name the same dep differently — at *build*,
  never as a runtime `AttributeError`. A node that needs an unprovisioned resource (an
  external endpoint, a store) fails loud here rather than faking one.
- **purity** — no process control / raw files / sockets / clock / random; a network client
  only when the node declared `network=True` (then it runs behind the container).
- **fail-loud** — no broad `except` that swallows the error (no silent fallback).

Third-party *imports* are NOT gated (the policy is open): a body may import any package. What
it needs beyond the base image is discovered (`provision.discover_requirements`) and installed
into a **per-system sandbox image** for the audit — see "The sandbox" below. A bogus/typo'd
package fails loud at image build, before the test run.

These also reflect the standing project rule: **fail loud, no fallbacks — in the code forge
*generates*, not just in forge itself.** A system that runs but does the wrong thing is
worse than one that stops. The deterministic gates alone decide fillability: a body that
passes every gate is filled; a body that never does within `attempt_cap` is honest
`unfillable` (its `NotImplementedError` restored).

## The sandbox

`sandbox.py` is the one seam through which LLM-written code executes — never in the forge
process. `make_sandbox(backend)` returns the Port; **Podman is the default** (host-isolated:
the package is COPIED IN over stdin as a tar — no host path shared with the VM — scrubbed
env, memory/pids caps; a pure run gets `--network none`, a declared-network node keeps
isolation but is allowed onto the net). `SubprocessSandbox` is the no-container fallback
(out-of-process, but **not** host-isolated — it refuses network execution).

**Per-system dependency provisioning.** A generated system may import beyond the base image
(the infra + baked commons). `provision.discover_requirements(pkg)` finds those long-tail
packages; `PodmanSandbox` bakes them into a small per-system image (`FROM base` + `pip
install`, content-addressed by the dep set so identical needs reuse one cached image) at
image-*build* time, so the audit test run stays fully offline. The policy is **open** — any
import is installed, not gated; a bogus name fails loud at the image build. `SubprocessSandbox`
ignores `extra_deps` (it runs in the host env). Build the base image from the repo root:

```
podman build -t andamentum-forge-sandbox -f src/andamentum/forge/Containerfile .
```

The backend is an explicit keyword arg (`--sandbox`), never an env var.

## Rules for working here

1. **Fail loud, no fallbacks — everywhere, including generated code.** Never silently drop,
   default, or invent data. A detected problem is surfaced (in the `DesignReport` / a gate
   violation) and repaired, or it raises. This is the load-bearing rule; most of the gate
   suite exists to enforce it on the code the model writes.
2. **Dialect-clean, both ways.** forge itself passes `andamentum-agentic-dialect check`, and
   so does every package it renders (the audit checks the output). `graph.py` is the only
   engine-aware file; everything else is an engine-free worker.
3. **Deterministic does the heavy lifting.** The model declares; code assembles, diagnoses,
   and verifies. Don't move correctness into a prompt-hope when a deterministic check can
   guarantee it.
4. **One code path; flat agent schemas; explicit `model=`; no env vars; bounded loops via
   named constants.** The standard andamentum/dialect conventions. The caps are module
   constants in their workers: `MAX_DESIGN_ROUNDS` (`decompose.py`), `ATTEMPT_CAP` / fan-out
   bounds (`graph.py`).

## File map

`graph.py` the pipeline (State / Deps / steps / `run_forge`) · `schemas.py` boundary types +
`ForgeResult` · `spec.py` the `SystemSpec` + recipe validators · `naming.py` identifier
helpers · `understand.py` / `frame.py` / `decompose.py` design workers · `fitness.py` the front
fitness gate (L9 — `assess_fitness` / `is_buildable` / `refusal_message`) · `review.py` the
deterministic per-area `plan_coverage` check · `assemble.py`
puzzle-fit DAG · `diagnose.py` the structural diagnostic engine · `compile_spec.py` board →
spec · `render.py` deterministic spec → package · `build.py` per-node authoring loop ·
`astcheck.py` / `patch.py` / `extract.py` the build gates + body editing · `verify.py`
render-stage checks · `audit.py` whole-system audit · `sandbox.py` + `Containerfile` the
execution seam · `runtime.py` the engine-free spine a **generated** package imports
(`run_head` / `loop_allowed` / `Store` — the rung-2 cross-run-memory Port: stdlib-sqlite
keyed CRUD, `Store(None)` in-memory, `Store(path)` durable) · `agents.py` the design + authoring heads (as data) ·
`reporter.py` the progress Port (`ForgeReporter` / `NoopReporter` default / `RichReporter`
live dashboard) · `cli.py` the `andamentum-forge` adapter.

`run_forge` takes an optional `reporter=` Port (default silent `NoopReporter`); it drives
the graph via `graph.iter` so each stage lights as it runs and its one-line summary is read
off the run state, while `build`/`audit` emit per-node and per-check sub-events. The CLI
installs the `RichReporter` live dashboard on `--verbose`. Following the `deep_research`
reporter pattern: keyword-only events, presentation-only leaf, no graph engine.

## Develop

```bash
uv run pytest src/andamentum/forge/tests
uv run pyright src/andamentum/forge
uv run ruff check src/andamentum/forge && uv run ruff format src/andamentum/forge
uv run andamentum-agentic-dialect check src/andamentum/forge   # forge stays dialect-clean
```

The whole forge graph is testable with no live model and no container: agents are stubbed
through the `AgentSink` Port and the sandbox through a `FakeSandbox` (see `tests/conftest.py`);
the real out-of-process path is exercised via `SubprocessSandbox`.

## Known scope (v1)

Deliberate exclusions, noted so they aren't mistaken for bugs: HITL persistence/resume
machinery is not generated (a consequential node renders as a spine hole); node I/O is
matched by name (string-similarity for near-misses), not by declared types; the generated
heads' user-prompt builders are reasonable defaults meant to be refined. The
`out_text`-style fallbacks have been removed; if you add generated-code templates, hold them
to the same fail-loud bar as hand-written code.

**Rung-2 status (single-record).** A stateful function — a brief whose output depends on
earlier runs — now builds and actually remembers:
- **Classification (deterministic, §7):** a datum a single node read-modify-writes (consumes
  ∩ produces) is durable — a value loaded, changed, saved. `compile_spec` requires it be
  declared an entity (`produces_kind=entity`) or fails loud (a read-modify-write *signal*
  would be forgotten); `diagnose` exempts the entity round-trip's self-edge from the cycle
  check (a round-trip is read-modify-write, not an unintended loop).
- **Wiring (dialect L1 / C §4 formula `save(store, f(text, load(store)))`):** the entity
  becomes a State field; the run entry loads the durable record into it at the start and saves
  it back at the end (single-record, constant key `"_"`), keeping every graph node pure. The
  generated smoke seeds entity fields to `""` (the first-run default).
- **Gate (Phase 5):** `BUILDABLE_RUNGS` now includes `stateful_function`; the fitness gate
  admits rung-2 and still refuses apps/agents/services. Proven by `tests/test_stateful.py` —
  a generated package called twice against one db file accumulates; an in-memory run forgets.

Not yet done: **multi-record by id** (uuid key — "update record #id", "append a note and
count many notes"); the design front-end producing rung-2 boards reliably from a live model
(needs calibration — single-node read-modify-write is the supported shape). See
`docs/plans/forge-functions/C-STORE-PRD.md` §6 (the key), §8 (the wall).
