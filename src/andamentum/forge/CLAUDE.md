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
Understand → Frame → Decompose → Compile → Render → Verify → Build → Audit → Finish → End
   └──── design heads ────┘        det      det     det     agents   sandbox + agents
```

- **Design** (`understand` / `frame` / `decompose`): the model declares the system —
  areas, steps, and each node's kind (spine/head) and its `consumes`/`produces` variable
  names, **freely**. Then `decompose` runs a bounded **assemble → diagnose → repair** loop:
  - `assemble.py` matches producers→consumers into a typed `DataGraph` (puzzle-fit). Fan-in,
    fan-out, and back-edges fall out of the matching — the full grammar, nothing forced linear.
  - `diagnose.py` (pure, `rapidfuzz`) gathers every structural problem with a concrete
    suggested fix: dangling read, near-miss name, orphan output, duplicate producer,
    multiple/zero sinks, unreachable, dead-end, disconnected, unintended cycle.
  - the repair loop re-types each flagged node with the finding+suggestion as feedback,
    bounded by `MAX_DESIGN_ROUNDS`. It converges to a clean design, or **fails loud** with
    the full report at the cap. The determinism does the heavy lifting; the model only
    applies targeted corrections (which is what makes it converge on small/local models).
- **Compile** (`compile_spec.py`, deterministic): assembles the validated `SystemSpec`.
  Canonicalises data names, promotes judgment-over-text spine nodes to heads (symmetric to
  the network/consequential→spine demotion), and keeps single-writer / dangling / orphan
  checks as fail-loud backstops.
- **Render** (`render.py`, deterministic, no LLM): the Assembly law — code, not an LLM,
  writes the package (models, prompts, Deps, graph wiring, single-successor heads, a smoke
  test). Spine bodies, routing, and gate decisions are left as `NotImplementedError` holes.
  Heads receive a **labelled** user prompt (`Label:\n<value>` per declared read), never a
  JSON dump.
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

- **contract** — reads/writes only *declared* fields; returns only *declared* successors;
  no dynamic (`getattr`/`setattr`) or bulk (`model_dump`) state access.
- **coverage** — actually **reads every declared input** and **sets every declared output**
  (catches a node that drops its input or never produces its output — faking).
- **purity** — no process control / raw files / sockets / clock / random; a network client
  only when the node declared `network=True` (then it runs behind the container).
- **fail-loud** — no broad `except` that swallows the error (no silent fallback).

These also reflect the standing project rule: **fail loud, no fallbacks — in the code forge
*generates*, not just in forge itself.** A system that runs but does the wrong thing is
worse than one that stops.

## The sandbox

`sandbox.py` is the one seam through which LLM-written code executes — never in the forge
process. `make_sandbox(backend)` returns the Port; **Podman is the default** (host-isolated:
read-only mount, scrubbed env, memory/pids caps; a pure run gets `--network none`, a
declared-network node keeps isolation but is allowed onto the net). `SubprocessSandbox` is
the no-container fallback (out-of-process, but **not** host-isolated — it refuses network
execution). Build the image from the repo root:

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
   named constants.** The standard andamentum/dialect conventions.

## File map

`graph.py` the pipeline (State / Deps / steps / `run_forge`) · `schemas.py` boundary types +
`ForgeResult` · `spec.py` the `SystemSpec` + recipe validators · `naming.py` identifier
helpers · `understand.py` / `frame.py` / `decompose.py` design workers · `assemble.py`
puzzle-fit DAG · `diagnose.py` the structural diagnostic engine · `compile_spec.py` board →
spec · `render.py` deterministic spec → package · `build.py` per-node authoring loop ·
`astcheck.py` / `patch.py` / `extract.py` the build gates + body editing · `verify.py`
render-stage checks · `audit.py` whole-system audit · `sandbox.py` + `Containerfile` the
execution seam · `runtime.py` the engine-free helpers (`run_head` / `loop_allowed`) a
**generated** package imports · `agents.py` the design + authoring heads (as data) ·
`cli.py` the `andamentum-forge` adapter.

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
