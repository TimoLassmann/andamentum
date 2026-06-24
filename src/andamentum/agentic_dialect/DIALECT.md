# The Agentic Dialect

A minimal way to build agentic systems on a graph engine and an LLM client. Not a framework — a
dialect: a few pieces, a fixed grammar, a handful of laws. Few primitives, orthogonal, no hidden
control flow, data and code kept apart. If a thing you are about to write is not obviously one of
the pieces below, it is in the wrong place.

Library-specific guidance is confined to the **Substrate** and **2026 model notes** sections; the
laws are engine- and library-neutral. Targets `pydantic-graph <2` and `pydantic-ai <2`.

---

## The mental model: a tape machine

Strip the engine to its core and you have a tape machine with three data surfaces and one
moving head:

- **Deps** — the environment. Frozen; what the machine was handed before it started.
- **State** — the tape. Mutable; the run-global record of what has been produced.
- **Inputs** — what the head carries between cells. Edge-local; one step hands it to the next.
- The **head** does one step of work, writes the tape, and moves.

Everything below makes the roles sharp enough that there is one right way to write each.

---

## The pieces

Three data surfaces, two code roles, one special worker, two disciplines. That is the whole
vocabulary.

| Piece | Kind | One job | The rule |
|---|---|---|---|
| **Deps** | surface | hold what the run was *given* | frozen handle; read everywhere, rebound nowhere |
| **State** | surface | hold what the run has *produced* | mutable plain data; written only by orchestrators |
| **Inputs** | surface | hand a value to the next step | edge-local; never holds pipeline progress |
| **Orchestrator** | code | advance the pipeline one step | route + dispatch only; computes nothing but the routing decision |
| **Worker** | code | do one unit of real work | explicit in → explicit out; never imports the graph engine |
| **Agent** | code | the one "ask the model" instruction | data (prompt + output schema), explicit `model=`, called through a shared builder |
| **Port** | discipline | a capability in Deps | a `Protocol` when >1 impl or test-substitution is needed |
| **Schema** | discipline | data on a boundary | a typed model (agent output, result, step IO); ≤1 level of nesting |

---

## The laws

**L1 — Surface placement.** Given → Deps. Produced and read widely → State. Produced for the next
step → Inputs. Every value also carries a *provenance*: **operator-trusted**, or **untrusted**
(derived, even transitively, from any fetch, tool, external document, or model generation).

**L2 — Thin orchestrator, fat worker.** Orchestrators route and dispatch; workers compute. A worker
never imports the graph engine. *Litmus: delete the engine and the workers + agents still compile
and pass their unit tests.*

**L3 — State is written only by orchestrators.** Workers return values; the orchestrator assigns
them. (A worker may write to a mutable store reached through a Deps **Port** — that store is not
State.) During parallelism, the **join is the sole State-write site**.

**L4 — Routing is static, declarative, and deterministic.** Every successor is known from the code;
you branch *among* declared edges, never synthesize one. The branch is a pure function of State,
Deps, Inputs — never the wall clock, randomness, unordered iteration, or a fresh external read. And
the selector is a trust boundary: route only on operator-trusted predicates or on closed-enum model
output gated by a deterministic check — never on raw untrusted text.

**L5 — Every loop, recursion, and fan-out is bounded.** The bound — iteration count *or* fan-out
width — traces to a Deps value or a named constant. Termination is structural, not hoped-for.

**L6 — The model is a component, not the controller.** Flow lives in the graph. An agent answers a
question; it does not drive the pipeline.

**L7 — Typed boundaries; fail loud.** A structured schema on every edge — no untyped `dict[str,
Any]`. A missing service raises. The one soft failure: a single item's bad data after the client's
retries — record it, continue. If the soft-failure rate crosses a Deps threshold, the run fails
loud with a typed partial result.

**L8 — Effects are idempotent.** A worker that changes the world must be safe to run more than once
— loops, retries, and resume all re-enter it. Guard every external effect with a stable key or a
done-set in State, checked before acting.

### Why these laws

Not arbitrary — each falls out of one goal: *understand and trust a run without running it.*
Data/code separation (1–3) lets you read the flow and test the work in isolation. Static,
deterministic routing (4) makes the topology inspectable and every run replayable — the
precondition for both resume and audit. Bounded everything (5) makes termination and cost
provable. Model-as-component (6) keeps control in code you can test, not a prompt you can't.
Typed, loud boundaries (7) turn failures into signals instead of silent corruption. Idempotent
effects (8) make the re-entry the other laws rely on safe in a world the system changes.

---

## Building a system: the procedure

1. Define the boundary schemas and the `Result` (Pydantic) in `schemas.py`.
2. Write each unit of work as **one engine-free worker function per verb-named file**.
3. Define `State` (three banners) and frozen `Deps` in `graph.py`.
4. Write thin steps: read surfaces → call one worker → assign → return a typed successor.
5. Add the entry function `run_<name>(<inputs>, *, model: str, ...) -> <Module>Result`.
6. For any branching graph, add the topology test.
7. For any world-changing worker, add its idempotency guard (Law 8).

**Pre-commit checklist** (each line is greppable):

- Delete every `pydantic_graph` import from worker/verb/schema files — they still compile. (L2)
- `grep` workers for `ctx` / `state` / `deps` parameters → zero hits. (L2)
- No `datetime` / `random` / unordered iteration in any `run()`. (L4)
- Every loop and fan-out bound is a Deps field or `SCREAMING_SNAKE` constant — no literals. (L5)
- `Result` is a Pydantic model returned via `End`. (L7)
- Every step's successors appear in its return type. (L4)

---

## Where a value lives (Law 1, in detail)

The test, when unsure: *could two independent runs share this value unchanged?* Yes → Deps.
Produced over the run, read by many steps → State. Produced by one step, consumed by its
successor → Inputs.

- **Config** (caps, flags, model id) was given → Deps. Config in State is mutable for no reason.
- **Open-ended produced entities** — a set that grows without bound — do not go in State. They
  live behind a **repository Port in Deps**; State then holds only progress: ids, counts,
  done-sets.
- **Handle vs contents.** A frozen Deps freezes the *binding*, not the bytes behind it: a Port may
  be a mutable write surface. Forbidden is rebinding a Deps field mid-run or stashing per-run
  scalars on the Deps object.
- **Provenance is orthogonal to placement.** A surface tells you where a value lives; its
  provenance tells you whether you may *trust* it. Untrusted values (fetched pages, tool results,
  free-text model output) may be read and stored, but must not reach a routing selector (Law 4) or
  an egress effect except through a closed-enum schema or a constrained capability Port. Treat
  closed-enum agent outputs as a security control, not only a small-model convenience.
- **Resumption is reconstruction, not a save file.** State must be rebuildable from the durable
  Ports plus a start node (whether you rebuild it or the engine persists it). State answers "where
  are we" *given* the durable Ports in Deps.

---

## Orchestrator discipline

An orchestrator step is **thin**: read the surfaces, call a worker, assign the result, return a
typed successor. One shape, no grades.

**It computes nothing but the routing decision itself.** Any computation whose output is *data*
is a worker. If choosing a successor needs work, that work is a worker whose return value the step
branches on. A loop over items or a multi-step sequence is a worker — never logic in `run()`. If
those per-item steps must be graph-visible (resumable independently, their own audit line, or
routed to a shared step), make them **real steps** joined by a fan-out/join — not a fat node.

Routing must be deterministic (Law 4): the branch reads only the data surfaces. Push every source
of nondeterminism — time, randomness, network reads — into a worker, and route on the value it
returns.

Artifact surfaces are `T | None` until produced; a downstream step opens with `x = state.x; assert
x is not None`. Better still, when a value has exactly one consumer, hand it forward via **Inputs**
(non-optional) instead of parking it in State and re-asserting.

---

## Worker discipline

**One unit of work = one public function = one file, named for the verb** — but this packaging
serves a deeper rule: a worker is a **deep module** (Ousterhout), one narrow interface hiding the
real complexity. When several operations share a private invariant, they are *one* module;
splitting them would leak that invariant across a file boundary. Default to one verb per file;
keep them together when a shared invariant says so.

- A worker takes **explicit, narrow inputs** and returns **explicit outputs**. Never the
  orchestration context, State, or Deps. Needs the model → `*, model: str`. Needs a capability →
  take the **Port**. Needs to report progress → take a `Protocol`-typed sink defaulted to no-op.
- **Pure vs effectful.** A *pure* worker only computes and returns. An *effectful* worker changes
  the world (sends, writes, charges, files). Effectful workers obey Law 8: idempotent by a stable
  natural key, or guarded by a done-set in State checked before the effect. At-least-once is the
  default, because every re-entry mechanism (loops, retries, resume) can run a worker again.
- **Time and waiting are effects, not routing.** A worker may read the clock or sleep; an
  orchestrator may not (Law 4). Any worker doing network IO carries a timeout traceable to a Deps
  value; a hung call is an expected exception (Law 7).

The litmus (Law 2) keeps this honest: the worker layer must stand without the graph *engine*. (The
LLM *client* is allowed in the worker layer — the line is the graph engine, not the model client.)

---

## State shape

```python
@dataclass
class State:
    # ── inputs (set once, at entry)
    source: str
    # ── artifacts (accumulated; T | None until produced)
    sections: list[Section] = field(default_factory=list)
    # ── flow control (small scalars/sets/maps the graph branches on)
    rounds: int = 0
    done: set[str] = field(default_factory=set)
```

- **Plain data, grouped: inputs, artifacts, flow-control.**
- **Typed containers are fine** — `dict[K, V]` / `set[T]` keyed by a runtime id. The ban is the
  untyped grab-bag. Keep derived bookkeeping reconstructible from the primary fields.
- **Recorder and predicate methods are allowed, nothing more**: append-only `log` / `record_error`
  / `quarantine`, pure read predicates over State's own fields, a derived index kept consistent by
  a thin mutator. No LLM / IO / domain work ever touches State.
- **Stays flat.** When flow-control fields develop a combinatorial legal-set you can't read off
  the dataclass, **split the pipeline** — do not add nested/hierarchical state types.

---

## The grammar: compositions from the control primitive

One control primitive — the declared successor. Everything composes from it.

| Shape | What it is |
|---|---|
| **Sequence** | a step returns the next step |
| **Branch** | a step selects among declared successors on a State/Inputs predicate |
| **Bounded loop** | re-enter an earlier step (counter on State, the step instance, or worker-local), capped per Law 5 |
| **Fan-out → join** | scatter work across parallel paths, then reduce the results back to one |
| **Sub-pipeline** | call another system's public entry function as one unit of work — never reach into its steps |

- **The join is the only safe write during parallelism.** Parallel paths produce values and write
  nothing to State; the join/reduce is the sole writer. Assign ids and counters in the
  deterministic reduce from a stable ordering of inputs — never inside a parallel path. (This is
  what keeps "only orchestrators write State" a guarantee rather than a lucky style.)
- **The join is also the supervision boundary.** A failed branch degrades to a typed partial and
  the join proceeds — it does not take down the run, and you do not wrap the whole fan-out in the
  bare `try/except` Law 7 forbids.
- **A sub-pipeline is opaque to the parent.** Its `Result` type *is* the boundary; the parent's
  topology test does not see inside it.

---

## Agents

The durable kernel — true regardless of model generation:

- **A model call is data**: a name, a prompt, an output schema. Explicit `model=`, threaded from
  the entry point. No hidden default, no ambient lookup.
- **The model is a component, not the controller** (Law 6).
- **Structured output is the default.** A schema is easier to validate, diff, and reason about
  than prose, and a closed-enum schema is a control-flow safety boundary (Law 4). Free-form text
  output is the exception and must be justified.
- **Call through a shared builder/runner**, never a hand-rolled client per call site, never a
  client built in an orchestrator body. Define an agent inline beside the worker that uses it;
  promote it to a module registry only when a second call site needs the same one.
- **Tool deps are the one mutable, per-call deps object** — separate from the graph's Deps,
  rebuilt per call, may hold per-conversation scratch. Not subject to Law 1.

> **2026 model notes** (expected to churn — not laws). Today's local/small models fill *flat* (≤1
> nesting, few fields, enums over free strings) schemas far more reliably, so prefer them and lean
> on a runner that retries by injecting the schema into the prompt when a model ignores the
> structured-output channel. Multi-turn/agentic loops still need a Law-5 bound; you *may* realize
> it via the client's usage-limit feature **only while** that feature raises on overflow — verify
> it does for your client, or the loop is unbounded. As native constrained decoding and default
> agentic loops mature, expect the fallback and the single-vs-multi-turn split to fade; the kernel
> above does not.

---

## Errors, retries, termination

- **Fail loud.** A missing required service raises. Catch the *expected* exception types around a
  unit of work, never a bare catch that swallows a missing-dependency or config error.
- **One soft failure:** one item's bad data after the client's retries — log, record in State,
  continue. Quarantine is per-item only, never around a step's control flow; a quarantined
  *untrusted* value must never be re-promoted into a routed prompt or an egress tool.
- **Aggregate is loud.** When the soft-failure rate crosses a Deps threshold, flip the result to
  the typed partial/failed kind — a run that skipped most of its work is not green.
- **Never silently drop produced data.** Return results whole; if a real constraint forces a slice,
  surface the total count beside it. A capped run still returns every signal it acquired — the cap
  filters further inquiry, not the output.
- **Three retry layers, and they compound** — the client's in-agent retries (Law 7's "after
  retries" means these; do not add a retry loop in a step), a per-item loop with skip-and-degrade,
  and the bounded re-entry cycle (Law 5). Each effectful re-entry is governed by Law 8.

---

## Driving, observability, audit, reproducibility

- **Two drivers.** One-shot (`run`) is the default. Iterating (`iter` + step) is for
  per-transition instrumentation, timing, a stop/start node, or a checkpoint hook — it may
  *observe* and break but **must not alter routing** (Law 4).
- **Log on entry and exit** of every step (run id + node name): an entry without an exit localizes
  a wedged run to one worker.
- **The transition log stream is the durable audit trail; State is the live in-process record**
  (it dies on a fail-loud abort, exactly when you need the trail most). Record each agent call's
  resolved input, raw output, and the model id actually used to an append-only sink, keyed by call
  id — emitted by the shared runner, so it is structural, not optional.
- **Cost is a run-scoped counter, not a State field.** Increment it at the call site (in the worker
  layer) from the call's usage object — a State counter cannot reach concurrently-gathered calls.
  The honest signal is input/output tokens per model, not request count.
- **Reproducibility has three levels.** Structural re-runnability is *guaranteed* by Laws 4–5.
  Replayable audit is *enabled* if agent IO is recorded (above). Bit-identical output is *out of
  scope* — it lives in the model. Note that a model id in Deps freezes the *binding*, not the
  *behavior* (a stable name over moving weights); a resumed run re-executes its tail live, so treat
  it as a new run for reproducibility, not a continuation.

---

## Testing

- **Workers** unit-test with no graph — call, assert the return (Law 2 makes this possible).
- **Agents** are stubbed by swapping a fake into Deps, so graph tests run without a live model.
- **Every branching graph ships a topology test** (reflection over declared successors): every step
  reachable from the entry, every terminal returns `End`, no dead-ends. A linear pipeline is
  exempt — and says so in a one-line comment.
- **The whole graph** integration-tests with stub Ports + stub agents via Deps.

---

## Enforcement

A rule with no guard rots. Each law and convention sits in a tier; the review-only ones are named
on purpose so they are not mistaken for guaranteed.

| Rule | Tier |
|---|---|
| L2 worker ∌ engine; no client in a node body | **lint/test** — import-linter contract or a small AST test |
| L4 static routing (reachability, terminals) | **test** — the topology test |
| L4 deterministic routing (no clock/random in `run()`) | **lint** ban + review |
| L5 no `while True`, no bare-literal bounds | **lint**; "bound traces to Deps" is review-only |
| L7 no untyped `dict[str, Any]` | **type-check** — ruff `ANN401` + pyright |
| `from __future__ import annotations` | **lint** — ruff `I002` required-import |
| dataclass / Pydantic forms | **type-check** — pyright (partial) |
| L1 placement, L3 concurrent-write, L6, L8, naming, banners | **review-only** |

> The machine-checkable guards: a ruff config with `I002` and `ANN401`, an import-linter contract
> for L2, two small AST tests (no engine import in worker files; no model client built in a node
> body), pyright blocking on `src/`, and the topology test generalized into a meta-test that
> discovers every graph and asserts one exists. The review-only tiers are review-only on purpose.
> `andamentum-agentic-dialect check <path>` runs the portable subset of these gates.

---

## Substrate: which engine, and the migration

| Dialect piece | Node-class runner (older) | Builder API (newer) |
|---|---|---|
| Orchestrator | a `BaseNode` subclass with `run()` | a step function `(ctx) -> out` |
| Successor | `run()` return-type annotation | return-type inference *or* a declared edge |
| Inputs (edge-local) | typed fields on the node instance | `ctx.inputs` (first-class) |
| Deps / State | `GraphRunContext.state` / `.deps` | `ctx.state` / `ctx.deps` |
| Branch | a `Union` return type | a first-class decision primitive |
| Fan-out / join | hand-rolled gather in a step | first-class fork / join with reducers |

**What survives:** State, Deps, return-type routing, the run/iter drivers, the core node/context
types. **What gets better:** edge-local Inputs, branching, and fan-out/join become first-class.
**What is in flux:** built-in persistence/resume — keep resumption reconstructible (Law 1).

Because workers and agents are engine-free (Law 2), moving between engine generations touches only
the thin orchestration layer.

---

## Canonical layout

One module = one pipeline. Files, fixed:

```
<module>/
  __init__.py    public surface — the entry function(s) and the Result type, nothing else
  graph.py       orchestration: State, Deps, the step classes, the Graph, the entry function
  <verb>.py      one worker per file, named for its verb
  schemas.py     boundary types and the Result
  cli.py         the command-line adapter, if any
  tests/         beside the code
```

A large entity-graph scheduler may add `operations/`, `entities/`, and `repository.py`.

## Locked conventions (decided for you)

Each is one fixed form chosen to remove a recurring micro-decision. Don't re-decide them.

- **Boilerplate.** Every `graph.py` and worker file opens with `from __future__ import
  annotations` — successor return-types then need no quoting.
- **Dataclass forms.** `State` → `@dataclass`. `Deps` → `@dataclass(frozen=True)`. Step classes →
  `@dataclass`. Boundary + result types → Pydantic `BaseModel`.
- **Names.** Steps `PascalCase` verb/verb-noun. Workers and files `snake_case`, matching the verb.
  `State` / `Deps` / `Result` prefixed with the module name. Surface fields `snake_case`. Caps are
  `SCREAMING_SNAKE` constants. Agent names `snake_case`.
- **State layout.** Three banner groups, fixed order: inputs, artifacts, flow-control. Standard
  names for universal slots: `errors`, and a quarantine list + `is_quarantined` predicate.
- **Result.** Always a Pydantic model — the `End[T]` payload. Never `str`, `dict`, or a bare
  dataclass.
- **Entry point.** `async def run_<name>(<inputs>, *, model: str, ...) -> <Module>Result`. Builds
  State and Deps, runs the graph, returns `.output`. `model` and all config keyword-only.
- **Caps.** Every loop and fan-out bound is a named Deps field or module constant — never a literal.

## Refuse list

Prohibitions not already obvious from a law:

- A **runtime-synthesized edge** — successors are statically declared (L4).
- **State as a validated model**, or any LLM / IO / domain work on State.
- **The graph engine imported into the worker layer**, or the orchestration context handed to a
  worker.
- A **model client constructed in an orchestrator body**, or hand-rolled per call site.
- **Routing on the clock, randomness, or raw untrusted text** (L4).
- An **untyped `dict[str, Any]`** standing in for named surface fields.
- **Re-promoting quarantined untrusted data** into a routed prompt or egress tool.
- **Reaching into another system's internal steps** to compose pipelines (call its entry point).

## Out of scope, on purpose

Deliberate exclusions, with where the concern lives instead:

- **State as a serializable checkpoint** (the LangGraph bet). We bet the opposite: State is
  rebuilt from durable Ports. Don't serialize State to resume.
- **Hierarchical / orthogonal statecharts.** State stays flat; split the pipeline instead.
- **Durable timers and cross-restart waiting.** Time is a worker/Port effect; durable waiting is an
  app concern, not a State field or a primitive.
- **A fourth surface for inbound events** (mailbox/signals). Model an external event as a durable
  Port the orchestrator polls at a declared step. Human-in-the-loop is the likely future revision,
  not today's grammar.
- **Cross-graph topology reachability**, **result caching of agent calls**, a **pricing table**,
  and a **full security framework** (sandboxing, secrets, authz, rate-limiting — deployment +
  capability-Port concerns). In scope for security: provenance, the routed-selector rule, egress
  via Port, and treating logs/State as a trust sink.

---

## Copy-paste skeleton (runs as-is)

A one-step `brief` module obeying every locked convention. The worker (`summarize`) is engine-free,
so it survives an engine migration untouched. (It is *pure*; an effectful worker would add an
idempotency guard per Law 8.)

```python
# brief/schemas.py
from pydantic import BaseModel

class Brief(BaseModel):          # the Result — a Pydantic model, the End[T] payload
    summary: str

# brief/summarize.py  (worker — no graph engine, no State/Deps)
from __future__ import annotations
from pydantic_ai import Agent
from .schemas import Brief

async def summarize(source: str, *, model: str) -> Brief:
    agent = Agent(model, instructions="Summarize the input in one sentence.", output_type=Brief)
    result = await agent.run(source)
    return result.output

# brief/graph.py  (orchestration — the only engine-aware layer)
from __future__ import annotations
from dataclasses import dataclass
from pydantic_graph import BaseNode, Graph, GraphRunContext, End
from .schemas import Brief
from .summarize import summarize

@dataclass(frozen=True)
class BriefDeps:
    model: str

@dataclass
class BriefState:
    # ── inputs
    source: str
    # ── artifacts
    brief: Brief | None = None

Ctx = GraphRunContext[BriefState, BriefDeps]

@dataclass
class Summarize(BaseNode[BriefState, BriefDeps, Brief]):
    async def run(self, ctx: Ctx) -> Done:
        ctx.state.brief = await summarize(ctx.state.source, model=ctx.deps.model)
        return Done()

@dataclass
class Done(BaseNode[BriefState, BriefDeps, Brief]):
    async def run(self, ctx: Ctx) -> End[Brief]:
        brief = ctx.state.brief
        assert brief is not None  # topology guarantees Summarize ran first
        return End(brief)

graph = Graph(nodes=[Summarize, Done])

async def run_brief(source: str, *, model: str) -> Brief:
    out = await graph.run(Summarize(), state=BriefState(source=source), deps=BriefDeps(model=model))
    return out.output
```
