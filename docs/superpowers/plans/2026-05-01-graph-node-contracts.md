# Graph Node Contracts — Move 3

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Do NOT execute this plan top-to-bottom in a single session — each phase has explicit acceptance criteria and a benchmark gate. Stop at the gate, run the benchmark, verify, then continue.**

**Goal:** Eliminate the recurring class of "silent dead zone" bugs in the epistemic graph by making each node's contract explicit (declared reads / writes / operations / successors / post-invariants), enforced by a contract validator test, and validated end-to-end by the existing reachability test (`test_graph_reachability.py`). The current monolithic `graph/nodes.py` (~1900 lines, ~20 nodes) is split into per-phase modules with declared contracts. Routing decisions become a *data structure* rather than 1900 lines of imperative `Union[NodeA, NodeB]` returns scattered across the file.

**Why:** Three routing bugs in the same week, all the same shape — a node returns to a successor that doesn't continue work the claim still needs, claim ends up stranded at a terminal it shouldn't reach, headline posterior decoheres from the prose verdict. Concrete instances:

- `DecomposeQuestionOperation` un-wired in v0.3 (registered, no graph caller)
- `ReflectOnGapsOperation` still dormant after the post-audit fix queue (same shape)
- `AbandonOrDemote → CheckCompletion` short-circuited soft-promoted claims past IBE
- `AbandonOrDemote → CheckCompletion` short-circuited HYPOTHESIS-with-pass claims when scrutiny found mixed outcomes (same node, same shape)

The structural-wiring test (`test_structural_wiring.py`) catches "operation has no graph caller". The reachability test (`test_graph_reachability.py`, added in commit `d280573`) catches "state pattern has no graph terminal" — but only for state patterns we've thought to enumerate. The class-of-bug remains: the routing topology is encoded implicitly across many `Union[...]` return types, and an explicit invariant per node would eliminate the implicit-knowledge gap that produces these bugs.

**Tech Stack:** Python 3.13, pydantic-graph, pydantic 2, dataclasses, the existing `andamentum.epistemic.graph` package.

---

## Architecture

### What pydantic-graph already provides

Before specifying the new layer, the parts of "topology as data" we **don't** need to build:

- **Successors are already declared by the `run()` method's return type annotation.** Pydantic-graph reads `Union["NodeA", "NodeB"]` and uses it to build the graph — quoting the docs, "the return type of the `run` method is used to determine the outgoing edges of the node." Pyright enforces that the body's `return ...` statements match the annotation. This means we get *static enforcement* of "body returns only declared successors" for free, with no extra metadata.

  The recurring routing bugs were not type violations — they were cases where the *annotation itself* permitted the wrong successor (e.g. `Union[A, B, CheckCompletion]` when `CheckCompletion` shouldn't have been allowed). The fix in those cases was to *tighten the annotation*. So the annotation IS the contract — duplicating it as `successors = frozenset({...})` would just add a second source of truth that has to be kept in sync.

- **Mermaid diagram rendering is built in.** `Graph.mermaid_code()`, `mermaid_image()`, and `mermaid_save()` produce diagrams from the type annotations directly. We don't need to build a Graphviz exporter; Mermaid output is sufficient for code review and onboarding diagrams.

- **Edge labels and docstring annotations** are supported via `pydantic_graph.Edge` and `BaseNode.docstring_notes`.

What pydantic-graph does NOT provide and we need to add:
- Read/write metadata for state fields
- Operation-dispatch metadata
- Post-condition / invariant declarations
- Static topology validators (e.g. reachability checks beyond the runtime edge enforcement)

### The contract

Each node carries class-level metadata declaring its state I/O, operation dispatch, and post-conditions. **Successors are NOT duplicated as metadata** — they live in the `run()` return type annotation, which both pyright and pydantic-graph already enforce.

```python
class Node(BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]):
    # The state fields this node reads. Validator asserts the body
    # only accesses ctx.state.<field> for fields in this set.
    reads: ClassVar[frozenset[str]] = frozenset()

    # The state fields this node writes. Validator asserts the body
    # only mutates ctx.state.<field> for fields in this set.
    writes: ClassVar[frozenset[str]] = frozenset()

    # The operations this node dispatches via _run_op. Validator
    # asserts the body only calls _run_op with these classes.
    operations: ClassVar[frozenset[type[BaseOperation]]] = frozenset()

    # Predicates that must hold over (state, claims) AFTER this node
    # runs. Checked at runtime in tests; checked statically where
    # possible.
    post_invariants: ClassVar[tuple[Invariant, ...]] = ()

    async def run(self, ctx: GraphRunContext) -> Union["NodeA", "NodeB"]:
        # Successors live HERE — in the return type annotation.
        # Pyright enforces; pydantic-graph reads it to build the graph.
        ...
```

`Invariant` is a callable taking `(state, claims) -> Optional[str]` — returns `None` when the invariant holds, returns a violation message when it doesn't. The reusable `_no_stranded_claims` invariant from the reachability test is the canonical example.

### The topology helper

A small reflection helper, `topology()`, builds a dict-of-sets from the node classes' return annotations. It uses `typing.get_type_hints` and `typing.get_args` on each `run()` method:

```python
def topology() -> dict[type[Node], frozenset[type[Node]]]:
    """Return {node_class: {successor_classes}} for every Node subclass.

    Reads the return type annotation of each node's run() method —
    no separate metadata required, no graph execution needed.
    """
    ...

# Usage:
>>> topo = topology()
>>> topo[AbandonOrDemote]
frozenset({Scrutinize, PromoteToSupported})
```

This is ~20 lines and lives in `graph/topology.py`. It's the single source of truth for "what does the graph look like as data."

### The validator

A new test, `test_node_contracts.py`, performs two checks per node (down from three — the successors check is unnecessary because pyright + pydantic-graph already enforce it):

1. **Body-only-touches-declared-state.** AST-walks the node's `run` method; every `ctx.state.X` access where `X` is read must appear in `reads`; every assignment to `ctx.state.X` must appear in `writes`. Allowlist for cross-cutting fields (`operations_log`, `successful`, `failed` — all written by `_run_op` itself, not by node bodies).
2. **Operations-only-from-declared-set.** Every `_run_op(OpClass, ...)` call's first argument must appear in `operations`.

For invariants, a separate test loops over the reachability test's state patterns and asserts every node's `post_invariants` hold after running.

For static reachability checks, a third test uses the `topology()` helper to verify properties like:
- "Every state pattern that produces a soft-promoted Claim has a graph path to `EnumerateCandidates`."
- "`CheckCompletion` is not in `AbandonOrDemote`'s reachable successors set." (The structural form of the recurring bug — fails at CI time if anyone widens the annotation back.)
- "Every node is reachable from `PrepareObjective`."
- "No accidental cycles" (intentional cycles like the inquiry loop are allowlisted).

### File layout

```
src/andamentum/epistemic/graph/
    __init__.py          # run_epistemic_graph (unchanged)
    base.py              # NEW — Node base class with contract metadata
    invariants.py        # NEW — _no_stranded_claims and other reusable invariants
    topology.py          # NEW — topology() reflection helper over run() annotations
    state.py             # unchanged
    deps.py              # unchanged
    quarantine.py        # unchanged
    result.py            # unchanged
    combination.py       # unchanged
    nodes/
        __init__.py      # re-exports + epistemic_graph build
        preplanning.py   # PrepareObjective, Decompose, PlanEvidence
        evidence.py      # ExtractEvidence, ExtractNewEvidence, ClusterEvidence
        claims.py        # CreateClaims
        scrutiny.py      # Scrutinize, Investigate, AbandonOrDemote
        verification.py  # PromoteToSupported, RunVerification, ResolveUncertainties
        integration.py   # EnumerateCandidates, ScoreLoveliness, ScoreLikeliness, SelectBestExplanation, PromoteSupported, CombineClaimVerdicts
        terminal.py      # CheckCompletion, Synthesize
```

`nodes/` (directory) replaces `nodes.py` (file). The directory `__init__.py` re-exports every node class so `from .nodes import X` continues to work for any X — *call sites do not change*.

`epistemic_graph` (the pydantic-graph builder) lives in `nodes/__init__.py` so the topology is visible in one place.

---

## Migration strategy

The refactor moves through nodes in **leaf-first order** so each migrated node has fully-migrated successors. Between phases, the existing benchmark must still produce the validation output from the v3 run (commit `d280573`'s baseline) — same posterior shape, same operation counts, no stranded claims. If the benchmark drifts, the phase blocks until the drift is investigated.

The structural wiring test (`test_structural_wiring.py`) and reachability test (`test_graph_reachability.py`) run after every phase. They assert outputs, not internals, so they survive the refactor.

### Phase 0 — Foundation (no behavior change)

- [ ] Create `graph/base.py` with the `Node` base class, `Invariant` typedef, and the contract metadata (`reads`, `writes`, `operations`, `post_invariants` — NOT `successors`, which lives in the `run()` return annotation). Empty defaults so existing nodes work unchanged.
- [ ] Create `graph/topology.py` with the `topology()` reflection helper. Uses `typing.get_type_hints` + `typing.get_args` over each `Node` subclass's `run()` return annotation; returns `dict[type[Node], frozenset[type[Node]]]`. ~20 lines. Plus a thin `mermaid()` wrapper that delegates to `epistemic_graph.mermaid_code()` if any callers want a one-liner instead of touching the graph object.
- [ ] Create `graph/invariants.py`. Move `_no_stranded_claims` from `tests/test_graph_reachability.py` into this module. Update the test to import from the new location.
- [ ] Create `graph/nodes/` directory with empty `__init__.py` that re-exports from the existing `graph/nodes.py`. **Do not move any code yet.** This step exists only so the import path can change atomically in Phase 1 without a code move.
- [ ] Add `test_node_contracts.py` skeleton with two checks (state I/O, operations) — successors check is unnecessary, pyright + pydantic-graph already cover it. Test class iterates over an empty registry on day one.
- [ ] Add `test_topology.py` — uses `topology()` to assert global reachability properties: `CheckCompletion not in topology()[AbandonOrDemote]` (the recurring-bug invariant); every node reachable from `PrepareObjective`; no unintentional cycles.
- [ ] Run full test suite + benchmark. **Acceptance:** all 1728+ tests pass; `topology()` returns the expected dict against the existing `nodes.py`; benchmark produces the same operation counts and posterior shape as the `d280573` baseline (allowing for LLM stochasticity in evidence retrieval but expecting same {claims_minted=4, IBE_runs=2-3, abandoned≤2, posterior_in_band=0.4-0.7} pattern).

### Phase 1 — Terminal nodes (Synthesize, CheckCompletion)

These are the leaves. They have the smallest contracts and minimal successor sets — `Synthesize` returns `End[...]` only, `CheckCompletion` returns `Union[Synthesize, End[...]]`.

- [ ] Move `CheckCompletion` and `Synthesize` from `graph/nodes.py` to `graph/nodes/terminal.py`. Keep imports/exports identical via `nodes/__init__.py`.
- [ ] Add `Node` base class metadata for both. Declare `reads`, `writes`, `operations`, `post_invariants` (successors are encoded in the existing return type annotation — leave that alone). For these two specifically:
  - `CheckCompletion.reads = {"retrieval_failed", "objective_id", ...}`
  - `Synthesize.operations = {FreezeSnapshotOperation, SynthesizeReportOperation}`
  - `Synthesize.post_invariants = (no_stranded_claims,)` — this is the load-bearing invariant for the whole graph
- [ ] Wire `test_node_contracts.py` to validate these two nodes. Include a deliberately-broken probe in a sibling test (e.g. a `BrokenNode` that declares `operations={A}` but calls `B`) to confirm the validator actually fails-loud.
- [ ] Run full test suite + benchmark. **Acceptance:** same as Phase 0; additionally, the contract validator passes for the two migrated nodes; the deliberately-broken probe fails as expected.

### Phase 2 — Routing hub (PromoteToSupported, AbandonOrDemote)

These are the high-density routing nodes where the recurring bug class lived. Migrating them is the highest-payoff step.

- [ ] Move `AbandonOrDemote` and `PromoteToSupported` to `graph/nodes/scrutiny.py` and `graph/nodes/verification.py` respectively.
- [ ] The recurring bug was a return type annotation that included `CheckCompletion`. The current annotations on these two nodes already exclude `CheckCompletion` (we tightened them in commits `e770e31` and `d280573`). Add a topology test in `test_topology.py` that asserts `CheckCompletion not in topology()[AbandonOrDemote]` — this prevents future widening of the annotation from re-introducing the bug class.
- [ ] Add `reads`, `writes`, `operations`, `post_invariants` metadata. Both nodes' `post_invariants` include `no_stranded_claims`.
- [ ] Run reachability test under the new contracts — must still pass.
- [ ] Run full test suite + benchmark. **Acceptance:** same as Phase 0; additionally, the contract validator passes for the four migrated nodes; the topology test asserts the routing-bug-class invariant.

### Phase 3 — Scrutiny + investigation (Scrutinize, Investigate, ResolveUncertainties)

- [ ] Move to appropriate files in `graph/nodes/`.
- [ ] Add contract metadata.
- [ ] Run all tests + benchmark. **Acceptance:** same as Phase 0.

### Phase 4 — Verification + IBE (RunVerification, ClusterEvidence, EnumerateCandidates, ScoreLoveliness, ScoreLikeliness, SelectBestExplanation, PromoteSupported)

The IBE chain. These are sequential and have well-defined inputs/outputs (each stage reads the previous stage's output from the Claim).

- [ ] Move to `graph/nodes/integration.py` and `graph/nodes/verification.py`.
- [ ] Add contract metadata. The IBE chain's invariants are particularly clear: each stage requires the previous stage's output present and writes only its own slot.
- [ ] Run all tests + benchmark. **Acceptance:** same as Phase 0.

### Phase 5 — Pre-claim phases (PrepareObjective, Decompose, PlanEvidence, ExtractEvidence, ExtractNewEvidence, CreateClaims, CombineClaimVerdicts)

The final batch. By this point the pattern is well-established and the migration is mechanical.

- [ ] Move all remaining nodes.
- [ ] Add contract metadata.
- [ ] Delete the original `graph/nodes.py` (now empty / shim).
- [ ] Run all tests + benchmark. **Acceptance:** same as Phase 0; the original `nodes.py` is gone; `from andamentum.epistemic.graph.nodes import X` still works for every X via `nodes/__init__.py`.

### Phase 6 — Type the load-bearing dictionaries

Once the node contracts are in place, the dictionaries that get accessed across multiple consumers can be typed without churning the routing logic:

- [ ] `Objective.decomposition: dict[str, Any]` → `Decomposition` Pydantic model. Replace dict access in `combine_claim_verdicts`, `compute_posterior`, `MultiSeedClaimOperation`, `CombineClaimVerdicts` node.
- [ ] `Claim.predictions: list[dict]` → `list[Prediction]`. Replace in `GeneratePredictionOperation`, the report renderer, the synthesis agent.
- [ ] `Claim.promotion_history: list[dict]` → `list[StageTransition]`. Replace in `PromoteClaimOperation`, the audit log, the report.
- [ ] `EpistemicGraphState.operations_log: list[dict]` → `list[OperationEvent]`. Replace in the operation profile renderer, the structural test.
- [ ] **Acceptance:** zero regression in test suite; typed access throughout; remove `dict.get()` calls on these surfaces; the structural-wiring + reachability tests still pass.

### Phase 7 — Closeout

- [ ] Update `CLAUDE.md`'s "Epistemic architecture principles" section to add P6: "Every node has an explicit contract; routing decisions are data, not implicit imperative flow."
- [ ] Update `handle_ask`'s docstring pipeline list if any node names changed.
- [ ] Run benchmark one final time. **Acceptance:** posterior + verdict shape consistent with the `d280573` baseline; operation profile contains the same set of ops (allowing LLM stochasticity in counts).
- [ ] Update memory at `~/.claude/projects/-Users-timo-code-andamentum/memory/` if any general lesson should persist across sessions (e.g. a feedback memory about explicit contracts beating implicit routing).

---

## Open decisions

These are deliberate gaps the executing session should make explicit choices about before writing code:

### 1. Validator implementation: AST-walk vs. dynamic instrumentation?

AST-walk is simpler but can't catch cases where state access goes through a helper function. Dynamic instrumentation (a `__setattr__` override on a wrapped state during a test run) catches helper indirection but is more complex.

**Recommendation:** start with AST-walk. The recurring bugs were direct accesses; if helper indirection becomes a problem we add dynamic instrumentation later.

### 2. Where do invariants live — per-node or global?

Some invariants (`no_stranded_claims`) apply to almost every node. Others (`every_claim_has_evidence_at_or_above_extracted`) apply to specific phases.

**Recommendation:** keep them as a flat module (`graph/invariants.py`); each node's `post_invariants` is a tuple of references. No inheritance/composition — duplication is fine when it's just a one-line reference.

### 3. Migration scope for the dictionary typing

Phase 6 lists four dicts. Are all four worth the churn, or just `decomposition` (the one that produced Bug C)?

**Recommendation:** type only `decomposition` and `predictions` for now. `promotion_history` is single-writer single-reader (audit log). `operations_log` is internal. Re-evaluate the latter two after the first two land.

### 4. What happens if a benchmark phase drifts but tests pass?

The benchmark is non-deterministic (LLM stochasticity in evidence retrieval and verdict computation). The acceptance criterion "same posterior + verdict shape" needs tolerance bands.

**Recommendation:** define the tolerance up front: `claims_minted in {3, 4}`, `IBE_runs >= 1` (i.e. at least one claim reached IBE), `posterior in {valid float ∈ [0, 1]}`, `n_no_verdict == 0` (the load-bearing invariant), `verdict ∈ {"supports", "contradicts", "insufficient", "no_data", "union"}`. If any of these fail, the phase is blocked.

---

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Validator catches false positives in existing nodes (overly strict reads/writes detection) | Medium | Low | Start with explicit allowlists for cross-cutting fields; validator failures are flagged as warnings until everyone's clear |
| Mid-refactor benchmark drift gets attributed to refactor when it's actually upstream stochasticity | Medium | Medium | Run benchmark 3× per phase; only block when 2/3 fail the acceptance bands |
| Phase 4 (IBE chain) reveals additional unstated invariants that aren't easy to express as `post_invariants` | Medium | Medium | Allow phase to land with a TODO; don't gate on completeness of invariants |
| Dictionary typing in Phase 6 surfaces serialization mismatches with the database (existing rows have dict-shaped metadata) | High | Medium | Add migration code that reads dict-shaped metadata into the typed model; new writes use the model |
| Refactor itself introduces a routing bug despite the test coverage | Medium | High | The reachability test is the gate; if it fires, stop, diagnose, fix before continuing |
| Plan takes much longer than estimated due to discovered complexity | Medium | Low | Each phase is independently committable; abandoning between phases is safe |

---

## Estimated effort

- **Phase 0** (foundation): half day
- **Phase 1** (terminals): half day
- **Phase 2** (routing hub): full day — this is the most subtle phase, with the highest blast radius if wrong
- **Phase 3** (scrutiny): half day
- **Phase 4** (verification + IBE): full day
- **Phase 5** (pre-claim): full day
- **Phase 6** (typing dicts): full day
- **Phase 7** (closeout): half day

**Total: ~5 working days, spread across multiple sessions with benchmark gates between phases.**

The "many sessions" framing matters: this plan is *not* meant to be executed top-to-bottom in one go. Each phase should produce a green commit and a benchmark run. If a phase gates on something unexpected, stop and re-plan rather than push through.

---

## Acceptance criteria for the whole refactor

When all phases complete, the following must be true:

1. `nodes/` directory exists; `nodes.py` does not.
2. Every node class has explicit `reads`, `writes`, `operations`, `post_invariants` metadata. Successors are encoded in the `run()` return type annotation (no duplicate metadata).
3. `graph/topology.py` exposes `topology() -> dict[type[Node], frozenset[type[Node]]]` for static graph inspection.
4. `test_node_contracts.py` passes, validating each node's body against its declared state I/O and operations.
5. `test_topology.py` passes, asserting global properties (e.g. `CheckCompletion not in topology()[AbandonOrDemote]`, every node reachable from `PrepareObjective`).
6. `test_graph_reachability.py` still passes (architecture-agnostic, by design).
7. `test_structural_wiring.py` still passes.
8. The full test suite passes (≥1728 tests; new contract tests add to the count).
9. The benchmark on `"Does intermittent fasting reduce all-cause mortality?"` with `--decompose` produces a posterior + verdict consistent with the baseline tolerance bands.
10. Pyright clean, ruff clean.
11. The `objective.decomposition` and `claim.predictions` dictionaries are typed.
12. `CLAUDE.md` reflects the new architecture in its principles section.

---

## What this plan does NOT cover

- Refactoring the operations themselves. Operations are already well-isolated and not the source of the recurring bug class.
- Changing the pydantic-graph dependency. The base class extends `BaseNode` from pydantic-graph; we add metadata but don't replace the runtime engine.
- Performance optimization. The operation-profile timing-shows-0ms cosmetic bug is out of scope.
- Renaming or restructuring the graph topology itself. This refactor preserves behavior; it changes how the topology is encoded, not what it is.

If any of these become necessary mid-refactor, write a follow-up plan rather than expanding this one.
