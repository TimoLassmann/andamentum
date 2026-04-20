# Graph Scheduler Migration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the pattern-based scheduler with a pydantic-graph DAG. Every scheduling bug found in this session (premature promotion, infinite loops, attempt counting, phase advancement hacks) stems from implicit ordering in the pattern scheduler. The graph makes dependencies explicit and type-checked.

**Architecture:** The existing operations stay — their `execute()` logic is correct. Only the scheduling layer changes. Each graph node wraps one or more operations and returns the next node to run, making the workflow visible in code and verifiable by pyright.

**Tech Stack:** pydantic-graph 1.84.1 (already installed), Python 3.12, dataclasses

---

## Design Principles

1. **Operations don't change.** The `execute()` methods in `operations/*.py` contain the domain logic. Graph nodes call them. No rewriting of operation internals.

2. **One graph per objective.** Each epistemic inquiry (one research question, one objective) is one graph execution. The graph runs to `End(result)`.

3. **State is explicit.** A single `EpistemicGraphState` dataclass replaces the implicit state-matching of the pattern scheduler. Nodes read and write this state.

4. **Deps carry infrastructure.** `EpistemicDeps` holds the repo, agent_runner, evidence_gatherer, embedding_model, quality_scorer — everything operations need. Immutable across the run.

5. **Cycles have explicit counters.** Investigation loop, uncertainty resolution loop — each has a counter in state, checked in the node, with `End()` as the escape.

6. **No attempt counting.** Nodes run exactly once per graph traversal. If the graph cycles (investigation), the cycle counter limits iterations. No `MAX_ENTITY_ATTEMPTS`, no `record_attempt`, no `reset_entity_attempts`.

7. **Promotion is a node, not a retry.** The promote node runs AFTER all prerequisites complete. It cannot fire early because it's downstream of verification and integration in the graph.

---

## Graph Topology

```
START
  │
  ▼
PrepareObjective ─────────────────────────────────────────────────────┐
  │ (clarify + classify + conceptual_analysis)                        │
  ▼                                                                   │
PlanEvidence                                                          │
  │ (plan_task → creates evidence stubs)                              │
  ▼                                                                   │
ExtractEvidence ◄─────────────────────────────────────────┐           │
  │ (extract all unextracted evidence)                     │           │
  ▼                                                       │           │
CreateClaims                                              │           │
  │ (seed_claim OR propose_claims)                        │           │
  ▼                                                       │           │
Scrutinize ◄──────────────────────────────────┐           │           │
  │ (scrutinise_claim on all claims)           │           │           │
  │                                            │           │           │
  ├── all pass ──────► PromoteToSupported      │           │           │
  │                         │                  │           │           │
  ├── needs_resolution ──► Investigate ────► ExtractEvidence           │
  │   (count < 3)           │ (creates stubs, increments count)       │
  │                         │                                         │
  ├── needs_resolution ──► AbandonOrDemote ──────────────────────────►│
  │   (count >= 3)                                                    │
  │                                                                   │
  └── fail@HYPOTHESIS ──► Investigate (count < 3)                     │
      fail@HYPOTHESIS ──► AbandonOrDemote (count >= 3)                │
      fail@SUPPORTED+ ─► Demote ──► Scrutinize                       │
                                                                      │
PromoteToSupported                                                    │
  │ (promote H→S, set routing defaults)                               │
  ▼                                                                   │
RunVerification                                                       │
  │ (adversarial + convergence + deductive + computational            │
  │  + contrastive + consistency + argument analysis)                 │
  │ Routing determines which tracks fire. All run sequentially.       │
  ▼                                                                   │
ResolveUncertainties ◄───────────────────────────┐                    │
  │ (resolve all blocking uncertainties)          │                    │
  │ (deduplicate remaining concerns)              │                    │
  │                                               │                    │
  ├── new blocking uncertainties created ────────►│                    │
  │   (from concern dedup, depth < 3)                                 │
  │                                                                   │
  └── all resolved ──► IntegrateEvidence                              │
                           │                                          │
                           ▼                                          │
                     PromoteToProvisional                             │
                           │ (gate check S→P)                         │
                           │                                          │
                           ├── pass ──► PromoteToRobust               │
                           │                 │                         │
                           │                 ├── pass ──► GeneratePredictions
                           │                 │                 │       │
                           │                 │                 ▼       │
                           │                 │          PromoteToActionable
                           │                 │                 │       │
                           │                 │                 ▼       │
                           │                 │          RecordDecision │
                           │                 │                 │       │
                           │                 └── fail ──►──────┤       │
                           │                                   │       │
                           └── fail ──►────────────────────────┤       │
                                                               │       │
◄──────────────────────────────────────────────────────────────┘       │
│                                                                      │
▼                                                                      │
CheckCompletion ◄──────────────────────────────────────────────────────┘
  │ (are all claims at terminal state?)
  ▼
Synthesize
  │ (freeze_snapshot + synthesize_report)
  ▼
End(EpistemicResult)
```

---

## File Structure

**New files:**
```
src/andamentum/epistemic/
  graph/
    __init__.py          — exports epistemic_graph, run_epistemic_graph
    state.py             — EpistemicGraphState dataclass
    deps.py              — EpistemicDeps dataclass
    nodes.py             — all graph node classes (~15 nodes)
    result.py            — EpistemicResult dataclass (End value)
  tests/
    test_graph.py        — graph node tests
```

**Modified files:**
- `operations_runner.py` — `run_research_question()` delegates to `run_epistemic_graph()` instead of the pattern scheduler loop
- `cli_handlers.py` — if it calls the runner directly

**Preserved files (no changes):**
- All `operations/*.py` — operation logic stays
- All `agents/*.py` — agent definitions stay
- All `entities/*.py` — entity types stay
- `gates.py` — gate logic stays (called by promote nodes)
- `routing.py` — routing logic stays (called by verification node)
- `confidence.py` — posterior computation stays (called after graph completes)

**Deprecated (can be removed after migration):**
- `patterns.py` — WORK_PATTERNS, PatternScheduler (replaced by graph)
- Attempt counting logic in operations_runner.py
- `_maybe_advance_phase` in scrutiny.py
- `_force_synthesis_if_needed` in operations_runner.py

---

## Task 1: Create state and deps dataclasses

**Files:**
- Create: `src/andamentum/epistemic/graph/__init__.py`
- Create: `src/andamentum/epistemic/graph/state.py`
- Create: `src/andamentum/epistemic/graph/deps.py`
- Create: `src/andamentum/epistemic/graph/result.py`

- [ ] **Step 1: Create `graph/state.py`**

```python
"""Mutable state for the epistemic graph.

Passed to every node via ctx.state. Nodes mutate it in place.
Tracks which phase the pipeline is in and per-claim progress.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EpistemicGraphState:
    """Shared mutable state for a single epistemic inquiry."""

    # Objective
    objective_id: str = ""
    question: str = ""
    question_type: str | None = None
    skip_preplanning: bool = False

    # Evidence
    evidence_extracted: bool = False

    # Claims
    claims_created: bool = False
    claim_ids: list[str] = field(default_factory=list)

    # Per-claim verification progress (claim_id -> done)
    verification_complete: dict[str, bool] = field(default_factory=dict)
    uncertainties_resolved: dict[str, bool] = field(default_factory=dict)
    integration_complete: dict[str, bool] = field(default_factory=dict)

    # Cycle counters (per claim_id)
    investigation_counts: dict[str, int] = field(default_factory=dict)

    # Synthesis
    synthesized: bool = False

    # Trace
    operations_log: list[dict[str, Any]] = field(default_factory=list)
```

- [ ] **Step 2: Create `graph/deps.py`**

```python
"""Immutable dependencies for the epistemic graph.

Passed to every node via ctx.deps. Not modified during execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class EpistemicDeps:
    """Infrastructure dependencies for graph execution."""

    repo: Any  # EpistemicRepository
    agent_runner: Any  # AgentRunner or None
    evidence_gatherer: Any | None = None
    quality_scorer: Any | None = None
    embedding_model: str | None = None
    provider: str = "all"
    verbose: bool = False
```

- [ ] **Step 3: Create `graph/result.py`**

```python
"""Result type for the epistemic graph End node."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EpistemicResult:
    """Final output of an epistemic graph run."""

    objective_id: str
    status: str  # "complete", "partial"
    iterations: int = 0
    successful: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
```

- [ ] **Step 4: Create `graph/__init__.py` with placeholder**

```python
"""Epistemic pipeline as a pydantic-graph DAG.

Replaces the pattern-based scheduler with explicit node dependencies.
Every scheduling decision is a typed return value, not a pattern match.
"""
```

- [ ] **Step 5: Test, commit**

```bash
uv run pyright src/andamentum/epistemic/graph/
uv run pytest src/andamentum/epistemic/tests/ -v
git commit -m "feat(epistemic): create graph state, deps, and result types"
```

---

## Task 2: Implement preplanning and evidence nodes

The entry path: PrepareObjective → PlanEvidence → ExtractEvidence → CreateClaims.

**Files:**
- Create: `src/andamentum/epistemic/graph/nodes.py`
- Test: `src/andamentum/epistemic/tests/test_graph.py`

- [ ] **Step 1: Implement PrepareObjective node**

```python
from pydantic_graph import BaseNode, End, GraphRunContext
from .state import EpistemicGraphState
from .deps import EpistemicDeps
from .result import EpistemicResult

@dataclass
class PrepareObjective(BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]):
    """Entry node: clarify question, classify type, run conceptual analysis."""

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "PlanEvidence":
        state = ctx.state
        deps = ctx.deps

        if not state.skip_preplanning:
            # Import and run operations
            from ..operations import (
                ClarifyQuestionOperation,
                ClassifyQuestionOperation,
                ConceptualAnalysisOperation,
            )
            # ... create WorkItem, execute each operation
            # Update objective phase as operations complete

        return PlanEvidence()
```

Each node follows this pattern: import the operation, create a WorkItem, call `op.execute(work)`, update state, return the next node.

- [ ] **Step 2: Implement PlanEvidence, ExtractEvidence, CreateClaims nodes**

Each wraps the corresponding operation. ExtractEvidence loops over all unextracted evidence entities.

- [ ] **Step 3: Write basic tests**

Test that nodes return the correct next node type for given states.

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(epistemic): implement preplanning and evidence graph nodes"
```

---

## Task 3: Implement scrutiny and investigation cycle

The core Peirce loop: Scrutinize ↔ Investigate ↔ ExtractEvidence.

- [ ] **Step 1: Implement Scrutinize node**

```python
@dataclass
class Scrutinize(BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]):
    """Run scrutiny on all claims. Branch based on verdict."""

    async def run(self, ctx: ...) -> Union[
        "PromoteToSupported",
        "Investigate",
        "AbandonOrDemote",
        "CheckCompletion",
    ]:
        # Run scrutinise_claim on each claim with verdict=None
        # Check verdicts:
        #   - all pass → PromoteToSupported
        #   - any needs_resolution with count < 3 → Investigate
        #   - any needs_resolution with count >= 3 → AbandonOrDemote
        #   - any fail at HYPOTHESIS with count < 3 → Investigate
        #   - any fail at SUPPORTED+ → DemoteThenScrutinize
        ...
```

- [ ] **Step 2: Implement Investigate node**

```python
@dataclass
class Investigate(BaseNode[...]):
    """Investigate evidence gaps (Peirce cycling). Creates evidence stubs."""

    async def run(self, ctx: ...) -> "ExtractNewEvidence":
        # Run investigate_claim for claims needing investigation
        # Increment investigation_counts in state
        return ExtractNewEvidence()
```

- [ ] **Step 3: Implement ExtractNewEvidence node**

Extracts newly created evidence stubs, then returns to Scrutinize.

```python
@dataclass
class ExtractNewEvidence(BaseNode[...]):
    """Extract evidence from investigation stubs, then re-scrutinize."""

    async def run(self, ctx: ...) -> "Scrutinize":
        # Extract all unextracted evidence
        return Scrutinize()
```

- [ ] **Step 4: Implement AbandonOrDemote node**

```python
@dataclass
class AbandonOrDemote(BaseNode[...]):
    """Handle exhausted claims: abandon at HYPOTHESIS, demote at SUPPORTED+."""

    async def run(self, ctx: ...) -> "Scrutinize" | "CheckCompletion":
        # For each exhausted claim:
        #   - HYPOTHESIS → abandon
        #   - SUPPORTED+ → demote to HYPOTHESIS (resets scrutiny)
        # If all claims are now terminal → CheckCompletion
        # Else → Scrutinize (re-scrutinize demoted claims)
        ...
```

- [ ] **Step 5: Test the investigation cycle**

Test that Scrutinize → Investigate → ExtractNewEvidence → Scrutinize cycles correctly, and terminates after 3 investigations.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(epistemic): implement scrutiny-investigation cycle as graph nodes"
```

---

## Task 4: Implement verification and integration nodes

After promotion to SUPPORTED: routing defaults → verification tracks → resolve uncertainties → integrate.

- [ ] **Step 1: Implement PromoteToSupported node**

```python
@dataclass
class PromoteToSupported(BaseNode[...]):
    """Promote passing HYPOTHESIS claims to SUPPORTED, set routing defaults."""

    async def run(self, ctx: ...) -> "RunVerification":
        # For each claim with scrutiny_verdict=pass at HYPOTHESIS:
        #   - Run promote_claim (H→S) — gate includes adversarial survival
        #   - Run set_routing_defaults
        return RunVerification()
```

- [ ] **Step 2: Implement RunVerification node**

This is the composite verification node. It runs each track sequentially based on routing activation.

```python
@dataclass
class RunVerification(BaseNode[...]):
    """Run all verification tracks on SUPPORTED claims."""

    async def run(self, ctx: ...) -> "ResolveUncertainties":
        # For each SUPPORTED claim:
        #   Get routing profile for question_type
        #   For each track (adversarial, convergence, deductive, etc.):
        #     If PRIMARY or SECONDARY (condition met): run the operation
        #   Run analyze_argument
        return ResolveUncertainties()
```

- [ ] **Step 3: Implement ResolveUncertainties node**

```python
@dataclass
class ResolveUncertainties(BaseNode[...]):
    """Resolve all blocking uncertainties, dedup concerns."""

    async def run(self, ctx: ...) -> "IntegrateEvidence" | "ResolveUncertainties":
        # Run resolve_uncertainty on all unresolved blocking uncertainties
        # Run deduplicate_concerns
        # If new blocking uncertainties were created → loop (self)
        # Else → IntegrateEvidence
        ...
```

- [ ] **Step 4: Implement IntegrateEvidence node**

```python
@dataclass
class IntegrateEvidence(BaseNode[...]):
    """Holistic evidence assessment (Peirce abduction)."""

    async def run(self, ctx: ...) -> "PromoteToProvisional":
        # Run integrate_evidence on each SUPPORTED claim
        return PromoteToProvisional()
```

- [ ] **Step 5: Test verification pipeline**

Test: PromoteToSupported → RunVerification → ResolveUncertainties → IntegrateEvidence → PromoteToProvisional.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(epistemic): implement verification and integration graph nodes"
```

---

## Task 5: Implement promotion chain and synthesis

The promotion ladder: S→P→R→A, with predictions and decisions.

- [ ] **Step 1: Implement PromoteToProvisional, PromoteToRobust, PromoteToActionable**

Each wraps `promote_claim` with the appropriate gate. If the gate fails, the node proceeds to CheckCompletion (the claim stays at its current stage — this is fine, not an error).

- [ ] **Step 2: Implement GeneratePredictions and RecordDecision**

Wrap the existing operations. Only fire at ROBUST and ACTIONABLE respectively.

- [ ] **Step 3: Implement CheckCompletion and Synthesize**

```python
@dataclass
class CheckCompletion(BaseNode[...]):
    """Check if all claims are at terminal state. If so, synthesize."""

    async def run(self, ctx: ...) -> "Synthesize" | End[EpistemicResult]:
        # If all claims are abandoned or at PROVISIONAL+:
        return Synthesize()
        # (No claims exist or all failed):
        # return End(EpistemicResult(status="partial"))

@dataclass
class Synthesize(BaseNode[...]):
    """Freeze snapshot and generate report. Terminal."""

    async def run(self, ctx: ...) -> End[EpistemicResult]:
        # freeze_snapshot
        # synthesize_report
        # compute_posterior
        return End(EpistemicResult(
            objective_id=ctx.state.objective_id,
            status="complete",
            ...
        ))
```

- [ ] **Step 4: Assemble the graph**

```python
from pydantic_graph import Graph

epistemic_graph = Graph(
    nodes=[
        PrepareObjective,
        PlanEvidence,
        ExtractEvidence,
        CreateClaims,
        Scrutinize,
        Investigate,
        ExtractNewEvidence,
        AbandonOrDemote,
        PromoteToSupported,
        RunVerification,
        ResolveUncertainties,
        IntegrateEvidence,
        PromoteToProvisional,
        PromoteToRobust,
        GeneratePredictions,
        PromoteToActionable,
        RecordDecision,
        CheckCompletion,
        Synthesize,
    ],
    name="epistemic_pipeline",
)
```

pydantic-graph validates all edges at construction time. If any node returns a type not in the list, it raises `GraphSetupError` immediately.

- [ ] **Step 5: Test the full graph with mock operations**

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(epistemic): implement promotion chain, synthesis, and graph assembly"
```

---

## Task 6: Wire the graph into the runner

Replace the `while True` scheduler loop in `operations_runner.py` with a graph execution call.

- [ ] **Step 1: Create `run_epistemic_graph()` function**

```python
async def run_epistemic_graph(
    question: str,
    database_name: str,
    verbose: bool = False,
    model: str | None = None,
    embedding_model: str | None = None,
    ...
) -> PatternSchedulerResult:
    """Run a research question through the epistemic graph."""

    # Initialize repo, runner, gatherer (same as current run_research_question)
    ...

    # Build state and deps
    state = EpistemicGraphState(
        objective_id=objective_id,
        question=question,
        skip_preplanning=skip_preplanning,
    )
    deps = EpistemicDeps(
        repo=repo,
        agent_runner=agent_runner,
        evidence_gatherer=evidence_gatherer,
        embedding_model=embedding_model,
        quality_scorer=quality_scorer,
        verbose=verbose,
    )

    # Run graph
    from .graph import epistemic_graph
    from pydantic_graph import FullStatePersistence

    persistence = FullStatePersistence()
    result = await epistemic_graph.run(
        PrepareObjective(),
        state=state,
        deps=deps,
        persistence=persistence,
    )

    # Convert to PatternSchedulerResult for backward compatibility
    return PatternSchedulerResult(
        objective_id=result.output.objective_id,
        iterations=len(persistence.history),
        successful=result.output.successful,
        failed=result.output.failed,
        status=result.output.status,
        errors=result.output.errors,
        posterior=posterior_report,
    )
```

- [ ] **Step 2: Update `run_research_question` to delegate**

```python
async def run_research_question(...) -> PatternSchedulerResult:
    return await run_epistemic_graph(...)
```

- [ ] **Step 3: Run the full test suite**

All existing tests should still pass because the operations they test haven't changed. The integration tests may need updates if they depend on pattern scheduler internals.

- [ ] **Step 4: Run the ADAR1 test case end-to-end**

```bash
uv run andamentum-epistemic ask "ADAR1 binds to Dicer to cleave pre-miRNA" \
    --model openai:gpt-5.4-nano --verbose --keep --name graph_test
```

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(epistemic): wire graph scheduler into operations runner"
```

---

## Task 7: Cleanup — remove pattern scheduler

After the graph is working:

- [ ] **Step 1: Mark `patterns.py` as deprecated**

Don't delete yet — keep for reference and in case rollback is needed. Add deprecation notice at top.

- [ ] **Step 2: Remove pattern scheduler hacks from operations**

Remove `_maybe_advance_phase` from scrutiny.py (the graph handles flow control).
Remove `_force_synthesis_if_needed` from operations_runner.py.
Remove attempt counting logic.

- [ ] **Step 3: Update CLAUDE.md**

Document the graph architecture, remove references to pattern scheduler.

- [ ] **Step 4: Final test suite run**

```bash
uv run pytest
uv run pyright
uv run ruff check
```

---

## Risk Mitigation

**Rollback path:** The pattern scheduler stays in `patterns.py` (deprecated, not deleted). If the graph has issues, `run_research_question` can be switched back to the old loop by changing one function call.

**Incremental testing:** Each task produces a working commit. The graph can be tested node-by-node with mock operations before wiring into the runner.

**Operation preservation:** No operation code changes. Every `execute()` method stays identical. The graph only changes WHEN operations are called, not WHAT they do.

**Type safety:** pydantic-graph validates all edges at graph construction time. If a node returns a type not in the graph, it fails immediately — not at runtime after 30 minutes of execution.

**Trace capability:** `FullStatePersistence` records every node execution with timestamps and state snapshots. This replaces the execution_step recording in the current runner and provides better debugging.
