# Clean Architecture: Separate Flow Control from Operations

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make operations pure transforms (read entity → do work → write result) and move all flow-control logic into the graph nodes. No operation should manipulate entity fields to signal what the graph should do next.

**Architecture:** Three changes: (1) enrich graph state so nodes track pipeline progress directly, (2) remove flow-control code from operations, (3) remove dead entity fields and rename stale identifiers.

**Tech Stack:** Python 3.12, pydantic-graph 1.84.1, pytest (asyncio_mode=auto)

---

## Principles

These five rules apply to ALL code in the epistemic module going forward.

### P1: Operations are pure transforms

An operation reads entities, does work (LLM calls, computations, validation), and writes the RESULT of that work back to the entity. It returns an `OperationResult` describing what happened.

An operation NEVER:
- Resets fields on OTHER entities to signal the graph
- Sets flags to trigger other operations
- Checks or changes the objective's phase
- Decides what should happen next

### P2: The graph is the sole flow controller

Only graph nodes decide what runs next. They decide based on:
- The `OperationResult` returned by the operation
- Entity state queried from the repo (read-only for decisions)
- Graph state (`EpistemicGraphState`) that nodes manage themselves

### P3: Entity fields are data, not signals

Every field on Claim, Evidence, Objective represents something real about that entity — a measurement, a verdict, a score. No field exists solely to tell the scheduler what to do.

Test: "If I deleted the graph and read this entity from the database, would this field still be meaningful?" If no, it's a signal, not data.

### P4: Graph state tracks pipeline progress

The graph state (`EpistemicGraphState`) tracks what work has been done and what needs doing. This is WHERE flow-control state lives — not on entities.

### P5: Operations don't reach across entity boundaries

An operation on a Claim should not modify a different Claim. An operation on an Uncertainty should not modify Claims. Cross-entity effects are the graph's job — the node calls one operation, reads the result, and decides whether to call another operation on a different entity.

---

## Current Violations

### V1: ResolveUncertaintyOperation resets scrutiny_verdict on other claims

**File:** `operations/uncertainty.py:195-203`
**What:** After resolving a blocking uncertainty, the operation iterates `uncertainty.affected_claim_ids` and sets `claim.scrutiny_verdict = None` on each.
**Why it's wrong:** This is P5 (cross-entity reach) + P2 (flow control in operation). The operation is telling the Scrutinize node "re-process these claims" by manipulating their state.
**Fix:** Remove the reset from the operation. The graph's ResolveUncertainties node reads `uncertainty.affected_claim_ids` from the resolved uncertainty and adds those claim IDs to `state.claims_needing_rescrutiny`. The Scrutinize node checks both `scrutiny_verdict is None` (never scrutinized) AND `state.claims_needing_rescrutiny` (need re-scrutiny).

### V2: InvestigateClaimOperation resets scrutiny_verdict and sets needs_revalidation

**File:** `operations/investigation.py:363-374`
**What:** After creating evidence stubs, the operation sets `claim.scrutiny_verdict = None` (flow signal for re-scrutiny) and `claim.needs_revalidation = True` (flow signal for TMS).
**Why it's wrong:** P2 + P3. Both are signals for the graph, not data about the investigation.
**Fix:** Remove both. The graph's Investigate node knows it just created evidence stubs (the `OperationResult.created_entities` tells it). It adds the claim to `state.claims_needing_rescrutiny`. It adds the claim to `state.claims_needing_tms` if the claim is promoted (stage != HYPOTHESIS). `investigation_count` stays — it IS data (how many times this claim was investigated).

### V3: ExtractEvidenceOperation triggers TMS via needs_revalidation

**File:** `operations/evidence.py:218-249`
**What:** After judging evidence as "contradicts," the operation checks the support/contradict balance and sets `linked_claim.needs_revalidation = True` if the balance tips.
**Why it's wrong:** P5 (evidence operation modifying a claim) + P2 (flow signal). The evidence operation's job is to extract and judge evidence. Whether that triggers TMS is the graph's decision.
**Fix:** Remove the TMS trigger. The graph's `_run_tms_sweep` already runs after extraction nodes and after verification. It handles the cascade.

### V4: routing_applied field on Claim

**File:** `entities/claim.py:122-125`
**What:** Set by `SetRoutingDefaultsOperation`, reset by `record_demotion()`. Never read by the graph.
**Why it's wrong:** P3 — exists solely to tell the old pattern scheduler "don't run routing defaults again."
**Fix:** Remove the field entirely.

### V5: needs_revalidation field on Claim

**File:** `entities/claim.py:148-149`
**What:** Set by operations (V2, V3) as a TMS signal. Read by `RevalidateClaimOperation` and `PromoteClaimOperation` (guard).
**Why it's wrong:** P3 — it's a flow signal, not data. "This claim needs revalidation" is a pipeline state, not an entity property.
**Fix:** Move to graph state: `state.claims_needing_tms: set[str]`. The TMS sweep reads this set. `PromoteClaimOperation`'s guard (`if claim.needs_revalidation: refuse`) can check graph state instead — but since promote is called FROM the graph, the graph can simply not call promote on claims that need TMS first.

### V6: WorkItem naming

**File:** `operations/base.py`
**What:** Named after the pattern scheduler concept of "work items." In the graph, it's just the input to an operation.
**Fix:** Rename to `OperationInput`.

### V7: OPERATION_CLASSES and create_operations in public API

**File:** `operations/__init__.py`, `__init__.py`
**What:** Registry for pattern scheduler lookup. Graph uses `_make_op()` directly.
**Fix:** Remove from `__init__.py` `__all__`. Keep in `operations/__init__.py` for tests.

---

## File Structure

**Modified files:**
- `src/andamentum/epistemic/graph/state.py` — add flow-control fields
- `src/andamentum/epistemic/graph/nodes.py` — use graph state for re-scrutiny and TMS
- `src/andamentum/epistemic/operations/uncertainty.py` — remove scrutiny_verdict reset
- `src/andamentum/epistemic/operations/investigation.py` — remove scrutiny_verdict reset + needs_revalidation
- `src/andamentum/epistemic/operations/evidence.py` — remove TMS trigger
- `src/andamentum/epistemic/operations/stage_management.py` — remove needs_revalidation guard (graph handles)
- `src/andamentum/epistemic/operations/base.py` — rename WorkItem to OperationInput
- `src/andamentum/epistemic/entities/claim.py` — remove routing_applied, needs_revalidation
- `src/andamentum/epistemic/patterns.py` — update re-export
- `src/andamentum/epistemic/__init__.py` — update exports
- All operation files + tests — WorkItem → OperationInput rename
- `CLAUDE.md` — update principles

---

## Task 1: Enrich graph state with flow-control fields

Add the fields that will replace entity-level flow signals.

**Files:**
- Modify: `src/andamentum/epistemic/graph/state.py`

- [ ] **Step 1: Add flow-control fields to EpistemicGraphState**

```python
    # ── Flow control (graph-managed, not on entities) ───────────
    # Claims that need re-scrutiny after uncertainty resolution
    # or investigation. The Scrutinize node checks this set in
    # addition to claims with scrutiny_verdict=None.
    claims_needing_rescrutiny: set[str] = field(default_factory=set)

    # Claims that need TMS revalidation after evidence changes.
    # The _run_tms_sweep helper checks this set.
    claims_needing_tms: set[str] = field(default_factory=set)
```

Add these after the existing `terminal_claims` field.

- [ ] **Step 2: Run tests, verify, commit**

```bash
uv run pytest src/andamentum/epistemic/tests/ -v
uv run pyright src/andamentum/epistemic/graph/state.py
git add -A && git commit -m "feat(graph): add flow-control fields to graph state"
```

---

## Task 2: Move flow control from operations to graph nodes

Remove flow-control code from three operations and move it into the graph nodes.

**Files:**
- Modify: `src/andamentum/epistemic/operations/uncertainty.py`
- Modify: `src/andamentum/epistemic/operations/investigation.py`
- Modify: `src/andamentum/epistemic/operations/evidence.py`
- Modify: `src/andamentum/epistemic/graph/nodes.py`

- [ ] **Step 1: Remove scrutiny_verdict reset from ResolveUncertaintyOperation**

In `src/andamentum/epistemic/operations/uncertainty.py`, delete lines 188-203 (the entire "Peirce cycling" block that resets `scrutiny_verdict = None` on affected claims):

```python
        # DELETE THIS BLOCK:
        # Peirce cycling: when a blocking uncertainty is resolved, the
        # epistemic landscape has changed. Reset scrutiny on affected claims
        # ...
        if uncertainty.resolution is not None and uncertainty.is_blocking:
            for cid in uncertainty.affected_claim_ids:
                try:
                    claim = await self.repo.get("claim", cid)
                    if isinstance(claim, Claim) and not claim.abandoned:
                        claim.scrutiny_verdict = None
                        await self.repo.save(claim)
                except Exception:
                    continue
```

The operation now just resolves the uncertainty and returns. Flow control moves to the graph.

- [ ] **Step 2: Remove scrutiny_verdict reset and needs_revalidation from InvestigateClaimOperation**

In `src/andamentum/epistemic/operations/investigation.py`, change lines 363-375 from:

```python
        # Reset scrutiny verdict so claim re-enters scrutiny after new evidence is extracted
        claim.scrutiny_verdict = None
        claim.investigation_count += 1
        claim.evidence_count = len(claim.evidence_ids)

        # TMS trigger: if the claim is already promoted...
        if created_entities and claim.stage != ClaimStage.HYPOTHESIS:
            claim.needs_revalidation = True
```

To:

```python
        claim.investigation_count += 1
        claim.evidence_count = len(claim.evidence_ids)
```

Keep `investigation_count` (data about the claim) and `evidence_count` (denormalized count). Remove `scrutiny_verdict = None` (flow signal) and `needs_revalidation = True` (flow signal).

- [ ] **Step 3: Remove TMS trigger from ExtractEvidenceOperation**

In `src/andamentum/epistemic/operations/evidence.py`, delete lines 218-249 (the TMS trigger block that sets `needs_revalidation` when contradicting evidence tips the balance). Keep the evidence judging code above it (lines 189-217) — judging IS the operation's job.

- [ ] **Step 4: Update graph nodes to handle flow control**

In `src/andamentum/epistemic/graph/nodes.py`:

**ResolveUncertainties node:** After calling `_run_op` for each uncertainty, read the resolved uncertainty's `affected_claim_ids` and add them to `state.claims_needing_rescrutiny`:

```python
        for unc in blocking:
            result = await _run_op(
                ResolveUncertaintyOperation, deps, state,
                unc.entity_id, "uncertainty", "resolve_uncertainty",
            )
            # Flow control: mark affected claims for re-scrutiny
            if result.success:
                unc_updated = await deps.repo.get("uncertainty", unc.entity_id)
                if unc_updated.is_blocking and unc_updated.resolution is not None:
                    for cid in unc_updated.affected_claim_ids:
                        state.claims_needing_rescrutiny.add(cid)
```

**Investigate node:** After calling investigate_claim, add the claim to `state.claims_needing_rescrutiny` and `state.claims_needing_tms` (if promoted):

```python
            result = await _run_op(
                InvestigateClaimOperation, deps, state,
                claim.entity_id, "claim", "investigate_claim",
            )
            if result.success:
                state.claims_needing_rescrutiny.add(claim.entity_id)
                # TMS: if claim is promoted and new evidence was created
                claim_updated = await deps.repo.get("claim", claim.entity_id)
                if claim_updated.stage != ClaimStage.HYPOTHESIS and result.created_entities:
                    state.claims_needing_tms.add(claim.entity_id)
```

**Scrutinize node:** Check BOTH `scrutiny_verdict is None` AND `state.claims_needing_rescrutiny`:

Change the filter from:
```python
        for claim in active_claims:
            if claim.scrutiny_verdict is None:
```

To:
```python
        for claim in active_claims:
            if claim.scrutiny_verdict is None or claim.entity_id in state.claims_needing_rescrutiny:
                # Reset verdict for re-scrutiny so the operation actually runs
                if claim.entity_id in state.claims_needing_rescrutiny:
                    claim.scrutiny_verdict = None
                    await deps.repo.save(claim)
                    state.claims_needing_rescrutiny.discard(claim.entity_id)
```

Note: we still set `scrutiny_verdict = None` on the entity — but now it's the GRAPH NODE doing it (the conductor), not the operation (the musician). The graph decides which claims need re-scrutiny; the entity field reflects the graph's decision.

**_run_tms_sweep:** Also process `state.claims_needing_tms`:

```python
    # Step 2b: Process claims flagged for TMS by graph nodes
    for cid in list(state.claims_needing_tms):
        try:
            claim = await deps.repo.get("claim", cid)
            if isinstance(claim, Claim) and not claim.abandoned:
                claim.needs_revalidation = True
                await deps.repo.save(claim)
        except Exception:
            pass
    state.claims_needing_tms.clear()
```

Wait — this still uses `needs_revalidation` on the entity. That's V5 which we handle in the next task. For now, keep this bridge — the TMS operations still read `needs_revalidation`. We'll clean that up in Task 3.

- [ ] **Step 5: Update tests**

The tests in `test_peirce_cycling.py` test that `ResolveUncertaintyOperation` resets `scrutiny_verdict`. Those tests need to be updated to verify the operation does NOT reset it (the graph does).

Similarly, tests for `InvestigateClaimOperation` that check `scrutiny_verdict = None` after investigation need updating.

- [ ] **Step 6: Run full test suite, verify, commit**

```bash
uv run pytest -v
uv run pyright
uv run ruff check
git add -A && git commit -m "refactor(epistemic): move flow control from operations to graph nodes"
```

---

## Task 3: Remove dead entity fields

Remove `routing_applied` and `needs_revalidation` from Claim. Move TMS triggering fully into graph state.

**Files:**
- Modify: `src/andamentum/epistemic/entities/claim.py`
- Modify: `src/andamentum/epistemic/operations/stage_management.py`
- Modify: `src/andamentum/epistemic/operations/belief_maintenance.py`
- Modify: `src/andamentum/epistemic/graph/nodes.py`
- Modify: tests

- [ ] **Step 1: Remove `routing_applied` from Claim**

In `entities/claim.py`:
- Delete the field definition (line 122-125)
- Delete from `record_demotion()` reset block
- Delete from `_extra_metadata()`
- Delete from `_from_metadata()`

In `operations/belief_maintenance.py` (SetRoutingDefaultsOperation):
- Remove `claim.routing_applied = True` line

- [ ] **Step 2: Replace needs_revalidation with graph-driven TMS**

In `entities/claim.py`:
- Delete the `needs_revalidation` field (line 148-149)
- Delete from `_extra_metadata()`
- Delete from `_from_metadata()`

In `operations/stage_management.py` (PromoteClaimOperation):
- Remove the `needs_revalidation` guard:
```python
        # DELETE:
        if claim.needs_revalidation:
            return OperationResult(
                success=False,
                entity_id=claim.entity_id,
                message="Revalidation pending — TMS must run first",
            )
```
The graph handles this — it runs TMS sweep before calling promote.

In `operations/belief_maintenance.py` (RevalidateClaimOperation):
- Change from checking `claim.needs_revalidation` to always running when called (the graph only calls it on claims that need it)
- Remove `claim.needs_revalidation = False` lines

In `graph/nodes.py` (_run_tms_sweep):
- Instead of checking `claim.needs_revalidation`, iterate `state.claims_needing_tms` directly and call RevalidateClaimOperation on each

- [ ] **Step 3: Update tests, verify, commit**

```bash
uv run pytest -v
uv run pyright
uv run ruff check
git add -A && git commit -m "refactor(epistemic): remove routing_applied and needs_revalidation from Claim"
```

---

## Task 4: Rename WorkItem to OperationInput

Mechanical rename across the codebase.

**Files:**
- Modify: `src/andamentum/epistemic/operations/base.py` — rename class
- Modify: `src/andamentum/epistemic/patterns.py` — update re-export
- Modify: `src/andamentum/epistemic/graph/nodes.py` — update import + usage
- Modify: All 13 operation files — update import + type annotations
- Modify: All test files that use WorkItem
- Modify: `src/andamentum/epistemic/__init__.py` — update export

- [ ] **Step 1: Rename the class in base.py**

```python
@dataclass
class OperationInput:
    """Input for an epistemic operation.

    Specifies which entity to process and which operation to run.
    """
    entity_id: str
    entity_type: str
    operation: str
    metadata: dict[str, Any] = field(default_factory=dict)

# Backward compatibility
WorkItem = OperationInput
```

Keep the alias for backward compat.

- [ ] **Step 2: Update imports across all files**

Replace `from ..patterns import WorkItem` and `from .base import WorkItem` with `from .base import OperationInput` in all operation files. Use find-and-replace.

- [ ] **Step 3: Update patterns.py re-export**

```python
from .operations.base import OperationInput, OperationInput as WorkItem  # backward compat

__all__ = ["OperationInput", "WorkItem"]
```

- [ ] **Step 4: Update __init__.py exports**

Replace `WorkItem` with `OperationInput` in `__all__`. Keep `WorkItem` as a re-export alias.

- [ ] **Step 5: Run tests, verify, commit**

```bash
uv run pytest -v
uv run pyright
uv run ruff check
git add -A && git commit -m "refactor(epistemic): rename WorkItem to OperationInput"
```

---

## Task 5: Remove OPERATION_CLASSES from public API and final cleanup

- [ ] **Step 1: Remove from `__init__.py` exports**

Remove `OPERATION_CLASSES` and `create_operations` from `__all__` in `src/andamentum/epistemic/__init__.py`. Keep them in `operations/__init__.py` for internal use and tests.

- [ ] **Step 2: Update CLAUDE.md with clean architecture principles**

Add the five principles (P1-P5) to the architectural conventions section. Update the description of `operations/` to note they are pure transforms.

- [ ] **Step 3: Final verification**

```bash
uv run pytest
uv run pyright
uv run ruff check
# Live test:
uv run andamentum-epistemic ask "Does exercise reduce depression?" \
    --model openai:gpt-5.4-mini --verbose --keep --name clean_test
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor(epistemic): clean architecture - operations as pure transforms"
```

---

## Self-Review

**Spec coverage:**
- V1 (uncertainty scrutiny reset) → Task 2, Step 1
- V2 (investigation scrutiny reset + needs_revalidation) → Task 2, Step 2
- V3 (evidence TMS trigger) → Task 2, Step 3
- V4 (routing_applied removal) → Task 3, Step 1
- V5 (needs_revalidation removal) → Task 3, Step 2
- V6 (WorkItem rename) → Task 4
- V7 (OPERATION_CLASSES from public API) → Task 5, Step 1
- P1-P5 principles documented → Task 5, Step 2

**Risk assessment:**
- Task 1 (graph state): Zero risk — adding fields
- Task 2 (move flow control): Medium risk — changes 3 operations + graph nodes + tests
- Task 3 (remove entity fields): Medium risk — changes entity schema, operations, graph
- Task 4 (rename): Low risk — mechanical, alias for backward compat
- Task 5 (cleanup): Low risk — removing exports, updating docs

**Test strategy:**
- Each task runs the full test suite
- Task 2 requires updating `test_peirce_cycling.py` (tests the old behavior)
- Task 3 requires updating `test_tms.py` (tests `needs_revalidation`)
- Task 5 includes a live end-to-end test
