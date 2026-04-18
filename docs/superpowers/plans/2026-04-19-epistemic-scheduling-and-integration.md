# Epistemic Scheduling Fixes and Abductive Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all pattern scheduler deadlock states, align gate logic with philosophical traditions, and add the missing abductive integration step that lets the system reason across evidence collectively.

**Architecture:** Three layers of changes, each independently testable:
1. **Deadlock elimination** (Tasks 1-3) — fix all known dead-end states in the pattern scheduler so no claim can get stuck in a non-terminal state with no matching pattern.
2. **Gate realignment** (Tasks 4-5) — fix the promote gate to accept adversarial survival as positive signal (Popper), and fix the predictive question circular dependency.
3. **Abductive integration** (Tasks 6-7) — add the missing step between per-item evidence judgment and posterior computation, where the system reasons holistically across all evidence.

**Tech Stack:** Python 3.12, pydantic, pytest (asyncio_mode=auto), pyright, ruff

**Key design principle:** Every claim state must either (a) match at least one pattern, or (b) be terminal (abandoned, or at the final stage with all operations complete). No silent dead ends.

---

## File Structure

**Modified files:**
- `src/andamentum/epistemic/patterns.py` — add abandonment pattern, remove saturation from investigation filter
- `src/andamentum/epistemic/operations/scrutiny.py` — remove saturation check entirely
- `src/andamentum/epistemic/entities/claim.py` — add `integrated_assessment` field, remove `saturated` field
- `src/andamentum/epistemic/gates.py` — add adversarial-survival gate path, fix predictive circular dep
- `src/andamentum/epistemic/confidence.py` — replace counting-based posterior with integration-informed posterior
- `src/andamentum/epistemic/operations/stage_management.py` — promote resets attempt counter after gate change
- `src/andamentum/epistemic/operations_runner.py` — reset entity attempts when new evidence is judged

**New files:**
- `src/andamentum/epistemic/operations/integration.py` — AbductiveIntegrationOperation
- `src/andamentum/epistemic/agents/integration.py` — agent definition for structured deliberation
- `src/andamentum/epistemic/agents/output_models.py` — IntegrationOutput model (append to existing)

**Test files:**
- `src/andamentum/epistemic/tests/test_deadlock_prevention.py` — new, tests all dead-end states are eliminated
- `src/andamentum/epistemic/tests/test_integration_operation.py` — new, tests abductive integration
- `src/andamentum/epistemic/tests/test_gates.py` — modify, add adversarial-survival tests

---

## Task 1: Eliminate saturation — replace with investigation cap

The `saturated` field and the saturation check in scrutiny cause two confirmed deadlock states (HYPOTHESIS/fail/saturated=True and HYPOTHESIS/needs_resolution/saturated=True). The saturation check is trying to prevent wasteful investigation, but `MAX_INVESTIGATION_ATTEMPTS=3` already does this. Saturation adds complexity and creates dead ends without preventing anything the investigation cap doesn't already prevent.

**Files:**
- Modify: `src/andamentum/epistemic/operations/scrutiny.py:364-382`
- Modify: `src/andamentum/epistemic/entities/claim.py` (remove `saturated` field)
- Modify: `src/andamentum/epistemic/patterns.py` (remove `saturated: False` from investigation filters)
- Modify: `src/andamentum/epistemic/tests/test_saturation.py` (rewrite — tests for investigation cap only)
- Test: `src/andamentum/epistemic/tests/test_deadlock_prevention.py`

- [ ] **Step 1: Remove the saturation check from scrutiny**

In `src/andamentum/epistemic/operations/scrutiny.py`, delete lines 364-382 (the entire saturation check block):

```python
# DELETE this entire block:
        # Saturation check: detect uninformative investigation cycles.
        # After at least 2 investigation cycles, if scrutiny still returns
        # ...
        if (
            claim.investigation_count >= 2
            and claim.scrutiny_verdict == "needs_resolution"
        ):
            blocking_unresolved = await self.repo.query(
                "uncertainty",
                affected_claim_ids__contains=claim.entity_id,
                resolution=None,
            )
            blocking_unresolved = [u for u in blocking_unresolved if u.is_blocking]

            if not blocking_unresolved:
                claim.saturated = True
```

- [ ] **Step 2: Remove `saturated` from investigation pattern filters**

In `src/andamentum/epistemic/patterns.py`, change the two investigation patterns to remove `"saturated": False`:

Pattern at line 187-197 (needs_resolution investigation):
```python
    Pattern(
        entity_type="claim",
        filters={
            "scrutiny_verdict": "needs_resolution",
            "investigation_count__lt": 3,
            "abandoned": False,
        },
        operation="investigate_claim",
        description="Investigate evidence gaps after ambiguous scrutiny",
    ),
```

Pattern at line 199-210 (fail at HYPOTHESIS investigation):
```python
    Pattern(
        entity_type="claim",
        filters={
            "scrutiny_verdict": "fail",
            "stage": ClaimStage.HYPOTHESIS.value,
            "investigation_count__lt": 3,
            "abandoned": False,
        },
        operation="investigate_claim",
        description="Investigate failed hypothesis before abandoning",
    ),
```

- [ ] **Step 3: Remove `saturated` field from Claim entity**

In `src/andamentum/epistemic/entities/claim.py`, remove the `saturated` field definition and all references to it in `record_demotion()` and `_extra_metadata()`. 

Remove from field definitions:
```python
    # DELETE:
    saturated: bool = Field(
        default=False,
        description="Whether investigation has stopped producing new information",
    )
```

Remove from `record_demotion()` (the line `self.saturated = False` inside the reset block).

Remove from `_extra_metadata()` (the line `"saturated": self.saturated`).

Remove from `_from_metadata()` (the line `saturated=metadata.get("saturated", False)`).

- [ ] **Step 4: Update test_saturation.py**

Rewrite `src/andamentum/epistemic/tests/test_saturation.py` to test that `MAX_INVESTIGATION_ATTEMPTS=3` is the sole investigation limiter. Remove all tests that reference `saturated`. The key test: after 3 investigation cycles, `InvestigateClaimOperation` sets `abandoned=True`.

- [ ] **Step 5: Run tests and verify**

```bash
uv run pytest src/andamentum/epistemic/tests/test_saturation.py -v
uv run pytest src/andamentum/epistemic/tests/ -v
uv run pyright src/andamentum/epistemic/
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(epistemic): remove saturation, rely on investigation cap alone"
```

---

## Task 2: Add abandonment pattern for exhausted hypotheses

After Task 1, claims at HYPOTHESIS with `scrutiny_verdict="fail"` and `investigation_count >= 3` should be abandoned by InvestigateClaimOperation. But if the investigation operation itself fails (exception, not `success=False`), the claim can be left at HYPOTHESIS/fail without being abandoned, because `investigation_count` stays below 3 while the entity attempt counter blocks scheduling.

Add a cleanup pattern that catches any HYPOTHESIS claim that can't make progress.

**Files:**
- Modify: `src/andamentum/epistemic/patterns.py`
- Create: `src/andamentum/epistemic/operations/cleanup.py`
- Modify: `src/andamentum/epistemic/operations/__init__.py`
- Test: `src/andamentum/epistemic/tests/test_deadlock_prevention.py`

- [ ] **Step 1: Create AbandonStaleClaimOperation**

Create `src/andamentum/epistemic/operations/cleanup.py`:

```python
"""Cleanup operations for stale claims.

Catches claims stuck in non-terminal states that no other operation
can advance. This is a safety net — if the system is working correctly,
this operation should rarely fire.

Depends on: base (BaseOperation, OperationResult)
Operates on: Claim entities
"""

from .base import BaseOperation, OperationResult, MAX_INVESTIGATION_ATTEMPTS
from ..entities import Claim, ClaimStage
from ..patterns import WorkItem


class AbandonStaleClaimOperation(BaseOperation):
    """Abandon claims stuck at HYPOTHESIS that cannot make progress.

    Matches claims where:
    - stage is HYPOTHESIS
    - scrutiny_verdict is "fail" or "needs_resolution"
    - investigation_count >= MAX_INVESTIGATION_ATTEMPTS
    - not already abandoned

    No LLM calls — purely structural cleanup.
    """

    entity_type = "claim"

    async def execute(self, work: WorkItem) -> OperationResult:
        claim = await self.repo.get("claim", work.entity_id)

        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
            )

        if claim.abandoned:
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message="Already abandoned",
            )

        claim.abandoned = True
        await self.repo.save(claim)

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=f"Abandoned stale claim at {claim.stage.value} "
            f"(scrutiny={claim.scrutiny_verdict}, investigations={claim.investigation_count})",
        )
```

- [ ] **Step 2: Register in operations/__init__.py**

Add `AbandonStaleClaimOperation` to `OPERATION_CLASSES`:
```python
from .cleanup import AbandonStaleClaimOperation
# In OPERATION_CLASSES dict:
"abandon_stale_claim": AbandonStaleClaimOperation,
```

- [ ] **Step 3: Add pattern to WORK_PATTERNS**

In `src/andamentum/epistemic/patterns.py`, add after the investigation patterns (after line 210):

```python
    # Abandonment safety net — catches HYPOTHESIS claims that exhausted investigation
    Pattern(
        entity_type="claim",
        filters={
            "stage": ClaimStage.HYPOTHESIS.value,
            "scrutiny_verdict__ne": "pass",
            "scrutiny_verdict__ne": None,
            "investigation_count__gte": 3,
            "abandoned": False,
        },
        operation="abandon_stale_claim",
        description="Abandon hypothesis that exhausted all investigation attempts",
    ),
```

Note: the filter `scrutiny_verdict__ne` can only appear once. Use a custom match function instead:

```python
    Pattern(
        entity_type="claim",
        filters={
            "stage": ClaimStage.HYPOTHESIS.value,
            "investigation_count__gte": 3,
            "abandoned": False,
        },
        operation="abandon_stale_claim",
        description="Abandon hypothesis that exhausted all investigation attempts",
        match_fn=lambda entity: getattr(entity, "scrutiny_verdict", None) not in ("pass", None),
    ),
```

Wait — `Pattern` doesn't support `match_fn`. Use the existing filter system. Since we need "verdict is fail OR needs_resolution", and both values are not None and not "pass", we can use:

Actually, the simplest approach: make two patterns, one for "fail" and one for "needs_resolution":

```python
    # Abandonment: HYPOTHESIS + fail + exhausted investigation
    Pattern(
        entity_type="claim",
        filters={
            "stage": ClaimStage.HYPOTHESIS.value,
            "scrutiny_verdict": "fail",
            "investigation_count__gte": 3,
            "abandoned": False,
        },
        operation="abandon_stale_claim",
        description="Abandon failed hypothesis after exhausting investigation",
    ),
    # Abandonment: HYPOTHESIS + needs_resolution + exhausted investigation
    Pattern(
        entity_type="claim",
        filters={
            "stage": ClaimStage.HYPOTHESIS.value,
            "scrutiny_verdict": "needs_resolution",
            "investigation_count__gte": 3,
            "abandoned": False,
        },
        operation="abandon_stale_claim",
        description="Abandon unresolved hypothesis after exhausting investigation",
    ),
```

- [ ] **Step 4: Write tests**

In `src/andamentum/epistemic/tests/test_deadlock_prevention.py`:

```python
"""Tests that no claim state is a dead end.

Every non-terminal claim state must either match a pattern or be
explicitly abandoned. These tests verify the dead-end states identified
in the scheduling audit are all handled.
"""

import pytest
from ..entities import Claim, ClaimStage
from ..operations.cleanup import AbandonStaleClaimOperation
from ..patterns import WorkItem, WORK_PATTERNS


class TestAbandonStaleClaim:
    @pytest.mark.asyncio
    async def test_fail_exhausted_gets_abandoned(self, repo):
        """HYPOTHESIS + fail + investigation_count=3 → abandoned."""
        claim = Claim(
            entity_id="cl-1",
            objective_id="obj-1",
            statement="Test",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="fail",
            investigation_count=3,
        )
        await repo.save(claim)

        op = AbandonStaleClaimOperation(repo=repo, agent_runner=None)
        work = WorkItem(entity_id="cl-1", entity_type="claim", operation="abandon_stale_claim")
        result = await op.execute(work)

        assert result.success
        updated = await repo.get("claim", "cl-1")
        assert updated.abandoned is True

    @pytest.mark.asyncio
    async def test_needs_resolution_exhausted_gets_abandoned(self, repo):
        """HYPOTHESIS + needs_resolution + investigation_count=3 → abandoned."""
        claim = Claim(
            entity_id="cl-1",
            objective_id="obj-1",
            statement="Test",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="needs_resolution",
            investigation_count=3,
        )
        await repo.save(claim)

        op = AbandonStaleClaimOperation(repo=repo, agent_runner=None)
        work = WorkItem(entity_id="cl-1", entity_type="claim", operation="abandon_stale_claim")
        result = await op.execute(work)

        assert result.success
        updated = await repo.get("claim", "cl-1")
        assert updated.abandoned is True

    def test_patterns_match_stale_claims(self):
        """Both stale-claim patterns exist and match correctly."""
        abandon_patterns = [p for p in WORK_PATTERNS if p.operation == "abandon_stale_claim"]
        assert len(abandon_patterns) >= 2

        # fail + exhausted
        claim_fail = Claim(
            statement="test", objective_id="obj-1",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="fail",
            investigation_count=3,
        )
        assert any(p.matches(claim_fail) for p in abandon_patterns)

        # needs_resolution + exhausted
        claim_nr = Claim(
            statement="test", objective_id="obj-1",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="needs_resolution",
            investigation_count=3,
        )
        assert any(p.matches(claim_nr) for p in abandon_patterns)

        # pass + exhausted should NOT match (pass claims should promote, not abandon)
        claim_pass = Claim(
            statement="test", objective_id="obj-1",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="pass",
            investigation_count=3,
        )
        assert not any(p.matches(claim_pass) for p in abandon_patterns)
```

- [ ] **Step 5: Run tests, verify, commit**

```bash
uv run pytest src/andamentum/epistemic/tests/test_deadlock_prevention.py -v
uv run pytest src/andamentum/epistemic/tests/ -v
git add -A && git commit -m "feat(epistemic): add abandonment pattern for exhausted hypotheses"
```

---

## Task 3: Reset promote attempts when evidence landscape changes

The scheduling deadlock: promote fires 3 times before investigation runs, exhausts `MAX_ENTITY_ATTEMPTS`, then can never fire again even after new supporting evidence appears. The fix: when `ExtractEvidenceOperation` judges evidence as "supports" for a linked claim, reset the promote attempt counter for that claim.

**Files:**
- Modify: `src/andamentum/epistemic/patterns.py` — add `reset_entity_attempts` method
- Modify: `src/andamentum/epistemic/operations_runner.py` — call reset after evidence judgment changes
- Test: `src/andamentum/epistemic/tests/test_patterns.py`

- [ ] **Step 1: Add `reset_entity_attempts` to PatternScheduler**

In `src/andamentum/epistemic/patterns.py`, add to `PatternScheduler`:

```python
    def reset_entity_attempts(self, entity_id: str, operation: str | None = None) -> None:
        """Reset attempt counters for an entity after the epistemic landscape changes.

        Called when new evidence is judged — the earlier promote failures
        may no longer be predictive because the gate inputs have changed.

        Args:
            entity_id: Entity to reset
            operation: If given, reset only this operation's counter.
                       If None, reset ALL operation counters for this entity.
        """
        if operation is not None:
            self._entity_attempts.pop((entity_id, operation), None)
        else:
            keys_to_remove = [k for k in self._entity_attempts if k[0] == entity_id]
            for k in keys_to_remove:
                del self._entity_attempts[k]
```

- [ ] **Step 2: Thread scheduler into operations that need it**

The scheduler needs to be accessible from operations. Currently operations only have `self.repo` and `self.agent_runner`. Rather than passing the scheduler, use a simpler approach: after each successful operation in the runner loop, check if the operation created new evidence or judged evidence. If so, reset promote attempts for affected claims.

In `src/andamentum/epistemic/operations_runner.py`, after `result.success` handling (around line 621):

```python
            if result.success:
                successful += 1
                scheduler.record_success(work.operation)

                # Reset promote attempts when evidence landscape changes.
                # When extraction judges evidence or investigation creates stubs,
                # earlier promote failures may no longer be predictive.
                if work.operation in ("extract_evidence", "investigate_claim", "seed_claim"):
                    for eid in result.created_entities or []:
                        try:
                            ev = await repo.get("evidence", eid)
                            if hasattr(ev, "depends_on_claim_id") and ev.depends_on_claim_id:
                                scheduler.reset_entity_attempts(
                                    ev.depends_on_claim_id, "promote_claim"
                                )
                        except Exception:
                            pass
                # Also reset when extract_evidence judges a "supports" verdict
                if work.operation == "extract_evidence":
                    # The entity IS the evidence. Check if it's linked to a claim.
                    try:
                        ev = await repo.get("evidence", work.entity_id)
                        if getattr(ev, "support_judgment", None) == "supports":
                            claims = await repo.query("claim", objective_id=ev.objective_id)
                            for c in claims:
                                if ev.entity_id in getattr(c, "evidence_ids", []):
                                    scheduler.reset_entity_attempts(
                                        c.entity_id, "promote_claim"
                                    )
                    except Exception:
                        pass
```

- [ ] **Step 3: Write tests**

Add to `src/andamentum/epistemic/tests/test_patterns.py`:

```python
class TestResetEntityAttempts:
    async def test_reset_specific_operation(self, repo):
        scheduler = PatternScheduler(repo)
        scheduler.record_attempt("e1", "promote_claim")
        scheduler.record_attempt("e1", "promote_claim")
        scheduler.record_attempt("e1", "promote_claim")
        assert scheduler._is_entity_exhausted("e1", "promote_claim")

        scheduler.reset_entity_attempts("e1", "promote_claim")
        assert not scheduler._is_entity_exhausted("e1", "promote_claim")

    async def test_reset_all_operations(self, repo):
        scheduler = PatternScheduler(repo)
        scheduler.record_attempt("e1", "promote_claim")
        scheduler.record_attempt("e1", "promote_claim")
        scheduler.record_attempt("e1", "promote_claim")
        scheduler.record_attempt("e1", "scrutinise_claim")
        scheduler.record_attempt("e1", "scrutinise_claim")
        scheduler.record_attempt("e1", "scrutinise_claim")

        scheduler.reset_entity_attempts("e1")
        assert not scheduler._is_entity_exhausted("e1", "promote_claim")
        assert not scheduler._is_entity_exhausted("e1", "scrutinise_claim")

    async def test_reset_does_not_affect_other_entities(self, repo):
        scheduler = PatternScheduler(repo)
        scheduler.record_attempt("e1", "promote_claim")
        scheduler.record_attempt("e1", "promote_claim")
        scheduler.record_attempt("e1", "promote_claim")
        scheduler.record_attempt("e2", "promote_claim")
        scheduler.record_attempt("e2", "promote_claim")
        scheduler.record_attempt("e2", "promote_claim")

        scheduler.reset_entity_attempts("e1", "promote_claim")
        assert not scheduler._is_entity_exhausted("e1", "promote_claim")
        assert scheduler._is_entity_exhausted("e2", "promote_claim")  # unaffected
```

- [ ] **Step 4: Run tests, verify, commit**

```bash
uv run pytest src/andamentum/epistemic/tests/test_patterns.py -v
uv run pytest src/andamentum/epistemic/tests/ -v
git add -A && git commit -m "feat(epistemic): reset promote attempts when evidence landscape changes"
```

---

## Task 4: Adversarial survival as positive gate signal (Popper)

When adversarial search runs and finds no strong counterevidence (adversarial_balance > 0.7), this should satisfy the `min_supporting_sources` requirement at the HYPOTHESIS → SUPPORTED gate. Philosophical justification: surviving a severe test is itself corroboration (Popper).

**Files:**
- Modify: `src/andamentum/epistemic/gates.py:536-549`
- Test: `src/andamentum/epistemic/tests/test_gates.py`

- [ ] **Step 1: Modify the supporting sources gate check**

In `src/andamentum/epistemic/gates.py`, change the supporting sources block (lines 536-549) to accept adversarial survival as an alternative:

```python
    # Supporting sources OR adversarial survival (Popper corroboration)
    # Direct supporting evidence is the primary path. But when adversarial
    # search has actively looked for counterevidence and found none
    # (high adversarial balance), that survival IS positive evidence.
    if gate.min_supporting_sources > 0:
        try:
            supporting = await count_supporting_sources(claim, repo)
            any_judged = await _any_evidence_judged(claim, repo)

            # Adversarial survival: if adversarial search ran and balance
            # is high, count it as satisfying the supporting sources gate.
            adversarial_survived = (
                claim.adversarial_checked
                and claim.adversarial_balance is not None
                and claim.adversarial_balance >= 0.7
            )

            if any_judged and supporting < gate.min_supporting_sources and not adversarial_survived:
                reasons.append(
                    f"Need {gate.min_supporting_sources} supporting sources, have {supporting}"
                )
        except Exception as e:
            warnings.append(f"Could not count supporting sources: {e}")
```

- [ ] **Step 2: Write tests**

Add to `src/andamentum/epistemic/tests/test_gates.py`:

```python
class TestAdversarialSurvivalGate:
    @pytest.mark.asyncio
    async def test_adversarial_survival_satisfies_supporting_sources(self, repo):
        """High adversarial balance with 0 supports should pass SUPPORTED gate."""
        ev = Evidence(
            entity_id="ev-1", objective_id="obj-1",
            extracted=True, extracted_content="test",
            quality_score=0.5, support_judgment="no_bearing",
        )
        await repo.save(ev)

        claim = Claim(
            entity_id="cl-1", objective_id="obj-1",
            statement="Test claim", evidence_ids=["ev-1"],
            stage=ClaimStage.HYPOTHESIS, scrutiny_verdict="pass",
            adversarial_checked=True, adversarial_balance=0.8,
        )

        from ..gates import validate_promotion, ClaimStage
        result = await validate_promotion(claim, ClaimStage.SUPPORTED, repo)
        # Should pass because adversarial survival compensates for 0 direct supports
        assert result.passed or "supporting sources" not in str(result.blocking_reasons)

    @pytest.mark.asyncio
    async def test_low_adversarial_balance_does_not_substitute(self, repo):
        """Low adversarial balance should NOT satisfy supporting sources."""
        ev = Evidence(
            entity_id="ev-1", objective_id="obj-1",
            extracted=True, extracted_content="test",
            quality_score=0.5, support_judgment="no_bearing",
        )
        await repo.save(ev)

        claim = Claim(
            entity_id="cl-1", objective_id="obj-1",
            statement="Test claim", evidence_ids=["ev-1"],
            stage=ClaimStage.HYPOTHESIS, scrutiny_verdict="pass",
            adversarial_checked=True, adversarial_balance=0.3,
        )

        from ..gates import validate_promotion, ClaimStage
        result = await validate_promotion(claim, ClaimStage.SUPPORTED, repo)
        assert not result.passed

    @pytest.mark.asyncio
    async def test_adversarial_not_run_does_not_substitute(self, repo):
        """If adversarial search hasn't run, can't claim survival."""
        ev = Evidence(
            entity_id="ev-1", objective_id="obj-1",
            extracted=True, extracted_content="test",
            quality_score=0.5, support_judgment="no_bearing",
        )
        await repo.save(ev)

        claim = Claim(
            entity_id="cl-1", objective_id="obj-1",
            statement="Test claim", evidence_ids=["ev-1"],
            stage=ClaimStage.HYPOTHESIS, scrutiny_verdict="pass",
            adversarial_checked=False,
        )

        from ..gates import validate_promotion, ClaimStage
        result = await validate_promotion(claim, ClaimStage.SUPPORTED, repo)
        assert not result.passed
```

- [ ] **Step 3: Run tests, verify, commit**

```bash
uv run pytest src/andamentum/epistemic/tests/test_gates.py -v -k adversarial_survival
uv run pytest src/andamentum/epistemic/tests/ -v
git add -A && git commit -m "feat(epistemic): adversarial survival satisfies supporting sources gate (Popper)"
```

---

## Task 5: Fix predictive question circular dependency

The gate audit found: for predictive questions, the PROVISIONAL → ROBUST gate requires `predictions_generated=True` (via `requires_falsification_criteria`), but `generate_prediction` only runs at ROBUST stage. Circular dependency — claim can never reach ROBUST.

**Files:**
- Modify: `src/andamentum/epistemic/routing.py` — move `requires_falsification_criteria` from `robust` to `actionable` thresholds
- Test: `src/andamentum/epistemic/tests/test_routing.py`

- [ ] **Step 1: Find and fix the routing override**

In `src/andamentum/epistemic/routing.py`, find the predictive question routing profile. The `requires_falsification_criteria` key is in the `robust` gate_thresholds. Move it to `actionable`:

Find the block that looks like:
```python
    "predictive": RoutingProfile(
        ...
        gate_thresholds={
            "robust": {
                "requires_falsification_criteria": True,
                ...
            },
        },
    ),
```

Change `"robust"` to `"actionable"` for the `requires_falsification_criteria` entry. This means predictions are required for ROBUST → ACTIONABLE (which is correct — predictions are generated AT ROBUST, then checked for ACTIONABLE).

- [ ] **Step 2: Write a test**

```python
class TestPredictiveQuestionGates:
    @pytest.mark.asyncio
    async def test_predictive_can_reach_robust_without_predictions(self, repo):
        """Predictive questions should not require predictions for PROVISIONAL→ROBUST."""
        from ..gates import validate_promotion
        from ..routing import get_routing_profile

        # Verify the routing profile doesn't block ROBUST with predictions
        profile = get_routing_profile("predictive")
        robust_overrides = profile.gate_thresholds.get("robust", {})
        assert "requires_falsification_criteria" not in robust_overrides
```

- [ ] **Step 3: Run tests, verify, commit**

```bash
uv run pytest src/andamentum/epistemic/tests/test_routing.py -v
uv run pytest src/andamentum/epistemic/tests/ -v
git add -A && git commit -m "fix(epistemic): move falsification requirement to actionable gate for predictive questions"
```

---

## Task 6: Abductive integration operation

The core missing step. After per-item evidence judgment, adversarial search, and convergence analysis, this operation takes ALL evidence (including no_bearing items) and the full epistemic context to produce a structured assessment. This is NOT a one-shot RAG synthesis — it reasons from the investigation's structured results.

**Files:**
- Create: `src/andamentum/epistemic/operations/integration.py`
- Modify: `src/andamentum/epistemic/agents/output_models.py` — add IntegrationAssessment
- Create: `src/andamentum/epistemic/agents/integration.py` — agent definition
- Modify: `src/andamentum/epistemic/entities/claim.py` — add `integrated_assessment` field
- Modify: `src/andamentum/epistemic/patterns.py` — add integration pattern
- Modify: `src/andamentum/epistemic/operations/__init__.py` — register
- Test: `src/andamentum/epistemic/tests/test_integration_operation.py`

- [ ] **Step 1: Add IntegrationAssessment output model**

In `src/andamentum/epistemic/agents/output_models.py`, add:

```python
class IntegrationAssessment(BaseModel):
    """Output of abductive integration: holistic evidence assessment.

    Unlike per-item judgment (supports/contradicts/no_bearing), this
    considers the TOTALITY of evidence and what it collectively implies.
    """

    verdict: str = Field(
        description="'supports', 'contradicts', or 'insufficient'. "
        "Based on collective evidence weight, not individual counts."
    )
    confidence: float = Field(
        description="0.0-1.0 confidence in the verdict, considering "
        "evidence quality, independence, adversarial testing outcome."
    )
    reasoning: str = Field(
        description="The evidential chain: what evidence was considered, "
        "how independent lines converge or diverge, and what the "
        "adversarial testing revealed."
    )
    key_support: list[str] = Field(
        default_factory=list,
        description="Evidence IDs that most strongly support the verdict."
    )
    key_concerns: list[str] = Field(
        default_factory=list,
        description="Remaining concerns that limit confidence."
    )
```

- [ ] **Step 2: Create the integration agent definition**

Create `src/andamentum/epistemic/agents/integration.py`:

```python
"""Abductive integration agent — holistic evidence assessment.

Reasons across ALL evidence (including no_bearing items) using the
structured results of the epistemic investigation: per-item judgments,
adversarial search outcome, convergence topology, and open uncertainties.

This is NOT a one-shot RAG synthesis. The agent has access to structured
investigation results that RAG never builds.

Architecture: Layer 2 (pydantic-ai agent)
"""

from pydantic_ai import Agent

from .output_models import IntegrationAssessment

epistemic_integrate_evidence = Agent(
    "openai:gpt-4o-mini",
    output_type=IntegrationAssessment,
    system_prompt=(
        "You are performing abductive integration: reasoning from the totality "
        "of evidence to the best-supported conclusion about a scientific claim.\n\n"
        "You have access to:\n"
        "1. INDIVIDUAL JUDGMENTS: each piece of evidence was independently assessed "
        "as 'supports', 'contradicts', or 'no_bearing' on the specific claim.\n"
        "2. ADVERSARIAL OUTCOME: a deliberate search for counterevidence was conducted. "
        "You know what was found and what was NOT found despite active searching.\n"
        "3. CONVERGENCE: whether the supporting evidence comes from independent domains "
        "or from a single line of work.\n"
        "4. OPEN UNCERTAINTIES: explicit knowledge gaps the system identified.\n\n"
        "Your task: considering ALL of this — not just the individual verdicts, but "
        "what the evidence collectively implies — assess whether the claim is supported, "
        "contradicted, or has insufficient evidence.\n\n"
        "KEY PRINCIPLE: Evidence marked 'no_bearing' individually may be COLLECTIVELY "
        "relevant. Three papers about podocyte actin, injury response, and cell motility "
        "machinery may individually not state 'podocytes migrate' but together they "
        "provide the mechanistic basis for the claim. Consider indirect evidence chains.\n\n"
        "KEY PRINCIPLE: The ABSENCE of counterevidence after active adversarial search "
        "is itself informative. 'We looked hard for refutation and found none' is "
        "stronger than 'we never looked.'\n\n"
        "Set confidence based on: number and independence of evidence lines, quality "
        "of sources, strength of adversarial testing, and remaining uncertainties."
    ),
)
```

- [ ] **Step 3: Create AbductiveIntegrationOperation**

Create `src/andamentum/epistemic/operations/integration.py`:

```python
"""Abductive integration operation.

Takes ALL evidence for a claim — including no_bearing items — along with
the adversarial search outcome, convergence assessment, and open
uncertainties. Produces a holistic IntegrationAssessment that captures
cross-evidence reasoning the per-item judge cannot.

Depends on: base (BaseOperation, OperationResult)
Operates on: Claim entities
"""

from .base import BaseOperation, OperationResult
from ..entities import Claim, Evidence, Uncertainty
from ..patterns import WorkItem


class AbductiveIntegrationOperation(BaseOperation):
    """Holistic evidence integration (Peirce abduction + Kahneman aggregation)."""

    entity_type = "claim"

    async def execute(self, work: WorkItem) -> OperationResult:
        claim = await self.repo.get("claim", work.entity_id)

        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
            )

        if claim.integrated_assessment is not None:
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message="Already integrated",
            )

        if not self.agent_runner:
            claim.integrated_assessment = "pass"  # No-agent fallback
            await self.repo.save(claim)
            return OperationResult(
                success=True,
                entity_id=claim.entity_id,
                message="Integration skipped (no agent runner)",
            )

        # Build the structured brief
        # 1. Individual judgments
        supports_items: list[str] = []
        contradicts_items: list[str] = []
        no_bearing_items: list[str] = []

        all_evidence = []
        for eid in claim.evidence_ids:
            try:
                ev = await self.repo.get("evidence", eid)
                if not isinstance(ev, Evidence) or ev.invalidated:
                    continue
                all_evidence.append(ev)
                summary = f"[{ev.source_type}] {(ev.extracted_content or '')[:300]}"
                if ev.support_judgment == "supports":
                    supports_items.append(summary)
                elif ev.support_judgment == "contradicts":
                    contradicts_items.append(summary)
                else:
                    no_bearing_items.append(summary)
            except Exception:
                continue

        # 2. Adversarial outcome
        adversarial_text = "Adversarial search has NOT been conducted."
        if claim.adversarial_checked:
            if claim.adversarial_balance is not None:
                if claim.adversarial_balance >= 0.7:
                    adversarial_text = (
                        f"Adversarial search was conducted and found NO strong "
                        f"counterevidence (balance: {claim.adversarial_balance:.2f}). "
                        f"The claim survived active attempts at refutation."
                    )
                else:
                    adversarial_text = (
                        f"Adversarial search found significant counterevidence "
                        f"(balance: {claim.adversarial_balance:.2f})."
                    )

        # 3. Convergence
        convergence_text = "Convergence has not been assessed."
        if claim.convergence_checked:
            convergence_text = "Cross-domain convergence has been assessed."

        # 4. Open uncertainties
        uncertainties = await self.repo.query(
            "uncertainty",
            objective_id=claim.objective_id,
        )
        open_uncertainties = [
            u for u in uncertainties
            if isinstance(u, Uncertainty)
            and claim.entity_id in u.affected_claim_ids
            and u.resolution is None
            and u.is_blocking
        ]
        if open_uncertainties:
            unc_text = "\n".join(f"- {u.description}" for u in open_uncertainties)
        else:
            unc_text = "No unresolved blocking uncertainties."

        # Run integration agent
        result = await self.run_agent(
            "epistemic_integrate_evidence",
            claim_statement=claim.statement,
            claim_scope=claim.scope,
            supporting_evidence="\n\n".join(supports_items) if supports_items else "None found.",
            contradicting_evidence="\n\n".join(contradicts_items) if contradicts_items else "None found.",
            no_bearing_evidence="\n\n".join(no_bearing_items[:20]) if no_bearing_items else "None.",
            adversarial_outcome=adversarial_text,
            convergence_assessment=convergence_text,
            open_uncertainties=unc_text,
            evidence_count=len(all_evidence),
            supporting_count=len(supports_items),
            contradicting_count=len(contradicts_items),
            no_bearing_count=len(no_bearing_items),
        )

        # Store the assessment
        claim.integrated_assessment = result.verdict
        claim.integrated_confidence = result.confidence
        claim.integrated_reasoning = result.reasoning
        await self.repo.save(claim)

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=f"Integration: {result.verdict} (confidence {result.confidence:.2f})",
        )
```

- [ ] **Step 4: Add fields to Claim entity**

In `src/andamentum/epistemic/entities/claim.py`, add:

```python
    # Abductive integration (Peirce + Kahneman)
    integrated_assessment: Optional[str] = Field(
        default=None,
        description="Holistic evidence verdict: 'supports', 'contradicts', 'insufficient', or None",
    )
    integrated_confidence: Optional[float] = Field(
        default=None,
        description="Confidence from abductive integration 0.0-1.0",
    )
    integrated_reasoning: Optional[str] = Field(
        default=None,
        description="Reasoning chain from integration assessment",
    )
```

Add to `_extra_metadata()`:
```python
    "integrated_assessment": self.integrated_assessment,
    "integrated_confidence": self.integrated_confidence,
```

Add to `record_demotion()` reset block:
```python
    self.integrated_assessment = None
    self.integrated_confidence = None
    self.integrated_reasoning = None
```

- [ ] **Step 5: Add pattern and register operation**

In `src/andamentum/epistemic/patterns.py`, add AFTER the verification tracks and BEFORE promotion:

```python
    # ══════════════════════════════════════════════════════════════════
    # PHASE 6.5: ABDUCTIVE INTEGRATION
    # After verification tracks complete, holistically assess evidence
    # ══════════════════════════════════════════════════════════════════
    Pattern(
        entity_type="claim",
        filters={
            "stage": ClaimStage.SUPPORTED.value,
            "adversarial_checked": True,
            "integrated_assessment": None,
        },
        operation="integrate_evidence",
        description="Holistic evidence integration (Peirce abduction)",
    ),
```

In `src/andamentum/epistemic/operations/__init__.py`:
```python
from .integration import AbductiveIntegrationOperation
# In OPERATION_CLASSES:
"integrate_evidence": AbductiveIntegrationOperation,
```

- [ ] **Step 6: Write tests**

Create `src/andamentum/epistemic/tests/test_integration_operation.py` with tests for:
- Operation sets `integrated_assessment` field
- Operation handles no evidence gracefully
- Operation is idempotent (already integrated → noop)
- Pattern matches at SUPPORTED with adversarial_checked=True and integrated_assessment=None
- Pattern does NOT match before adversarial search completes

- [ ] **Step 7: Run tests, verify, commit**

```bash
uv run pytest src/andamentum/epistemic/tests/test_integration_operation.py -v
uv run pytest src/andamentum/epistemic/tests/ -v
uv run pyright
git add -A && git commit -m "feat(epistemic): add abductive integration operation (Peirce + Kahneman)"
```

---

## Task 7: Wire posterior to integration assessment

Replace the counting-based posterior (`logistic(supports - contradicts)`) with one informed by the integration assessment. When `integrated_assessment` is available, use its verdict and confidence directly. Fall back to counting when integration hasn't run.

**Files:**
- Modify: `src/andamentum/epistemic/confidence.py`
- Test: `src/andamentum/epistemic/tests/test_posterior.py`

- [ ] **Step 1: Modify compute_posterior**

In `src/andamentum/epistemic/confidence.py`, change the posterior computation to prefer integration assessment:

```python
    # 5. Compute posterior from integration assessment if available,
    #    otherwise fall back to evidence counting.
    integrated_claims = [c for c in active_claims if c.integrated_assessment is not None]

    if integrated_claims:
        # Integration-informed posterior: use the holistic assessment
        # that considered ALL evidence including no_bearing items
        for claim in integrated_claims:
            if claim.integrated_assessment == "supports":
                supporting += 1
            elif claim.integrated_assessment == "contradicts":
                contradicting += 1
            # "insufficient" → neither, stays at prior
    else:
        # Fallback: count per-item judgments (original behavior)
        for claim in active_claims:
            claim_evidence = [
                e for e in evidence
                if e.entity_id in claim.evidence_ids
                and not e.invalidated
                and e.cluster_status not in ("corroborative", "deferred")
            ]
            for e in claim_evidence:
                if e.support_judgment == "supports":
                    supporting += 1
                elif e.support_judgment == "contradicts":
                    contradicting += 1
```

- [ ] **Step 2: Update tests, verify, commit**

Add tests for integration-informed posterior vs counting fallback.

```bash
uv run pytest src/andamentum/epistemic/tests/test_posterior.py -v
uv run pytest src/andamentum/epistemic/tests/ -v
uv run pyright && uv run ruff check
git add -A && git commit -m "feat(epistemic): posterior uses integration assessment when available"
```

---

## Self-Review

**Spec coverage:**
- Deadlock state 1 (HYPOTHESIS/fail/saturated) → Task 1 (remove saturation)
- Deadlock state 2 (entity-exhausted investigation) → Task 2 (abandonment pattern)
- Deadlock state 3 (HYPOTHESIS/needs_resolution/saturated) → Task 1 (remove saturation)
- Deadlock state 6 (ROBUST/predictions exhausted) → covered by existing MAX_INVESTIGATION_ATTEMPTS; Task 2's pattern could be extended later
- Scheduling deadlock (promote exhausted before evidence) → Task 3 (reset attempts)
- Popper absence-of-refutation → Task 4 (adversarial survival gate)
- Predictive circular dependency → Task 5 (move falsification to actionable)
- Peirce abductive integration → Task 6 (integration operation)
- Kahneman structured aggregation → Task 6 (integration operation)
- Wimsatt full-evidence convergence → Task 6 (no_bearing evidence included in integration)
- Hempel evidential entailment → Task 6 (reasoning chain in integration)
- Integration-informed posterior → Task 7

**Gate audit Finding 5** (no question_type → contrastive/consistency block promotion): This is handled by the existing SUPPORTED→PROVISIONAL promotion pattern requiring `contrastive_checked=True` and `consistency_checked=True`. Without question_type, these tracks fire normally (all tracks active). With question_type, SetRoutingDefaults marks SKIP tracks as checked. The only edge case is if SetRoutingDefaults fails — but that's an operation failure, not a design flaw. No additional fix needed.

**Simplification achieved:**
- Removed `saturated` field and saturation check (Task 1) — one less entity field, one less code path, two fewer deadlock states
- Investigation is now controlled by a single mechanism (`MAX_INVESTIGATION_ATTEMPTS=3`) instead of two (`saturated` + `investigation_count`)
- The abandonment pattern (Task 2) is a universal safety net rather than per-state special cases

**Edge cases considered:**
- Integration fires AFTER adversarial search, so it has the adversarial outcome as input
- Integration only fires at SUPPORTED stage, so it doesn't run on HYPOTHESIS claims (which might not have enough evidence)
- If integration agent fails, the claim proceeds without integration (fallback to counting)
- Peirce cycling (demotion → reset) clears `integrated_assessment`, so integration re-runs after evidence changes
- The promote attempt reset (Task 3) only triggers on evidence-changing operations, not on every operation

Plan complete and saved to `docs/superpowers/plans/2026-04-19-epistemic-scheduling-and-integration.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?