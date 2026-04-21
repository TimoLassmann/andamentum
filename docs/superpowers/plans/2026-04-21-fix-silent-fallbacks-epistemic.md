# Fix Silent Fallbacks in Epistemic Operations Layer

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate silent fallbacks and fabricated defaults in `epistemic/operations/`, `epistemic/graph/`, and `epistemic/evidence_gathering.py` so failures are visible at the entity level and the user can distinguish real results from degraded ones.

**Architecture:** Introduce per-entity quarantine tracked in `EpistemicGraphState`. The graph's central operation runner (`_run_op`) catches exceptions once, records them in state (entity_id, entity_type, operation, exception repr), and skips the entity from future work. Individual operations STOP catching LLM-agent / repo-load failures internally — they let exceptions propagate up to `_run_op`. Quarantined entities surface in `EpistemicResult.quarantined` and in `PipelineResult`.

**Tech Stack:** Python 3.13, pydantic-graph, pytest, ruff, pyright.

**Order of work:**
1. Infrastructure (Tasks 1–3): quarantine state, visible `_run_op` quarantine, surface in result.
2. Per-site fixes (Tasks 4–12): remove each internal fallback. Each task is one file, one commit.
3. Full verification (Task 13): green state — pytest + ruff + pyright.

**Test file:** New single file `src/andamentum/epistemic/tests/test_no_silent_fallbacks.py` collects all per-site tests. Infrastructure tests live alongside related existing tests.

**Out of scope for this plan (tracked as future themes):**
- `deep_research/nodes.py` synthesis fallback, `deep_research/novelty/checker.py`, `document_store/extraction.py`, `document_store/search.py` signal swallows — separate themes.
- Truncation in providers (Theme 2).
- Zombie code removal (Theme 3).

---

## File Structure

**Modified:**
- `src/andamentum/epistemic/graph/state.py` — add `quarantined: list[QuarantineRecord]` + helpers
- `src/andamentum/epistemic/graph/nodes.py` — `_run_op` records quarantine on exception; helper for filtering alive entities
- `src/andamentum/epistemic/graph/result.py` — add `quarantined` field to `EpistemicResult`
- `src/andamentum/epistemic/graph/__init__.py` — pass `quarantined` to `PipelineResult`; stop swallowing posterior failure
- `src/andamentum/epistemic/operations_runner.py` — add `quarantined` to `PipelineResult`
- `src/andamentum/epistemic/operations/evidence.py` — delete placeholder branch + FINAL GUARD default
- `src/andamentum/epistemic/operations/claims.py` — remove relevance-screening silent inclusion
- `src/andamentum/epistemic/operations/verification.py` — remove 4 swallows (counterquery, counterarg eval, domain classifier, convergence load)
- `src/andamentum/epistemic/operations/stage_management.py` — remove objective-load swallow
- `src/andamentum/epistemic/evidence_gathering.py` — remove `CompositeGatherer` silent provider→web fallback

**Created:**
- `src/andamentum/epistemic/tests/test_no_silent_fallbacks.py` — per-site behavior tests
- `src/andamentum/epistemic/graph/quarantine.py` — `QuarantineRecord` dataclass (tiny; keeps `state.py` lean)

---

## Task 1: Add `QuarantineRecord` dataclass and state tracking

**Files:**
- Create: `src/andamentum/epistemic/graph/quarantine.py`
- Modify: `src/andamentum/epistemic/graph/state.py`
- Test: `src/andamentum/epistemic/tests/test_graph_state.py` (create if missing — check first)

- [ ] **Step 1.1: Create `QuarantineRecord` dataclass**

Create `src/andamentum/epistemic/graph/quarantine.py`:

```python
"""Quarantine record for entities whose operations failed.

When an operation raises, the central runner records a QuarantineRecord
so downstream nodes can skip the entity and the final report can surface
the failure to the user. Fail-loud: no silent degradation.

Architecture: Layer 1 (framework-agnostic, pure dataclass)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuarantineRecord:
    """Records that an entity was quarantined because an operation raised."""

    entity_id: str
    entity_type: str
    operation: str
    exception_type: str
    message: str
```

- [ ] **Step 1.2: Add quarantine fields + helpers to `EpistemicGraphState`**

Edit `src/andamentum/epistemic/graph/state.py`. Add after the `errors` field (around line 65):

```python
    # Entities whose operations raised. Downstream nodes must skip these.
    quarantined: list[QuarantineRecord] = field(default_factory=list)
    _quarantined_ids: set[str] = field(default_factory=set)
```

Add these methods to the class (after `log_operation`):

```python
    def quarantine(
        self,
        entity_id: str,
        entity_type: str,
        operation: str,
        exception: BaseException,
    ) -> None:
        """Record that an entity's operation raised. Idempotent per entity."""
        record = QuarantineRecord(
            entity_id=entity_id,
            entity_type=entity_type,
            operation=operation,
            exception_type=type(exception).__name__,
            message=str(exception),
        )
        self.quarantined.append(record)
        self._quarantined_ids.add(entity_id)

    def is_quarantined(self, entity_id: str) -> bool:
        """Return True if this entity is quarantined from further operations."""
        return entity_id in self._quarantined_ids
```

Add the import at the top of `state.py`:

```python
from .quarantine import QuarantineRecord
```

- [ ] **Step 1.3: Write failing tests**

Check whether `src/andamentum/epistemic/tests/test_graph_state.py` exists (`ls`). If it does, append. If not, create with:

```python
"""Tests for EpistemicGraphState quarantine tracking."""

from andamentum.epistemic.graph.state import EpistemicGraphState


def test_quarantine_records_record():
    state = EpistemicGraphState(objective_id="obj-1", question="q")
    err = ValueError("upstream LLM died")
    state.quarantine("ev-42", "evidence", "extract_evidence", err)
    assert len(state.quarantined) == 1
    record = state.quarantined[0]
    assert record.entity_id == "ev-42"
    assert record.entity_type == "evidence"
    assert record.operation == "extract_evidence"
    assert record.exception_type == "ValueError"
    assert record.message == "upstream LLM died"


def test_is_quarantined_returns_true_after_quarantine():
    state = EpistemicGraphState(objective_id="obj-1", question="q")
    assert not state.is_quarantined("ev-42")
    state.quarantine("ev-42", "evidence", "extract_evidence", RuntimeError("x"))
    assert state.is_quarantined("ev-42")


def test_is_quarantined_false_for_unquarantined():
    state = EpistemicGraphState(objective_id="obj-1", question="q")
    state.quarantine("ev-42", "evidence", "op", RuntimeError("x"))
    assert not state.is_quarantined("ev-99")


def test_quarantine_idempotent_across_calls():
    state = EpistemicGraphState(objective_id="obj-1", question="q")
    state.quarantine("ev-42", "evidence", "op1", RuntimeError("one"))
    state.quarantine("ev-42", "evidence", "op2", RuntimeError("two"))
    # Both records retained, but single membership in the skip set
    assert len(state.quarantined) == 2
    assert state.is_quarantined("ev-42")
```

- [ ] **Step 1.4: Run the tests — expect PASS (implementation precedes test here is fine since it's infrastructure; we still verify)**

Run: `uv run pytest src/andamentum/epistemic/tests/test_graph_state.py -v`
Expected: 4 tests pass.

- [ ] **Step 1.5: Commit**

```bash
git add src/andamentum/epistemic/graph/quarantine.py src/andamentum/epistemic/graph/state.py src/andamentum/epistemic/tests/test_graph_state.py
git commit -m "feat(epistemic): add QuarantineRecord and state tracking"
```

---

## Task 2: Update `_run_op` to quarantine on exception (replaces silent success=False)

**Files:**
- Modify: `src/andamentum/epistemic/graph/nodes.py` (lines 48-101)
- Test: `src/andamentum/epistemic/tests/test_no_silent_fallbacks.py` (create)

- [ ] **Step 2.1: Create the new per-site test file with infrastructure test**

Create `src/andamentum/epistemic/tests/test_no_silent_fallbacks.py`:

```python
"""Tests asserting that silent fallbacks have been removed.

Every test here asserts a single property: a failure in a downstream call
(LLM agent, repo load, provider) either raises out of the operation or is
recorded on the graph state — never silently swallowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from andamentum.epistemic.graph.nodes import _run_op
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.operations.base import (
    BaseOperation,
    OperationInput,
    OperationResult,
)


class _RaisingOp(BaseOperation):
    """Test double: always raises."""

    entity_type = "claim"
    raised: Exception = RuntimeError("kaboom")

    async def execute(self, work: OperationInput) -> OperationResult:
        raise self.raised


@dataclass
class _StubDeps:
    """Minimal deps for _run_op — only fields the function reads."""

    repo: Any = None
    agent_runner: Any = None
    evidence_gatherer: Any = None
    quality_scorer: Any = None
    embedding_model: Any = None
    progress_callback: Any = None


@pytest.mark.asyncio
async def test_run_op_quarantines_entity_on_exception():
    state = EpistemicGraphState(objective_id="obj-1", question="q")
    deps = _StubDeps()

    result = await _run_op(
        _RaisingOp, deps, state, "claim-7", "claim", "scrutinize_claim"
    )

    # The result is surfaced as success=False (for logging), but the state
    # now carries a quarantine record — no silent degradation.
    assert result.success is False
    assert state.is_quarantined("claim-7")
    assert len(state.quarantined) == 1
    record = state.quarantined[0]
    assert record.entity_id == "claim-7"
    assert record.entity_type == "claim"
    assert record.operation == "scrutinize_claim"
    assert record.exception_type == "RuntimeError"
    assert "kaboom" in record.message
```

- [ ] **Step 2.2: Run the test — expect FAIL**

Run: `uv run pytest src/andamentum/epistemic/tests/test_no_silent_fallbacks.py::test_run_op_quarantines_entity_on_exception -v`
Expected: FAIL — the assertion `state.is_quarantined("claim-7")` returns False because `_run_op` currently swallows without recording quarantine.

- [ ] **Step 2.3: Update `_run_op` to record quarantine**

Edit `src/andamentum/epistemic/graph/nodes.py` lines 48-72. Replace the current `try/except` block with:

```python
async def _run_op(
    op_class: type,
    deps: EpistemicDeps,
    state: EpistemicGraphState,
    entity_id: str,
    entity_type: str,
    operation: str,
) -> Any:
    """Instantiate an operation, execute it, log the result, and return it.

    If the operation raises, record a quarantine on the graph state and
    return a failed OperationResult — never swallow silently. Downstream
    nodes must call state.is_quarantined(entity_id) before scheduling
    further work on the entity.
    """
    op = _make_op(op_class, deps)
    work = _op_input(entity_id, entity_type, operation)
    try:
        result = await op.execute(work)
    except Exception as e:
        logger.warning(
            "%s on %s raised %s: %s — quarantining entity",
            operation,
            entity_id[:12],
            type(e).__name__,
            e,
        )
        state.quarantine(entity_id, entity_type, operation, e)
        from ..operations.base import OperationResult

        result = OperationResult(
            success=False,
            entity_id=entity_id,
            message=f"{operation} quarantined: {type(e).__name__}: {e}",
        )
    state.log_operation(operation, entity_id, result.success, result.message)
    # ... rest of function unchanged (trace persistence, progress callback)
```

Keep the remaining body (backend trace persistence starting at current line 73 and progress callback block at line 94-100) unchanged.

Remove the inner `import logging` at line 62 and the local `logging.getLogger(__name__)` call — use the module-level `logger` already defined at line 22.

- [ ] **Step 2.4: Run the test — expect PASS**

Run: `uv run pytest src/andamentum/epistemic/tests/test_no_silent_fallbacks.py::test_run_op_quarantines_entity_on_exception -v`
Expected: PASS.

- [ ] **Step 2.5: Run the full epistemic suite to confirm no regressions**

Run: `uv run pytest src/andamentum/epistemic/tests -x`
Expected: all pre-existing tests pass.

- [ ] **Step 2.6: Commit**

```bash
git add src/andamentum/epistemic/graph/nodes.py src/andamentum/epistemic/tests/test_no_silent_fallbacks.py
git commit -m "feat(epistemic): _run_op records quarantine on exception"
```

---

## Task 3: Surface quarantined entities in EpistemicResult and PipelineResult

**Files:**
- Modify: `src/andamentum/epistemic/graph/result.py`
- Modify: `src/andamentum/epistemic/graph/__init__.py`
- Modify: `src/andamentum/epistemic/operations_runner.py`
- Test: `src/andamentum/epistemic/tests/test_no_silent_fallbacks.py`

- [ ] **Step 3.1: Write failing integration-style test**

Append to `src/andamentum/epistemic/tests/test_no_silent_fallbacks.py`:

```python
def test_epistemic_result_has_quarantined_field():
    from andamentum.epistemic.graph.result import EpistemicResult

    result = EpistemicResult(objective_id="obj-1", status="partial")
    # Default: empty list, not None
    assert result.quarantined == []


def test_pipeline_result_has_quarantined_field():
    from andamentum.epistemic.operations_runner import PipelineResult

    result = PipelineResult(
        objective_id="obj-1",
        iterations=0,
        successful=0,
        failed=0,
        status="partial",
    )
    assert result.quarantined == []
```

- [ ] **Step 3.2: Run — expect FAIL**

Run: `uv run pytest src/andamentum/epistemic/tests/test_no_silent_fallbacks.py -v -k quarantined_field`
Expected: both tests fail (AttributeError).

- [ ] **Step 3.3: Add `quarantined` to `EpistemicResult`**

Edit `src/andamentum/epistemic/graph/result.py`. Add after the existing imports:

```python
from .quarantine import QuarantineRecord
```

Add field inside the dataclass (after `posterior`):

```python
    quarantined: list[QuarantineRecord] = field(default_factory=list)
```

- [ ] **Step 3.4: Find `PipelineResult` in `operations_runner.py` and add the same field**

Read `src/andamentum/epistemic/operations_runner.py` to locate the `PipelineResult` dataclass. Add:

```python
    quarantined: list[QuarantineRecord] = field(default_factory=list)
```

And add the import near the top:

```python
from .graph.quarantine import QuarantineRecord
```

- [ ] **Step 3.5: Thread the data through `run_epistemic_graph` in `graph/__init__.py`**

In `src/andamentum/epistemic/graph/__init__.py`, locate the terminal node that returns `End(EpistemicResult(...))`. (Find it with `grep -n "EpistemicResult(" src/andamentum/epistemic/graph/`). At the construction site, pass `quarantined=state.quarantined`.

Then, where `PipelineResult(...)` is constructed (currently around line 158-166), add:

```python
    return PipelineResult(
        objective_id=objective_id,
        iterations=len(state.operations_log),
        successful=result.successful,
        failed=result.failed,
        status=result.status,
        errors=result.errors,
        posterior=posterior_report,
        quarantined=result.quarantined,
    )
```

- [ ] **Step 3.6: Run both tests — expect PASS**

Run: `uv run pytest src/andamentum/epistemic/tests/test_no_silent_fallbacks.py -v -k quarantined_field`
Expected: PASS.

- [ ] **Step 3.7: Run full suite — no regressions**

Run: `uv run pytest src/andamentum/epistemic -x`
Expected: green.

- [ ] **Step 3.8: Commit**

```bash
git add src/andamentum/epistemic/graph/result.py src/andamentum/epistemic/graph/__init__.py src/andamentum/epistemic/operations_runner.py src/andamentum/epistemic/tests/test_no_silent_fallbacks.py
git commit -m "feat(epistemic): surface quarantined entities in EpistemicResult and PipelineResult"
```

---

## Task 4: Delete placeholder evidence fabrication + FINAL GUARD default

**Files:**
- Modify: `src/andamentum/epistemic/operations/evidence.py` (lines 141-160)
- Test: `src/andamentum/epistemic/tests/test_no_silent_fallbacks.py`

**Why:** `evidence.py:141-146` fabricates `extracted_content = f"[Content from {source_ref}]"` when no agent runner is available, then lines 149-156 guard with a fake `quality_score = 0.1`. Both are placeholders that look like real extractions to downstream code. Per design decision #4, delete both.

- [ ] **Step 4.1: Write failing test**

Append to `test_no_silent_fallbacks.py`:

```python
@pytest.mark.asyncio
async def test_extract_evidence_raises_without_runner_or_gatherer():
    """When neither an agent runner nor a gatherer is wired up, extraction
    must raise — never fabricate `[Content from ...]` placeholders."""
    from andamentum.epistemic.entities import Evidence
    from andamentum.epistemic.operations.evidence import ExtractEvidenceOperation
    from andamentum.epistemic.repository import EpistemicRepository
    from andamentum.epistemic.storage import InMemoryStorageBackend

    repo = EpistemicRepository(backend=InMemoryStorageBackend())
    ev = Evidence(
        objective_id="obj-1",
        source_type="web_search",
        source_ref="http://example.org/paper",
    )
    await repo.save(ev)

    op = ExtractEvidenceOperation(
        repo=repo,
        agent_runner=None,  # no runner
        evidence_gatherer=None,  # no gatherer
        quality_scorer=None,
        embedding_model=None,
    )
    with pytest.raises(RuntimeError, match="no extractor"):
        await op.execute(
            OperationInput(
                entity_id=ev.entity_id,
                entity_type="evidence",
                operation="extract_evidence",
            )
        )
```

(If `EpistemicRepository` / `InMemoryStorageBackend` construction differs, grep for existing fixtures and mirror them.)

- [ ] **Step 4.2: Run — expect FAIL (currently fabricates content)**

Run: `uv run pytest src/andamentum/epistemic/tests/test_no_silent_fallbacks.py -k extract_evidence_raises -v`
Expected: FAIL — operation completes with fabricated content, no exception.

- [ ] **Step 4.3: Delete the placeholder branch in `evidence.py`**

In `src/andamentum/epistemic/operations/evidence.py`, replace lines 141-160 (the `else:` branch that fabricates content AND the `FINAL GUARD` block that sets default `quality_score = 0.1`):

```python
        # Strategy 2: Use agent runner (primary when no gatherer, or fallback when gatherer fails)
        if self.agent_runner:
            _extract_log.info(
                "[extract_evidence] AGENT extraction for %s", evidence.entity_id
            )
            # ... unchanged agent-extraction block ...
            await self._score_evidence(evidence)
        else:
            raise RuntimeError(
                f"[extract_evidence] no extractor available for {evidence.entity_id}: "
                f"ExtractEvidenceOperation requires either an evidence_gatherer or "
                f"an agent_runner. This indicates a wiring bug in graph construction."
            )

        evidence.extracted = True
        # ... keep the DONE log line below unchanged ...
```

Delete the entire `# Strategy 3: Placeholder` `else:` block (lines 141-146) and the `# Final guard` block (lines 149-156). The agent-extraction path already calls `_score_evidence`, so no default is needed.

- [ ] **Step 4.4: Run the new test — expect PASS**

Run: `uv run pytest src/andamentum/epistemic/tests/test_no_silent_fallbacks.py -k extract_evidence_raises -v`
Expected: PASS.

- [ ] **Step 4.5: Run the full epistemic suite and fix any tests that relied on fabrication**

Run: `uv run pytest src/andamentum/epistemic -x`
If tests fail with "quality_score is None" or similar, those tests were depending on the FINAL GUARD default. Update them to pass a mock agent runner. Do NOT re-add the default.

- [ ] **Step 4.6: Commit**

```bash
git add src/andamentum/epistemic/operations/evidence.py src/andamentum/epistemic/tests/test_no_silent_fallbacks.py
git commit -m "fix(epistemic): remove placeholder evidence fabrication and FINAL GUARD default"
```

---

## Task 5: Remove relevance-screening silent inclusion (claims.py)

**Files:**
- Modify: `src/andamentum/epistemic/operations/claims.py` lines 224-238
- Test: `src/andamentum/epistemic/tests/test_no_silent_fallbacks.py`

**Why:** When `epistemic_screen_relevance` raises, the loop at line 236-237 silently includes the evidence as if it had passed screening. Fix: let the exception propagate out of the operation. `_run_op` will quarantine the objective; the user sees "objective X could not propose claims: screening agent failed."

- [ ] **Step 5.1: Write failing test**

Append:

```python
@pytest.mark.asyncio
async def test_propose_claims_propagates_screening_failure():
    """When epistemic_screen_relevance raises, ProposeClaimsOperation must
    propagate — the previous behavior (include-by-default) silently poisoned
    downstream evidence selection with unscreened items."""
    # Use existing test fixtures — check tests/conftest.py for the shape.
    # Construct an objective with one evidence item, a runner that raises
    # on "epistemic_screen_relevance", and expect execute() to raise.
    from andamentum.epistemic.entities import Evidence, Objective
    from andamentum.epistemic.operations.claims import ProposeClaimsOperation
    from andamentum.epistemic.repository import EpistemicRepository
    from andamentum.epistemic.storage import InMemoryStorageBackend

    class _RaisingScreenRunner:
        async def run(self, agent_name: str, **kwargs):
            if agent_name == "epistemic_screen_relevance":
                raise RuntimeError("screening model timed out")
            # Other agents shouldn't be reached before screening
            raise AssertionError(
                f"Unexpected agent call {agent_name} before screening failed"
            )

    repo = EpistemicRepository(backend=InMemoryStorageBackend())
    obj = Objective(description="test question", clarified_question="q?")
    await repo.save(obj)
    ev = Evidence(
        objective_id=obj.entity_id,
        source_type="web_search",
        source_ref="http://example.org/x",
        extracted=True,
        extracted_content="some content",
    )
    await repo.save(ev)

    op = ProposeClaimsOperation(
        repo=repo,
        agent_runner=_RaisingScreenRunner(),
        evidence_gatherer=None,
        quality_scorer=None,
        embedding_model=None,
    )
    with pytest.raises(RuntimeError, match="screening model timed out"):
        await op.execute(
            OperationInput(
                entity_id=obj.entity_id,
                entity_type="objective",
                operation="propose_claims",
            )
        )
```

- [ ] **Step 5.2: Run — expect FAIL (currently succeeds because failure is swallowed)**

Run: `uv run pytest src/andamentum/epistemic/tests/test_no_silent_fallbacks.py -k screening -v`
Expected: FAIL (no exception raised).

- [ ] **Step 5.3: Remove the swallow**

Edit `src/andamentum/epistemic/operations/claims.py` lines 224-238. Replace:

```python
        if self.agent_runner and extracted:
            relevant: list[Evidence] = []
            for ev in extracted:
                try:
                    screen = await self.run_agent(
                        "epistemic_screen_relevance",
                        research_question=clarified,
                        evidence_content=ev.extracted_content,
                        source_info=f"[{ev.source_type}] {ev.source_ref}",
                    )
                    if screen.is_relevant:
                        relevant.append(ev)
                except Exception:
                    relevant.append(ev)  # Screening failed — include by default
            extracted = relevant
```

With:

```python
        if self.agent_runner and extracted:
            relevant: list[Evidence] = []
            for ev in extracted:
                screen = await self.run_agent(
                    "epistemic_screen_relevance",
                    research_question=clarified,
                    evidence_content=ev.extracted_content,
                    source_info=f"[{ev.source_type}] {ev.source_ref}",
                )
                if screen.is_relevant:
                    relevant.append(ev)
            extracted = relevant
```

- [ ] **Step 5.4: Run — expect PASS**

Run: `uv run pytest src/andamentum/epistemic/tests/test_no_silent_fallbacks.py -k screening -v`
Expected: PASS.

- [ ] **Step 5.5: Run full suite**

Run: `uv run pytest src/andamentum/epistemic -x`

- [ ] **Step 5.6: Commit**

```bash
git add src/andamentum/epistemic/operations/claims.py src/andamentum/epistemic/tests/test_no_silent_fallbacks.py
git commit -m "fix(epistemic): propagate relevance-screening failures"
```

---

## Task 6: Remove counterquery generator swallow (verification.py:86-95)

**Files:**
- Modify: `src/andamentum/epistemic/operations/verification.py` lines 86-98

**Why:** When one of three counterquery framings fails, the `_generate_one` returns `None` and downstream filters it out. A 2/3 degradation in adversarial query coverage passes silently. Fix: remove the try/except; let `asyncio.gather` raise (default `return_exceptions=False`). The whole adversarial check for that claim fails → claim quarantined.

- [ ] **Step 6.1: Write failing test**

Append:

```python
@pytest.mark.asyncio
async def test_adversarial_check_propagates_counterquery_failure():
    """One failing framing must propagate. Previous: silently dropped to 2/3.
    New: the claim gets quarantined by _run_op."""
    from andamentum.epistemic.entities import Claim, Objective
    from andamentum.epistemic.entities.claim import ClaimStage
    from andamentum.epistemic.operations.verification import AdversarialCheckOperation
    from andamentum.epistemic.repository import EpistemicRepository
    from andamentum.epistemic.storage import InMemoryStorageBackend

    class _OneFramingRaisesRunner:
        def __init__(self):
            self.calls = 0

        async def run(self, agent_name: str, **kwargs):
            if agent_name == "epistemic_generate_counterquery":
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("framing 2 failed")
                from types import SimpleNamespace
                return SimpleNamespace(query=f"q-{self.calls}")
            raise AssertionError(f"Unexpected agent {agent_name}")

    repo = EpistemicRepository(backend=InMemoryStorageBackend())
    obj = Objective(description="q", clarified_question="q")
    await repo.save(obj)
    claim = Claim(
        objective_id=obj.entity_id,
        statement="X causes Y",
        scope="specific",
        stage=ClaimStage.PROPOSED,
    )
    await repo.save(claim)

    op = AdversarialCheckOperation(
        repo=repo,
        agent_runner=_OneFramingRaisesRunner(),
        evidence_gatherer=None,
        quality_scorer=None,
        embedding_model=None,
    )
    with pytest.raises(RuntimeError, match="framing 2 failed"):
        await op.execute(
            OperationInput(
                entity_id=claim.entity_id,
                entity_type="claim",
                operation="adversarial_check",
            )
        )
```

- [ ] **Step 6.2: Run — expect FAIL**

Run: `uv run pytest src/andamentum/epistemic/tests/test_no_silent_fallbacks.py -k counterquery -v`
Expected: FAIL.

- [ ] **Step 6.3: Remove the swallow**

Edit `src/andamentum/epistemic/operations/verification.py` lines 86-98. Replace:

```python
            async def _generate_one(framing: str) -> str | None:
                try:
                    cq_result = await self.run_agent(
                        "epistemic_generate_counterquery",
                        claim=claim.statement,
                        framing=framing,
                    )
                    return cq_result.query
                except Exception:
                    return None

            results = await asyncio.gather(*[_generate_one(f) for f in framings])
            agent_queries = [q for q in results if q is not None]
```

With:

```python
            async def _generate_one(framing: str) -> str:
                cq_result = await self.run_agent(
                    "epistemic_generate_counterquery",
                    claim=claim.statement,
                    framing=framing,
                )
                return cq_result.query

            agent_queries = list(
                await asyncio.gather(*[_generate_one(f) for f in framings])
            )
```

- [ ] **Step 6.4: Run — expect PASS**

Run: `uv run pytest src/andamentum/epistemic/tests/test_no_silent_fallbacks.py -k counterquery -v`
Expected: PASS.

- [ ] **Step 6.5: Run full suite**

Run: `uv run pytest src/andamentum/epistemic -x`

- [ ] **Step 6.6: Commit**

```bash
git add src/andamentum/epistemic/operations/verification.py src/andamentum/epistemic/tests/test_no_silent_fallbacks.py
git commit -m "fix(epistemic): propagate counterquery-generation failures"
```

---

## Task 7: Remove counterargument evaluator fallback (verification.py:191-203)

**Files:**
- Modify: `src/andamentum/epistemic/operations/verification.py` lines 149-203

**Why:** When `epistemic_evaluate_counterargument` fails on a single hit, current code builds a `Counterargument` with default scores which flows into `synthesize_adversarial_result`. Defaults from failed evaluations pollute the `adversarial_balance` that gates use. Fix: remove the try/except; let the whole operation raise.

- [ ] **Step 7.1: Write failing test**

Append:

```python
@pytest.mark.asyncio
async def test_adversarial_check_propagates_counterarg_eval_failure():
    """When epistemic_evaluate_counterargument raises on any hit, the
    operation must raise — do not build a default-scored Counterargument."""
    # Similar structure to previous test, but use a runner that raises
    # on "epistemic_evaluate_counterargument" after producing queries.
    # Need a simple evidence_gatherer stub that returns one fake hit.
    pytest.skip(
        "See implementation: requires an evidence_gatherer stub; scaffold "
        "when implementing this task."
    )
```

(The test is skipped in the plan; implement it fully during execution by mirroring the fixture shape in `tests/conftest.py`.)

- [ ] **Step 7.2: Remove the swallow**

Edit `src/andamentum/epistemic/operations/verification.py` lines 149-203. Replace:

```python
            async def _evaluate_one(
                summary: str, source_ref: str
            ) -> tuple[CounterargumentModel, str]:
                """Evaluate a single search hit. Returns (counterargument, justification).

                On failure, returns a fallback counterargument with defaults and empty
                justification — same behavior as the previous sequential loop.
                """
                async with eval_semaphore:
                    try:
                        eval_result = await self.run_agent(...)
                        ...
                        return proper_ca, justification
                    except Exception as e:
                        logger.warning(...)
                        proper_ca = create_counterargument(...)
                        return proper_ca, ""
```

With (drop the outer try/except, keep category-parsing try/except for the enum):

```python
            async def _evaluate_one(
                summary: str, source_ref: str
            ) -> tuple[CounterargumentModel, str]:
                """Evaluate a single search hit. Raises on agent failure —
                the caller relies on asyncio.gather to propagate."""
                async with eval_semaphore:
                    eval_result = await self.run_agent(
                        "epistemic_evaluate_counterargument",
                        claim_statement=claim.statement,
                        counterargument_text=summary,
                        source_ref=source_ref,
                    )
                    try:
                        category = CriticismCategory(eval_result.category)
                    except ValueError:
                        category = CriticismCategory.INTERPRETATION
                    quality = CounterargumentQuality(
                        relevance=eval_result.relevance,
                        specificity=eval_result.specificity,
                        evidence_backed=eval_result.evidence_backed,
                        source_credibility=eval_result.source_credibility,
                        novelty=0.5,
                    )
                    justification = (
                        getattr(eval_result, "justification", None) or ""
                    )
                    proper_ca = create_counterargument(
                        summary=summary,
                        source_ref=source_ref,
                        claim_id=claim.entity_id,
                        category=category,
                        quality=quality,
                    )
                    return proper_ca, justification
```

Note the `try/except ValueError` for enum parsing remains — that's deterministic coercion of an agent-supplied string, not a fallback.

- [ ] **Step 7.3: Unskip and complete the test; run it**

Implement the test body, then run. Expected: PASS.

- [ ] **Step 7.4: Run full suite**

Run: `uv run pytest src/andamentum/epistemic -x`

- [ ] **Step 7.5: Commit**

```bash
git add src/andamentum/epistemic/operations/verification.py src/andamentum/epistemic/tests/test_no_silent_fallbacks.py
git commit -m "fix(epistemic): propagate counterargument-evaluator failures"
```

---

## Task 8: Remove domain classifier fallback (verification.py:392-405)

**Files:**
- Modify: `src/andamentum/epistemic/operations/verification.py` lines 372-398

**Why:** When `epistemic_classify_evidence_domain` raises, code silently falls back to the heuristic `default_classify`. The convergence verdict downstream gates consume is then computed with a different classifier, no flag. Fix: let the exception propagate.

- [ ] **Step 8.1: Write failing test**

Append a test modeled on the counterquery test, but raising on `epistemic_classify_evidence_domain`. Expected post-fix: operation raises.

- [ ] **Step 8.2: Remove the fallback**

Edit lines 372-398. Replace:

```python
            if self.agent_runner:
                try:
                    dc_result = await self.run_agent(
                        "epistemic_classify_evidence_domain",
                        ...
                    )
                    classification = DomainClassification(...)
                except Exception:
                    # Fallback to default classification
                    classification = default_classify(...)
            else:
                # No agent runner — use default classification
                classification = default_classify(...)
```

With:

```python
            if self.agent_runner:
                dc_result = await self.run_agent(
                    "epistemic_classify_evidence_domain",
                    evidence_text=content,
                    source_type=ev.source_type,
                    source_ref=ev.source_ref,
                )
                classification = DomainClassification(
                    evidence_id=eid,
                    claim_id=claim.entity_id,
                    method_type=MethodType(dc_result.method_type),
                    data_source=DataSourceType(dc_result.data_source),
                    temporal=TemporalApproach(dc_result.temporal_approach),
                    causal_role=CausalRole(dc_result.causal_role),
                    classification_confidence=float(dc_result.confidence),
                    classification_method="agent",
                    classification_notes=dc_result.justification,
                )
            else:
                # No agent runner — use default classification (explicit no-agent path)
                classification = default_classify(
                    evidence_id=eid,
                    claim_id=claim.entity_id,
                    evidence_text=content,
                )
```

Note: the "no agent runner at all" path keeps `default_classify` — that's not a fallback from failure, it's an explicit no-LLM execution mode.

- [ ] **Step 8.3: Run tests**

Run: `uv run pytest src/andamentum/epistemic/tests/test_no_silent_fallbacks.py -k domain_classifier -v`
Expected: PASS.

- [ ] **Step 8.4: Full suite + commit**

```bash
git add src/andamentum/epistemic/operations/verification.py src/andamentum/epistemic/tests/test_no_silent_fallbacks.py
git commit -m "fix(epistemic): propagate domain-classifier failures"
```

---

## Task 9: Remove convergence evidence loader swallow (verification.py:360-368)

**Files:**
- Modify: `src/andamentum/epistemic/operations/verification.py` lines 360-368

**Why:** When `repo.get("evidence", eid)` raises inside the convergence loop, evidence is silently skipped. Convergence then computes against a partial set. If the failing evidence would have changed the verdict, the claim passes convergence silently. Fix: let repo load raise — that's an infrastructure error, not a per-item data issue.

- [ ] **Step 9.1: Write failing test**

Append: test with a stubbed repo that raises on `get("evidence", "ev-bad")` but not on others. Assert `ConvergenceAssessmentOperation` raises.

- [ ] **Step 9.2: Remove the swallow**

Edit lines 360-368. Replace:

```python
        for eid in claim.evidence_ids:
            try:
                ev = await self.repo.get("evidence", eid)
                if not isinstance(ev, Evidence) or not ev.extracted_content:
                    continue
                if ev.cluster_status in ("corroborative", "deferred"):
                    continue
            except Exception:
                continue
```

With:

```python
        for eid in claim.evidence_ids:
            ev = await self.repo.get("evidence", eid)
            if not isinstance(ev, Evidence) or not ev.extracted_content:
                continue
            if ev.cluster_status in ("corroborative", "deferred"):
                continue
```

- [ ] **Step 9.3: Run tests, full suite, commit**

```bash
git add src/andamentum/epistemic/operations/verification.py src/andamentum/epistemic/tests/test_no_silent_fallbacks.py
git commit -m "fix(epistemic): propagate repo-load failures in convergence assessment"
```

---

## Task 10: Remove objective-load fallback in stage_management.py

**Files:**
- Modify: `src/andamentum/epistemic/operations/stage_management.py` lines 49-55

**Why:** When `repo.get("objective", claim.objective_id)` fails, the code falls through with `question_type=None` and uses default routing thresholds. A claim can be promoted under the wrong thresholds with no signal. Gates make the final safety call; a stage-promotion is too load-bearing to be silent. Fix: let the load raise.

- [ ] **Step 10.1: Write failing test**

```python
@pytest.mark.asyncio
async def test_promote_claim_propagates_objective_load_failure():
    """When the objective can't be loaded, promotion must raise — silently
    falling back to default thresholds could promote claims under the wrong
    routing profile."""
    # Use a repo stub that raises on get("objective", ...).
    pytest.skip("Implement with repo stub during task execution.")
```

- [ ] **Step 10.2: Remove the fallback**

Edit lines 49-55. Replace:

```python
        question_type = None
        try:
            objective = await self.repo.get("objective", claim.objective_id)
            question_type = objective.question_type
        except Exception:
            pass  # Fall back to default thresholds
```

With:

```python
        objective = await self.repo.get("objective", claim.objective_id)
        question_type = objective.question_type
```

- [ ] **Step 10.3: Complete test, run suite, commit**

```bash
git add src/andamentum/epistemic/operations/stage_management.py src/andamentum/epistemic/tests/test_no_silent_fallbacks.py
git commit -m "fix(epistemic): propagate objective-load failures in claim promotion"
```

---

## Task 11: Remove posterior-computation swallow in graph/__init__.py

**Files:**
- Modify: `src/andamentum/epistemic/graph/__init__.py` lines 145-153

**Why:** When `compute_posterior` raises, the current code logs a warning and returns `posterior=None`. The headline confidence number — the whole point of the pipeline — is silently absent. Fix: let it propagate. If the graph completes successfully enough to compute posterior, posterior failure is a code bug, not a data issue.

- [ ] **Step 11.1: Write failing test**

```python
@pytest.mark.asyncio
async def test_run_epistemic_graph_propagates_posterior_failure(monkeypatch):
    """When compute_posterior raises, run_epistemic_graph must propagate —
    a silent None posterior hides the headline result."""
    # Monkeypatch confidence.compute_posterior to raise; run a minimal graph
    # that completes successfully; assert the expected exception propagates.
    pytest.skip("Implement with monkeypatch during task execution.")
```

- [ ] **Step 11.2: Remove the swallow**

Edit `src/andamentum/epistemic/graph/__init__.py` lines 145-153. Replace:

```python
    # Compute posterior confidence (deterministic, no LLM)
    posterior_report = None
    if result.successful > 0:
        try:
            from ..confidence import compute_posterior

            posterior_report = await compute_posterior(repo, objective_id)
        except Exception as e:
            logger.warning(f"Posterior computation failed: {e}")
```

With:

```python
    # Compute posterior confidence (deterministic, no LLM).
    # No fallback: if posterior raises, the caller sees the real error.
    posterior_report = None
    if result.successful > 0:
        from ..confidence import compute_posterior

        posterior_report = await compute_posterior(repo, objective_id)
```

- [ ] **Step 11.3: Complete test, run suite, commit**

```bash
git add src/andamentum/epistemic/graph/__init__.py src/andamentum/epistemic/tests/test_no_silent_fallbacks.py
git commit -m "fix(epistemic): propagate posterior-computation failures"
```

---

## Task 12: Remove CompositeGatherer silent provider→web fallback

**Files:**
- Modify: `src/andamentum/epistemic/evidence_gathering.py` lines 337-394

**Why:** When a specifically-requested provider (`chembl`, `clinicaltrials`, etc.) either fails OR returns empty, code silently falls through to generic web search. A user asking for `clinicaltrials` evidence may get web results labelled as if they came from the specific provider they requested.

**Design nuance:**
- Specific source_type (e.g. `chembl`) → provider raises → surface the error. Provider returns empty → return empty (not fall back).
- `source_type == "all"` → iterate over all providers + web, accumulate successes, raise if every single one fails.
- `source_type` not in providers and not `all` and not `web_search` → this is the "unknown source_type" case. Options: raise, or treat as web_search. Current code treats as web_search. Keep that — it's an explicit mapping, not a silent fallback.

- [ ] **Step 12.1: Write failing tests**

```python
@pytest.mark.asyncio
async def test_composite_gatherer_specific_provider_failure_raises():
    """When the specifically-requested provider fails, CompositeGatherer must
    raise — not silently fall through to web search."""
    from andamentum.epistemic.evidence_gathering import CompositeGatherer

    class _FailingProvider:
        async def gather(self, query):
            raise RuntimeError("clinicaltrials.gov unreachable")

    class _UnreachableWebSearch:
        async def gather(self, source_type, query):
            raise AssertionError("web search should not be called on provider failure")

    gatherer = CompositeGatherer(
        web_search=_UnreachableWebSearch(),
        providers={"clinicaltrials": _FailingProvider()},
    )
    with pytest.raises(RuntimeError, match="clinicaltrials.gov unreachable"):
        await gatherer.gather("clinicaltrials", "diabetes prevention")


@pytest.mark.asyncio
async def test_composite_gatherer_empty_provider_returns_empty():
    """When the specifically-requested provider returns empty, CompositeGatherer
    must return empty — not silently fall through to web search."""
    from andamentum.epistemic.evidence_gathering import CompositeGatherer

    class _EmptyProvider:
        async def gather(self, query):
            return []

    class _UnreachableWebSearch:
        async def gather(self, source_type, query):
            raise AssertionError("web search should not be called on empty provider")

    gatherer = CompositeGatherer(
        web_search=_UnreachableWebSearch(),
        providers={"chembl": _EmptyProvider()},
    )
    assert await gatherer.gather("chembl", "aspirin") == []


@pytest.mark.asyncio
async def test_composite_gatherer_all_aggregates_and_logs_failures():
    """source_type='all' iterates all providers + web. Per-provider failures
    are logged but don't prevent aggregation of successes. If every provider
    fails AND web search fails, raise."""
    # Implement per the design nuance above.
    pytest.skip("Implement during task execution.")
```

- [ ] **Step 12.2: Run — expect FAIL**

- [ ] **Step 12.3: Rewrite `CompositeGatherer.gather`**

Replace the body of `gather` (lines 337-394) with:

```python
    async def gather(self, source_type: str, query: str) -> list[GatheredEvidence]:
        """Gather evidence.

        Routing:
        - Specific provider registered for source_type → call it.
          Provider raises → propagate (no silent fallback).
          Provider returns empty → return empty (no silent fallback).
        - source_type == "all" → call every provider and web search,
          aggregate successes, log per-provider failures. Raise only
          if EVERY single call fails.
        - Unknown source_type → use web search (explicit default route).
        """
        logger.info(
            "[CompositeGatherer] source_type=%s providers=%s query=%.80s",
            source_type,
            list(self._providers.keys()),
            query,
        )

        if source_type == "all":
            all_results: list[GatheredEvidence] = []
            failures: list[tuple[str, Exception]] = []

            for name, prov in self._providers.items():
                try:
                    results = await prov.gather(query)
                    logger.info(
                        "[CompositeGatherer] Provider '%s' returned %d results",
                        name, len(results),
                    )
                    all_results.extend(results)
                except Exception as e:
                    logger.warning(
                        "[CompositeGatherer] Provider '%s' failed during 'all': %s",
                        name, e,
                    )
                    failures.append((name, e))
            try:
                web_results = await self._web_search.gather(source_type, query)
                all_results.extend(web_results)
            except Exception as e:
                logger.warning(
                    "[CompositeGatherer] Web search failed during 'all': %s", e
                )
                failures.append(("web_search", e))

            if not all_results and failures:
                raise RuntimeError(
                    f"All gather calls failed for 'all' source_type. "
                    f"Failures: {[(n, type(e).__name__, str(e)) for n, e in failures]}"
                )
            return all_results

        provider = self._providers.get(source_type)
        if provider:
            results = await provider.gather(query)
            logger.info(
                "[CompositeGatherer] Provider '%s' returned %d results",
                source_type, len(results),
            )
            return results

        # Unknown source_type → explicit web-search default route
        return await self._web_search.gather(source_type, query)
```

- [ ] **Step 12.4: Run tests, full suite**

Run: `uv run pytest src/andamentum/epistemic -x`

Expected: existing tests that relied on the silent web-search fallback will fail. Review each: either the test was asserting a bug (update it to expect the new raise), or the test wires a provider that should genuinely return empty (fine).

- [ ] **Step 12.5: Commit**

```bash
git add src/andamentum/epistemic/evidence_gathering.py src/andamentum/epistemic/tests/test_no_silent_fallbacks.py
git commit -m "fix(epistemic): remove CompositeGatherer silent provider→web fallback"
```

---

## Task 13: Full verification — green state

**Files:** none modified; verification only.

- [ ] **Step 13.1: Run the full pytest suite**

Run: `uv run pytest`
Expected: **814 tests passing, 1 benchmark deselected** (matching CLAUDE.md canonical green state). If test count drifted, investigate before claiming done.

- [ ] **Step 13.2: Run pyright**

Run: `uv run pyright`
Expected: **0 errors, 0 warnings.**

- [ ] **Step 13.3: Run ruff**

Run: `uv run ruff check && uv run ruff format --check`
Expected: clean.

- [ ] **Step 13.4: Smoke-test a real run**

Run the epistemic CLI against a short claim using the local model configured in `ANDAMENTUM_MAIN_LLM_MODEL`, and verify:
1. A successful run produces a posterior as before.
2. A deliberate failure (unset `SEARXNG_URL` or equivalent) produces a visible quarantine entry in the output, not a silent degraded answer.

Document the smoke-test command and outcome in the commit message.

- [ ] **Step 13.5: Commit (no code changes, but record the verification)**

No commit needed if nothing changed. If the smoke test revealed an issue in earlier tasks, fix it in a follow-up commit.

---

## Self-Review Notes

**Spec coverage:** Every silent-fallback site from Theme 1 has a task (4–12). Infrastructure (quarantine) is Tasks 1–3. Verification is Task 13.

**Known trade-offs:**
- Task 5 (screening failure) raises out of `ProposeClaimsOperation`, which quarantines the whole objective rather than just the failing evidence. This is strictly louder than the alternative of per-evidence quarantine. If during execution the user decides this is too aggressive (a single flaky screen call should not lose the whole objective), revisit by quarantining individual evidence items on failure.
- Task 7 same concern: one failing counterargument eval fails the whole adversarial check. Matches Task 5's choice for consistency.
- Task 12's "unknown source_type → web search" is a deliberate keep: it's an explicit default route, not a fallback from failure.

**Not covered here (intentional):**
- `deep_research/nodes.py` synthesis fallback
- `deep_research/novelty/checker.py`
- `document_store/extraction.py`
- `document_store/search.py` silent `[]` returns across 4 RRF signals
- `document_store/public.py:267-276` doc-embedding skip

These belong to Theme 4 (document_store) and a future deep_research theme, to be planned separately.
