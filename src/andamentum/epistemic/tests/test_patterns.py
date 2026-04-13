"""Tests for pattern matching and scheduler."""

from ..entities import Claim, ClaimStage, Objective
from ..patterns import (
    Pattern,
    PatternScheduler,
    WORK_PATTERNS,
    DEFAULT_OPERATION_BUDGETS,
    MAX_ENTITY_ATTEMPTS,
)


class TestPatternMatching:
    def test_exact_match(self):
        p = Pattern(
            entity_type="claim", filters={"stage": "hypothesis"}, operation="scrutinise"
        )
        c = Claim(statement="X", objective_id="o", stage=ClaimStage.HYPOTHESIS)
        assert p.matches(c)

    def test_exact_mismatch(self):
        p = Pattern(
            entity_type="claim", filters={"stage": "supported"}, operation="promote"
        )
        c = Claim(statement="X", objective_id="o", stage=ClaimStage.HYPOTHESIS)
        assert not p.matches(c)

    def test_none_filter(self):
        p = Pattern(
            entity_type="claim",
            filters={"scrutiny_verdict": None},
            operation="scrutinise",
        )
        c = Claim(statement="X", objective_id="o")
        assert p.matches(c)

    def test_gte_filter(self):
        p = Pattern(
            entity_type="claim", filters={"evidence_count__gte": 3}, operation="promote"
        )
        c = Claim(statement="X", objective_id="o", evidence_ids=["e1", "e2", "e3"])
        assert p.matches(c)

    def test_gte_filter_fails(self):
        p = Pattern(
            entity_type="claim", filters={"evidence_count__gte": 3}, operation="promote"
        )
        c = Claim(statement="X", objective_id="o", evidence_ids=["e1"])
        assert not p.matches(c)

    def test_lte_filter(self):
        p = Pattern(
            entity_type="claim", filters={"modification_count__lte": 2}, operation="ok"
        )
        c = Claim(statement="X", objective_id="o", modification_count=1)
        assert p.matches(c)

    def test_gt_filter(self):
        p = Pattern(
            entity_type="claim", filters={"evidence_count__gt": 0}, operation="ok"
        )
        c = Claim(statement="X", objective_id="o", evidence_ids=["e1"])
        assert p.matches(c)

    def test_lt_filter(self):
        p = Pattern(
            entity_type="claim", filters={"modification_count__lt": 3}, operation="ok"
        )
        c = Claim(statement="X", objective_id="o", modification_count=2)
        assert p.matches(c)

    def test_ne_filter(self):
        p = Pattern(
            entity_type="claim", filters={"stage__ne": "hypothesis"}, operation="demote"
        )
        c = Claim(statement="X", objective_id="o", stage=ClaimStage.SUPPORTED)
        assert p.matches(c)

    def test_ne_filter_fails(self):
        p = Pattern(
            entity_type="claim", filters={"stage__ne": "hypothesis"}, operation="demote"
        )
        c = Claim(statement="X", objective_id="o", stage=ClaimStage.HYPOTHESIS)
        assert not p.matches(c)

    def test_contains_filter(self):
        p = Pattern(
            entity_type="claim",
            filters={"evidence_ids__contains": "e-1"},
            operation="ok",
        )
        c = Claim(statement="X", objective_id="o", evidence_ids=["e-1", "e-2"])
        assert p.matches(c)

    def test_contains_filter_fails(self):
        p = Pattern(
            entity_type="claim",
            filters={"evidence_ids__contains": "e-99"},
            operation="ok",
        )
        c = Claim(statement="X", objective_id="o", evidence_ids=["e-1"])
        assert not p.matches(c)

    def test_multiple_filters_all_must_match(self):
        p = Pattern(
            entity_type="claim",
            filters={"stage": "hypothesis", "scrutiny_verdict": "pass"},
            operation="promote",
        )
        c1 = Claim(
            statement="X",
            objective_id="o",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="pass",
        )
        c2 = Claim(
            statement="X",
            objective_id="o",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict=None,
        )
        assert p.matches(c1)
        assert not p.matches(c2)


class TestWorkPatterns:
    def test_patterns_not_empty(self):
        assert len(WORK_PATTERNS) > 0

    def test_all_patterns_have_operations(self):
        for p in WORK_PATTERNS:
            assert p.operation, f"Pattern {p.description} has no operation"

    def test_preplanning_patterns_exist(self):
        ops = {p.operation for p in WORK_PATTERNS}
        assert "clarify_question" in ops
        assert "conceptual_analysis" in ops
        assert "plan_task" in ops

    def test_synthesis_patterns_exist(self):
        ops = {p.operation for p in WORK_PATTERNS}
        assert "freeze_snapshot" in ops
        assert "synthesize_report" in ops

    def test_scope_creating_ops_have_budgets(self):
        """Scope-creating operations must have budget caps; processing operations must not.

        Design invariant: budgets gate SCOPE (entity creation), not EXECUTION
        (processing of existing entities). Processing ops (extract_evidence,
        scrutinise_claim, adversarial_search, etc.) are idempotent via pattern
        filters — they fire at most once per entity based on entity state flags.
        Capping them strands claims mid-pipeline, producing incomplete benchmark data.
        """
        # Scope-creating ops: must be capped
        assert DEFAULT_OPERATION_BUDGETS["plan_task"] == 2
        assert DEFAULT_OPERATION_BUDGETS["propose_claims"] == 2
        assert DEFAULT_OPERATION_BUDGETS["clarify_question"] == 2
        assert DEFAULT_OPERATION_BUDGETS["conceptual_analysis"] == 2

        # Processing ops: must NOT be in default budgets (idempotent via pattern filters)
        processing_ops = [
            "extract_evidence",
            "scrutinise_claim",
            "adversarial_search",
            "assess_convergence",
            "validate_deductively",
            "verify_computationally",
            "promote_claim",
            "demote_claim",
            "analyze_argument",
            "generate_prediction",
            "record_decision",
        ]
        for op in processing_ops:
            assert op not in DEFAULT_OPERATION_BUDGETS, (
                f"'{op}' should not be budget-capped — it is idempotent via pattern "
                f"filters and must always complete for existing entities."
            )


class TestPatternScheduler:
    async def test_new_objective_generates_clarify_work(self, repo):
        o = Objective(
            entity_id="obj-1", objective_id="obj-1", description="Test Q", phase="new"
        )
        await repo.save(o)
        scheduler = PatternScheduler(repo)
        work = await scheduler.get_pending_work(objective_id="obj-1")
        ops = [w.operation for w in work]
        assert "clarify_question" in ops

    async def test_no_work_for_empty_repo(self, repo):
        scheduler = PatternScheduler(repo)
        work = await scheduler.get_pending_work()
        assert len(work) == 0

    async def test_get_next_work_returns_first_pattern_work(self, repo):
        o = Objective(
            entity_id="obj-1", objective_id="obj-1", description="Q", phase="new"
        )
        await repo.save(o)
        scheduler = PatternScheduler(repo)
        item = await scheduler.get_next_work(objective_id="obj-1")
        assert item is not None
        assert item.operation == "clarify_question"

    async def test_has_pending_work(self, repo):
        scheduler = PatternScheduler(repo)
        assert not await scheduler.has_pending_work()
        o = Objective(
            entity_id="obj-1", objective_id="obj-1", description="Q", phase="new"
        )
        await repo.save(o)
        assert await scheduler.has_pending_work(objective_id="obj-1")


class TestBudgetExhaustion:
    async def test_budget_limits_operations(self, repo):
        scheduler = PatternScheduler(repo, operation_budgets={"clarify_question": 1})
        o = Objective(
            entity_id="obj-1", objective_id="obj-1", description="Q", phase="new"
        )
        await repo.save(o)

        # Before recording, work exists
        work = await scheduler.get_pending_work(objective_id="obj-1")
        assert any(w.operation == "clarify_question" for w in work)

        # After exhausting budget via successful execution, work is gone
        scheduler.record_attempt("obj-1", "clarify_question")
        scheduler.record_success("clarify_question")
        work = await scheduler.get_pending_work(objective_id="obj-1")
        assert not any(w.operation == "clarify_question" for w in work)

    async def test_synthesis_ops_never_budget_limited(self, repo):
        scheduler = PatternScheduler(repo, operation_budgets={"freeze_snapshot": 0})
        o = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="Q",
            phase="claims_done",
            snapshot_id=None,
        )
        await repo.save(o)
        work = await scheduler.get_pending_work(objective_id="obj-1")
        assert any(w.operation == "freeze_snapshot" for w in work)


class TestEntityAttemptLimits:
    async def test_entity_exhausted_after_max_attempts(self, repo):
        # Use empty operation_budgets so only entity-level limits are tested
        scheduler = PatternScheduler(repo, operation_budgets={})
        o = Objective(
            entity_id="obj-1", objective_id="obj-1", description="Q", phase="new"
        )
        await repo.save(o)

        for _ in range(MAX_ENTITY_ATTEMPTS):
            scheduler.record_attempt("obj-1", "clarify_question")

        work = await scheduler.get_pending_work(objective_id="obj-1")
        assert not any(
            w.operation == "clarify_question" and w.entity_id == "obj-1" for w in work
        )

    async def test_entity_not_exhausted_below_limit(self, repo):
        # Use empty operation_budgets so only entity-level limits are tested
        scheduler = PatternScheduler(repo, operation_budgets={})
        o = Objective(
            entity_id="obj-1", objective_id="obj-1", description="Q", phase="new"
        )
        await repo.save(o)

        for _ in range(MAX_ENTITY_ATTEMPTS - 1):
            scheduler.record_attempt("obj-1", "clarify_question")

        work = await scheduler.get_pending_work(objective_id="obj-1")
        assert any(w.operation == "clarify_question" for w in work)
