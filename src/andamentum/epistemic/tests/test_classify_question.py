"""Tests for question type classification."""

import pytest
from epistemic.primitives import QuestionType


class TestQuestionTypeEnum:
    def test_has_seven_values(self):
        assert len(QuestionType) == 7

    def test_values(self):
        expected = {
            "verificatory", "explanatory", "exploratory",
            "comparative", "predictive", "compositional", "normative",
        }
        assert {qt.value for qt in QuestionType} == expected

    def test_is_string_enum(self):
        assert QuestionType.VERIFICATORY == "verificatory"
        assert isinstance(QuestionType.VERIFICATORY, str)


from epistemic.entities.objective import Objective


class TestObjectiveQuestionType:
    def test_defaults_to_none(self):
        obj = Objective(description="test question")
        assert obj.question_type is None

    def test_can_set_question_type(self):
        obj = Objective(
            description="Is P true?",
            question_type="verificatory",
        )
        assert obj.question_type == "verificatory"

    def test_question_type_in_metadata(self):
        obj = Objective(
            description="test",
            question_type="explanatory",
        )
        content, metadata = obj.to_document()
        assert metadata["question_type"] == "explanatory"

    def test_question_type_none_not_in_metadata(self):
        obj = Objective(description="test")
        content, metadata = obj.to_document()
        assert metadata.get("question_type") is None

    def test_roundtrip_from_metadata(self):
        obj = Objective(
            description="test",
            question_type="predictive",
        )
        content, metadata = obj.to_document()
        restored = Objective.from_document(content, metadata)
        assert restored.question_type == "predictive"

    def test_roundtrip_none_from_metadata(self):
        obj = Objective(description="test")
        content, metadata = obj.to_document()
        restored = Objective.from_document(content, metadata)
        assert restored.question_type is None


from epistemic.agents import get_agent
from epistemic.agents.output_models import ClassifyQuestionOutput


class TestClassifyQuestionAgent:
    def test_agent_registered(self):
        agent = get_agent("epistemic_classify_question")
        assert agent.name == "epistemic_classify_question"

    def test_output_model_is_correct(self):
        agent = get_agent("epistemic_classify_question")
        assert agent.output_model is ClassifyQuestionOutput

    def test_output_model_fields(self):
        """Output model should be narrow: just question_type + reasoning."""
        fields = ClassifyQuestionOutput.model_fields
        assert "question_type" in fields
        assert "reasoning" in fields
        assert len(fields) == 2

    def test_output_model_accepts_valid_type(self):
        output = ClassifyQuestionOutput(
            question_type="verificatory",
            reasoning="Binary truth claim",
        )
        assert output.question_type == "verificatory"


from types import SimpleNamespace
from epistemic.adapters import adapt_agent_output


class TestClassifyQuestionAdapter:
    def test_adapts_valid_output(self):
        raw = SimpleNamespace(
            question_type="verificatory",
            reasoning="Binary truth claim about efficacy",
        )
        result = adapt_agent_output("epistemic_classify_question", raw)
        assert result.question_type == "verificatory"
        assert result.reasoning == "Binary truth claim about efficacy"

    def test_normalizes_case(self):
        raw = SimpleNamespace(
            question_type="VERIFICATORY",
            reasoning="test",
        )
        result = adapt_agent_output("epistemic_classify_question", raw)
        assert result.question_type == "verificatory"

    def test_strips_whitespace(self):
        raw = SimpleNamespace(
            question_type="  explanatory  ",
            reasoning="test",
        )
        result = adapt_agent_output("epistemic_classify_question", raw)
        assert result.question_type == "explanatory"


from epistemic.storage import InMemoryStorageBackend
from epistemic.repository import EpistemicRepository
from epistemic.operations import ClassifyQuestionOperation, OPERATION_CLASSES
from epistemic.patterns import WorkItem


class TestClassifyQuestionOperation:
    @pytest.fixture
    def backend(self):
        return InMemoryStorageBackend()

    @pytest.fixture
    def repo(self, backend):
        return EpistemicRepository(backend)

    @pytest.fixture
    def mock_runner(self):
        class Runner:
            def __init__(self):
                self.calls = []
                self._agent_calls = []
            async def run(self, agent_name, **kwargs):
                self.calls.append((agent_name, kwargs))
                return SimpleNamespace(
                    question_type="explanatory",
                    reasoning="Asks how a mechanism works",
                )
        return Runner()

    @pytest.mark.asyncio
    async def test_sets_question_type_on_objective(self, repo, mock_runner):
        obj = Objective(
            entity_id="obj-cls-1",
            objective_id="obj-cls-1",
            description="How does spaced repetition work?",
            phase="clarified",
        )
        await repo.save(obj)

        op = ClassifyQuestionOperation(repo, mock_runner)
        work = WorkItem(
            entity_id="obj-cls-1",
            entity_type="objective",
            operation="classify_question",
        )
        result = await op.execute(work)

        assert result.success
        updated = await repo.get("objective", "obj-cls-1")
        assert updated.question_type == "explanatory"

    @pytest.mark.asyncio
    async def test_does_not_change_phase(self, repo, mock_runner):
        obj = Objective(
            entity_id="obj-cls-2",
            objective_id="obj-cls-2",
            description="test",
            phase="clarified",
        )
        await repo.save(obj)

        op = ClassifyQuestionOperation(repo, mock_runner)
        work = WorkItem(
            entity_id="obj-cls-2",
            entity_type="objective",
            operation="classify_question",
        )
        await op.execute(work)

        updated = await repo.get("objective", "obj-cls-2")
        assert updated.phase == "clarified"

    @pytest.mark.asyncio
    async def test_calls_correct_agent(self, repo, mock_runner):
        obj = Objective(
            entity_id="obj-cls-3",
            objective_id="obj-cls-3",
            description="Is coffee healthy?",
            phase="clarified",
            clarified_question="Does regular coffee consumption improve health outcomes?",
        )
        await repo.save(obj)

        op = ClassifyQuestionOperation(repo, mock_runner)
        work = WorkItem(
            entity_id="obj-cls-3",
            entity_type="objective",
            operation="classify_question",
        )
        await op.execute(work)

        assert len(mock_runner.calls) == 1
        agent_name, kwargs = mock_runner.calls[0]
        assert agent_name == "epistemic_classify_question"
        assert "Does regular coffee" in kwargs.get("question", "")

    @pytest.mark.asyncio
    async def test_registered_in_operation_classes(self):
        assert "classify_question" in OPERATION_CLASSES

    @pytest.mark.asyncio
    async def test_skips_if_already_classified(self, repo, mock_runner):
        obj = Objective(
            entity_id="obj-cls-4",
            objective_id="obj-cls-4",
            description="test",
            phase="clarified",
            question_type="verificatory",
        )
        await repo.save(obj)

        op = ClassifyQuestionOperation(repo, mock_runner)
        work = WorkItem(
            entity_id="obj-cls-4",
            entity_type="objective",
            operation="classify_question",
        )
        result = await op.execute(work)

        assert result.success
        assert len(mock_runner.calls) == 0  # No agent call


from epistemic.patterns import PatternScheduler, WORK_PATTERNS, Pattern


class TestClassifyQuestionPattern:
    def test_pattern_exists(self):
        names = [p.operation for p in WORK_PATTERNS]
        assert "classify_question" in names

    def test_pattern_matches_clarified_unclassified(self):
        pattern = next(p for p in WORK_PATTERNS if p.operation == "classify_question")
        obj = Objective(description="test", phase="clarified")
        assert pattern.matches(obj)

    def test_pattern_does_not_match_already_classified(self):
        pattern = next(p for p in WORK_PATTERNS if p.operation == "classify_question")
        obj = Objective(description="test", phase="clarified", question_type="verificatory")
        assert not pattern.matches(obj)

    def test_pattern_does_not_match_new_phase(self):
        pattern = next(p for p in WORK_PATTERNS if p.operation == "classify_question")
        obj = Objective(description="test", phase="new")
        assert not pattern.matches(obj)

    def test_classify_before_conceptual_analysis(self):
        classify_idx = next(i for i, p in enumerate(WORK_PATTERNS) if p.operation == "classify_question")
        conceptual_idx = next(i for i, p in enumerate(WORK_PATTERNS) if p.operation == "conceptual_analysis")
        assert classify_idx < conceptual_idx

    @pytest.mark.asyncio
    async def test_scheduler_picks_classify_after_clarify(self):
        backend = InMemoryStorageBackend()
        repo = EpistemicRepository(backend)
        obj = Objective(
            entity_id="obj-sched-1",
            objective_id="obj-sched-1",
            description="test",
            phase="clarified",
        )
        await repo.save(obj)

        scheduler = PatternScheduler(repo)
        work = await scheduler.get_next_work(obj.entity_id)
        assert work is not None
        assert work.operation == "classify_question"

    @pytest.mark.asyncio
    async def test_scheduler_picks_conceptual_after_classification(self):
        backend = InMemoryStorageBackend()
        repo = EpistemicRepository(backend)
        obj = Objective(
            entity_id="obj-sched-2",
            objective_id="obj-sched-2",
            description="test",
            phase="clarified",
            question_type="verificatory",
        )
        await repo.save(obj)

        scheduler = PatternScheduler(repo)
        work = await scheduler.get_next_work(obj.entity_id)
        assert work is not None
        assert work.operation == "conceptual_analysis"
