"""Tests for question type classification."""

import pytest
from ..primitives import QuestionType


class TestQuestionTypeEnum:
    def test_has_seven_values(self):
        assert len(QuestionType) == 7

    def test_values(self):
        expected = {
            "verificatory",
            "explanatory",
            "exploratory",
            "comparative",
            "predictive",
            "compositional",
            "normative",
        }
        assert {qt.value for qt in QuestionType} == expected

    def test_is_string_enum(self):
        assert QuestionType.VERIFICATORY == "verificatory"
        assert isinstance(QuestionType.VERIFICATORY, str)


from ..entities.objective import Objective  # noqa: E402


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


from ..agents import get_agent  # noqa: E402
from ..agents.output_models import ClassifyQuestionOutput  # noqa: E402


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


from types import SimpleNamespace  # noqa: E402
from ..adapters import adapt_agent_output  # noqa: E402


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


from andamentum.document_store import DocumentStore  # noqa: E402
from ..repository import EpistemicRepository  # noqa: E402
from ..operations import ClassifyQuestionOperation, OPERATION_CLASSES  # noqa: E402
from ..operations.base import OperationInput  # noqa: E402


class TestClassifyQuestionOperation:
    @pytest.fixture
    async def store(self, tmp_path):
        s = DocumentStore.for_database("test", db_dir=tmp_path)
        await s.initialize()
        return s

    @pytest.fixture
    async def repo(self, store):
        return EpistemicRepository(store)

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
        work = OperationInput(
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
        work = OperationInput(
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
        work = OperationInput(
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
        work = OperationInput(
            entity_id="obj-cls-4",
            entity_type="objective",
            operation="classify_question",
        )
        result = await op.execute(work)

        assert result.success
        assert len(mock_runner.calls) == 0  # No agent call

