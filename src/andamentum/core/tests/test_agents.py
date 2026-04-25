"""Tests for core agent runner with PromptedOutput fallback."""

import pytest
from pydantic import BaseModel, Field

from andamentum.core.agents import AgentDefinition, AgentRunner


class SimpleOutput(BaseModel):
    """Test output model."""

    answer: str = Field(description="The answer")
    confidence: float = Field(description="0-1 confidence")


TEST_DEFN = AgentDefinition(
    name="test_agent",
    prompt="You are a test agent. Answer concisely.",
    output_model=SimpleOutput,
    retries=1,
    output_retries=1,
)


class TestAgentDefinition:
    def test_frozen(self):
        """AgentDefinition is immutable."""
        with pytest.raises(AttributeError):
            TEST_DEFN.name = "changed"  # type: ignore[misc]

    def test_fields(self):
        """All fields are accessible."""
        assert TEST_DEFN.name == "test_agent"
        assert TEST_DEFN.output_model is SimpleOutput
        assert TEST_DEFN.retries == 1
        assert TEST_DEFN.output_retries == 1

    def test_defaults(self):
        """Default retries are 3 and 5."""
        defn = AgentDefinition(
            name="default_test",
            prompt="test",
            output_model=SimpleOutput,
        )
        assert defn.retries == 3
        assert defn.output_retries == 5


class TestAgentRunner:
    def test_init_with_string_model(self):
        """Runner accepts a model string."""
        runner = AgentRunner(model="openai:gpt-4o")
        assert runner.model == "openai:gpt-4o"  # passthrough

    def test_init_resolves_model(self):
        """Runner resolves ollama: prefix."""
        runner = AgentRunner(model="ollama:llama3")
        assert not isinstance(runner.model, str)

    def test_cache_starts_empty(self):
        """Cache is empty on init."""
        runner = AgentRunner(model="openai:gpt-4o")
        assert len(runner._cache) == 0

    def test_clear_cache(self):
        """clear_cache empties the cache."""
        runner = AgentRunner(model="openai:gpt-4o")
        runner._cache["test"] = "value"
        runner.clear_cache()
        assert len(runner._cache) == 0

    def test_is_local_true_for_ollama_string(self):
        runner = AgentRunner(model="ollama:llama3")
        assert runner.is_local is True

    def test_is_local_false_for_openai_string(self):
        runner = AgentRunner(model="openai:gpt-4o")
        assert runner.is_local is False

    def test_is_local_false_for_anthropic_string(self):
        runner = AgentRunner(model="anthropic:claude-haiku-4-5")
        assert runner.is_local is False
