"""Tests for core model resolution."""

from andamentum.core.models import (
    resolve_model,
    resolve_model_from_args,
    BEDROCK_MODEL_MAP,
    REGION_PREFIX_MAP,
)


class TestResolveModel:
    def test_passthrough_openai(self):
        """openai: prefix passes through as string."""
        result = resolve_model("openai:gpt-4o")
        assert result == "openai:gpt-4o"

    def test_passthrough_anthropic(self):
        """anthropic: prefix passes through as string."""
        result = resolve_model("anthropic:claude-haiku-4-5")
        assert result == "anthropic:claude-haiku-4-5"

    def test_passthrough_unknown_prefix(self):
        """Unknown prefix passes through as string."""
        result = resolve_model("google:gemini-pro")
        assert result == "google:gemini-pro"

    def test_ollama_creates_model_object(self):
        """ollama: prefix creates OllamaModel, not a string."""
        result = resolve_model("ollama:llama3")
        assert not isinstance(result, str)

    def test_ollama_uses_ollama_model_class(self):
        """ollama: prefix creates OllamaModel specifically."""
        from pydantic_ai.models.ollama import OllamaModel

        result = resolve_model("ollama:llama3")
        assert isinstance(result, OllamaModel)

    def test_ollama_respects_env_var(self, monkeypatch):
        """OLLAMA_BASE_URL env var is used for ollama models."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://custom:1234/v1")
        result = resolve_model("ollama:llama3")
        assert not isinstance(result, str)

    def test_ollama_default_base_url(self, monkeypatch):
        """Default OLLAMA_BASE_URL is localhost:11434/v1."""
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        result = resolve_model("ollama:llama3")
        assert not isinstance(result, str)


class TestBedrockModelMap:
    def test_known_models_in_map(self):
        """Bedrock map should have key models."""
        assert "claude-haiku-4-5" in BEDROCK_MODEL_MAP
        assert "claude-sonnet-4-5" in BEDROCK_MODEL_MAP
        assert "claude-opus-4-5" in BEDROCK_MODEL_MAP

    def test_region_prefix_map_has_entries(self):
        """Region prefix map should have key regions."""
        assert "ap-southeast-2" in REGION_PREFIX_MAP
        assert REGION_PREFIX_MAP["ap-southeast-2"] == "au"


class TestResolveModelFromArgs:
    def test_returns_arg_when_provided(self):
        result = resolve_model_from_args("openai:gpt-4o")
        assert result == "openai:gpt-4o"

    def test_falls_back_to_env_var(self, monkeypatch):
        monkeypatch.setenv("ANDAMENTUM_MAIN_LLM_MODEL", "anthropic:claude-haiku-4-5")
        result = resolve_model_from_args(None)
        assert result == "anthropic:claude-haiku-4-5"

    def test_exits_when_neither_provided(self, monkeypatch):
        monkeypatch.delenv("ANDAMENTUM_MAIN_LLM_MODEL", raising=False)
        import pytest

        with pytest.raises(SystemExit):
            resolve_model_from_args(None)
