"""Tests for andamentum.core.cloud_gate."""

from __future__ import annotations

import pytest

from andamentum.core.cloud_gate import is_cloud_model, provider_name


class TestIsCloudModel:
    @pytest.mark.parametrize(
        "model",
        [
            "ollama:llama3",
            "ollama:gemma4:e4b-it-q4_K_M",
            "passthrough:test:anything",
            "OLLAMA:LLAMA3",  # case insensitive
        ],
    )
    def test_local_providers_return_false(self, model: str) -> None:
        assert is_cloud_model(model) is False

    @pytest.mark.parametrize(
        "model",
        [
            "openai:gpt-5.4-nano",
            "anthropic:claude-haiku-4-5",
            "bedrock:claude-haiku-4-5",
            "gemini:gemini-2.0-flash",
            "mistral:mistral-small",
            "groq:llama-3.1-70b",
            "cohere:command-r",
            "ANTHROPIC:claude-haiku-4-5",  # case insensitive
        ],
    )
    def test_known_cloud_providers_return_true(self, model: str) -> None:
        assert is_cloud_model(model) is True

    def test_none_returns_false(self) -> None:
        assert is_cloud_model(None) is False

    def test_empty_string_returns_false(self) -> None:
        assert is_cloud_model("") is False

    def test_unknown_provider_treated_as_cloud(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = is_cloud_model("brand-new-cloud-thing:foo")
        captured = capsys.readouterr()
        assert result is True
        assert "unknown model provider" in captured.err
        assert "brand-new-cloud-thing" in captured.err


class TestProviderName:
    def test_simple(self) -> None:
        assert provider_name("openai:gpt-5.4-nano") == "openai"

    def test_no_colon(self) -> None:
        assert provider_name("just-a-name") == "just-a-name"

    def test_multi_colon(self) -> None:
        assert provider_name("ollama:gemma4:e4b-it-q4_K_M") == "ollama"
