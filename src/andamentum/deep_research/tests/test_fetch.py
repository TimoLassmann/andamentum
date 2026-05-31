"""Tests for run_fetch — URL → structured summary one-shot."""

from __future__ import annotations

from unittest import mock

import pytest

from ..fetch import run_fetch
from ..models import FetchSummary


class _FakeAgent:
    def __init__(self, output: FetchSummary):
        self.output = output

    async def run(self, _prompt: str):
        captured = self.output

        class _R:
            output = captured

        return _R()


@pytest.mark.asyncio
async def test_run_fetch_composes_extract_and_summariser():
    """harvest.extract result is fed to the summariser; url/title get filled in."""
    agent_output = FetchSummary(
        url="",  # agent leaves blank — run_fetch fills it from the URL arg
        title="",  # also blank — run_fetch falls back to first heading
        summary="A 200-word summary",
        key_points=["point one", "point two"],
    )

    async def fake_extract(_url: str) -> str:
        return "# Big Important Title\n\nBody paragraph."

    fake_agent = _FakeAgent(agent_output)

    with (
        mock.patch("andamentum.harvest.extract", fake_extract),
        mock.patch(
            "andamentum.core.agents.build_pydantic_ai_agent",
            return_value=fake_agent,
        ),
    ):
        result = await run_fetch("https://example.com/page", model="fake:test")

    assert result.url == "https://example.com/page"
    # Title was empty in agent output, so first markdown heading wins.
    assert result.title == "Big Important Title"
    assert result.summary == "A 200-word summary"
    assert result.key_points == ["point one", "point two"]


@pytest.mark.asyncio
async def test_run_fetch_preserves_agent_title_when_present():
    """If the agent fills in a title, run_fetch keeps it."""
    agent_output = FetchSummary(
        url="ignored",
        title="Agent-supplied title",
        summary="s",
        key_points=["k"],
    )

    async def fake_extract(_url: str) -> str:
        return "# A different heading\n\nBody."

    with (
        mock.patch("andamentum.harvest.extract", fake_extract),
        mock.patch(
            "andamentum.core.agents.build_pydantic_ai_agent",
            return_value=_FakeAgent(agent_output),
        ),
    ):
        result = await run_fetch("https://example.com", model="fake:test")

    assert result.title == "Agent-supplied title"
    # url is always overridden with the authoritative input.
    assert result.url == "https://example.com"


@pytest.mark.asyncio
async def test_run_fetch_falls_back_to_url_when_no_title_anywhere():
    """No agent title, no markdown heading → url is used as title."""
    agent_output = FetchSummary(url="", title="", summary="s", key_points=["k"])

    async def fake_extract(_url: str) -> str:
        return "Just a paragraph, no heading."

    with (
        mock.patch("andamentum.harvest.extract", fake_extract),
        mock.patch(
            "andamentum.core.agents.build_pydantic_ai_agent",
            return_value=_FakeAgent(agent_output),
        ),
    ):
        result = await run_fetch("https://example.com/x", model="fake:test")

    assert result.title == "https://example.com/x"
