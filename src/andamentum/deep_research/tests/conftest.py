"""Shared fixtures for deep-research package tests."""

import pytest

from .. import ResearchState


@pytest.fixture
def research_state() -> ResearchState:
    """Minimal ResearchState for testing."""
    return ResearchState(query="test query")
