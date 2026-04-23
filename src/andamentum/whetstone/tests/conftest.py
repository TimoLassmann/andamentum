"""Shared pytest fixtures for whetstone tests."""

import pytest


@pytest.fixture
def sample_text() -> str:
    """A short paragraph used by multiple renderer tests."""
    return (
        "The experiment show a significant effect. "
        "The data was collected over three weeks."
    )
