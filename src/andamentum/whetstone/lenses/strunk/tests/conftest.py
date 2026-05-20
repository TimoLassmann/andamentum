"""Shared test fixtures for the Strunk lens.

The ``StubAgentExecutor`` is the test seam: tests inject one in place
of a real ``AgentRunner`` and the agent nodes call ``.run`` exactly
the same way they would in production. Each test supplies a
``responder`` callable that maps ``(agent_definition, kwargs)`` to a
canned verdict (or raises to simulate schema-validation failure).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest


class StubAgentExecutor:
    """Test double matching the ``AgentExecutor`` protocol from ``state.py``.

    Pass a ``responder`` that maps ``(definition, kwargs) -> verdict``
    or ``-> Exception`` (the latter is re-raised to simulate a real
    schema-validation failure). Every call is logged in ``.calls``.
    """

    def __init__(self, responder: Callable[[Any, dict[str, Any]], Any]):
        self.responder = responder
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def run(self, defn: Any, /, **kwargs: Any) -> Any:
        self.calls.append((defn.name, dict(kwargs)))
        result = self.responder(defn, kwargs)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def stub_executor():
    """Factory: ``executor = stub_executor(responder)``."""

    def _make(
        responder: Callable[[Any, dict[str, Any]], Any],
    ) -> StubAgentExecutor:
        return StubAgentExecutor(responder)

    return _make
