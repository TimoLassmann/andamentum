"""These tests never reach the network.

Every test that needs an HTTP answer injects an ``httpx.MockTransport`` (or a
recorded fixture); a client built on the real transport would send a request to
a public service, so the real transports refuse here. A test that uses a mock
transport is unaffected. Tests that carry a live marker (they call a real
model or service on purpose) are left alone.
"""

from __future__ import annotations

import httpx
import pytest


_LIVE_MARKERS = ("compose_eval", "ollama", "cloud", "live_provider")


@pytest.fixture(autouse=True)
def _no_network(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    if any(request.node.get_closest_marker(name) for name in _LIVE_MARKERS):
        return

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(
            "a test reached the real network; inject an httpx.MockTransport "
            "or the recorded fixtures"
        )

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refuse)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)
