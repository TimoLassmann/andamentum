"""Whetstone's tests never reach Crossref.

The default review runs the reference check with its Crossref lookup on, so a
test that runs the whole command line over a document with a reference list
would otherwise send that list to a public service on every run — slow, flaky
and outward-facing. A test that wants the lookup replaces this stub itself.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_crossref(monkeypatch: pytest.MonkeyPatch) -> None:
    async def refuse(*_args, **_kwargs):
        raise AssertionError(
            "a whetstone test reached the Crossref lookup; pass --offline "
            "or stub andamentum.references.lookup_entries"
        )

    monkeypatch.setattr("andamentum.references.lookup_entries", refuse)
