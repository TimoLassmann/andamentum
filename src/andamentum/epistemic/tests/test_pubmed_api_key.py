"""How PubMedProvider gets its NCBI key.

The key is optional and only changes the rate limit (3 -> 10 req/s), so
nothing here needs a real one. What is pinned is the precedence and the
fact that the value never reaches a log.
"""

from __future__ import annotations

import logging

from ..providers.pubmed import PubMedProvider

_FAKE = "fake-key-not-a-real-credential"


def test_explicit_argument_wins_over_the_environment(monkeypatch):
    """The repo's usual precedence — compare core.models.resolve_model_from_args."""
    monkeypatch.setenv("NCBI_API_KEY", "from-environment")
    provider = PubMedProvider(ncbi_api_key=_FAKE)
    assert provider.api_key == _FAKE


def test_environment_is_the_fallback(monkeypatch):
    """This is what the .env workflow fills, so it has to keep working."""
    monkeypatch.setenv("NCBI_API_KEY", _FAKE)
    assert PubMedProvider().api_key == _FAKE


def test_no_key_anywhere_is_not_an_error(monkeypatch):
    """Without a key the provider still works, just at the lower rate limit."""
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    assert PubMedProvider().api_key is None


def test_the_key_is_keyword_only():
    """A bare second positional would be confusable with max_results."""
    import inspect

    parameter = inspect.signature(PubMedProvider).parameters["ncbi_api_key"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_the_key_value_is_never_logged(monkeypatch, caplog):
    """Whether a key was used is worth recording; the key itself is not."""
    monkeypatch.setenv("NCBI_API_KEY", _FAKE)
    logger = logging.getLogger("andamentum.epistemic.providers.pubmed")
    monkeypatch.setattr(logger, "propagate", True)
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        PubMedProvider()
    logged = " ".join(record.getMessage() for record in caplog.records)
    assert _FAKE not in logged
    assert "in use" in logged
