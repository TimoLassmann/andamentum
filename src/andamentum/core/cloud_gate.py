"""Detect whether a model id points at an external (cloud) inference provider.

Used by CLIs to decide whether to print a warning or prompt before sending
document content to a third party. Pure function — no I/O.

The classification is conservative: unknown providers are treated as cloud
unless explicitly recognised as local. This is the safe default for the
responsible-release gate — false negatives (failing to warn about a cloud
provider) are worse than false positives.
"""

from __future__ import annotations

import sys

# Providers that always run locally on the user's machine.
_LOCAL_PREFIXES: frozenset[str] = frozenset(
    {
        "ollama:",
        "passthrough:test:",  # pydantic-ai's in-test fake; never hits the network
    }
)

# Providers known to send content over the network to a third party.
_CLOUD_PREFIXES: frozenset[str] = frozenset(
    {
        "openai:",
        "anthropic:",
        "claude:",  # alias some callers use
        "bedrock:",
        "gemini:",
        "google:",
        "mistral:",
        "groq:",
        "cohere:",
        "perplexity:",
        "deepseek:",
        "openrouter:",
        "fireworks:",
        "together:",
    }
)


def is_cloud_model(model: str | None) -> bool:
    """Return True if ``model`` resolves to an external inference provider.

    Unknown providers are treated as cloud — we'd rather over-warn than
    silently leak content. Pass ``None`` returns False (no model = no call).
    """
    if model is None or not model:
        return False
    lowered = model.lower()
    for prefix in _LOCAL_PREFIXES:
        if lowered.startswith(prefix):
            return False
    for prefix in _CLOUD_PREFIXES:
        if lowered.startswith(prefix):
            return True
    # Unknown provider — treat as cloud, note to stderr so the user can tell
    # us if classification needs updating.
    print(
        f"andamentum: unknown model provider '{model.split(':', 1)[0]}' — "
        f"treating as cloud (external inference). If this is wrong, please "
        f"file an issue.",
        file=sys.stderr,
    )
    return True


def provider_name(model: str) -> str:
    """Extract the provider prefix from a model id (e.g. 'openai:gpt-x' → 'openai')."""
    return model.split(":", 1)[0] if ":" in model else model
