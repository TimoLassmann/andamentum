"""Secrets for the command-line tools: API keys from a ``.env`` file.

Secrets (API keys, a contact mail) come from the environment; configuration
never does, it comes from flags. A command calls :func:`load_secrets` once at
start, and only a command or a script does: library code reads
``os.environ`` where a key is needed, never a ``.env`` file.

The ``.env`` file is looked for in the working folder and then in each folder
above it, so a key kept beside a project is found wherever andamentum is
installed. A variable already set in the environment wins over the file.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

from dotenv import find_dotenv, load_dotenv


def load_secrets() -> str | None:
    """Load the nearest ``.env`` at or above the working folder.

    Returns the file's path, or None when there is none. Variables already
    set are kept.
    """
    path = find_dotenv(usecwd=True)
    if not path:
        return None
    load_dotenv(path, override=False)
    return path


def key_status(names: Iterable[str]) -> str:
    """One line naming which of *names* are set, never their values."""
    return "keys: " + ", ".join(
        f"{name} {'set' if os.environ.get(name) else 'not set'}" for name in names
    )
