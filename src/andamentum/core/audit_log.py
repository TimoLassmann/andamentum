"""Opt-in local audit log for cloud LLM calls.

The audit log is OFF by default. It writes nothing unless the user
explicitly sets ``ANDAMENTUM_AUDIT_LOG`` to a writable path of their
choice. There is no XDG fallback and no hidden default directory.

Use case: an institutional question comes up six months later
("did you send patient data to OpenAI on the morning of 2026-05-16?")
and the user wants their own paper trail to answer it.

Format (one line per call, append-only):

    2026-05-16T14:23:01Z whetstone --mode panel anthropic:claude-haiku-4-5 sha256:abc123 6234B

Field 1: ISO-8601 UTC timestamp
Field 2: CLI / caller name (e.g. ``whetstone``)
Field 3: operation tag (free text — e.g. ``--mode panel`` or ``embed``)
Field 4: model identifier
Field 5: ``sha256:<digest>`` of the content sent, or ``n/a``
Field 6: payload byte count + literal ``B``

Failures (path unwritable, disk full) emit a stderr warning and
return. They never block the calling operation.
"""

from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_ENV_VAR = "ANDAMENTUM_AUDIT_LOG"


def _resolve_log_path() -> Path | None:
    """Return the user-chosen audit-log path, or None if opt-in is off.

    Empty string and the literal ``"0"`` both count as off, so a user
    can disable a previously-set env var via ``ANDAMENTUM_AUDIT_LOG=0``.
    """
    raw = os.environ.get(_ENV_VAR)
    if raw is None or raw == "" or raw == "0":
        return None
    return Path(raw).expanduser()


def is_enabled() -> bool:
    """Cheap predicate callers can check before computing payload hashes."""
    return _resolve_log_path() is not None


def log_cloud_call(
    *,
    cli_name: str,
    operation: str,
    model: str,
    content: bytes | str | None = None,
    byte_count: int | None = None,
) -> None:
    """Append one audit entry. No-op when opt-in is off.

    ``content`` is hashed (SHA-256) and discarded — it is not stored.
    Pass ``byte_count`` explicitly if you have it already; otherwise it
    is derived from ``content`` if given, else recorded as ``0``.
    """
    path = _resolve_log_path()
    if path is None:
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if content is None:
        digest = "n/a"
        n_bytes = byte_count if byte_count is not None else 0
    else:
        data = content.encode("utf-8") if isinstance(content, str) else content
        digest = f"sha256:{hashlib.sha256(data).hexdigest()}"
        n_bytes = byte_count if byte_count is not None else len(data)

    op = operation.strip() or "-"
    line = f"{timestamp} {cli_name} {op} {model} {digest} {n_bytes}B\n"

    try:
        existed = path.exists()
        if not existed:
            # Create with restrictive perms BEFORE first write so an interrupted
            # creation can't leave a world-readable file behind.
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(mode=0o600, exist_ok=False)
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    except OSError as exc:
        print(
            f"andamentum: failed to write audit log at {path}: {exc}. "
            f"Continuing without logging.",
            file=sys.stderr,
        )
