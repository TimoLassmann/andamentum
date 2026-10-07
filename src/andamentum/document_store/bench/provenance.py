"""Where a run or a built store came from: git state and the dependency lock."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

_HERE = Path(__file__).parent


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=_HERE, check=True
    ).stdout


def git_state() -> dict[str, str | bool | None]:
    """Commit, dirty flag, and the sha256 of the uncommitted diff (None when clean)."""
    dirty = bool(_git("status", "--porcelain").strip())
    return {
        "commit": _git("rev-parse", "HEAD").strip(),
        "dirty": dirty,
        "diff_sha256": (
            hashlib.sha256(_git("diff", "HEAD").encode()).hexdigest() if dirty else None
        ),
    }


def uv_lock_sha256() -> str | None:
    """sha256 of the repository's ``uv.lock``, or None when there is none."""
    root = Path(_git("rev-parse", "--show-toplevel").strip())
    lock = root / "uv.lock"
    return hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None
