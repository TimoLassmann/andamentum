"""Named epistemic databases: where they live on disk, and how to open one.

An epistemic database is one :class:`andamentum.document_store.Store` file,
``<directory>/<name>.db``. This module owns the directory layout that the
document store itself no longer knows about:

- Permanent databases: ``~/.local/share/document-store/{name}.db``
- Ephemeral databases (names starting ``ask_``, ``test_``, ``varfolders`` or
  ``tmp``): ``~/.local/share/document-store/.ephemeral/{name}.db``

Override the directory with the ``DOCUMENT_STORE_DIR`` env var (or the older
``ANDAMENTUM_DATABASES_DIR``). An explicit ``db_dir`` argument beats both and
is never treated as ephemeral.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from andamentum.document_store import Store

logger = logging.getLogger(__name__)

#: Name prefixes that put a database in the ephemeral directory.
EPHEMERAL_PREFIXES = ("ask_", "test_", "varfolders", "tmp")


def get_databases_dir() -> Path:
    """The permanent databases directory.

    ``DOCUMENT_STORE_DIR`` (or ``ANDAMENTUM_DATABASES_DIR``) overrides the
    default ``~/.local/share/document-store/``.
    """
    override = os.environ.get("DOCUMENT_STORE_DIR") or os.environ.get(
        "ANDAMENTUM_DATABASES_DIR"
    )
    if override:
        return Path(override)
    return Path.home() / ".local" / "share" / "document-store"


def get_ephemeral_dir() -> Path:
    """The ephemeral databases directory (``<databases dir>/.ephemeral``)."""
    return get_databases_dir() / ".ephemeral"


def is_ephemeral_name(database_name: str) -> bool:
    """True if the name matches an ephemeral pattern (ask_*, test_*, ...)."""
    if not database_name:
        return False
    return database_name.lower().startswith(EPHEMERAL_PREFIXES)


def _validate_database_name(database_name: str) -> str:
    """Return the sanitised database name; raise ValueError if unusable."""
    if not database_name or not database_name.strip():
        raise ValueError("database_name cannot be empty")

    # Reject path-like names (common bug: passing a full path as the name)
    if "/" in database_name or "\\" in database_name:
        raise ValueError(
            f"database_name cannot contain path separators. "
            f"Got '{database_name}'. Use just the name, not the full path."
        )

    sanitized = "".join(c for c in database_name if c.isalnum() or c in "-_")
    if not sanitized:
        raise ValueError(f"Invalid database name: {database_name}")

    if len(sanitized) > 50 and len(sanitized) > len(database_name) * 0.8:
        logger.warning(
            f"Database name '{database_name[:50]}...' is unusually long. "
            "Make sure you're passing a name, not a path."
        )

    return sanitized


def get_db_path(database_name: str, ephemeral: bool = False) -> Path:
    """The file for a named database in the default directory."""
    sanitized = _validate_database_name(database_name)
    if ephemeral:
        return get_ephemeral_dir() / f"{sanitized}.db"
    return get_databases_dir() / f"{sanitized}.db"


def resolve_db_path(name: str, db_dir: str | Path | None = None) -> Path:
    """The file :func:`open_store` opens for ``name``.

    With ``db_dir`` the file is ``db_dir/{name}.db``; without it, the
    default directory (ephemeral names go to the ephemeral directory).
    """
    if db_dir is not None:
        return Path(db_dir) / f"{name}.db"
    return get_db_path(name, ephemeral=is_ephemeral_name(name))


async def open_store(name: str, db_dir: str | Path | None = None) -> Store:
    """Open (creating if absent) the store for the named database.

    The caller closes it. A file written by the pre-``Store`` document store
    is not readable; opening one raises.
    """
    path = resolve_db_path(name, db_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    return await Store.open(path)


def delete_database(database_name: str, ephemeral: bool = False) -> bool:
    """Delete a named database in the default directory.

    Close any open :class:`Store` on it first. Returns False if not found.
    """
    return delete_store_files(get_db_path(database_name, ephemeral=ephemeral))


def delete_store_files(db_path: Path) -> bool:
    """Delete a database file, its WAL sidecars and its ``_raw`` directory.

    Close any open :class:`Store` on it first. Returns False if none of them
    existed.
    """
    found = False
    raw_dir = db_path.with_name(f"{db_path.stem}_raw")
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
        found = True
    # WAL mode leaves sidecar files beside the database
    for path in (db_path, *(db_path.with_name(db_path.name + s) for s in ("-wal", "-shm"))):
        if path.exists():
            path.unlink()
            found = True
    return found
