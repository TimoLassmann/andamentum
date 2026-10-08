"""Opening and deleting an epistemic database file.

An epistemic database is one :class:`andamentum.document_store.Store` file at
a path the caller names. There is no default location: the command writes it
into its ``--out`` folder (or a temporary folder it removes afterwards), and
Python callers pass the path.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from andamentum.document_store import Store

#: The database file name inside a run folder (``--out DIR`` -> ``DIR/epistemic.db``).
DATABASE_FILE = "epistemic.db"


def database_path(folder: str | Path) -> Path:
    """The database file inside a run folder."""
    return Path(folder) / DATABASE_FILE


async def open_store(database: str | Path) -> Store:
    """Open (creating if absent) the store at ``database``.

    The caller closes it. A file written by the pre-``Store`` document store
    is not readable; opening one raises.
    """
    path = Path(database)
    path.parent.mkdir(parents=True, exist_ok=True)
    return await Store.open(path)


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
