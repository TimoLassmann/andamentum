"""Central database lifecycle management for DocumentStore.

This module is the SINGLE SOURCE OF TRUTH for:
- Database directory structure
- Database path construction
- Database metadata management
- Database cleanup logic

Architecture: Named Database System
- Permanent databases: ~/.local/share/document-store/{name}.db
- Ephemeral databases: ~/.local/share/document-store/.ephemeral/{name}.db

Override with DOCUMENT_STORE_DIR env var to write databases elsewhere.

Permanent databases are user-curated and persist indefinitely.
Ephemeral databases are auto-generated (ask_*, test_*) and can be auto-cleaned.
"""

import logging
import os
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Patterns that indicate ephemeral databases
EPHEMERAL_PREFIXES = ("ask_", "test_", "varfolders", "tmp")


def get_databases_dir() -> Path:
    """Get the permanent databases directory path.

    Override with DOCUMENT_STORE_DIR env var to write databases elsewhere.
    Falls back to ANDAMENTUM_DATABASES_DIR for backward compatibility.
    Default: ~/.local/share/document-store/

    Returns:
        Path to databases directory
    """
    override = os.environ.get("DOCUMENT_STORE_DIR") or os.environ.get("ANDAMENTUM_DATABASES_DIR")
    if override:
        return Path(override)
    return Path.home() / ".local" / "share" / "document-store"


def get_ephemeral_dir() -> Path:
    """Get the ephemeral databases directory path.

    Ephemeral databases are auto-generated and can be auto-cleaned.

    Returns:
        Path to databases dir / .ephemeral/
    """
    return get_databases_dir() / ".ephemeral"


def is_ephemeral_name(database_name: str) -> bool:
    """Check if a database name suggests it should be ephemeral.

    Args:
        database_name: Name of the database

    Returns:
        True if name matches ephemeral patterns (ask_*, test_*, var*, tmp*)
    """
    if not database_name:
        return False
    name_lower = database_name.lower()
    return any(name_lower.startswith(prefix) for prefix in EPHEMERAL_PREFIXES)


def _validate_database_name(database_name: str) -> str:
    """Validate and sanitize a database name.

    Args:
        database_name: Raw database name

    Returns:
        Sanitized database name

    Raises:
        ValueError: If name is empty, contains path separators, or is invalid
    """
    if not database_name or not database_name.strip():
        raise ValueError("database_name cannot be empty")

    # Reject path-like names (common bug: passing full path as name)
    if "/" in database_name or "\\" in database_name:
        raise ValueError(
            f"database_name cannot contain path separators. "
            f"Got '{database_name}'. Use just the name, not the full path."
        )

    # Sanitize name (allow alphanumeric, dash, underscore)
    sanitized = "".join(c for c in database_name if c.isalnum() or c in "-_")
    if not sanitized:
        raise ValueError(f"Invalid database name: {database_name}")

    # Warn if sanitized name is very different (suggests path was passed)
    if len(sanitized) > 50 and len(sanitized) > len(database_name) * 0.8:
        logger.warning(
            f"Database name '{database_name[:50]}...' is unusually long. "
            "Make sure you're passing a name, not a path."
        )

    return sanitized


def get_db_path(database_name: str, ephemeral: bool = False) -> Path:
    """Get database path for a named database.

    This is the ONLY place that knows the directory structure.
    All other code must call this function for paths.

    Args:
        database_name: Name of the database (e.g., "research", "project-x")
        ephemeral: If True, use ephemeral directory (default: False)

    Returns:
        Path to database file

    Raises:
        ValueError: If database_name is empty or invalid
    """
    sanitized = _validate_database_name(database_name)

    if ephemeral:
        return get_ephemeral_dir() / f"{sanitized}.db"
    return get_databases_dir() / f"{sanitized}.db"


def list_databases() -> list[str]:
    """List all available databases.

    Returns:
        List of database names (without .db extension)
    """
    db_dir = get_databases_dir()
    if not db_dir.exists():
        return []

    return [f.stem for f in db_dir.glob("*.db")]


def database_exists(database_name: str) -> bool:
    """Check if a database exists.

    Args:
        database_name: Name of the database

    Returns:
        True if database exists
    """
    return get_db_path(database_name).exists()


def init_database_metadata(db_path: str, database_name: str) -> None:
    """Initialize database metadata table - makes database self-describing.

    Creates a _database_metadata table that stores:
    - database_name: Name of this database
    - created_at: timestamp

    This allows the database to know its identity without external context.

    Args:
        db_path: Path to database file
        database_name: Name of this database
    """
    conn = sqlite3.connect(db_path)
    try:
        # Create metadata table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _database_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        # Store metadata
        created_at = datetime.now(timezone.utc).isoformat()

        metadata = [
            ("database_name", database_name),
            ("created_at", created_at),
        ]

        for key, value in metadata:
            conn.execute(
                "INSERT OR REPLACE INTO _database_metadata (key, value) VALUES (?, ?)",
                (key, value)
            )

        conn.commit()
    finally:
        conn.close()


def get_database_metadata(db_path: str) -> dict:
    """Read metadata from database - database tells us its name.

    Args:
        db_path: Path to database file

    Returns:
        Dictionary with metadata:
        - database_name: Name of this database
        - created_at: ISO format timestamp

    Raises:
        FileNotFoundError: If database doesn't exist
        ValueError: If metadata table missing or corrupted
    """
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        # Check if metadata table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='_database_metadata'"
        )
        if not cursor.fetchone():
            raise ValueError(f"Database missing metadata table: {db_path}")

        # Read all metadata
        cursor = conn.execute("SELECT key, value FROM _database_metadata")
        metadata = dict(cursor.fetchall())

        # Validate required fields
        required = ["database_name", "created_at"]
        missing = [k for k in required if k not in metadata]
        if missing:
            raise ValueError(f"Database metadata missing fields: {missing}")

        return metadata
    finally:
        conn.close()


def delete_database(database_name: str, ephemeral: bool = False) -> bool:
    """Delete a named database.

    Args:
        database_name: Name of the database to delete
        ephemeral: If True, look in ephemeral directory

    Returns:
        True if deleted, False if not found
    """
    db_path = get_db_path(database_name, ephemeral=ephemeral)
    if not db_path.exists():
        return False

    # Also delete the raw files directory for this database
    raw_dir = db_path.parent / f"{database_name}_raw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)

    db_path.unlink()
    return True


def list_ephemeral_databases() -> list[str]:
    """List all ephemeral databases.

    Returns:
        List of ephemeral database names (without .db extension)
    """
    eph_dir = get_ephemeral_dir()
    if not eph_dir.exists():
        return []

    return [f.stem for f in eph_dir.glob("*.db")]


def cleanup_ephemeral_databases(max_age_days: int = 7) -> list[str]:
    """Clean up old ephemeral databases.

    Args:
        max_age_days: Delete databases older than this many days

    Returns:
        List of deleted database names
    """
    eph_dir = get_ephemeral_dir()
    if not eph_dir.exists():
        return []

    deleted = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

    for db_file in eph_dir.glob("*.db"):
        try:
            # Check file modification time
            mtime = datetime.fromtimestamp(db_file.stat().st_mtime, timezone.utc)
            if mtime < cutoff:
                name = db_file.stem
                delete_database(name, ephemeral=True)
                deleted.append(name)
                logger.info(f"Cleaned up old ephemeral database: {name}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {db_file}: {e}")

    return deleted


def migrate_to_ephemeral() -> dict[str, list[str]]:
    """Move databases with ephemeral-pattern names to ephemeral directory.

    This is a one-time migration to clean up existing databases.

    Returns:
        Dictionary with 'moved' and 'failed' lists
    """
    db_dir = get_databases_dir()
    eph_dir = get_ephemeral_dir()

    # Ensure ephemeral directory exists
    eph_dir.mkdir(parents=True, exist_ok=True)

    moved = []
    failed = []

    for db_file in db_dir.glob("*.db"):
        name = db_file.stem
        if is_ephemeral_name(name):
            try:
                dest = eph_dir / db_file.name
                shutil.move(str(db_file), str(dest))

                # Also move raw files directory if it exists
                raw_dir = db_dir / f"{name}_raw"
                if raw_dir.exists():
                    shutil.move(str(raw_dir), str(eph_dir / f"{name}_raw"))

                moved.append(name)
                logger.info(f"Migrated to ephemeral: {name}")
            except Exception as e:
                failed.append(name)
                logger.error(f"Failed to migrate {name}: {e}")

    return {"moved": moved, "failed": failed}
