"""File hashing utilities for deduplication.

Provides efficient hash-based deduplication for document processing.
Calculates content hashes to detect unchanged files before expensive operations.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of file content.

    Uses chunked reading for memory efficiency with large files.

    Args:
        file_path: Path to file

    Returns:
        Hex digest of SHA256 hash

    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file can't be read
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_file_metadata(file_path: Path) -> dict[str, Any]:
    """Get file metadata for deduplication.

    Args:
        file_path: Path to file

    Returns:
        Dict with:
        - file_hash: SHA256 hash of content
        - file_size: Size in bytes
        - file_mtime: Modification timestamp

    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file can't be accessed
    """
    stat = file_path.stat()
    return {
        "file_hash": calculate_file_hash(file_path),
        "file_size": stat.st_size,
        "file_mtime": stat.st_mtime,
    }
