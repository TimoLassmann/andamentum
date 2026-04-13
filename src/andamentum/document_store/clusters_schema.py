"""Cluster table schema and initialization for DHP temporal clustering.

Creates the database tables needed for storing cluster state and audit trail.
The documents table already has a cluster_id column (added in Phase 1 schema).
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3


def init_cluster_tables(cursor: sqlite3.Cursor) -> None:
    """Initialize DHP clustering tables.

    Creates:
    - clusters: Stores cluster centroids, kernel parameters, and metadata
    - cluster_runs: Audit trail of re-clustering runs with full config

    The documents.cluster_id column and its index are created by
    documents_schema.init_documents_table() and are not duplicated here.
    """

    # Cluster definitions with learned parameters
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            centroid TEXT NOT NULL,
            decay_rate REAL NOT NULL,
            kernel_params TEXT NOT NULL,
            doc_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            last_active_at TEXT NOT NULL
        )
    """)

    # Audit trail for re-clustering runs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cluster_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config TEXT NOT NULL,
            doc_count INTEGER NOT NULL,
            cluster_count INTEGER NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            duration_seconds REAL
        )
    """)
