"""Pattern-based scheduler (DEPRECATED).

This module is kept only for backward-compatible re-export of WorkItem.
The pattern scheduler has been replaced by the pydantic-graph DAG in
``andamentum.epistemic.graph``.
"""

# WorkItem lives in operations.base — re-export for backward compatibility
from .operations.base import WorkItem

__all__ = ["WorkItem"]
