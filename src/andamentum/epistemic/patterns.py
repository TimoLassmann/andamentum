"""Pattern-based scheduler (DEPRECATED).

This module is kept only for backward-compatible re-export of OperationInput
(formerly WorkItem). The pattern scheduler has been replaced by the
pydantic-graph DAG in ``andamentum.epistemic.graph``.
"""

from .operations.base import OperationInput, OperationInput as WorkItem  # noqa: F401

__all__ = ["OperationInput", "WorkItem"]
