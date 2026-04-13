"""Validation types for epistemic constitution.

Core validation primitives used across all validation modules:
- ValidationSeverity: Error level enum
- ValidationError: Single validation issue
- ValidationResult: Collection of errors/warnings with helper methods

Architecture: Layer 1 (framework-agnostic, no model calls)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class ValidationSeverity(str, Enum):
    """Severity level of validation errors."""

    ERROR = "error"  # Blocks operation
    WARNING = "warning"  # Logged but allows operation
    INFO = "info"  # Informational only


@dataclass
class ValidationError:
    """A single validation error or warning."""

    code: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    field_path: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of validating an output or state transition.

    Soft Validation Model:
    - `valid`: True if no issues at all (errors or warnings)
    - `can_proceed`: True if no blocking ERRORs (warnings don't block)

    The orchestrator should use `can_proceed` to decide whether to continue,
    and log warnings for transparency.
    """

    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)

    @property
    def can_proceed(self) -> bool:
        """Can we proceed with this result? (No blocking errors)

        Unlike `valid`, this returns True if we only have warnings.
        Use this for soft validation where warnings don't block the pipeline.
        """
        return len(self.errors) == 0

    @property
    def has_warnings(self) -> bool:
        """Are there any warnings?"""
        return len(self.warnings) > 0

    @property
    def has_errors(self) -> bool:
        """Are there any blocking errors?"""
        return len(self.errors) > 0

    def add_error(self, code: str, message: str, **details: Any) -> None:
        """Add an error (blocks operation)."""
        self.valid = False
        self.errors.append(ValidationError(code=code, message=message, severity=ValidationSeverity.ERROR, details=details))

    def add_warning(self, code: str, message: str, **details: Any) -> None:
        """Add a warning (logged but doesn't block)."""
        self.valid = False  # Not fully valid, but can_proceed is still True
        self.warnings.append(
            ValidationError(code=code, message=message, severity=ValidationSeverity.WARNING, details=details)
        )

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge another validation result into this one."""
        self.valid = self.valid and other.valid
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        return self

    def get_error_messages(self) -> List[str]:
        """Get list of error messages for logging."""
        return [e.message for e in self.errors]

    def get_warning_messages(self) -> List[str]:
        """Get list of warning messages for logging."""
        return [w.message for w in self.warnings]
