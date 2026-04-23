#!/usr/bin/env python3
"""
Common result types for standardized error handling and return values.

This module provides standardized result types that eliminate duplication
in error handling patterns across the editor components.
"""

from typing import Optional, List, Any
from dataclasses import dataclass
from enum import Enum


class ValidationStatus(Enum):
    """Status codes for validation operations."""

    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"


class LocationStatus(Enum):
    """Status codes for location operations."""

    FOUND = "found"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"


@dataclass
class ValidationResult:
    """
    Standardized result for validation operations.

    Replaces inconsistent bool/tuple/dict return patterns.
    """

    status: ValidationStatus
    message: str = ""
    details: Optional[dict] = None

    @property
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return self.status == ValidationStatus.VALID

    @property
    def is_invalid(self) -> bool:
        """Check if validation failed."""
        return self.status == ValidationStatus.INVALID

    @property
    def has_warning(self) -> bool:
        """Check if validation has warnings."""
        return self.status == ValidationStatus.WARNING

    @classmethod
    def valid(cls, message: str = "Validation passed") -> "ValidationResult":
        """Create a valid result."""
        return cls(ValidationStatus.VALID, message)

    @classmethod
    def invalid(cls, message: str, details: Optional[dict] = None) -> "ValidationResult":
        """Create an invalid result."""
        return cls(ValidationStatus.INVALID, message, details)

    @classmethod
    def warning(cls, message: str, details: Optional[dict] = None) -> "ValidationResult":
        """Create a warning result."""
        return cls(ValidationStatus.WARNING, message, details)


@dataclass
class LocationResult:
    """
    Standardized result for location/search operations.

    Consolidates various location finding patterns.
    """

    status: LocationStatus
    index: Optional[int] = None
    confidence: float = 0.0
    message: str = ""
    alternatives: Optional[List[int]] = None

    @property
    def found(self) -> bool:
        """Check if location was found."""
        return self.status == LocationStatus.FOUND

    @property
    def is_not_found(self) -> bool:
        """Check if location was not found."""
        return self.status == LocationStatus.NOT_FOUND

    @property
    def is_ambiguous(self) -> bool:
        """Check if multiple matches were found."""
        return self.status == LocationStatus.AMBIGUOUS

    @classmethod
    def found_at(cls, index: int, confidence: float = 1.0, message: str = "") -> "LocationResult":
        """Create a successful location result."""
        return cls(LocationStatus.FOUND, index, confidence, message)

    @classmethod
    def not_found(cls, message: str = "Location not found") -> "LocationResult":
        """Create a not found result."""
        return cls(LocationStatus.NOT_FOUND, None, 0.0, message)

    @classmethod
    def ambiguous(cls, alternatives: List[int], message: str = "Multiple matches found") -> "LocationResult":
        """Create an ambiguous result."""
        return cls(LocationStatus.AMBIGUOUS, None, 0.0, message, alternatives)


@dataclass
class ProcessingResult:
    """
    Standardized result for processing operations.

    Replaces various custom result types with consistent interface.
    """

    success: bool
    message: str = ""
    data: Optional[Any] = None
    error_code: Optional[str] = None
    processing_time: float = 0.0
    metadata: Optional[dict] = None

    @classmethod
    def succeeded(
        cls,
        message: str = "Processing completed",
        data: Optional[Any] = None,
        processing_time: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> "ProcessingResult":
        """Create a successful processing result."""
        return cls(True, message, data, None, processing_time, metadata)

    @classmethod
    def failure(
        cls,
        message: str,
        error_code: Optional[str] = None,
        processing_time: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> "ProcessingResult":
        """Create a failed processing result."""
        return cls(False, message, None, error_code, processing_time, metadata)

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the result."""
        if self.metadata is None:
            self.metadata = {}
        self.metadata[key] = value


# Helper functions for common patterns


def combine_validation_results(results: List[ValidationResult]) -> ValidationResult:
    """
    Combine multiple validation results into a single result.

    Args:
        results: List of validation results to combine

    Returns:
        Combined validation result
    """
    if not results:
        return ValidationResult.valid("No validations performed")

    # Check for any invalid results
    invalid_results = [r for r in results if r.is_invalid]
    if invalid_results:
        messages = [r.message for r in invalid_results]
        return ValidationResult.invalid(f"Validation failed: {'; '.join(messages)}")

    # Check for warnings
    warning_results = [r for r in results if r.has_warning]
    if warning_results:
        messages = [r.message for r in warning_results]
        return ValidationResult.warning(f"Validation warnings: {'; '.join(messages)}")

    # All valid
    return ValidationResult.valid(f"All {len(results)} validations passed")


def to_legacy_bool_tuple(result: ValidationResult) -> tuple[bool, str]:
    """
    Convert ValidationResult to legacy (bool, str) tuple format.

    For backward compatibility during migration.

    Args:
        result: ValidationResult to convert

    Returns:
        Tuple of (success, message)
    """
    return result.is_valid, result.message


def from_legacy_bool_tuple(success: bool, message: str) -> ValidationResult:
    """
    Convert legacy (bool, str) tuple to ValidationResult.

    For migrating existing code patterns.

    Args:
        success: Whether operation succeeded
        message: Result message

    Returns:
        ValidationResult object
    """
    if success:
        return ValidationResult.valid(message)
    else:
        return ValidationResult.invalid(message)
