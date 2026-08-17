"""Exceptions for document review system."""


class DocumentReviewError(Exception):
    """Base exception for document review errors."""

    pass


class FileNotFoundError(DocumentReviewError):
    """Document file not found."""

    pass


class ConversionError(DocumentReviewError):
    """Document conversion failed."""

    pass


class FinalizationError(DocumentReviewError):
    """Document finalization failed."""

    pass
