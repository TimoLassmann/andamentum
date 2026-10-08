"""Exceptions for document review system."""

from ..errors import WhetstoneError


class DocumentReviewError(WhetstoneError):
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
