"""DOCX manipulation for whetstone — track changes, XML patching, attribution.

Framework-agnostic code using python-docx + lxml.
"""

from .attribution import ChangeAttributionTracker, TokenAttribution
from .exceptions import FinalizationError
from .finalization import finalize_reviewed_document
from .low_level import DocxEditor, ParagraphContext, ParagraphData
from .patch_editor import PatchDocxEditor
from .result_types import (
    LocationResult,
    LocationStatus,
    ProcessingResult,
    ValidationResult,
    ValidationStatus,
    combine_validation_results,
)
from .text_processor import FuzzyMatchResult, SimilarityResult, TextProcessor
from .token_processor import TokenData, TokenIterator, TokenProcessor
from .validator import PatchValidator
from .xml_builder import XMLElementBuilder, XMLPatternMatcher

__all__ = [
    "ChangeAttributionTracker",
    "TokenAttribution",
    "FinalizationError",
    "finalize_reviewed_document",
    "DocxEditor",
    "ParagraphContext",
    "ParagraphData",
    "PatchDocxEditor",
    "LocationResult",
    "LocationStatus",
    "ProcessingResult",
    "ValidationResult",
    "ValidationStatus",
    "combine_validation_results",
    "FuzzyMatchResult",
    "SimilarityResult",
    "TextProcessor",
    "TokenData",
    "TokenIterator",
    "TokenProcessor",
    "PatchValidator",
    "XMLElementBuilder",
    "XMLPatternMatcher",
]
