"""DOCX manipulation for whetstone — track changes and comment placement.

Framework-agnostic code using python-docx + lxml.

The document has one text representation: the atom stream in :mod:`.atoms`.
:mod:`.textmap` indexes it, and :mod:`.surgery` edits it by cutting only the
runs a resolved span covers.
"""

from .annotate import (
    AnnotationReport,
    CoordinateError,
    Note,
    Rewrite,
    apply_annotations,
)
from .atoms import Atom, AtomKind, document_atoms, paragraph_atoms, projection
from .exceptions import FinalizationError
from .finalization import finalize_annotated_document, finalize_reviewed_document
from .headings import (
    DocxHeading,
    ParagraphSpan,
    StyleLevels,
    document_headings,
    heading_level,
    paragraph_spans,
)
from .reading import (
    DocxParts,
    auxiliary_text,
    body_paragraphs,
    main_document_part,
    read_docx,
)
from .low_level import DocxEditor, ParagraphContext, ParagraphData, PlacedComment
from .patch_editor import PatchDocxEditor
from .surgery import Edit, EditReport, apply_edits, isolate_span, split_run_at
from .textmap import AtomSpan, TextMap
from .result_types import (
    LocationResult,
    LocationStatus,
    ProcessingResult,
    ValidationResult,
    ValidationStatus,
    combine_validation_results,
)
from .token_processor import TokenData, TokenIterator, TokenProcessor
from .validator import PatchValidator
from .xml_builder import XMLElementBuilder, XMLPatternMatcher

__all__ = [
    "AnnotationReport",
    "CoordinateError",
    "Note",
    "Rewrite",
    "apply_annotations",
    "finalize_annotated_document",
    "auxiliary_text",
    "Atom",
    "AtomKind",
    "AtomSpan",
    "Edit",
    "EditReport",
    "TextMap",
    "apply_edits",
    "document_atoms",
    "isolate_span",
    "paragraph_atoms",
    "projection",
    "split_run_at",
    "DocxHeading",
    "DocxParts",
    "ParagraphSpan",
    "StyleLevels",
    "body_paragraphs",
    "document_headings",
    "heading_level",
    "main_document_part",
    "paragraph_spans",
    "read_docx",
    "PlacedComment",
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
    "TokenData",
    "TokenIterator",
    "TokenProcessor",
    "PatchValidator",
    "XMLElementBuilder",
    "XMLPatternMatcher",
]
