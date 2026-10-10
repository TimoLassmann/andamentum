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
from .finalization import finalize_annotated_document
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
from .low_level import DocxEditor, ParagraphData, PlacedComment
from .surgery import (
    Edit,
    EditReport,
    apply_edits,
    run_bounds,
    split_at_boundaries,
    split_run_at,
)
from .textmap import AtomSpan, TextMap

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
    "run_bounds",
    "split_at_boundaries",
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
    "DocxEditor",
    "ParagraphData",
]
