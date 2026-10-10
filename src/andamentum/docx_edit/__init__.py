"""Tracked-change editing of a .docx: the one implementation shared by whetstone and compose.

This package is a leaf. It imports no other andamentum sub-module, apart from
``core.text_match`` (a shared utility). Reading the body and its paragraphs,
turning paragraphs into a character stream, cutting runs and writing tracked
insertions and deletions all live here.
"""

from .atoms import (
    EXCLUDED_SUBTREES,
    Atom,
    AtomKind,
    document_atoms,
    paragraph_atoms,
    projection,
    qn,
)
from .body import body_paragraphs, main_document_part
from .surgery import (
    Edit,
    EditReport,
    apply_edits,
    next_change_id,
    run_bounds,
    split_at_boundaries,
    tracked_paragraph,
    translate_span,
)
from .textmap import AtomSpan, TextMap
from .xml_builder import XMLElementBuilder

__all__ = [
    "EXCLUDED_SUBTREES",
    "Atom",
    "AtomKind",
    "AtomSpan",
    "Edit",
    "EditReport",
    "TextMap",
    "XMLElementBuilder",
    "apply_edits",
    "body_paragraphs",
    "document_atoms",
    "main_document_part",
    "next_change_id",
    "paragraph_atoms",
    "projection",
    "qn",
    "run_bounds",
    "split_at_boundaries",
    "tracked_paragraph",
    "translate_span",
]
