"""Revise a draft until a stated condition holds.

Review -> edit -> re-review over whetstone's own machinery, until the
condition holds for two rounds running, no round changes anything, or a round
cap is hit. Every check is built from data whetstone already produces, or a
small, narrowly-scoped deterministic function — see each submodule's
docstring.
"""

from .figures import (
    FigureCheckResult,
    FigureTextMatch,
    FigureTextReport,
    check_figure_text_match,
    locate_figure_mentions,
)
from .loop import RevisionResult, RoundRecord, StoppedReason, revise_document
from .references import ReferenceCheckReport, check_references_clean
from .spec import FIXED_CONDITION_TEXT, FigureRef, RevisionCheck, RevisionSpec
from .structure import StructureReport, structure_clean
from .verdict import RoundVerdict

__all__ = [
    "FIXED_CONDITION_TEXT",
    "FigureCheckResult",
    "FigureRef",
    "FigureTextMatch",
    "FigureTextReport",
    "ReferenceCheckReport",
    "RevisionCheck",
    "RevisionResult",
    "RevisionSpec",
    "RoundRecord",
    "RoundVerdict",
    "StoppedReason",
    "StructureReport",
    "check_figure_text_match",
    "check_references_clean",
    "locate_figure_mentions",
    "revise_document",
    "structure_clean",
]
