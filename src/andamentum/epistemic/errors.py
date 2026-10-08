"""The base error of ``andamentum.epistemic``.

Every specific error the package raises on purpose derives from
``EpistemicError``: ``DegenerateQuestionError`` (the question cannot be a
research question), ``EntityNotFoundError`` (an entity is not in the
database) and ``StageInvariantError`` (a stage ended without what it must
leave behind).
"""

from __future__ import annotations


class EpistemicError(Exception):
    """Any error raised by ``andamentum.epistemic`` on purpose."""
