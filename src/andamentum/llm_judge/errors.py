"""Errors raised by andamentum.llm_judge."""

from __future__ import annotations


class JudgeError(ValueError):
    """Base class for llm_judge errors.

    A ``ValueError`` subclass, so callers that catch ``ValueError`` keep working.
    """


class InvalidJudgeInputError(JudgeError):
    """The model list, a model id or the criteria are unusable (caught before any model call)."""
