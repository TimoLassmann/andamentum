"""The one base error of ``deep_research``; the specific errors derive from it."""

from __future__ import annotations


class ResearchError(RuntimeError):
    """Any failure ``deep_research`` reports on purpose.

    A spent bound (calls, seconds, bytes) is not an error: the run stops and
    the result's ``stop_reason`` says so.
    """
