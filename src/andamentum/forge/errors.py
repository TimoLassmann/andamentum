"""The errors of ``andamentum.forge``.

One base class, ``ForgeError``; the specific errors derive from it.
"""

from __future__ import annotations


class ForgeError(Exception):
    """Any error forge raises on purpose."""


class BriefRefusedError(ForgeError, ValueError):
    """The brief is not a small function: an app, agent or service hands the loop to
    an outside driver, so forge refuses it and says how to reshape it."""


class DesignError(ForgeError, ValueError):
    """The design could not be completed: the model produced no valid output after
    retries, or a design stage came back empty."""
