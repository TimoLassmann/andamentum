"""The base error of ``andamentum.core``.

Every specific error in core derives from ``CoreError``: the fetch gate
(``FetchGateError``, ``RobotsBlocked``, ``PaywallBlocked``), URL safety
(``SsrfBlocked``, ``ResponseTooLarge``) and the decision models
(``DecisionError`` and its subclasses).
"""

from __future__ import annotations


class CoreError(Exception):
    """Any error raised by ``andamentum.core``."""
