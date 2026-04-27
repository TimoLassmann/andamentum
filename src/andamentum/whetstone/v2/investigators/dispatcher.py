"""Deterministic hypothesis classifier.

Inspects a hypothesis's text and returns its ``investigation_type``.
This runs AFTER skim_agent emits the hypotheses but BEFORE the
InvestigateLoop dispatches them — so we can classify based on what the
hypothesis is asking, without bothering the LLM with another decision.

For Phase 2 only the default ``"internal"`` type ships. Future
extensions add patterns here:

  • ``novelty``   — claims like "novel", "first to", "no prior"
  • ``factual``   — questions of fact better answered by a KB lookup
  • ``statistical`` — claims about numbers re-derivable from the document

Add a new type by adding a (compiled regex, type_name) entry to
``_PATTERNS``.
"""

from __future__ import annotations

import re

from ..schemas import Hypothesis

# (pattern, investigation_type). First match wins. Patterns checked in
# the order they appear here. Default type is "internal" if none match.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Reserved for future extension. Example:
    # (re.compile(r"\bnovel\b|\bfirst to\b|\bno prior\b", re.IGNORECASE), "novelty"),
]


def classify_hypothesis(hypothesis: Hypothesis) -> str:
    """Return the ``investigation_type`` for this hypothesis.

    Pure function — no LLM, no state. Always safe to call repeatedly.
    """
    text = hypothesis.text or ""
    for pattern, type_name in _PATTERNS:
        if pattern.search(text):
            return type_name
    return "internal"
