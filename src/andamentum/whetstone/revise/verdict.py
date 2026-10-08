"""The per-round verdict — a boolean and why, nothing else.

Structurally the same shape as ``epistemic.demand.Demand`` and
``epistemic.gates.GateResult`` elsewhere in this repo: a check reports
whether it is satisfied and the concrete reason, so the next round — or the
author, if the loop stops short — knows what is still missing. Whetstone
must not depend on ``epistemic`` (the repo's own layering rule), so this is
whetstone's own copy of the same idea, not a shared import.
"""

from __future__ import annotations

from pydantic import BaseModel


class RoundVerdict(BaseModel):
    satisfied: bool
    reason: str
    progress_score: float | None = None
    """Informational only (e.g. ``judge_score``'s ``expected_score``, or a
    mismatch ratio). Never triggers a stop by itself — see
    ``revise.loop.revise_document`` for what actually stops the loop."""
