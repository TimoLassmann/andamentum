"""The shared eligibility-verdict type every principle contract returns.

Plan §4.1 (``docs/epistemic/PRINCIPLE_FIDELITY_PLAN.md``): "``EligibilityVerdict``
is a small typed result (``eligible: bool``, ``reason: str``) — never a bare
boolean. A principle that declines to fire must say why, in a form the canary
test (§4.3) can print on failure." This mirrors the existing ``Invariant``
typedef in ``graph/base.py`` / ``graph/invariants.py``, which already returns
``None | str`` for exactly this reason: a routing/eligibility signal should
never be silent about why.

Lives in its own leaf module (rather than inside ``peirce.py`` or ``agm.py``)
because every principle module returns it AND ``adjudicate.py`` needs it to
type ``AdjudicationDecision.considered`` — defining it in either principle
module would make the other one import from it for no principled reason, and
``adjudicate.py`` importing the principle modules while also depending on one
of them for a shared type would be a needless coupling.

Architecture: Layer 1 (framework-agnostic, pure data).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EligibilityVerdict(BaseModel):
    """One principle's answer to "do you want this claim, and why".

    Never reduced to a bare ``bool``: ``reason`` is populated on both the
    ``eligible=True`` and ``eligible=False`` path, so a declined principle's
    rationale is always available — for the firing-rate canary
    (plan §4.3) and for adjudication history
    (``adjudicate.AdjudicationDecision.considered``).
    """

    eligible: bool = Field(
        description="Whether this principle claims eligibility for the claim."
    )
    reason: str = Field(
        min_length=1,
        description="Why — populated whether eligible is True or False, never empty.",
    )


__all__ = ["EligibilityVerdict"]
