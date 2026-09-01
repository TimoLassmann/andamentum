"""Principle contracts — plan §4.1-4.2 (``docs/epistemic/PRINCIPLE_FIDELITY_PLAN.md``).

The codebase's principle-fidelity defects (``paper/FINDINGS.md`` §9) shared
one root cause: no seam where a philosophical principle was a bounded,
ownable, testable unit, so conflicts between two principles were resolved by
whichever ``if`` got written first, not by a decision anyone made. This
package is that seam.

Each principle module (``peirce``, ``agm``, ...) exports exactly three
things: a pure ``eligible(claim, state) -> EligibilityVerdict``, the
operation(s) it dispatches (re-exported from wherever they already live —
nothing is relocated), and an ``ACTIVATION_NAME`` string naming the
operation-log entry the firing-rate canary counts. ``adjudicate()`` asks
every principle, in a fixed declared order, whether it wants a claim, and
records which one won and why — preserving the losing principle's reason
rather than letting it disappear into a buried conditional.

Stage 2a (plan §5) builds ``peirce`` and ``agm`` plus ``adjudicate`` to name
the one existing conflict — Peirce's cycle-cap termination vs. AGM's belief
contraction in ``graph/nodes.py:AbandonOrDemote`` — as data instead of an
inline ``if``. It changes nothing about which principle wins.
"""

from __future__ import annotations

from . import agm, peirce
from .adjudicate import AdjudicationDecision, adjudicate
from .eligibility import EligibilityVerdict

__all__ = [
    "EligibilityVerdict",
    "AdjudicationDecision",
    "adjudicate",
    "peirce",
    "agm",
]
