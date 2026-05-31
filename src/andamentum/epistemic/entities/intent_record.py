"""Typed record for one prior investigation intent.

Each round, the gap-analysis agent (``epistemic_investigate_claim``)
proposes one or more **intents** — natural-language descriptions of
evidence-search angles to try. Each intent is dispatched through the
routing layer; the number of ``GatheredEvidence`` items returned is
recorded here.

The yield signal is **reachability**, not quality: it tells the next
round's gap-analysis agent whether this kind of intent connected to
any indexed evidence at all. A 0-yield intent means the angle was a
dead end — not that the underlying claim is false (Quine-Duhem), but
that the routing didn't find papers on this particular framing.

The agent uses this as a Lakatos-style signal: response to a dead-end
intent must be *substantively different* (changing method, population,
temporal frame, control comparison, or level of analysis), not a
lexical permutation of the same angle.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IntentRecord(BaseModel):
    """One prior investigation intent and the reachability signal it
    produced.

    ``text`` is the intent string the agent generated, verbatim. The
    next round's prompt sees this list and is expected to propose
    intents whose angle is substantively different from anything here.

    ``evidence_count`` is the number of Evidence entities the routing
    layer persisted for this intent. **A 0 means the routing found
    nothing reachable** — every provider either abstained on the intent
    or returned empty. The next round's agent should not propose
    variants of a 0-yield intent.
    """

    text: str = Field(description="The intent string as the agent generated it.")
    evidence_count: int = Field(
        default=0,
        description=(
            "Number of Evidence entities the routing layer persisted "
            "for this intent. 0 means the intent was unreachable "
            "through the current provider catalogue."
        ),
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IntentRecord":
        """Reconstruct from a dict (legacy/metadata storage path)."""
        return cls(
            text=str(data.get("text", "")),
            evidence_count=int(data.get("evidence_count", 0)),
        )
