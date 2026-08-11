"""The probe vocabulary — the contract every other checklist module builds on.

A :class:`Probe` names ONE defect class precisely enough that finding an
instance is *extraction* rather than *judgment*. That distinction is the whole
reason this mode exists: asked "is the evaluation adequate?", a small model
answers "yes, broadly" and produces nothing; asked "find the 3 claims the
reported results support least well", the same model produces three anchored
findings.

Two shapes, because a document can fail in two directions:

``presence``
    The defect is *in* the text. The probe returns a verbatim quote, and
    :mod:`.verify` drops any finding whose quote is not in the source. This is
    the hallucination gate the review pipeline already relies on.

``absence``
    The defect is something *missing*. There is no quote to anchor, so the
    probe instead names the missing item and lists the search terms that would
    appear if it were present. :mod:`.verify` searches the source for those
    terms and drops the finding when any of them turns up — the mirror image of
    quote anchoring, and the reason absence findings can be trusted at all.

The response *format* is fixed machinery; the probe *list* is data. Swapping
banks (see :mod:`.bank`) changes what gets checked without touching any code
here, which is what keeps the mode usable beyond academic drafts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ..model import Severity, Span

ProbeShape = Literal["presence", "absence"]


class Probe(BaseModel):
    """One defect class, phrased so a small model can extract instances of it.

    Attributes
    ----------
    id:
        Stable snake_case identifier (``"unmotivated_choice"``). It is the
        join key for corroboration, the coverage table row label, and the
        ``category`` on the rendered finding — so renaming one is a breaking
        change to saved output, not a cosmetic edit.
    title:
        Short human label for the coverage table ("Unmotivated choices").
    instruction:
        The defect definition, written as an imperative to the model. Must
        define the defect well enough that two readers would agree on what
        counts; a probe that needs the model to invent its own bar is not a
        probe.
    shape:
        ``presence`` or ``absence`` — selects the output schema and the
        verification route. See the module docstring.
    quota:
        ``3`` asks for "the 3 most serious", which turns the task into a
        ranking with a fixed output size — the point of the mode. ``None``
        asks for every instance and explicitly permits an empty list.

        Use a quota only where the defect is near-universal (every real draft
        contains unmotivated choices, so ranking them is informative). Use
        ``None`` where the defect is genuinely often absent — forcing a quota
        there manufactures findings, which is the failure mode this design is
        most exposed to.
    tags:
        Optional closed label set. When non-empty the model is asked to pick
        one, and :func:`normalise_tag` discards anything outside the set. Tags
        are an opt-in precision layer for domains whose failure modes you
        actually know; a probe with no tags works everywhere.
    """

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    title: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    shape: ProbeShape = "presence"
    quota: int | None = Field(default=3, ge=1)
    tags: tuple[str, ...] = ()

    model_config = {"frozen": True}

    @property
    def allows_empty(self) -> bool:
        """True when the probe may legitimately return nothing.

        Derived from ``quota`` rather than stored separately: an unquota'd
        probe is exactly one that reports what it finds, including nothing.
        Keeping it derived means the two cannot drift apart.
        """
        return self.quota is None


class ProbeFinding(BaseModel):
    """One verified defect instance, before conversion to a public ``Finding``.

    Deliberately lean and internal. The public
    :class:`andamentum.whetstone.schemas.Finding` is the renderer-facing type;
    this one carries only what the checklist pipeline needs to verify,
    corroborate and tabulate.
    """

    probe_id: str
    issue: str
    severity: Severity = "moderate"
    tag: str = ""

    # presence findings only — the verbatim span the defect sits in.
    quote: str = ""

    # absence findings only — what is missing, and the terms whose absence
    # from the source is the evidence for it.
    missing_item: str = ""
    search_terms: tuple[str, ...] = ()

    # Filled by .verify — the located span (presence findings only).
    span: Span | None = None

    # Filled by .corroborate — ids of OTHER probes that landed on the same
    # passage. A passage more than one check reached is worth reading first, so
    # it is recorded rather than merged away. It is NOT a claim that the checks
    # agree: they have different instructions, and telling "same defect" from
    # "two defects, one sentence" needs their prose compared — a model call this
    # mode deliberately does not make.
    corroborated_by: tuple[str, ...] = ()

    @property
    def shape(self) -> ProbeShape:
        """Which verification route produced (or must produce) this finding."""
        return "absence" if self.missing_item else "presence"

    @model_validator(mode="after")
    def _check_shape_invariant(self) -> ProbeFinding:
        """A finding must be exactly one shape, fully populated.

        Enforced here rather than at the call site because both shapes flow
        through the same list: an under-populated finding would otherwise fail
        far downstream, in verification or rendering, where the cause is no
        longer visible.
        """
        if self.quote and self.missing_item:
            raise ValueError(
                f"ProbeFinding[{self.probe_id}]: has both a quote and a "
                "missing_item. A finding is either presence-shaped (quote) or "
                "absence-shaped (missing_item), never both."
            )
        if not self.quote and not self.missing_item:
            raise ValueError(
                f"ProbeFinding[{self.probe_id}]: has neither a quote nor a "
                "missing_item. Presence findings need a verbatim quote to "
                "anchor; absence findings need the missing item to search for."
            )
        if self.missing_item and not self.search_terms:
            raise ValueError(
                f"ProbeFinding[{self.probe_id}]: absence finding has no "
                "search_terms. Without them the claim that something is "
                "missing cannot be checked, and an unverifiable absence "
                "finding is indistinguishable from a guess."
            )
        return self


def normalise_tag(raw: str, allowed: tuple[str, ...]) -> str:
    """Coerce a model-emitted tag to the probe's closed set, or drop it.

    Returns the canonical tag when *raw* matches one ignoring case and
    SURROUNDING whitespace, and ``""`` otherwise — including when *allowed*
    is empty, since an untagged probe has nothing to match against.

    Internal whitespace is significant, and deliberately so: the allowed tags
    are snake_case identifiers, so a model that answers ``"no result"`` for
    ``no_result`` is dropped like any other miss rather than repaired. That
    keeps the drop rate readable as what it is — a measure of whether the
    model can follow the closed set — instead of hiding near-misses inside it.

    Dropping rather than repairing is deliberate. The tag is a label, not a
    control signal, so a wrong one costs nothing downstream; and the rate at
    which tags get dropped is a direct read on whether a given model can run
    the bank at all, which is only measurable if bad tags are counted rather
    than silently fixed up.
    """
    if not allowed:
        return ""
    needle = raw.strip().casefold()
    return next((tag for tag in allowed if tag.casefold() == needle), "")
