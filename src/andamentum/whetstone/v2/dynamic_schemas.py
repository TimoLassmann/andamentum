"""Runtime Pydantic schema construction for custom-criteria mode.

v1's ``dynamic_models.py`` accepted an open-ended dict-of-dicts spec and
produced a Pydantic model with arbitrary field shapes. v2 narrows the
contract: we know exactly what custom-criteria mode needs — one
status + one notes field per criterion, plus a single overall_assessment
field. That tighter contract removes the field-type plumbing and the
ad-hoc kwargs dispatch.

Public surface
--------------

* :func:`slugify_criterion` — deterministic, total criterion-name → snake_case
  identifier. Strips non-ascii, lowercases, replaces non-word characters
  with underscores. Raises ``ValueError`` with a clear message when a
  criterion produces an empty identifier (e.g. all-non-ascii input).

* :func:`create_custom_evaluation_model` — given a list of criterion
  strings, returns a Pydantic model class whose fields are
  ``<slug>_status``, ``<slug>_notes`` for each criterion, plus a single
  ``overall_assessment: str`` summarising the whole review.

The returned model is for the LLM call only. The orchestrator unpacks
its values into a flat ``list[CustomEvaluation]`` via
:func:`unpack_custom_evaluations` so downstream consumers (renderers,
test assertions) never need to know the runtime schema.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

from .schemas import CustomEvaluation

__all__ = [
    "MAX_CUSTOM_CRITERIA",
    "create_custom_evaluation_model",
    "slugify_criterion",
    "unpack_custom_evaluations",
]


# Hard cap — anything more bloats the prompt and small models stop
# filling 50+ flat fields reliably.
MAX_CUSTOM_CRITERIA = 30


def slugify_criterion(criterion: str) -> str:
    """Produce a deterministic snake_case identifier from a criterion string.

    Algorithm
    ---------
    1. Unicode-normalise (NFKD) and drop combining marks → strips
       accents but keeps ascii letters underneath.
    2. Lowercase.
    3. Replace any non-``[a-z0-9]`` run with a single underscore.
    4. Strip leading / trailing underscores.

    Raises
    ------
    ValueError
        If the input collapses to an empty string (e.g. it is purely
        whitespace or non-ascii symbols that have no ascii equivalent).
        We never silently fall back to a placeholder name — the caller
        passed a criterion the LLM cannot reliably reference, and we
        want the failure loud at the validation boundary.
    """
    if not isinstance(criterion, str):
        raise TypeError(f"criterion must be str, got {type(criterion).__name__}")

    # NFKD then drop combining marks → "café" → "cafe", "ürlich" → "urlich".
    normalised = unicodedata.normalize("NFKD", criterion)
    ascii_only = "".join(ch for ch in normalised if not unicodedata.combining(ch))
    lowered = ascii_only.lower()
    # Anything that isn't [a-z0-9] becomes a single underscore.
    slug = re.sub(r"[^a-z0-9]+", "_", lowered)
    slug = slug.strip("_")
    if not slug:
        raise ValueError(
            f"criterion {criterion!r} produces an empty slug; supply a "
            "criterion with at least one ascii letter or digit."
        )
    return slug


def create_custom_evaluation_model(criteria: list[str]) -> type[BaseModel]:
    """Build a Pydantic model for evaluating a list of custom criteria.

    For each criterion, two fields are created:
      * ``<slug>_status`` — ``Literal["pass", "fail", "unclear"]``
      * ``<slug>_notes`` — ``str`` (1-2 sentences justifying the verdict)

    Plus one always-present field:
      * ``overall_assessment`` — ``str`` (a 2-3 sentence holistic summary)

    Parameters
    ----------
    criteria:
        Raw criterion strings as supplied by the caller. Order is
        preserved — the resulting model has fields in input order.

    Raises
    ------
    ValueError
        If ``criteria`` is empty, exceeds :data:`MAX_CUSTOM_CRITERIA`,
        or contains any criterion whose slug collides with another.
        (Slug collisions are reported with both offending strings so
        the caller can see which two need disambiguation.)
    """
    if not criteria:
        raise ValueError("criteria must contain at least one criterion")
    if len(criteria) > MAX_CUSTOM_CRITERIA:
        raise ValueError(
            f"too many criteria ({len(criteria)}); limit is "
            f"{MAX_CUSTOM_CRITERIA}. Combine related criteria or split "
            "the review into multiple runs."
        )

    field_definitions: dict[str, Any] = {}
    seen: dict[str, str] = {}  # slug -> original criterion

    for criterion in criteria:
        criterion_clean = criterion.strip()
        if not criterion_clean:
            raise ValueError(
                "criteria must not contain empty / whitespace-only strings"
            )
        slug = slugify_criterion(criterion_clean)
        if slug in seen:
            raise ValueError(
                f"criterion slug collision: {seen[slug]!r} and "
                f"{criterion_clean!r} both slugify to {slug!r}. "
                "Disambiguate by rephrasing one of them."
            )
        seen[slug] = criterion_clean

        status_field_name = f"{slug}_status"
        notes_field_name = f"{slug}_notes"
        # Preserve the human criterion text in field descriptions so the
        # LLM sees the original wording even though the field name is
        # slugified.
        field_definitions[status_field_name] = (
            Literal["pass", "fail", "unclear"],
            Field(
                description=(
                    f"Verdict for criterion {criterion_clean!r}: "
                    "pass = clearly met, fail = clearly not met, "
                    "unclear = ambiguous or the criterion does not apply."
                ),
            ),
        )
        field_definitions[notes_field_name] = (
            str,
            Field(
                description=(
                    f"1-2 sentences explaining the verdict for criterion "
                    f"{criterion_clean!r}, grounded in the document."
                ),
            ),
        )

    field_definitions["overall_assessment"] = (
        str,
        Field(
            description=(
                "2-3 sentence holistic summary across all criteria. "
                "Should be consistent with the per-criterion verdicts."
            ),
        ),
    )

    return create_model("CustomReviewerOutput", **field_definitions)


def unpack_custom_evaluations(
    criteria: list[str], filled_model: BaseModel
) -> list[CustomEvaluation]:
    """Translate a filled runtime model into a flat list of CustomEvaluation.

    The criteria list must match the one used to build ``filled_model``
    (same strings, same order). Re-slugifying here keeps the unpacker
    self-contained — no need to thread the original slug map through.
    """
    out: list[CustomEvaluation] = []
    data = filled_model.model_dump()
    for criterion in criteria:
        slug = slugify_criterion(criterion.strip())
        out.append(
            CustomEvaluation(
                criterion=criterion.strip(),
                status=data[f"{slug}_status"],
                notes=str(data.get(f"{slug}_notes", "")).strip(),
            )
        )
    return out
