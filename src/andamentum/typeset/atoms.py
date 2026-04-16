"""Atom validation for the andamentum.typeset 7-atom document system.

Each atom is a plain dict with a ``kind`` key drawn from :data:`ATOM_KINDS`.
This module validates and normalises atom dicts before they reach the renderer.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

ATOM_KINDS: frozenset[str] = frozenset(
    {"heading", "prose", "callout", "items", "aside", "card", "reference"}
)

CALLOUT_TONES: frozenset[str] = frozenset(
    {"info", "warning", "success", "note", "quote"}
)

ITEMS_VARIANTS: frozenset[str] = frozenset({"pairs", "right", "left"})

# Kinds that require a specific field to be present.
_REQUIRED_FIELDS: dict[str, str | None] = {
    "heading": "content",
    "prose": "content",
    "callout": "content",
    "items": "entries",
    "aside": None,  # accepts content OR groups — nothing strictly required
    "card": "content",
    "reference": "content",
}


def validate_atom(atom: Mapping[str, object], index: int) -> dict[str, object]:
    """Validate one atom dict and return a normalised copy.

    Unknown ``kind`` values fall back to ``"prose"`` with a :mod:`warnings`
    warning.  Missing required fields raise :exc:`ValueError` with the atom
    index and field name included in the message.

    Parameters
    ----------
    atom:
        Raw atom dict (not mutated).
    index:
        Zero-based position of the atom in the document list (used in error
        messages).

    Returns
    -------
    dict
        Normalised copy of *atom*.
    """
    result: dict[str, object] = dict(atom)

    # ── Kind normalisation ─────────────────────────────────────────────────
    kind = result.get("kind")
    if kind is None:
        result["kind"] = "prose"
        kind = "prose"
    elif kind not in ATOM_KINDS:
        warnings.warn(
            f"Atom {index}: unknown kind {kind!r}; falling back to 'prose'.",
            stacklevel=3,
        )
        result["kind"] = "prose"
        kind = "prose"

    # ── Required field check ───────────────────────────────────────────────
    required = _REQUIRED_FIELDS[str(kind)]
    if required is not None and required not in result:
        raise ValueError(
            f"Atom {index} (kind={kind!r}) is missing required field {required!r}."
        )

    # ── Enum validation ────────────────────────────────────────────────────
    if kind == "callout":
        tone = result.get("tone")
        if tone is not None and tone not in CALLOUT_TONES:
            raise ValueError(
                f"Atom {index}: invalid callout tone {tone!r}. "
                f"Expected one of {sorted(CALLOUT_TONES)}."
            )

    if kind == "items":
        variant = result.get("variant")
        if variant is not None and variant not in ITEMS_VARIANTS:
            raise ValueError(
                f"Atom {index}: invalid items variant {variant!r}. "
                f"Expected one of {sorted(ITEMS_VARIANTS)}."
            )

    return result


def validate_document(atoms: list[Mapping[str, object]]) -> list[dict[str, object]]:
    """Validate a full document (list of atom dicts).

    Parameters
    ----------
    atoms:
        List of raw atom dicts.

    Returns
    -------
    list[dict]
        List of normalised atom dicts.

    Raises
    ------
    ValueError
        If *atoms* is not a list, or if any element is not a dict, or if any
        atom fails validation.
    """
    if not isinstance(atoms, list):
        raise ValueError(
            f"Document must be a list of atom dicts; got {type(atoms).__name__!r}."
        )

    result: list[dict[str, object]] = []
    for i, atom in enumerate(atoms):
        if not isinstance(atom, dict):
            raise ValueError(
                f"Atom {i} must be a dict; got {type(atom).__name__!r}."
            )
        result.append(validate_atom(atom, i))
    return result
