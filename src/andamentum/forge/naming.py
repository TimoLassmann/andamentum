"""Shared naming helpers — the recipe's identifier rules in one place.

Pydantic-free, stdlib-only. Three jobs that would otherwise be duplicated across
modules live here:

  - validate an identifier is snake_case / PascalCase  (spec validators)
  - generate an identifier from free text              (design compile)
  - convert PascalCase → snake_case                    (per-node test naming)

Ported from the `forge` exploratory dump (already dialect-clean: a leaf worker
file that imports no graph engine).
"""

from __future__ import annotations

import re

PASCAL = re.compile(r"^[A-Z][A-Za-z0-9]*$")  # PascalCase type / class name
IDENT = re.compile(r"^[a-z][a-z0-9_]*$")  # snake_case identifier


def is_pascal(v: str) -> bool:
    return bool(PASCAL.match(v))


def is_ident(v: str) -> bool:
    return bool(IDENT.match(v))


def to_snake(text: str, fallback: str, *, max_words: int = 4) -> str:
    """Free text → a snake_case identifier, or ``fallback``.

    ``max_words`` caps the words used (default 4, for brevity in system/agent
    names); pass ``0`` for no cap, e.g. when a State field name must faithfully
    and uniquely represent its data name (truncation there caused field-name
    collisions).
    """
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    chosen = words if max_words <= 0 else words[:max_words]
    s = "_".join(chosen)
    return s if re.match(r"^[a-z]", s) else fallback


def to_pascal(text: str, fallback: str) -> str:
    """Free text → a PascalCase identifier (first four words), or ``fallback``."""
    words = re.findall(r"[A-Za-z0-9]+", text)
    p = "".join(w[:1].upper() + w[1:] for w in words[:4])
    return p if re.match(r"^[A-Za-z]", p) else fallback


def pascal_to_snake(name: str) -> str:
    """PascalCase → snake_case (e.g. 'FooBar' → 'foo_bar')."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def canonical_datum(name: str, input_tokens: frozenset[str]) -> str:
    """Canonical form of a data name: a graph-input token kept (lowercased), everything
    else snake_cased so the SAME concept named with different casing/spacing maps to one
    identifier. This is a normalisation (deterministic, lossless intent), not a fallback —
    it lets the variable registry match a selection to a produced name exactly.
    """
    if name.lower() in input_tokens:
        return name.lower()
    return to_snake(name, name, max_words=0)
