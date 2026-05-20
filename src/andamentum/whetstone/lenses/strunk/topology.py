"""Reflect the Strunk sub-graph as a Python value.

Mirrors the discipline used by ``andamentum.epistemic.graph.topology``:
each node's ``run()`` return-type annotation is parsed to recover the
successor list, and the result is a plain ``dict`` you can diff,
assert against, or print. This lets structural tests catch "routing
bugs become static" issues — e.g. accidentally re-introducing a node
whose successor is unreachable, or breaking the linear chain.

The reflection deliberately uses string parsing rather than fully
evaluating annotations. Annotations live as forward references
(``-> "R13OmitNeedlessWords"``) so node modules don't have to import
their successor — that would create circular-import noise. Parsing
extracts only identifiers that are members of ``NODE_CLASSES`` or
the literal ``End``, ignoring everything else (type arguments,
``list``, ``Finding``).
"""

from __future__ import annotations

import re
from inspect import signature
from typing import Any

from .graph import NODE_CLASSES


_CLASS_NAME = re.compile(r"\b([A-Z][a-zA-Z0-9_]*)\b")


def _annotation_to_string(ann: Any) -> str:
    """Best-effort stringify of an annotation (raw type or forward ref)."""
    if ann is None:
        return ""
    if isinstance(ann, str):
        return ann
    # Resolved type — use its repr; this captures End[list[Finding]] etc.
    return str(ann)


def _extract_successors(ann: Any, known: set[str]) -> list[str]:
    """Return successor node names mentioned in the annotation.

    ``known`` is the set of valid successor symbols (every node class
    name plus the literal ``End``). Everything else in the annotation
    (``list``, ``Finding``, ``Union``, ...) is dropped.
    """
    text = _annotation_to_string(ann)
    if not text:
        return []
    tokens = _CLASS_NAME.findall(text)
    return [t for t in tokens if t in known]


def topology() -> dict[str, dict[str, Any]]:
    """Reflect the sub-graph as a dict.

    Each entry::

        "<NodeName>": {
            "kind": "deterministic" | "agent" | "control",
            "reads": [...],          # sorted list copy of the ClassVar frozenset
            "writes": [...],
            "successors": [...],     # node class names reachable from this node's run()
            "rule_number": int | None,   # AGENT nodes only
            "rule_source": str | None,   # AGENT nodes only
            "model": str | None,         # AGENT nodes only
            "output_model": str | None,  # AGENT nodes only
        }
    """
    known = {cls.__name__ for cls in NODE_CLASSES} | {"End"}
    out: dict[str, dict[str, Any]] = {}
    for cls in NODE_CLASSES:
        ret = signature(cls.run).return_annotation
        successors = _extract_successors(ret, known)
        entry: dict[str, Any] = {
            "kind": cls.kind.value,  # type: ignore[attr-defined]
            "reads": sorted(cls.reads),  # type: ignore[attr-defined]
            "writes": sorted(cls.writes),  # type: ignore[attr-defined]
            "successors": successors,
        }
        # Agent-only metadata
        rule_number = getattr(cls, "rule_number", None)
        if rule_number is not None:
            entry["rule_number"] = rule_number
        rule_source = getattr(cls, "rule_source", None)
        if rule_source is not None:
            entry["rule_source"] = rule_source
        model = getattr(cls, "model", None)
        if model is not None:
            entry["model"] = model
        output_model = getattr(cls, "output_model", None)
        if output_model is not None:
            entry["output_model"] = output_model.__name__
        out[cls.__name__] = entry
    return out
