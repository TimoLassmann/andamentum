"""The agentic dialect: the conventions for building agentic graph systems, as
data plus portable conformance checks.

One core, many thin adapters. The structured laws (``laws``/``law``), role briefs
(``for_role``), checklist, and skeleton are the authoritative enforceable surface;
the prose canon lives in ``DIALECT.md`` and is drift-tested against this kernel.
``check_code`` runs the portable static gates.

This package is a leaf: it depends only on ``pydantic`` and the standard library,
imports nothing from another andamentum sub-module, and uses only relative imports
internally — so it stays independently extractable.
"""

from __future__ import annotations

from ._laws import Law, checklist, law, laws
from ._roles import ROLES, for_role, roles
from .checks import Violation, check_code
from .doc import doc_path, read_doc, skeleton

__all__ = [
    "Law",
    "Violation",
    "ROLES",
    "check_code",
    "checklist",
    "doc_path",
    "for_role",
    "law",
    "laws",
    "read_doc",
    "roles",
    "skeleton",
]
