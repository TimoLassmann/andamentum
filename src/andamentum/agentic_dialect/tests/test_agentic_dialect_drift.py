"""The anti-drift binding: the Python kernel and DIALECT.md cannot diverge."""

from __future__ import annotations

import re

from andamentum.agentic_dialect import checklist, laws, skeleton
from andamentum.agentic_dialect.doc import normalize, read_doc


def test_every_law_has_a_tagged_section_with_its_statement() -> None:
    doc = normalize(read_doc())
    for lw in laws():
        header = normalize(f"{lw.id} — {lw.name}.")
        assert header in doc, f"missing law header in DIALECT.md: {lw.id} — {lw.name}"
        assert normalize(lw.statement) in doc, (
            f"law {lw.id} statement not found verbatim in DIALECT.md"
        )


def test_doc_law_ids_match_the_kernel_exactly() -> None:
    raw = read_doc()
    known = {lw.id for lw in laws()}
    found = set(re.findall(r"\*\*(L\d+) —", raw))
    assert found == known, f"law ids in doc {sorted(found)} != kernel {sorted(known)}"


def test_checklist_items_present_in_doc() -> None:
    doc = normalize(read_doc())
    for item, _law_id in checklist():
        assert normalize(item) in doc, f"checklist item not in DIALECT.md: {item}"


def test_skeleton_extracts_and_looks_runnable() -> None:
    s = skeleton()
    assert "async def run_brief" in s
    assert "from pydantic_graph import" in s
    assert s.strip().endswith("return out.output")
