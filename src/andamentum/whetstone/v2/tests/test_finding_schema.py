"""Schema-level tests for ``Finding`` after the critical-review refactor.

The Finding type gained two fields:
  • ``id`` — auto-generated short uuid, stable across the reflection loop
  • ``category`` — short tag chosen by the lens

These tests cover the field defaults; behaviour involving them is covered
by the node-level tests for critical_read and reflect_and_investigate.
"""

from __future__ import annotations

from andamentum.whetstone.v2.schemas import Finding


def _basic_finding(**overrides) -> Finding:
    defaults = dict(
        title="missing citation",
        severity="moderate",
        confidence="medium",
        rationale="Section 4 cites Smith 2020 but no entry in the bibliography.",
    )
    defaults.update(overrides)
    return Finding(**defaults)  # type: ignore[arg-type]


def test_finding_gets_auto_id() -> None:
    f = _basic_finding()
    assert f.id, "id should auto-generate to a non-empty string"
    assert len(f.id) <= 12, "id should be short (≤12 chars)"


def test_finding_gets_default_empty_category() -> None:
    f = _basic_finding()
    assert f.category == ""


def test_finding_accepts_explicit_category() -> None:
    f = _basic_finding(category="evidence")
    assert f.category == "evidence"


def test_finding_ids_are_unique_across_many() -> None:
    ids = {_basic_finding(title=str(i)).id for i in range(100)}
    assert len(ids) == 100, "100 freshly built Findings should all get distinct ids"


def test_finding_id_round_trips_through_model_dump() -> None:
    f = _basic_finding()
    rebuilt = Finding(**f.model_dump())
    assert rebuilt.id == f.id, "id must survive model_dump → __init__ round trip"
