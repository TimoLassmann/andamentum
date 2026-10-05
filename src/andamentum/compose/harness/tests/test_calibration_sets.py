"""The synthetic labelled sets used to calibrate the decision model."""

from __future__ import annotations


def test_calibration_sets_shape() -> None:
    from andamentum.compose.harness.bench.calibration import load_labelled_set

    for task, fields in (
        ("needs_citation", {"id", "text", "label"}),
        ("supported_by_passage", {"id", "passage", "sentence", "label"}),
    ):
        rows = load_labelled_set(task)  # type: ignore[arg-type]
        assert len(rows) == 30
        assert all(set(r) == fields for r in rows)
        assert sum(bool(r["label"]) for r in rows) == 15
        assert len({r["id"] for r in rows}) == 30
