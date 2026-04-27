"""Load benchmark cases from disk: pair .input + .truth.json, resolve anchors."""

from __future__ import annotations

import json
from pathlib import Path

from andamentum.chunker.validation import find_anchor

from .types import (
    BenchmarkCase,
    ResolvedTruth,
    ResolvedTruthUnit,
)


class LoadError(Exception):
    """Raised when a case file is malformed or anchors don't resolve."""


def load_case(truth_path: Path | str) -> BenchmarkCase:
    """Load a benchmark case from a .truth.json path.

    Looks for the sibling .input.<ext> file, reads source, resolves
    anchors to char offsets, returns a fully-loaded BenchmarkCase.
    """
    truth_path = Path(truth_path)
    if not truth_path.name.endswith(".truth.json"):
        raise LoadError(f"Truth file must be named *.truth.json, got {truth_path.name}")

    # Find the input file (any of several extensions)
    stem_name = truth_path.name.removesuffix(".truth.json")
    input_file = None
    for ext in ("md", "txt", "html", "py", "rst", "json"):
        candidate = truth_path.parent / f"{stem_name}.input.{ext}"
        if candidate.exists():
            input_file = candidate
            break
    if input_file is None:
        raise LoadError(f"No matching .input.<ext> file for {truth_path}")

    source = input_file.read_text()
    truth_data = json.loads(truth_path.read_text())

    # Resolve anchors monotonically
    resolved_units: list[ResolvedTruthUnit] = []
    cursor = 0
    for u in truth_data["units"]:
        start_match = find_anchor(u["start_anchor"], source, search_from=cursor)
        if start_match is None:
            raise LoadError(
                f"Case {stem_name!r} unit {u.get('title')!r}: "
                f"start_anchor {u['start_anchor']!r} not found in source after offset {cursor}"
            )
        end_match = find_anchor(u["end_anchor"], source, search_from=start_match.start)
        if end_match is None:
            raise LoadError(
                f"Case {stem_name!r} unit {u.get('title')!r}: "
                f"end_anchor {u['end_anchor']!r} not found after start_anchor"
            )
        resolved_units.append(
            ResolvedTruthUnit(
                title=u.get("title", ""),
                start_offset=start_match.start,
                end_offset=end_match.end,
            )
        )
        cursor = end_match.end

    return BenchmarkCase(
        name=stem_name,
        source=source,
        domain=truth_data.get("domain", "general"),
        expected_f1_floor=float(truth_data["expected_f1_floor"]),
        boundary_tolerance_chars=int(truth_data["boundary_tolerance_chars"]),
        truth=ResolvedTruth(
            convention=truth_data["convention"],
            units=resolved_units,
        ),
    )
