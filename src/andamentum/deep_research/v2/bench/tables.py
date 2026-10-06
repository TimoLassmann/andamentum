"""Markdown tables shared by the experiment report and the pool tables."""

from __future__ import annotations

from collections.abc import Sequence

NOTHING = "n/a"


def fmt(value: float | None, digits: int = 3) -> str:
    return NOTHING if value is None else f"{value:.{digits}f}"


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)
