"""Translate a ``where`` filter into SQL over a revision's metadata JSON."""

from __future__ import annotations

import json

from .types import Where


def json_path(field: str) -> str:
    """``"bib.year"`` -> ``'$."bib"."year"'``; every segment is quoted."""
    parts = field.split(".")
    if not field or any(not p for p in parts):
        raise ValueError(f"invalid metadata field name {field!r}")
    return "$" + "".join("." + json.dumps(p) for p in parts)


def _check_scalar(field: str, value: object) -> None:
    if not isinstance(value, str | int | float | bool):
        raise TypeError(
            f"where[{field!r}]: {type(value).__name__} is not a str, int, float or bool"
        )


def where_sql(where: Where | None, *, column: str) -> tuple[str, list[object]]:
    """SQL condition and parameters for ``where`` over the JSON in ``column``.

    Returns ``("1", [])`` for an empty filter.
    """
    if not where:
        return "1", []
    clauses: list[str] = []
    params: list[object] = []
    for field, value in where.items():
        path = json_path(field)
        if value is None:
            clauses.append(f"json_type({column}, ?) IS NULL OR json_type({column}, ?) = 'null'")
            clauses[-1] = f"({clauses[-1]})"
            params += [path, path]
        elif isinstance(value, list):
            if not value:
                raise ValueError(f"where[{field!r}]: an empty list matches nothing")
            for v in value:
                _check_scalar(field, v)
            marks = ",".join("?" * len(value))
            clauses.append(f"json_extract({column}, ?) IN ({marks})")
            params += [path, *value]
        else:
            _check_scalar(field, value)
            clauses.append(f"json_extract({column}, ?) = ?")
            params += [path, value]
    return " AND ".join(clauses), params
