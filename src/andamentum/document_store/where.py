"""Translate a ``where`` filter into SQL over a revision's metadata JSON."""

from __future__ import annotations

import json
from collections.abc import Iterable

from .types import Where


def json_path(field: str) -> str:
    """``"bib.year"`` -> ``'$."bib"."year"'``; every segment is quoted."""
    parts = field.split(".")
    if not field or any(not p for p in parts):
        raise ValueError(f"invalid metadata field name {field!r}")
    # ensure_ascii=False: SQLite's JSON paths do not decode \uXXXX escapes.
    return "$" + "".join("." + json.dumps(p, ensure_ascii=False) for p in parts)


def _scalar_sql(column: str, path: str, value: object, field: str) -> tuple[str, list[object]]:
    """Equality that keeps JSON booleans and numbers apart.

    SQLite stores JSON true/false as 1/0, so ``= 1`` alone would match both
    ``true`` and ``1``; the ``json_type`` check tells them apart.
    """
    if isinstance(value, bool):
        return (
            f"json_type({column}, ?) = ?",
            [path, "true" if value else "false"],
        )
    if isinstance(value, int | float):
        return (
            f"(json_extract({column}, ?) = ? AND json_type({column}, ?) IN ('integer', 'real'))",
            [path, value, path],
        )
    if isinstance(value, str):
        return (
            f"(json_extract({column}, ?) = ? AND json_type({column}, ?) = 'text')",
            [path, value, path],
        )
    raise TypeError(
        f"where[{field!r}]: {type(value).__name__} is not a str, int, float or bool"
    )


def where_sql(where: Where | None, *, column: str) -> tuple[str, list[object]]:
    """SQL condition and parameters for ``where`` over the JSON in ``column``.

    Returns ``("1", [])`` for an empty filter. A list (or tuple, or set) value
    matches any of its members; an empty one matches nothing.
    """
    if not where:
        return "1", []
    clauses: list[str] = []
    params: list[object] = []
    for field, value in where.items():
        path = json_path(field)
        if value is None:
            clauses.append(f"(json_type({column}, ?) IS NULL OR json_type({column}, ?) = 'null')")
            params += [path, path]
        elif isinstance(value, list | tuple | set | frozenset):
            members = list(value) if not isinstance(value, set | frozenset) else sorted(value, key=repr)
            if not members:
                clauses.append("0")
                continue
            parts = [_scalar_sql(column, path, v, field) for v in _members(members)]
            clauses.append("(" + " OR ".join(sql for sql, _ in parts) + ")")
            for _, p in parts:
                params += p
        else:
            sql, p = _scalar_sql(column, path, value, field)
            clauses.append(sql)
            params += p
    return " AND ".join(clauses), params


def _members(values: Iterable[object]) -> list[object]:
    out = []
    for v in values:
        if v is None:
            raise TypeError("a where list cannot hold None; use {field: None} instead")
        out.append(v)
    return out
