"""Discover ``NotImplementedError`` holes in a generated package.

Parses ``nodes.py`` with ``ast`` so the result is robust to whitespace/formatting.
The hole kinds come directly from the renderer: a spine body or a multi-successor
head's routing. Ported from the ``forge`` dump. Leaf worker.
"""

from __future__ import annotations

import ast
from pathlib import Path

from .contract import Hole, HoleKind


def discover_holes(package_dir: Path) -> list[Hole]:
    """Find all NotImplementedError holes in ``nodes.py`` files under ``package_dir``."""
    holes: list[Hole] = []
    for nodes_file in sorted(package_dir.rglob("nodes.py")):
        holes.extend(_extract_from_file(nodes_file))
    return holes


def _extract_from_file(path: Path) -> list[Hole]:
    source = path.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    holes: list[Hole] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for method in node.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            hole = _find_hole(path, source, node, method)
            if hole is not None:
                holes.append(hole)
    return holes


def _find_hole(
    path: Path,
    source: str,
    cls: ast.ClassDef,
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Hole | None:
    raise_stmt: ast.Raise | None = None
    for stmt in method.body:
        if isinstance(stmt, ast.Raise) and _is_not_impl(stmt):
            raise_stmt = stmt
            break
    if raise_stmt is None:
        return None

    hint = _extract_hint(raise_stmt)
    kind = _infer_kind(method.name, hint)
    lines = source.splitlines()
    signature = lines[method.lineno - 1]

    context_parts: list[str] = []
    for stmt in method.body:
        if stmt is raise_stmt:
            break
        seg = ast.get_source_segment(source, stmt)
        if seg:
            context_parts.append(seg)

    return Hole(
        kind=kind,
        node=cls.name,
        method=method.name,
        file=path,
        hint=hint,
        signature=signature,
        context="\n".join(context_parts),
    )


def _is_not_impl(stmt: ast.Raise) -> bool:
    exc = stmt.exc
    if exc is None:
        return False
    if isinstance(exc, ast.Call):
        func = exc.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        return name == "NotImplementedError"
    if isinstance(exc, ast.Name):
        return exc.id == "NotImplementedError"
    return False


def _extract_hint(stmt: ast.Raise) -> str:
    exc = stmt.exc
    if not isinstance(exc, ast.Call) or not exc.args:
        return ""
    arg = exc.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    return ast.unparse(arg)


def _infer_kind(method_name: str, hint: str) -> HoleKind:
    if method_name == "_map_one":
        # The renderer's map scaffold leaves exactly one hole on an 'each' node: the
        # pure per-item transform. Keyed on the method name — the one thing the
        # renderer guarantees — not on hint prose.
        return HoleKind.MAP_ITEM
    if "Route on `out`" in hint:
        return HoleKind.ROUTING
    return HoleKind.SPINE_BODY
