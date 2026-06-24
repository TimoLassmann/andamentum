"""Portable, conservative static gates over a dialect-conforming codebase.

Pure-stdlib AST. Each gate maps to a law; a violation carries the law id so the
fix is greppable. Conservative by design — these gate builds, so a false
positive is worse than a missed one. Heavier checks (topology reflection,
Result-is-Pydantic) belong in a graph's own tests, not here.

File classification is by name: ``graph.py`` / ``nodes.py`` hold orchestration
(and so may import the engine, define ``run`` bodies); everything else under a
module is a worker/leaf file. ``tests/``, ``cli.py``, ``__init__.py`` are skipped.
"""

from __future__ import annotations

import ast
from pathlib import Path

from pydantic import BaseModel

_ORCHESTRATION_FILES = {"graph.py", "nodes.py"}
_SKIP_FILES = {"cli.py", "__init__.py", "conftest.py"}
_CLIENT_CONSTRUCTORS = {"Agent"}


class Violation(BaseModel):
    """One dialect violation found by ``check_code``."""

    file: str
    line: int
    law: str
    code: str
    message: str


def check_code(path: str | Path) -> list[Violation]:
    """Run the portable gates over a file or directory tree.

    Returns violations sorted by ``(file, line)``. An empty list means the gates
    pass — it does not prove conformance, since the review-only laws are
    unchecked.
    """
    root = Path(path)
    files = [root] if root.is_file() else sorted(root.rglob("*.py"))
    out: list[Violation] = []
    for f in files:
        if _skip(f):
            continue
        out.extend(_check_file(f))
    out.sort(key=lambda v: (v.file, v.line))
    return out


def _skip(f: Path) -> bool:
    if f.name in _SKIP_FILES or f.name.startswith("test_"):
        return True
    return "tests" in f.parts


def _is_orchestration(f: Path) -> bool:
    return f.name in _ORCHESTRATION_FILES


def _check_file(f: Path) -> list[Violation]:
    src = f.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [
            Violation(
                file=str(f),
                line=e.lineno or 0,
                law="",
                code="parse-error",
                message=f"could not parse: {e.msg}",
            )
        ]
    out: list[Violation] = []
    out.extend(_check_no_untyped_dict(f, tree))
    if _is_orchestration(f):
        out.extend(_check_future_annotations(f, tree))
        out.extend(_check_run_bodies(f, tree))
    else:
        out.extend(_check_no_engine_import(f, tree))
    return out


def _dotted(node: ast.AST) -> str:
    """Dotted name of an attribute/name expression (e.g. ``datetime.datetime.now``)."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _check_no_engine_import(f: Path, tree: ast.AST) -> list[Violation]:
    out: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Import):
            hit = any(
                a.name == "pydantic_graph" or a.name.startswith("pydantic_graph.")
                for a in node.names
            )
        else:
            mod = node.module or ""
            hit = mod == "pydantic_graph" or mod.startswith("pydantic_graph.")
        if hit:
            out.append(
                Violation(
                    file=str(f),
                    line=node.lineno,
                    law="L2",
                    code="engine-import-in-worker",
                    message="worker/leaf file imports the graph engine (pydantic_graph)",
                )
            )
    return out


def _check_no_untyped_dict(f: Path, tree: ast.AST) -> list[Violation]:
    out: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        base = node.value
        base_name = (
            base.id
            if isinstance(base, ast.Name)
            else base.attr
            if isinstance(base, ast.Attribute)
            else ""
        )
        if base_name not in {"dict", "Dict"}:
            continue
        sl = node.slice
        if isinstance(sl, ast.Tuple) and len(sl.elts) == 2:
            v = sl.elts[1]
            vname = (
                v.id
                if isinstance(v, ast.Name)
                else v.attr
                if isinstance(v, ast.Attribute)
                else ""
            )
            if vname == "Any":
                out.append(
                    Violation(
                        file=str(f),
                        line=node.lineno,
                        law="L7",
                        code="untyped-dict-any",
                        message="untyped dict[str, Any] — use a typed schema",
                    )
                )
    return out


def _check_future_annotations(f: Path, tree: ast.Module | ast.AST) -> list[Violation]:
    body = getattr(tree, "body", [])
    for node in body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            if any(a.name == "annotations" for a in node.names):
                return []
    return [
        Violation(
            file=str(f),
            line=1,
            law="",
            code="missing-future-annotations",
            message="orchestration file must open with `from __future__ import annotations`",
        )
    ]


def _run_methods(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run":
            yield node


def _check_run_bodies(f: Path, tree: ast.AST) -> list[Violation]:
    out: list[Violation] = []
    for run in _run_methods(tree):
        for node in ast.walk(run):
            if isinstance(node, ast.Call):
                name = _dotted(node.func)
                root = name.split(".")[0]
                if (
                    root == "random"
                    or name in {"time.time", "time.monotonic"}
                    or (name.endswith(".now") and "datetime" in name)
                ):
                    out.append(
                        Violation(
                            file=str(f),
                            line=node.lineno,
                            law="L4",
                            code="nondeterminism-in-routing",
                            message=f"routing reads a nondeterministic source: {name}()",
                        )
                    )
                if name.split(".")[-1] in _CLIENT_CONSTRUCTORS:
                    out.append(
                        Violation(
                            file=str(f),
                            line=node.lineno,
                            law="L2",
                            code="client-in-node-body",
                            message=f"model client {name}(...) constructed in a node body",
                        )
                    )
            elif isinstance(node, ast.While):
                t = node.test
                if (isinstance(t, ast.Constant) and t.value is True) or (
                    isinstance(t, ast.Name) and t.id == "True"
                ):
                    out.append(
                        Violation(
                            file=str(f),
                            line=node.lineno,
                            law="L5",
                            code="unbounded-loop",
                            message="while True in a node body — bound the loop (L5)",
                        )
                    )
            elif isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
                if _dotted(node.iter.func) == "range":
                    args = node.iter.args
                    if args and all(
                        isinstance(a, ast.Constant) and isinstance(a.value, int) for a in args
                    ):
                        out.append(
                            Violation(
                                file=str(f),
                                line=node.lineno,
                                law="L5",
                                code="literal-loop-bound",
                                message="range() with a literal bound in a node body — "
                                "the bound must trace to Deps or a named constant (L5)",
                            )
                        )
    return out
