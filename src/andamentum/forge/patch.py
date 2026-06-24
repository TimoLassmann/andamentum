"""Replace a method body in a Python source file using AST line ranges.

``apply_body`` finds the named class + method via the AST, normalises the caller's
``new_body`` to the correct indent (parse-based, via ``ast.unparse`` — so inconsistent
model indentation can't corrupt the file), and writes it back. The signature line is
never touched; a body that won't parse is left for the compile gate to reject cleanly.

Ported from the ``forge`` dump. Leaf worker: ``stdlib`` only, no graph engine.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path


def apply_body(path: Path, class_name: str, method_name: str, new_body: str) -> None:
    """Overwrite the body of ``class_name.method_name`` in ``path`` with ``new_body``.

    Raises ``ValueError`` if the class or method is not found.
    """
    source = path.read_text()
    tree = ast.parse(source)

    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef) or cls.name != class_name:
            continue
        for method in cls.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if method.name != method_name:
                continue

            lines = source.splitlines(keepends=True)
            def_line = lines[method.lineno - 1]
            def_indent = len(def_line) - len(def_line.lstrip())
            body_indent = " " * (def_indent + 4)

            body_start = method.body[0].lineno - 1  # 0-indexed, inclusive
            body_end = method.end_lineno  # 0-indexed, exclusive

            new_lines = _normalise(new_body, body_indent)
            result = "".join(lines[:body_start] + new_lines + lines[body_end:])
            path.write_text(result)
            return

    raise ValueError(f"{class_name}.{method_name} not found in {path}")


def _normalise(body: str, indent: str) -> list[str]:
    """Re-indent ``body`` to ``indent``, canonicalising via the AST when it parses."""
    stripped = textwrap.dedent(body).strip("\n")
    if not stripped.strip():
        return [indent + "pass\n"]

    canon = stripped
    try:
        wrapped = "async def _f():\n" + textwrap.indent(stripped, "    ")
        func = ast.parse(wrapped).body[0]
        assert isinstance(func, ast.AsyncFunctionDef)
        canon = "\n".join(ast.unparse(stmt) for stmt in func.body)
    except (SyntaxError, AssertionError, ValueError):
        canon = stripped

    return [
        indent + line + "\n" if line.strip() else "\n" for line in canon.splitlines()
    ]
