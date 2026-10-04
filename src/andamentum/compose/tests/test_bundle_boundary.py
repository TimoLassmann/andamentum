"""The bundle stays model-free: nothing under ``compose/bundle/`` may import a
model library, the graph engine, core, whetstone, llm_judge or any compose
module outside ``bundle/``.
"""

from __future__ import annotations

import ast
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parents[1] / "bundle"

FORBIDDEN = (
    "pydantic_ai",
    "pydantic_graph",
    "andamentum.core",
    "andamentum.whetstone",
    "andamentum.llm_judge",
)
COMPOSE = "andamentum.compose"
BUNDLE = "andamentum.compose.bundle"


def _absolute_imports(path: Path) -> list[str]:
    """Every imported module name, relative imports resolved against the file."""
    package = [
        "andamentum",
        "compose",
        *path.relative_to(BUNDLE_ROOT.parent).parts[:-1],
    ]
    names: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                anchor = package[: len(package) - (node.level - 1)]
                base = ".".join(anchor + ([node.module] if node.module else []))
            names.append(base)
            # `from .. import x` may name a sibling module
            names += [f"{base}.{a.name}" for a in node.names]
    return names


def _is(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def test_bundle_imports_no_model_or_compose_code():
    files = sorted(BUNDLE_ROOT.rglob("*.py"))
    assert files, "bundle sources not found"
    for path in files:
        for name in _absolute_imports(path):
            for forbidden in FORBIDDEN:
                assert not _is(name, forbidden), f"{path.name} imports {name!r}"
            if _is(name, COMPOSE) and not _is(name, BUNDLE) and name != COMPOSE:
                raise AssertionError(f"{path.name} imports {name!r} outside bundle/")
