"""deep_research imports `core`, `references` and `harvest` among the sub-modules, and nothing else.

In particular never `epistemic`, `document_store`, `compose` or `whetstone`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import andamentum.deep_research as package

ALLOWED = {"core", "references", "harvest", "deep_research"}
FORBIDDEN = {"epistemic", "document_store", "compose", "whetstone"}


def test_deep_research_imports_no_other_sub_module() -> None:
    root = Path(package.__file__).parent
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        if "tests" in path.relative_to(root).parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            for name in names:
                parts = name.split(".")
                if (
                    parts[0] == "andamentum"
                    and len(parts) > 1
                    and parts[1] not in ALLOWED
                ):
                    offenders.append(f"{path.name}: {name}")
    assert offenders == []


def test_the_forbidden_modules_are_not_in_the_allowed_set() -> None:
    assert not ALLOWED & FORBIDDEN
