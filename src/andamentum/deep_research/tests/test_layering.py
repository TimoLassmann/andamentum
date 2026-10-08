"""v2 imports `core` and `references` among the sub-modules, and nothing else."""

from __future__ import annotations

import ast
from pathlib import Path

import andamentum.deep_research as v2

ALLOWED = {"core", "references", "deep_research"}


def test_v2_imports_no_other_sub_module() -> None:
    root = Path(v2.__file__).parent
    offenders: list[str] = []
    for path in root.rglob("*.py"):
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
