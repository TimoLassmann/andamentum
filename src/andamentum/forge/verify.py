"""Worker: verify a rendered package — deterministic, no LLM.

Four checks, escalating in strength:

  - **files** — the expected package files were written.
  - **parses** — every generated ``.py`` is syntactically valid (``ast.parse``).
  - **assembles** — the package imports and its ``Graph`` assembles with every node
    (the strong check: the generated system is a real, importable agentic graph).
  - **recipe** — the frozen ``spec.json`` re-validates against ``SystemSpec``.

``works`` is True only when every check passes; ``score`` is the pass fraction. The
assemble check imports the generated package in-process and restores ``sys.modules``
afterwards, so it is safe to run repeatedly (dialect Law 8).

Engine-free leaf worker — it touches the filesystem and the import system, but never
the graph engine of *this* module.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

from .schemas import CheckResult, VerificationReport
from .spec import SystemSpec


def _files_check(pkg: Path) -> CheckResult:
    required = [
        "__init__.py",
        "models.py",
        "deps.py",
        "nodes.py",
        "graph.py",
        "spec.json",
    ]
    missing = [f for f in required if not (pkg / f).exists()]
    return CheckResult(
        name="files",
        passed=not missing,
        detail="all present" if not missing else f"missing: {missing}",
    )


def _parse_check(pkg: Path) -> CheckResult:
    bad: list[str] = []
    for py in sorted(pkg.rglob("*.py")):
        try:
            ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError as e:
            bad.append(f"{py.name}:{e.lineno} {e.msg}")
    return CheckResult(
        name="parses", passed=not bad, detail="all parse" if not bad else "; ".join(bad)
    )


def _assemble_check(dest: Path, name: str, expected_nodes: int) -> CheckResult:
    """Import the generated package and assert its Graph assembles with every node."""
    dest_str = str(dest)
    added = dest_str not in sys.path
    if added:
        sys.path.insert(0, dest_str)
    saved = {
        k: v for k, v in sys.modules.items() if k == name or k.startswith(name + ".")
    }
    for k in list(saved):
        del sys.modules[k]
    try:
        graph_mod = importlib.import_module(f"{name}.graph")
        n = len(graph_mod.graph.node_defs)
        ok = n == expected_nodes
        return CheckResult(
            name="assembles",
            passed=ok,
            detail=f"{n} nodes assembled (expected {expected_nodes})",
        )
    except Exception as e:  # noqa: BLE001 — any import/assembly failure is a verification failure
        return CheckResult(
            name="assembles", passed=False, detail=f"{type(e).__name__}: {e}"
        )
    finally:
        for k in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
            del sys.modules[k]
        sys.modules.update(saved)
        if added and dest_str in sys.path:
            sys.path.remove(dest_str)


def _recipe_check(pkg: Path) -> CheckResult:
    spec_json = pkg / "spec.json"
    if not spec_json.exists():
        return CheckResult(name="recipe", passed=False, detail="spec.json absent")
    try:
        SystemSpec.model_validate_json(spec_json.read_text(encoding="utf-8"))
        return CheckResult(
            name="recipe", passed=True, detail="frozen spec re-validates"
        )
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            name="recipe", passed=False, detail=f"{type(e).__name__}: {e}"
        )


def verify_package(spec: SystemSpec, dest: Path) -> VerificationReport:
    """Run every deterministic check over the package rendered at ``dest/<spec.name>``."""
    pkg = dest / spec.name
    checks = [
        _files_check(pkg),
        _parse_check(pkg),
        _assemble_check(dest, spec.name, len(spec.nodes)),
        _recipe_check(pkg),
    ]
    passed = sum(1 for c in checks if c.passed)
    return VerificationReport(
        works=all(c.passed for c in checks), score=passed / len(checks), checks=checks
    )
