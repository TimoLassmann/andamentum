"""The docx_edit package stays a leaf, and nothing imports its old paths.

docx_edit is shared by whetstone and compose. It may import no other andamentum
module except the shared utility ``core.text_match``. Without this check a
later import of a whetstone or compose helper would quietly turn it into a
cycle.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve().parents[2]
ALLOWED = ("andamentum.docx_edit", "andamentum.core.text_match")
OLD_PATHS = (
    "andamentum.whetstone.docx.atoms",
    "andamentum.whetstone.docx.surgery",
    "andamentum.whetstone.docx.textmap",
    "andamentum.whetstone.docx.xml_builder",
)


def _imports(path: Path) -> list[str]:
    """Every absolute module name imported by *path*, relative imports resolved."""
    names: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
        elif isinstance(node, ast.ImportFrom) and node.level > 0:
            package = list(path.relative_to(SOURCE).with_suffix("").parts)
            if path.name != "__init__.py":
                package = package[:-1]
            base = package[: len(package) - (node.level - 1)]
            names.append(".".join(base + ([node.module] if node.module else [])))
    return names


def _package_sources() -> list[Path]:
    return [p for p in PACKAGE.rglob("*.py") if "tests" not in p.parts]


def test_docx_edit_imports_no_other_andamentum_module() -> None:
    offenders = {}
    for path in _package_sources():
        for name in _imports(path):
            if name.startswith("andamentum") and not name.startswith(ALLOWED):
                offenders.setdefault(str(path.relative_to(SOURCE)), []).append(name)
    assert not offenders, f"docx_edit must stay a leaf: {offenders}"


def test_nothing_imports_the_old_module_paths() -> None:
    offenders = []
    for path in SOURCE.rglob("*.py"):
        if "docx_edit" in path.parts and path.parent.name == "docx_edit":
            continue
        for name in _imports(path):
            if name.startswith(OLD_PATHS):
                offenders.append(str(path.relative_to(SOURCE)))
    assert not offenders, f"old docx paths still imported: {offenders}"
