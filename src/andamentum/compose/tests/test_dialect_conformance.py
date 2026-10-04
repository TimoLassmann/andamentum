"""Guards that ``andamentum.compose`` stays dialect-conforming and correctly
layered: fast, static checks, no LLM calls.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from andamentum.agentic_dialect.checks import check_code

MODULE_ROOT = Path(__file__).resolve().parents[1]

# compose may use: core, scribe, whetstone (public API), proofread, llm_judge,
# harvest, figures; deep_research only behind --research (deferred import).
FORBIDDEN_MODULES = {
    "andamentum.epistemic",
    "andamentum.document_store",
    "andamentum.forge",
    "andamentum.chunker",
    "andamentum.vision_critique",
}
# The graph engine belongs to graph.py alone (dialect L2).
ENGINE = "pydantic_graph"


def _source_files() -> list[Path]:
    return sorted(
        p
        for p in MODULE_ROOT.rglob("*.py")
        if "tests" not in p.relative_to(MODULE_ROOT).parts
    )


def _imports(path: Path) -> list[tuple[str, bool]]:
    """(module name, imported at module top level) for every import in a file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top = {id(n) for n in tree.body}
    found: list[tuple[str, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(a.name, id(node) in top) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.module, id(node) in top))
    return found


def test_check_code_passes_default():
    assert check_code(MODULE_ROOT) == []


def test_check_code_passes_strict():
    assert check_code(MODULE_ROOT, strict=True) == []


def test_no_forbidden_imports():
    for path in _source_files():
        for name, _top in _imports(path):
            for forbidden in FORBIDDEN_MODULES:
                assert not (name == forbidden or name.startswith(forbidden + ".")), (
                    f"{path.name} imports forbidden module {name!r}"
                )


def test_whetstone_is_used_through_its_public_api_only():
    for path in _source_files():
        for name, _top in _imports(path):
            assert not name.startswith("andamentum.whetstone."), (
                f"{path.name} imports {name!r}; import from andamentum.whetstone"
            )


def test_only_the_graph_imports_the_engine():
    for path in _source_files():
        if path.name == "graph.py":
            continue
        for name, _top in _imports(path):
            assert not name.startswith(ENGINE), f"{path.name} imports {name!r}"


def test_deep_research_is_only_imported_lazily():
    for path in _source_files():
        for name, top in _imports(path):
            if name.startswith("andamentum.deep_research"):
                assert not top, f"{path.name} imports deep_research at module level"


def test_nothing_else_in_the_package_imports_compose():
    package_root = MODULE_ROOT.parent
    for path in package_root.rglob("*.py"):
        relative = path.relative_to(package_root)
        if relative.parts[0] == "compose":
            continue
        text = path.read_text(encoding="utf-8")
        assert "andamentum.compose" not in text, f"{relative} references compose"


def test_public_surface_matches_documented_set():
    import andamentum.compose as compose

    expected = {
        "write_document",
        "compose_bundle",
        "compile_brief",
        "ComposeResult",
        "Brief",
        "Roles",
        "Genre",
        "GENRES",
        "genre_for",
        "StopReason",
    }
    assert set(compose.__all__) == expected
    for name in expected:
        assert getattr(compose, name) is not None


def test_entry_points_are_coroutines_with_keyword_only_roles():
    import andamentum.compose as compose

    for fn in (compose.write_document, compose.compose_bundle):
        assert inspect.iscoroutinefunction(fn)
        assert inspect.signature(fn).parameters["roles"].kind is (
            inspect.Parameter.KEYWORD_ONLY
        )
