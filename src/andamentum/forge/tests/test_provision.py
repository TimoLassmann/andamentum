"""Dependency discovery: what a generated package needs beyond the base image.

Pure AST — no container. ``discover_requirements`` scans a rendered package, drops the
stdlib / base-image / andamentum / intra-package imports, and maps the long-tail imports to
pip names for per-system provisioning.
"""

from __future__ import annotations

from pathlib import Path

from andamentum.forge.provision import IMPORT_TO_PIP, discover_requirements


def _pkg(tmp_path: Path, name: str, modules: dict[str, str]) -> Path:
    pkg = tmp_path / name
    (pkg / "tests").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "tests" / "__init__.py").write_text("")
    for fname, src in modules.items():
        (pkg / fname).write_text(src)
    return pkg


def test_long_tail_third_party_is_discovered(tmp_path: Path) -> None:
    pkg = _pkg(
        tmp_path,
        "widget",
        {"nodes.py": "import lxml\nimport sqlalchemy\n\ndef f():\n    return 1\n"},
    )
    assert discover_requirements(pkg) == frozenset({"lxml", "sqlalchemy"})


def test_import_names_map_to_pip_names(tmp_path: Path) -> None:
    pkg = _pkg(tmp_path, "widget", {"nodes.py": "import PIL\nfrom cv2 import imread\n"})
    assert discover_requirements(pkg) == frozenset({"pillow", "opencv-python"})


def test_stdlib_base_image_and_andamentum_are_excluded(tmp_path: Path) -> None:
    # json/asyncio (stdlib), bs4/feedparser/numpy (base image), andamentum, and the
    # package's own name are all provisioned already — none is a per-system dependency.
    src = (
        "import json\n"
        "import asyncio\n"
        "import bs4\n"
        "import feedparser\n"
        "import numpy\n"
        "from andamentum.forge.runtime import run_head\n"
        "import widget\n"
    )
    pkg = _pkg(tmp_path, "widget", {"nodes.py": src})
    assert discover_requirements(pkg) == frozenset()


def test_relative_imports_are_not_dependencies(tmp_path: Path) -> None:
    pkg = _pkg(
        tmp_path,
        "widget",
        {"nodes.py": "from .models import Foo\nfrom . import deps\n"},
    )
    assert discover_requirements(pkg) == frozenset()


def test_the_packages_own_tests_are_not_scanned(tmp_path: Path) -> None:
    # A test module importing pytest (and a stray lib) must not pollute the runtime deps —
    # the test tree is provisioned separately and runs pytest, which the base image carries.
    pkg = _pkg(tmp_path, "widget", {"nodes.py": "x = 1\n"})
    (pkg / "tests" / "test_it.py").write_text("import pytest\nimport hypothesis\n")
    assert discover_requirements(pkg) == frozenset()


def test_mapping_covers_the_common_mismatches() -> None:
    # A guard on the table itself — the well-known import≠pip cases stay correct.
    assert IMPORT_TO_PIP["bs4"] == "beautifulsoup4"
    assert IMPORT_TO_PIP["yaml"] == "pyyaml"
    assert IMPORT_TO_PIP["sklearn"] == "scikit-learn"
