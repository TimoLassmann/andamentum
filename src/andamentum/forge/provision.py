"""What a generated system needs installed, and the name it goes by on PyPI.

Stage 2 of the sandbox capability model. The base image bakes the infra + a handful of
common domain libraries (see ``Containerfile``); anything *else* an authored body imports is
the **long tail** — discovered here from the rendered package and provisioned on demand into
a per-system sandbox image (``sandbox.PodmanSandbox``). So forge can build a system that
needs, say, ``lxml`` or ``pandas`` without those living in every sandbox.

The policy is **open**: any third-party import is discovered and installed, not gated — a
bogus/typo'd name fails loud at *image build* (pip can't resolve it), which is earlier and
clearer than a runtime ImportError at the test run. Leaf worker: ``stdlib`` only.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Import name → the name a generated body uses; pip name → what `pip install` wants. Most
# match; these are the common mismatches an authored body is likely to hit.
IMPORT_TO_PIP: dict[str, str] = {
    "bs4": "beautifulsoup4",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "yaml": "pyyaml",
    "dateutil": "python-dateutil",
    "sklearn": "scikit-learn",
    "dotenv": "python-dotenv",
    "jwt": "pyjwt",
    "git": "gitpython",
    "OpenSSL": "pyopenssl",
    "docx": "python-docx",
    "pptx": "python-pptx",
    "attr": "attrs",
}

# Import names the base image already provides — infra the runtime/builder chain needs, plus
# the light domain commons baked in the Containerfile. Discovered imports in this set are
# NOT reinstalled (already present); everything else is a per-system extra. KEEP IN SYNC with
# the two pip lines in ``Containerfile``.
_BASE_IMAGE_IMPORTS: frozenset[str] = frozenset(
    {
        # infra
        "pydantic",
        "pydantic_graph",
        "pydantic_ai",
        "httpx",
        "numpy",
        "rich",
        "rapidfuzz",
        "pytest",
        "andamentum",
        # domain commons (baked)
        "bs4",
        "feedparser",
        "dateutil",
        "yaml",
        "requests",
        "markdown",
        "pypdf",
    }
)


def _pip_name(import_top: str) -> str:
    return IMPORT_TO_PIP.get(import_top, import_top)


def _top_level_imports(tree: ast.Module) -> set[str]:
    """Absolute top-level module names imported anywhere in a module (relative imports skipped)."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # `from .x import y` — intra-package, never a dependency
                continue
            if node.module:
                out.add(node.module.split(".")[0])
    return out


def discover_requirements(pkg_dir: Path) -> frozenset[str]:
    """The pip packages a generated package needs beyond the base image (its long tail).

    Scans every module of the package (its own tests excluded — they import only the package
    + pytest, both provisioned), collects absolute third-party imports, drops the stdlib, the
    base-image set, ``andamentum``, and the package's own name, and maps the rest to pip
    names. An empty result means the base image already covers the system (no per-system
    image needed — the fast path).
    """
    own = pkg_dir.name
    excluded = _BASE_IMAGE_IMPORTS | sys.stdlib_module_names | {own}
    tops: set[str] = set()
    for py in pkg_dir.rglob("*.py"):
        if "tests" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue  # a half-authored hole; nothing to discover, not our error to raise
        tops |= _top_level_imports(tree)
    return frozenset(_pip_name(t) for t in tops if t not in excluded)
