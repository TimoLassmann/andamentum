"""Step 5 — attribution: turn a red audit into the node(s) to re-author (§4.4).

A red test names no culprit. ``attribute_failures`` is the pure, ranked, deterministic
function that computes ``rebuild_targets``: it consumes the already-produced audit
findings and the build report and returns the sorted node names a rebuild should
target. Its only I/O is reading the filled ``nodes.py`` under ``pkg`` to AST-map a
failing ``line`` to its owning node class (signals 1 and 2 both need the source that
Render is about to overwrite back to holes — hence attribution runs inside the ``Audit``
node). It makes **no** model call: the model contributes at most one of four ranked
signals (the reconciled critic finding), and only ever as data.

The four signals, ranked so deterministic ones lead and the model is a supplement:

  1. ``nodes.py``-scoped ``check_code`` ``Violation``s → AST line→owning-class map.
     A violation outside ``nodes.py`` is a forge render bug, not a node-body fix, so it
     is dropped here (the caller treats it as a loud terminal).
  2. failing-test traceback ``nodes.py:line`` frames → the same AST map. A frame landing
     in a render-owned top-level helper (e.g. ``_user_prompt_*``) maps to no class span
     and is correctly dropped.
  3. ``BuildConcern`` nodes — the component manager's already-recorded suspects.
  4. reconciled critic ``NodeFinding``s — the one LLM signal, its free node name
     reconciled to a real spec node via rapidfuzz.

``targets = ((1 ∪ 2 ∪ 3 ∪ 4) ∩ authored_nodes) ∪ remaining_holes``, sorted, where
``authored_nodes = filled ∪ unfillable`` and ``remaining_holes`` (the still-unfilled
``unfillable`` nodes) are unconditionally eligible. On an empty attributed set with a
red audit, a fallback ladder: all ``BuildConcern`` nodes → all authored spine nodes.
Still empty ⇒ ``[]`` (the caller's loud terminal — re-authoring cannot fix it).

Pure module: ``ast`` / ``re`` / stdlib plus the sibling schemas and the reconciliation
helper; no graph engine.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from .audit import _reconcile_node
from .schemas import AuditReport, BuildReport, CheckResult
from .spec import SystemSpec

# A `nodes.py:<line>` traceback frame. The negative lookbehind for a word char keeps a
# frame in `test_nodes.py` (or any `*_nodes.py`) from matching — only the generated
# `nodes.py` counts.
_FRAME_RE = re.compile(r"(?<!\w)nodes\.py:(\d+)")


def _class_spans(source: str) -> dict[str, tuple[int, int]]:
    """Map each top-level ``ClassDef`` name to its ``(start, end)`` line span.

    The generated node classes are top-level ``@dataclass`` ``BaseNode`` subclasses, so
    the spans are clean. ``start`` includes any decorator lines (``ClassDef.lineno``
    points at ``class``, not ``@dataclass``) so a line on the decorator still attributes
    to the class. Returns an empty map on a syntax error (a package that will not parse
    attributes to nothing → loud terminal)."""
    spans: dict[str, tuple[int, int]] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return spans
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            spans[node.name] = (start, node.end_lineno or node.lineno)
    return spans


def _owning_class(line: int, spans: dict[str, tuple[int, int]]) -> str:
    """The class whose span contains ``line`` — or ``""`` when the line is in a
    render-owned top-level helper (a ``_user_prompt_*`` function) or module scope, which
    is not an authored body and so is dropped rather than mis-attributed."""
    for name, (start, end) in spans.items():
        if start <= line <= end:
            return name
    return ""


def _check(audit: AuditReport, name: str) -> CheckResult | None:
    for c in audit.checks:
        if c.name == name:
            return c
    return None


def _violation_nodes(audit: AuditReport, spans: dict[str, tuple[int, int]]) -> set[str]:
    """Signal 1: ``nodes.py``-scoped dialect violations → owning node classes."""
    dialect = _check(audit, "dialect")
    out: set[str] = set()
    if dialect is None:
        return out
    for v in dialect.violations:
        if Path(v.file).name != "nodes.py":
            continue
        owner = _owning_class(v.line, spans)
        if owner:
            out.add(owner)
    return out


def _traceback_nodes(audit: AuditReport, spans: dict[str, tuple[int, int]]) -> set[str]:
    """Signal 2: ``nodes.py:line`` traceback frames → owning node classes."""
    tests = _check(audit, "tests")
    out: set[str] = set()
    if tests is None:
        return out
    for m in _FRAME_RE.finditer(tests.raw_output):
        owner = _owning_class(int(m.group(1)), spans)
        if owner:
            out.add(owner)
    return out


def _render_owned_violation(audit: AuditReport) -> bool:
    """True if the dialect check flagged a violation *outside* ``nodes.py`` — a
    render-owned file (``deps.py`` / ``graph.py`` / ``models.py`` / …).

    Such a violation is a forge *render* bug, not something re-authoring a node body can
    fix (§6.1 / §12.4): re-rendering reproduces it and re-authoring cannot touch it. So it
    forces a **loud terminal** — ``attribute_failures`` returns ``[]`` rather than falling
    through to the spine fallback (which would burn a rebuild round that cannot help)."""
    dialect = _check(audit, "dialect")
    if dialect is None:
        return False
    return any(Path(v.file).name != "nodes.py" for v in dialect.violations)


def _import_or_collection_failure(audit: AuditReport) -> bool:
    """True if pytest reported a collection/import ERROR and no ``nodes.py`` frame — the
    signature of a package that failed to import/collect (a missing dependency, a
    render-owned import error). Keyed on pytest's distinct ``error`` count, NOT a 0/0
    passed/failed inference: an ``error`` is unambiguously a collection/import/setup
    failure, whereas a ``failed`` assertion (a node produced the wrong value, tests_failed>0)
    is a behavioural failure that IS fixable by re-authoring and must still flow to the
    normal rebuild path. Re-authoring a body's *logic* cannot fix a package that will not
    import, so this forces a **loud terminal** (§4.4's "package-level import error")."""
    tests = _check(audit, "tests")
    if tests is None or tests.passed:
        return False
    return tests.tests_errored > 0 and not _FRAME_RE.search(tests.raw_output)


def _concern_nodes(build: BuildReport | None) -> set[str]:
    """Signal 3: nodes the component manager flagged as suspect bodies."""
    return {c.node for c in build.concerns} if build is not None else set()


def _critic_nodes(audit: AuditReport, node_names: list[str]) -> set[str]:
    """Signal 4: critic findings reconciled to real spec node names (the one LLM signal)."""
    out: set[str] = set()
    if audit.critic is None:
        return out
    for finding in audit.critic.issues:
        real = _reconcile_node(finding.node, node_names)
        if real:
            out.add(real)
    return out


def attribute_failures(
    audit: AuditReport,
    build: BuildReport | None,
    spec: SystemSpec,
    pkg: Path,
) -> list[str]:
    """The pure ranked-signal attribution (§4.4). ``pkg`` is the destination root; the
    filled source read for the AST line-map is ``pkg/<spec.name>/nodes.py``."""
    node_names = [n.name for n in spec.nodes]
    nodes_file = pkg / spec.name / "nodes.py"
    source = nodes_file.read_text(encoding="utf-8") if nodes_file.exists() else ""
    spans = _class_spans(source)

    # A render-owned dialect violation OR an import/collection failure is a forge/packaging
    # bug, not a node-body fix (§6.1/§12.4): surface as a loud terminal, never a rebuild
    # target — re-authoring cannot help, so a rebuild round would only be burned.
    if _render_owned_violation(audit) or _import_or_collection_failure(audit):
        return []

    signals = (
        _violation_nodes(audit, spans)
        | _traceback_nodes(audit, spans)
        | _concern_nodes(build)
        | _critic_nodes(audit, node_names)
    )

    filled = {f.node for f in build.filled} if build is not None else set()
    unfillable = {u.node for u in build.unfillable} if build is not None else set()
    authored_nodes = filled | unfillable
    remaining_holes = unfillable

    targets = (signals & authored_nodes) | remaining_holes
    if targets:
        return sorted(targets)

    # Fallback ladder — only when nothing attributed AND the audit is red.
    if audit.works:
        return []
    concern_nodes = _concern_nodes(build) & authored_nodes
    if concern_nodes:
        return sorted(concern_nodes)
    authored_spine = {n.name for n in spec.spine_nodes} & authored_nodes
    if authored_spine:
        return sorted(authored_spine)
    return []
