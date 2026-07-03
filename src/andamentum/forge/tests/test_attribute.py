"""Step 5 — ``attribute_failures``: the pure ranked-signal attribution (§4.4 + §10.5).

Each signal is tested offline with a minimal fixture. Signals 1 and 2 need a real
``nodes.py`` on disk for the AST line→owning-class map, so those tests compile + render
the two-spine reading-list plan (no build, no model, no sandbox) — the render leaves the
node classes at real line spans. The remaining signals and the fallback ladder are pure
schema fixtures.
"""

from __future__ import annotations

import ast
from pathlib import Path

from andamentum.agentic_dialect import Violation
from andamentum.forge import compile_spec, render
from andamentum.forge.attribute import (
    _class_spans,
    _owning_class,
    attribute_failures,
)
from andamentum.forge.schemas import (
    AuditReport,
    BuildConcern,
    BuildReport,
    CheckResult,
    CriticVerdict,
    DesignPlan,
    FilledNode,
    ForgeWhy,
    NodeDraft,
    NodeFinding,
    UnfillableNode,
)
from andamentum.forge.spec import NodeKind, SystemSpec

# Node classes rendered from the plan below (matches test_build_audit.py).
_PARSE = "ParseTheRequest"
_NORMALISE = "NormaliseTheRequest"
_ANSWER = "AnswerTheRequest"  # the HEAD


def _two_spine_plan() -> DesignPlan:
    return DesignPlan(
        why=ForgeWhy(
            purpose="Help manage a reading list.",
            boundary_in="a request",
            boundary_out="an answer",
        ),
        nodes=[
            NodeDraft(
                id="n1",
                area="core",
                job="Parse the request.",
                kind=NodeKind.SPINE,
                consumes=["input"],
                produces=["parsed_request"],
            ),
            NodeDraft(
                id="n2",
                area="core",
                job="Normalise the request.",
                kind=NodeKind.SPINE,
                consumes=["parsed_request"],
                produces=["normalised"],
            ),
            NodeDraft(
                id="n3",
                area="core",
                job="Answer the request.",
                kind=NodeKind.HEAD,
                consumes=["normalised"],
                produces=["answer"],
            ),
        ],
    )


def _rendered(tmp_path: Path) -> tuple[SystemSpec, Path]:
    """Compile + render the plan; return the spec and the destination root ``pkg``."""
    spec = compile_spec(_two_spine_plan())
    render(spec, tmp_path)
    return spec, tmp_path


def _line_in(source: str, class_name: str) -> int:
    """A line strictly inside ``class_name``'s body (its second line)."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node.lineno + 1
    raise AssertionError(f"class {class_name} not found")


def _first_top_level_func_line(source: str) -> int | None:
    """A line inside the first top-level function (a render-owned helper) — or None."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.lineno + 1
    return None


def _nodes_source(spec: SystemSpec, pkg: Path) -> str:
    return (pkg / spec.name / "nodes.py").read_text(encoding="utf-8")


def _filled(*names: str) -> list[FilledNode]:
    return [FilledNode(node=n, attempts=1, body="x = 1\n") for n in names]


# --- the AST helper --------------------------------------------------------------


def test_class_spans_and_owning_class() -> None:
    source = (
        "import os\n"  # line 1 — module scope, no class
        "\n"
        "class Alpha:\n"  # line 3
        "    def run(self):\n"  # 4
        "        return 1\n"  # 5
        "\n"
        "def helper():\n"  # line 7 — top-level function, no class
        "    return 2\n"  # 8
        "\n"
        "class Beta:\n"  # line 10
        "    x = 1\n"  # 11
    )
    spans = _class_spans(source)
    assert set(spans) == {"Alpha", "Beta"}
    assert _owning_class(4, spans) == "Alpha"
    assert _owning_class(5, spans) == "Alpha"
    assert _owning_class(11, spans) == "Beta"
    assert _owning_class(1, spans) == ""  # module scope
    assert _owning_class(8, spans) == ""  # top-level helper → dropped


def test_class_spans_includes_decorator_line() -> None:
    source = "@dataclass\nclass Gamma:\n    x = 1\n"  # decorator on line 1
    spans = _class_spans(source)
    assert _owning_class(1, spans) == "Gamma"


def test_class_spans_syntax_error_is_empty() -> None:
    assert _class_spans("def (: broken") == {}


# --- signal 1: nodes.py-scoped dialect violations --------------------------------


def test_signal1_nodes_py_violation_attributes_to_class(tmp_path: Path) -> None:
    spec, pkg = _rendered(tmp_path)
    source = _nodes_source(spec, pkg)
    line = _line_in(source, _PARSE)
    audit = AuditReport(
        works=False,
        checks=[
            CheckResult(
                name="dialect",
                passed=False,
                violations=[
                    Violation(
                        file=str(pkg / spec.name / "nodes.py"),
                        line=line,
                        law="R2",
                        code="raise NotImplementedError",
                        message="unfilled",
                    )
                ],
            )
        ],
    )
    build = BuildReport(filled=_filled(_PARSE, _NORMALISE))
    assert attribute_failures(audit, build, spec, pkg) == [_PARSE]


def test_signal1_render_owned_violation_is_a_loud_terminal(tmp_path: Path) -> None:
    spec, pkg = _rendered(tmp_path)
    # A violation in a render-owned file (graph.py) is a forge render bug: re-authoring a
    # node body cannot fix it, so it forces a loud terminal (§6.1/§12.4) — NOT a rebuild
    # target and NOT the spine fallback (which would burn a round that cannot help).
    audit = AuditReport(
        works=False,
        checks=[
            CheckResult(
                name="dialect",
                passed=False,
                violations=[
                    Violation(
                        file=str(pkg / spec.name / "graph.py"),
                        line=10,
                        law="L2",
                        code="import engine",
                        message="render-owned",
                    )
                ],
            )
        ],
    )
    build = BuildReport(filled=_filled(_PARSE, _NORMALISE))
    assert attribute_failures(audit, build, spec, pkg) == []


# --- signal 2: failing-test traceback frames -------------------------------------


def test_signal2_traceback_frame_attributes_to_class(tmp_path: Path) -> None:
    spec, pkg = _rendered(tmp_path)
    source = _nodes_source(spec, pkg)
    line = _line_in(source, _NORMALISE)
    raw = (
        "FAILED tests/test_graph.py::test_smoke\n"
        f"{pkg}/{spec.name}/nodes.py:{line}: in run\n"
        "    raise ValueError('boom')\n"
        "=== 0 passed, 1 failed ===\n"
    )
    audit = AuditReport(
        works=False,
        checks=[CheckResult(name="tests", passed=False, raw_output=raw)],
    )
    build = BuildReport(filled=_filled(_PARSE, _NORMALISE))
    assert attribute_failures(audit, build, spec, pkg) == [_NORMALISE]


def test_signal2_helper_frame_maps_to_no_class(tmp_path: Path) -> None:
    spec, pkg = _rendered(tmp_path)
    source = _nodes_source(spec, pkg)
    helper_line = _first_top_level_func_line(source)
    assert helper_line is not None, (
        "expected a render-owned top-level helper in nodes.py"
    )
    raw = f"{pkg}/{spec.name}/nodes.py:{helper_line}: in _user_prompt_helper\n"
    audit = AuditReport(
        works=False,
        checks=[CheckResult(name="tests", passed=False, raw_output=raw)],
    )
    build = BuildReport(filled=_filled(_PARSE, _NORMALISE))
    # The helper frame attributes to no class → attributed set empty → spine fallback.
    assert attribute_failures(audit, build, spec, pkg) == sorted([_PARSE, _NORMALISE])


def test_signal2_test_nodes_py_frame_is_not_matched(tmp_path: Path) -> None:
    spec, pkg = _rendered(tmp_path)
    source = _nodes_source(spec, pkg)
    line = _line_in(source, _PARSE)
    # A frame in `test_nodes.py` must NOT be treated as the generated nodes.py.
    raw = f"{pkg}/{spec.name}/tests/test_nodes.py:{line}: in test_it\n"
    audit = AuditReport(
        works=False,
        checks=[CheckResult(name="tests", passed=False, raw_output=raw)],
    )
    build = BuildReport(filled=_filled(_PARSE, _NORMALISE))
    # No real nodes.py frame → spine fallback, not an attribution to _PARSE from the frame.
    assert attribute_failures(audit, build, spec, pkg) == sorted([_PARSE, _NORMALISE])


# --- import/collection ERROR vs behavioural FAILURE (crash-awareness, review P1) ---


def test_pytest_collection_error_is_a_loud_terminal(tmp_path: Path) -> None:
    spec, pkg = _rendered(tmp_path)
    # A package that fails to import: pytest reports an ERROR, no test ran, no nodes.py
    # frame. Re-authoring a body's logic cannot fix an import failure → loud terminal ([]).
    raw = "ImportError: No module named 'nope'\n=== 1 error in 0.1s ===\n"
    audit = AuditReport(
        works=False,
        checks=[
            CheckResult(
                name="tests",
                passed=False,
                raw_output=raw,
                tests_passed=0,
                tests_failed=0,
                tests_errored=1,
            )
        ],
    )
    build = BuildReport(filled=_filled(_PARSE, _NORMALISE))
    assert attribute_failures(audit, build, spec, pkg) == []


def test_behavioural_failure_still_rebuilds_not_a_loud_terminal(tmp_path: Path) -> None:
    spec, pkg = _rendered(tmp_path)
    # A test ran and asserted false (a node produced a wrong value): pytest reports a
    # FAILURE, not an error. Even with no nodes.py frame, this is fixable by re-authoring,
    # so it must fall through to the spine fallback — NOT be misread as an import failure.
    raw = f"{pkg}/{spec.name}/tests/test_smoke.py:5: in test_it\n=== 1 failed in 0.1s ===\n"
    audit = AuditReport(
        works=False,
        checks=[
            CheckResult(
                name="tests",
                passed=False,
                raw_output=raw,
                tests_passed=0,
                tests_failed=1,
                tests_errored=0,
            )
        ],
    )
    build = BuildReport(filled=_filled(_PARSE, _NORMALISE))
    assert attribute_failures(audit, build, spec, pkg) == sorted([_PARSE, _NORMALISE])


# --- signal 3: BuildConcern nodes ------------------------------------------------


def test_signal3_build_concern_is_targeted(tmp_path: Path) -> None:
    spec, pkg = _rendered(tmp_path)
    audit = AuditReport(works=False)
    build = BuildReport(
        filled=_filled(_PARSE, _NORMALISE),
        concerns=[BuildConcern(node=_NORMALISE, issue="ignores its input")],
    )
    assert attribute_failures(audit, build, spec, pkg) == [_NORMALISE]


# --- signal 4: reconciled critic findings ----------------------------------------


def test_signal4_critic_finding_reconciled_and_targeted(tmp_path: Path) -> None:
    spec, pkg = _rendered(tmp_path)
    audit = AuditReport(
        works=False,
        critic=CriticVerdict(
            issues=[NodeFinding(node="ParseRequest", issue="hardcoded stand-in")]
        ),
    )
    build = BuildReport(filled=_filled(_PARSE, _NORMALISE))
    # "ParseRequest" reconciles (rapidfuzz) to the real node ParseTheRequest.
    assert attribute_failures(audit, build, spec, pkg) == [_PARSE]


def test_signal4_unreconcilable_critic_finding_is_dropped(tmp_path: Path) -> None:
    spec, pkg = _rendered(tmp_path)
    audit = AuditReport(
        works=False,
        critic=CriticVerdict(
            issues=[NodeFinding(node="TotallyUnrelated", issue="who knows")]
        ),
    )
    build = BuildReport(filled=_filled(_PARSE, _NORMALISE))
    # The name does not reconcile → dropped → attributed set empty → spine fallback.
    assert attribute_failures(audit, build, spec, pkg) == sorted([_PARSE, _NORMALISE])


# --- remaining holes + intersection + fallback + terminal ------------------------


def test_remaining_holes_are_unconditionally_targeted(tmp_path: Path) -> None:
    spec, pkg = _rendered(tmp_path)
    audit = AuditReport(works=False)  # no signals at all
    build = BuildReport(
        filled=_filled(_PARSE),
        unfillable=[UnfillableNode(node=_NORMALISE, last_error="boom", attempts=3)],
    )
    # An unfilled hole is definitionally work that needs doing — always a target.
    assert attribute_failures(audit, build, spec, pkg) == [_NORMALISE]


def test_signal_intersected_with_authored_nodes(tmp_path: Path) -> None:
    spec, pkg = _rendered(tmp_path)
    # A concern naming a node that was NOT authored is intersected away.
    audit = AuditReport(works=False)
    build = BuildReport(
        filled=_filled(_PARSE),
        concerns=[BuildConcern(node="GhostNode", issue="not built")],
    )
    # GhostNode ∉ authored_nodes → dropped → spine fallback among authored nodes (_PARSE).
    assert attribute_failures(audit, build, spec, pkg) == [_PARSE]


def test_fallback_spine_nodes_when_red_and_unattributed(tmp_path: Path) -> None:
    spec, pkg = _rendered(tmp_path)
    audit = AuditReport(works=False)  # red, no signals, no concerns
    build = BuildReport(filled=_filled(_PARSE, _NORMALISE, _ANSWER))
    # Fallback ladder terminal tier: all AUTHORED spine nodes (the HEAD _ANSWER excluded).
    assert attribute_failures(audit, build, spec, pkg) == sorted([_PARSE, _NORMALISE])


def test_non_attributable_red_returns_empty(tmp_path: Path) -> None:
    spec, pkg = _rendered(tmp_path)
    # Red audit, no signals, no authored nodes at all → nothing attributable → [].
    audit = AuditReport(works=False)
    build = BuildReport()
    assert attribute_failures(audit, build, spec, pkg) == []


def test_green_audit_returns_empty(tmp_path: Path) -> None:
    spec, pkg = _rendered(tmp_path)
    audit = AuditReport(works=True)
    build = BuildReport(filled=_filled(_PARSE, _NORMALISE))
    assert attribute_failures(audit, build, spec, pkg) == []


def test_result_is_sorted_and_deduplicated(tmp_path: Path) -> None:
    spec, pkg = _rendered(tmp_path)
    # The same node named by two signals appears once; order is sorted.
    audit = AuditReport(
        works=False,
        critic=CriticVerdict(
            issues=[
                NodeFinding(node=_NORMALISE, issue="a"),
                NodeFinding(node=_PARSE, issue="b"),
            ]
        ),
    )
    build = BuildReport(
        filled=_filled(_PARSE, _NORMALISE),
        concerns=[BuildConcern(node=_NORMALISE, issue="dup")],
    )
    assert attribute_failures(audit, build, spec, pkg) == sorted([_PARSE, _NORMALISE])
