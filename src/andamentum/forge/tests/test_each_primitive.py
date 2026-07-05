"""Offline, model-free proof of the map-over-items ('each') primitive.

The failure this primitive removes: forge's data grammar was scalar-only, so a brief like
"summarise EACH article and combine" collapsed the list to one string and the generated
workflow processed ONE item. The fix follows the house philosophy — THE MODEL SELECTS FROM
CLOSED CHOICES (``mode: whole | each`` per node, ``input_is_collection`` on the why);
DETERMINISTIC CODE DOES THE HEAVY LIFTING:

  - collection-ness of every datum is COMPUTED (``assemble.collection_data``), never
    declared per-datum — field annotations (``list[str]``) follow it;
  - ``diagnose`` flags an 'each' node whose inputs are not exactly one collection
    (``each_needs_collection``), feeding the existing repair loop; ``compile_spec``
    keeps the same rule as a fail-loud backstop;
  - the RENDERER writes the whole iteration scaffold (max_items bound, per-item
    soft-fail into item_failures, all-fail raise, declaration-order join) — an EACH
    head is fully rendered, an EACH spine leaves ONE hole, the pure per-item transform
    ``_map_one(item)``, gated by its own simplified contract (uses `item`, returns a
    value, no ctx at all).

All deterministic — no live model, no container; the e2e drives the real pipeline
through the ScriptedSink and executes the generated package's own tests.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from andamentum.forge import compile_spec, render, run_forge, verify_package
from andamentum.forge.assemble import assemble, collection_data
from andamentum.forge.astcheck import check_map_item_body
from andamentum.forge.contract import HoleKind
from andamentum.forge.diagnose import diagnose
from andamentum.forge.extract import discover_holes
from andamentum.forge.schemas import (
    DataKind,
    DesignPlan,
    FindingKind,
    ForgeWhy,
    NodeDraft,
)
from andamentum.forge.spec import NodeControl, NodeKind, NodeMode

from .conftest import FakeSandbox, NodeScript, ScriptedSink


def _why(*, collection: bool) -> ForgeWhy:
    return ForgeWhy(
        purpose="Summarize each article and combine the summaries.",
        boundary_in="a list of article texts",
        boundary_out="one combined summary",
        input_is_collection=collection,
    )


def _draft(
    node_id: str,
    job: str,
    *,
    kind: NodeKind = NodeKind.SPINE,
    consumes: list[str],
    produces: list[str],
    mode: NodeMode = NodeMode.WHOLE,
    produces_kind: DataKind = DataKind.SIGNAL,
    control: NodeControl = NodeControl.NONE,
) -> NodeDraft:
    return NodeDraft(
        id=node_id,
        area="core",
        job=job,
        kind=kind,
        consumes=consumes,
        produces=produces,
        mode=mode,
        produces_kind=produces_kind,
        control=control,
    )


def _map_plan(*, collection: bool = True) -> DesignPlan:
    """The canonical map shape: EACH spine → EACH head → WHOLE combine head."""
    return DesignPlan(
        why=_why(collection=collection),
        nodes=[
            _draft(
                "n1",
                "Normalise each article.",
                consumes=["input"],
                produces=["normalised_articles"],
                mode=NodeMode.EACH,
            ),
            _draft(
                "n2",
                "Summarise each article.",
                kind=NodeKind.HEAD,
                consumes=["normalised_articles"],
                produces=["article_summaries"],
                mode=NodeMode.EACH,
            ),
            _draft(
                "n3",
                "Combine the summaries.",
                kind=NodeKind.HEAD,
                consumes=["article_summaries"],
                produces=["combined_summary"],
            ),
        ],
    )


# --- collection propagation: computed, never declared per-datum -------------------------


def test_collection_data_is_computed_from_flag_and_modes() -> None:
    nodes = _map_plan().nodes
    # input flagged a collection → every input token is a collection; EACH produces are
    # collections; the WHOLE combine's produce is a scalar.
    coll = collection_data(nodes, input_is_collection=True)
    assert "input" in coll
    assert "normalised_articles" in coll and "article_summaries" in coll
    assert "combined_summary" not in coll
    # flag off → the input is a scalar; the EACH produces are still lists.
    coll_off = collection_data(nodes, input_is_collection=False)
    assert "input" not in coll_off
    assert "normalised_articles" in coll_off


def test_compiled_spec_annotations_follow_collection_ness() -> None:
    spec = compile_spec(_map_plan())
    # Input model: the primary field became a list of items.
    primary = next(
        f for f in spec.input.model.fields if f.name == spec.input.primary_text_field
    )
    assert primary.annotation == "list[str]"
    # State: EACH produces are list[str]; the WHOLE produce is str; the shared per-item
    # failure log exists because the system maps.
    ann = {f.name: f.annotation for f in spec.state.fields}
    assert ann["normalised_articles"] == "list[str]"
    assert ann["article_summaries"] == "list[str]"
    assert ann["combined_summary"] == "str"
    assert ann["item_failures"] == "list[str]"
    # The mode rides onto the NodeSpec (the renderer keys the scaffold off it).
    modes = {n.consumes[0]: n.mode for n in spec.nodes if n.consumes}
    assert modes["input"] is NodeMode.EACH
    assert modes["article_summaries"] is NodeMode.WHOLE


def test_scalar_system_is_unchanged(tmp_path: Path) -> None:
    # No EACH node, no collection flag → nothing about the map primitive leaks in.
    plan = DesignPlan(
        why=ForgeWhy(purpose="Do a thing.", boundary_in="x", boundary_out="y"),
        nodes=[
            _draft("n1", "Parse the request.", consumes=["input"], produces=["parsed"]),
            _draft(
                "n2",
                "Answer the request.",
                kind=NodeKind.HEAD,
                consumes=["parsed"],
                produces=["answer"],
            ),
        ],
    )
    spec = compile_spec(plan)
    primary = next(
        f for f in spec.input.model.fields if f.name == spec.input.primary_text_field
    )
    assert primary.annotation == "str"
    assert all(f.name != "item_failures" for f in spec.state.fields)
    render(spec, tmp_path)
    pkg = tmp_path / spec.name
    assert "max_items" not in (pkg / "deps.py").read_text()
    assert "asyncio.gather" not in (pkg / "nodes.py").read_text()


# --- the rendered scaffold: deterministic code owns the iteration -----------------------


def test_each_spine_scaffold_is_rendered_and_hole_is_map_one(tmp_path: Path) -> None:
    spec = compile_spec(_map_plan())
    render(spec, tmp_path)
    pkg = tmp_path / spec.name
    src = (pkg / "nodes.py").read_text()

    each_spine = next(
        n for n in spec.nodes if n.mode is NodeMode.EACH and n.kind is NodeKind.SPINE
    )
    # The scaffold: gather over the per-item hole, bounded, soft-fail, all-fail raise.
    assert "*(self._map_one(item) for item in items), return_exceptions=True" in src
    assert "if len(items) > ctx.deps.max_items:" in src
    assert "ctx.state.item_failures.append(" in src
    assert f'"{each_spine.name}: every item failed: "' in src
    # The ONLY hole on the node is `_map_one` — run() itself is fully rendered.
    holes = discover_holes(pkg)
    spine_holes = [h for h in holes if h.node == each_spine.name]
    assert [h.method for h in spine_holes] == ["_map_one"]
    assert spine_holes[0].kind is HoleKind.MAP_ITEM
    import ast

    tree = ast.parse(src)
    cls = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == each_spine.name
    )
    run = next(
        m for m in cls.body if isinstance(m, ast.AsyncFunctionDef) and m.name == "run"
    )
    assert not any(
        isinstance(n, ast.Raise)
        and isinstance(n.exc, ast.Call)
        and getattr(n.exc.func, "id", "") == "NotImplementedError"
        for n in ast.walk(run)
    ), "an EACH spine's run() must be fully rendered — the hole is _map_one only"


def test_each_head_is_fully_rendered_with_per_item_prompt(tmp_path: Path) -> None:
    spec = compile_spec(_map_plan())
    render(spec, tmp_path)
    pkg = tmp_path / spec.name
    src = (pkg / "nodes.py").read_text()

    each_head = next(
        n for n in spec.nodes if n.mode is NodeMode.EACH and n.kind is NodeKind.HEAD
    )
    agent = each_head.agent
    # Fully rendered: run_head per item with the labelled per-item prompt — no hole.
    assert f"_item_prompt_{agent}(item)" in src
    assert f"def _item_prompt_{agent}(item: str) -> str:" in src
    assert 'return f"Item:\\n{item}"' in src
    assert all(h.node != each_head.name for h in discover_holes(pkg))


def test_whole_head_sees_collection_items_labelled(tmp_path: Path) -> None:
    spec = compile_spec(_map_plan())
    render(spec, tmp_path)
    src = (tmp_path / spec.name / "nodes.py").read_text()
    # The combine head reads a collection → its user prompt goes through the numbered
    # item helper, never a Python-list repr.
    assert "def _label_items(label: str, items: list[str]) -> str:" in src
    assert '_label_items("Article summaries", ctx.state.article_summaries)' in src


def test_map_package_renders_verifies_and_stays_dialect_clean(tmp_path: Path) -> None:
    from andamentum.agentic_dialect import check_code

    spec = compile_spec(_map_plan())
    render(spec, tmp_path)
    report = verify_package(spec, tmp_path)
    failed = [f"{c.name}: {c.detail}" for c in report.checks if not c.passed]
    assert report.works, failed
    assert not check_code(tmp_path / spec.name)
    # The Deps bound is provisioned: named constant + field, and the entry splits lines.
    deps_src = (tmp_path / spec.name / "deps.py").read_text()
    assert "MAX_ITEMS = 50" in deps_src
    assert "max_items: int = MAX_ITEMS" in deps_src
    graph_src = (tmp_path / spec.name / "graph.py").read_text()
    assert "text.splitlines()" in graph_src


# --- diagnose: each_needs_collection feeds the repair loop ------------------------------


def test_diagnose_flags_each_consuming_a_scalar() -> None:
    # n1 (WHOLE) produces a scalar; n2 (EACH) maps over it — the error case.
    nodes = [
        _draft("n1", "Extract the topic.", consumes=["input"], produces=["topic"]),
        _draft(
            "n2",
            "Summarise each item.",
            consumes=["topic"],
            produces=["summaries"],
            mode=NodeMode.EACH,
        ),
    ]
    report = diagnose(nodes, assemble(nodes), input_is_collection=False)
    kinds = {f.kind for f in report.findings}
    assert FindingKind.EACH_NEEDS_COLLECTION in kinds
    finding = next(
        f for f in report.findings if f.kind is FindingKind.EACH_NEEDS_COLLECTION
    )
    assert finding.node == "n2"
    assert "'whole'" in finding.suggestion  # the concrete fix is named


def test_diagnose_clean_each_and_whole_reduce() -> None:
    # EACH over the (collection) input, then a WHOLE reduce over the EACH output —
    # both legitimate; no finding.
    nodes = [
        _draft(
            "n1",
            "Summarise each article.",
            consumes=["input"],
            produces=["summaries"],
            mode=NodeMode.EACH,
        ),
        _draft(
            "n2", "Combine the summaries.", consumes=["summaries"], produces=["answer"]
        ),
    ]
    report = diagnose(nodes, assemble(nodes), input_is_collection=True)
    assert FindingKind.EACH_NEEDS_COLLECTION not in {f.kind for f in report.findings}


def test_diagnose_flags_each_with_extra_scalar_context() -> None:
    # v1: an EACH node reads its ONE stream and nothing else — a scalar rider is flagged.
    nodes = [
        _draft("n1", "Extract the tone.", consumes=["input"], produces=["tone"]),
        _draft(
            "n2",
            "Rewrite each note.",
            consumes=["input", "tone"],
            produces=["rewritten_notes"],
            mode=NodeMode.EACH,
        ),
        _draft(
            "n3",
            "Combine the notes.",
            consumes=["rewritten_notes", "tone"],
            produces=["answer"],
        ),
    ]
    report = diagnose(nodes, assemble(nodes), input_is_collection=True)
    flagged = [
        f for f in report.findings if f.kind is FindingKind.EACH_NEEDS_COLLECTION
    ]
    assert [f.node for f in flagged] == ["n2"]
    assert flagged[0].variable == "tone"


# --- compile backstops: fail loud, mirroring diagnose -----------------------------------


def test_compile_rejects_each_consuming_a_scalar() -> None:
    plan = DesignPlan(
        why=_why(collection=False),
        nodes=[
            _draft("n1", "Extract the topic.", consumes=["input"], produces=["topic"]),
            _draft(
                "n2",
                "Summarise each item.",
                consumes=["topic"],
                produces=["summaries"],
                mode=NodeMode.EACH,
            ),
        ],
    )
    with pytest.raises(ValueError, match="exactly ONE"):
        compile_spec(plan)


def test_compile_rejects_each_with_control_or_entity() -> None:
    checkpoint = DesignPlan(
        why=_why(collection=True),
        nodes=[
            _draft(
                "n1",
                "Process each item again.",
                consumes=["input"],
                produces=["processed"],
                mode=NodeMode.EACH,
                control=NodeControl.CHECKPOINT,
            ),
        ],
    )
    with pytest.raises(ValueError, match="control="):
        compile_spec(checkpoint)

    entity = DesignPlan(
        why=_why(collection=True),
        nodes=[
            _draft(
                "n1",
                "Store each item.",
                consumes=["input"],
                produces=["stored_items"],
                mode=NodeMode.EACH,
                produces_kind=DataKind.ENTITY,
            ),
        ],
    )
    with pytest.raises(ValueError, match="entity"):
        compile_spec(entity)


# --- the MAP_ITEM gate: uses `item`, returns a value, no ctx at all ---------------------


def _map_item_violations(tmp_path: Path, body: str) -> list[str]:
    file = tmp_path / "probe.py"
    indented = "".join(f"        {line}\n" for line in body.splitlines())
    file.write_text(
        "class N:\n    async def _map_one(self, item: str) -> str:\n" + indented
    )
    return check_map_item_body(file, "N", "_map_one")


def test_map_item_gate_rejects_ignoring_item(tmp_path: Path) -> None:
    violations = _map_item_violations(tmp_path, 'return "constant"')
    assert any("`item` parameter" in v for v in violations)


def test_map_item_gate_rejects_ctx_and_self(tmp_path: Path) -> None:
    violations = _map_item_violations(tmp_path, "return item + ctx.state.answer")
    assert any("references ctx" in v for v in violations)
    violations = _map_item_violations(tmp_path, "return item + self.extra")
    assert any("references self" in v for v in violations)


def test_map_item_gate_rejects_missing_return(tmp_path: Path) -> None:
    violations = _map_item_violations(tmp_path, "cleaned = item.strip()")
    assert any("returns a value" in v for v in violations)


def test_map_item_gate_accepts_a_pure_transform(tmp_path: Path) -> None:
    assert _map_item_violations(tmp_path, "return item.strip().lower()") == []


# --- offline end-to-end: brief → design → render → build, through the real pipeline -----


def _map_sink() -> ScriptedSink:
    return ScriptedSink(
        why=_why(collection=True),
        areas=["core"],
        jobs_by_area={
            "core": [
                "Normalise each article.",
                "Summarise each article.",
                "Combine the summaries.",
            ]
        },
        typings={
            "n1": NodeScript(
                kind=NodeKind.SPINE,
                consumes=["input"],
                produces=["normalised_articles"],
                mode=NodeMode.EACH,
            ),
            "n2": NodeScript(
                kind=NodeKind.HEAD,
                consumes=["normalised_articles"],
                produces=["article_summaries"],
                mode=NodeMode.EACH,
            ),
            "n3": NodeScript(
                kind=NodeKind.HEAD,
                consumes=["article_summaries"],
                produces=["combined_summary"],
            ),
        },
    )


async def test_map_brief_builds_end_to_end(tmp_path: Path) -> None:
    out_dir = tmp_path / "pkg"
    result = await run_forge(
        "Given a list of article texts, summarise each and combine the summaries.",
        model="test",
        dest=out_dir,
        stop_after="build",
        sink=_map_sink(),
        sandbox=FakeSandbox(),
    )
    assert result.build is not None and result.build.all_filled, result.build
    spec = result.spec
    pkg = out_dir / spec.name
    src = (pkg / "nodes.py").read_text()

    # The scaffold survived the build (gather, bound, soft-fail) and the ONE hole —
    # the EACH spine's `_map_one` — was authored: no NotImplementedError remains.
    assert "asyncio.gather" in src
    assert "ctx.deps.max_items" in src
    assert "item_failures" in src
    assert (
        "raise NotImplementedError" not in src
    )  # (the module docstring mentions the word)
    filled = {f.node for f in result.build.filled}
    each_spine = next(
        n for n in spec.nodes if n.mode is NodeMode.EACH and n.kind is NodeKind.SPINE
    )
    assert each_spine.name in filled

    # Execute the generated package's OWN tests (assembly + recipe + stub-driven smoke):
    # the map scaffolds actually run — two seeded items flow through EACH spine, EACH
    # head, and the WHOLE combine to End. Real execution, still no live model.
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(pkg / "tests")],
        cwd=out_dir,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
