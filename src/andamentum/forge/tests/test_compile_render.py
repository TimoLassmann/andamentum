"""compile_spec → render → verify over a hand-built design board (no model)."""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.forge import compile_spec, render, verify_package
from andamentum.forge.schemas import DataKind, DesignPlan, ForgeWhy, NodeDraft
from andamentum.forge.spec import NodeKind


def test_dangling_read_fails_loud() -> None:
    # A read that no node produces must RAISE — never be silently dropped or rewired to
    # the input. A system that runs but doesn't pass its data is worse than one that stops.
    plan = DesignPlan(
        why=ForgeWhy(purpose="Do a thing.", boundary_in="x", boundary_out="y"),
        nodes=[
            NodeDraft(
                id="n1",
                area="core",
                job="Step one.",
                kind=NodeKind.SPINE,
                consumes=["input"],
                produces=["alpha"],
            ),
            NodeDraft(
                id="n2",
                area="core",
                job="Step two.",
                kind=NodeKind.SPINE,
                consumes=["beta"],
                produces=["gamma"],
            ),  # 'beta' is produced by no one
        ],
    )
    with pytest.raises(ValueError, match="no node produces"):
        compile_spec(plan)


def test_duplicate_producer_fails_loud() -> None:
    # Two nodes writing one signal would make the last write silently win — reject it.
    plan = DesignPlan(
        why=ForgeWhy(purpose="Do a thing.", boundary_in="x", boundary_out="y"),
        nodes=[
            NodeDraft(
                id="n1",
                area="core",
                job="Step one.",
                kind=NodeKind.SPINE,
                consumes=["input"],
                produces=["x"],
            ),
            NodeDraft(
                id="n2",
                area="core",
                job="Step two.",
                kind=NodeKind.SPINE,
                consumes=["input"],
                produces=["x"],
            ),  # second writer of x
        ],
    )
    with pytest.raises(ValueError, match="single-writer"):
        compile_spec(plan)


def test_orphan_output_fails_loud() -> None:
    # A signal produced but read by nobody and not the system output is discarded work.
    plan = DesignPlan(
        why=ForgeWhy(purpose="Do a thing.", boundary_in="x", boundary_out="y"),
        nodes=[
            NodeDraft(
                id="n1",
                area="core",
                job="Step one.",
                kind=NodeKind.SPINE,
                consumes=["input"],
                produces=["alpha"],
            ),  # alpha is never read and isn't the output
            NodeDraft(
                id="n2",
                area="core",
                job="Step two.",
                kind=NodeKind.SPINE,
                consumes=["input"],
                produces=["beta"],
            ),
        ],
    )
    with pytest.raises(ValueError, match="orphan"):
        compile_spec(plan)


def test_judgment_over_text_is_promoted_to_head() -> None:
    # "Summarize the text" reads prose and judges meaning → it must be a HEAD (LLM), not a
    # deterministic spine body that would fake the summary with string slicing.
    plan = DesignPlan(
        why=ForgeWhy(
            purpose="Summarize text.", boundary_in="text", boundary_out="summary"
        ),
        nodes=[
            NodeDraft(
                id="n1",
                area="core",
                job="Summarize the text.",
                kind=NodeKind.SPINE,
                consumes=["input"],
                produces=["summary"],
            ),
        ],
    )
    spec = compile_spec(plan)
    assert spec.nodes[0].kind.value == "head", (
        "a summarize-over-text node must be promoted to head"
    )
    assert len(spec.agents) == 1


def _plan() -> DesignPlan:
    return DesignPlan(
        why=ForgeWhy(
            purpose="Help the user manage a personal reading list.",
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
                produces_kind=DataKind.SIGNAL,
            ),
            NodeDraft(
                id="n2",
                area="core",
                job="Answer the request.",
                kind=NodeKind.HEAD,
                consumes=["parsed_request"],
                produces=["answer"],
                produces_kind=DataKind.SIGNAL,
            ),
        ],
    )


def test_compile_produces_a_valid_spec() -> None:
    spec = compile_spec(_plan())
    assert spec.name  # derived snake_case import name
    assert len(spec.nodes) == 2
    assert len(spec.agents) == 1  # one head → one agent
    assert spec.entry_node  # reachability enforced by SystemSpec validators


def test_render_then_verify_works(tmp_path: Path) -> None:
    spec = compile_spec(_plan())
    files = render(spec, tmp_path)
    assert files
    assert (tmp_path / spec.name / "graph.py").exists()

    report = verify_package(spec, tmp_path)
    failed = [f"{c.name}: {c.detail}" for c in report.checks if not c.passed]
    assert report.works, failed
    assert report.score == 1.0


def test_render_emits_a_dialect_clean_cli_launcher(tmp_path: Path) -> None:
    # render emits __main__.py so the system runs as `python -m <name>` — and the launcher
    # is an adapter (imports the run entry, not the engine), so it stays dialect-clean.
    from andamentum.agentic_dialect import check_code

    spec = compile_spec(_plan())
    render(spec, tmp_path)
    pkg = tmp_path / spec.name
    main = pkg / "__main__.py"
    assert main.exists()
    src = main.read_text()
    assert f"run_{spec.name}" in src  # delegates to the package's run entry
    assert "required=True" in src  # --model has no hidden default
    assert not check_code(
        pkg
    )  # the whole package, launcher included, stays dialect-clean


def test_rendered_cli_is_runnable_as_a_module(tmp_path: Path) -> None:
    # `python -m <name> --help` must import the package and wire argparse without a model
    # call — proof the launcher is actually runnable, not just present.
    import subprocess
    import sys

    spec = compile_spec(_plan())
    render(spec, tmp_path)
    proc = subprocess.run(
        [sys.executable, "-m", spec.name, "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--model" in proc.stdout

    # Missing the required --model fails loud (argparse exit 2), never a silent default.
    missing = subprocess.run(
        [sys.executable, "-m", spec.name, "some text"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert missing.returncode != 0


def test_render_provisions_the_store(tmp_path: Path) -> None:
    # render wires the cross-run-memory Store onto Deps (default in-memory), threads a
    # --store path through the run entry, and the deps gate admits ctx.deps.store.
    from andamentum.forge.astcheck import check_deps_access
    from andamentum.forge.build import _declared_deps

    spec = compile_spec(_plan())
    render(spec, tmp_path)
    pkg = tmp_path / spec.name

    deps_src = (pkg / "deps.py").read_text()
    assert "from andamentum.forge.runtime import Store" in deps_src
    assert "store: Store = field(default_factory=Store)" in deps_src

    graph_src = (pkg / "graph.py").read_text()
    assert "store: str | None = None" in graph_src  # run entry takes a path
    assert "store=Store(store)" in graph_src  # resolved once, above the nodes

    # The deps gate reads the allowed set off deps.py, so it now permits ctx.deps.store
    # (and still forbids any undeclared handle).
    assert "store" in _declared_deps(pkg)
    probe = tmp_path / "probe.py"
    probe.write_text(
        "class N:\n    async def run(self, ctx):\n"
        "        _ = ctx.deps.store\n        return None\n"
    )
    assert check_deps_access(probe, "N", "run", allowed=_declared_deps(pkg)) == []
