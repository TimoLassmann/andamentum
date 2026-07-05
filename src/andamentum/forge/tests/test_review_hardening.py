"""Regression tests for the cycle-1 review-and-fix hardening.

Covers the fixes made in the iterative review pass: the purity-gate bypass closures, the
sandbox's no-silent-downgrade rule, the CLI ``--json`` / summary / error paths, and the
generated package's own ``validate_input`` door.
"""

from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

import pytest

import andamentum.forge.cli as forge_cli
from andamentum.forge import compile_spec, render
from andamentum.forge.astcheck import check_purity
from andamentum.forge.sandbox import PodmanSandbox, SandboxUnavailableError
from andamentum.forge.schemas import (
    DesignPlan,
    ForgeResult,
    ForgeWhy,
    NodeDraft,
    NodeKind,
)


def _plan() -> DesignPlan:
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
                job="Answer the request.",
                kind=NodeKind.HEAD,
                consumes=["parsed_request"],
                produces=["answer"],
            ),
        ],
    )


def _purity(body_src: str) -> list[str]:
    f = Path(tempfile.mktemp(suffix=".py"))
    f.write_text(f"class N:\n    async def run(self, ctx):\n{body_src}\n")
    return check_purity(f, "N", "run", allow_network=False)


# --- purity gate: the closed bypasses (P0/P1 security) ---------------------------


def test_purity_rejects_dynamic_import_in_body() -> None:
    assert _purity(
        "        import importlib\n        importlib.import_module('os')\n        return E()"
    )
    assert _purity(
        "        from importlib import import_module\n        import_module('os')\n        return E()"
    )


def test_purity_rejects_dunder_escape() -> None:
    v = _purity(
        "        x = ().__class__.__bases__[0].__subclasses__()\n        return E()"
    )
    assert v and any("__subclasses__" in m or "__bases__" in m for m in v)


def test_purity_rejects_aliased_banned_builtin() -> None:
    # Aliasing eval to dodge the direct-call check must still be caught.
    v = _purity("        bad = eval\n        return bad('1')")
    assert v and any("eval" in m for m in v)


def test_purity_still_passes_a_legitimate_pure_body() -> None:
    # No false positive: ordinary deterministic logic stays clean.
    assert (
        _purity(
            "        ctx.state.n = len(ctx.state.items.split(','))\n        return Done()"
        )
        == []
    )


def test_purity_allows_a_local_shadowing_a_builtin_name() -> None:
    # A body may bind `input`/`open`/`vars` as ordinary locals — the load of the LOCAL must
    # not fire the aliasing check (cycle-2 regression). The name is bound (assigned), so it
    # shadows the builtin; only an UNBOUND load of a banned builtin (aliasing) is flagged.
    assert (
        _purity(
            "        input = ctx.state.raw_text\n        ctx.state.out = input.strip()\n        return Done()"
        )
        == []
    )
    assert (
        _purity(
            "        open = ctx.state.is_open\n        ctx.state.status = 'open' if open else 'closed'\n        return Done()"
        )
        == []
    )


# --- audit: pytest count parsing is scoped to the summary line (cycle-2 regression) ---


def test_parse_counts_ignores_error_prose_in_assertion_output() -> None:
    from andamentum.forge.audit import _parse_counts

    # A behavioural failure whose assertion message mentions "errors" must NOT be counted
    # as a pytest ERROR (which would misroute it to a loud terminal instead of a rebuild).
    out = "AssertionError: got 2 errors\n=== 1 failed, 4 passed in 0.12s ===\n"
    assert _parse_counts(out) == (4, 1, 0)


def test_parse_counts_reads_a_real_collection_error() -> None:
    from andamentum.forge.audit import _parse_counts

    assert _parse_counts("ImportError: no module\n=== 1 error in 0.10s ===\n") == (
        0,
        0,
        1,
    )


def test_parse_counts_handles_a_bare_summary_line_without_equals() -> None:
    from andamentum.forge.audit import _parse_counts

    # pytest's real summary is ``===``-wrapped, but the duration alone identifies it too.
    assert _parse_counts("nodes.py:5: in run\n0 passed, 3 failed in 0.10s\n") == (
        0,
        3,
        0,
    )


# --- sandbox: no silent downgrade to an unisolated subprocess (P0 security) -------


def test_podman_missing_fails_loud_for_a_pure_run(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("andamentum.forge.sandbox.shutil.which", lambda _name: None)
    with pytest.raises(SandboxUnavailableError):
        PodmanSandbox().run(["python", "-c", "1"], cwd=tmp_path, allow_network=False)


def test_podman_missing_fails_loud_for_a_network_run(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("andamentum.forge.sandbox.shutil.which", lambda _name: None)
    with pytest.raises(SandboxUnavailableError):
        PodmanSandbox().run(["python", "-c", "1"], cwd=tmp_path, allow_network=True)


# --- CLI: --json / human summary / error paths (previously untested) --------------


def _canned_result() -> ForgeResult:
    return ForgeResult(spec=compile_spec(_plan()), stage_reached="design")


def test_cli_json_emits_parseable_forgeresult(monkeypatch, capsys) -> None:
    async def _fake_run(*_a, **_k):
        return _canned_result()

    monkeypatch.setattr(forge_cli, "resolve_model_from_args", lambda _m: "stub")
    monkeypatch.setattr(forge_cli, "run_forge", _fake_run)

    rc = forge_cli.main(["design", "a reading list assistant", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)  # must be clean, parseable JSON
    assert "spec" in payload and payload["stage_reached"] == "design"


def test_cli_human_summary_path(monkeypatch, capsys) -> None:
    async def _fake_run(*_a, **_k):
        return _canned_result()

    monkeypatch.setattr(forge_cli, "resolve_model_from_args", lambda _m: "stub")
    monkeypatch.setattr(forge_cli, "run_forge", _fake_run)

    rc = forge_cli.main(["design", "a reading list assistant"])
    assert rc == 0
    assert "system:" in capsys.readouterr().out  # the human summary, not JSON


def test_cli_error_path_exits_nonzero_without_traceback(monkeypatch, capsys) -> None:
    async def _boom(*_a, **_k):
        raise ValueError("Understand: the model failed to produce valid output")

    monkeypatch.setattr(forge_cli, "resolve_model_from_args", lambda _m: "stub")
    monkeypatch.setattr(forge_cli, "run_forge", _boom)

    rc = forge_cli.main(["design", "a reading list assistant"])
    assert rc == 1
    assert "Error:" in capsys.readouterr().err


# --- generated package: validate_input door (P1 ingestion) ------------------------


def test_generated_validate_input_rejects_blank_and_accepts_text(
    tmp_path: Path,
) -> None:
    spec = compile_spec(_plan())
    render(spec, tmp_path)
    graph_src = (tmp_path / spec.name / "graph.py").read_text()
    # The door is generated and fail-loud: blank raises, valid returns the input model.
    assert "def validate_input(" in graph_src
    tree = ast.parse(graph_src)
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "validate_input"
    )
    body_src = ast.get_source_segment(graph_src, fn) or ""
    assert 'raise ValueError("input must not be blank")' in body_src
    assert "if not text or not text.strip():" in body_src
