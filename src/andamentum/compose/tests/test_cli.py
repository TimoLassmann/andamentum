"""CLI: argument errors, role resolution, genres, status. No model is ever called."""

from __future__ import annotations

import pytest

from andamentum.compose import cli
from andamentum.compose.schemas import Roles


def run(argv: list[str]) -> int:
    return cli.main(argv)


def test_missing_role_model_exits_2_and_names_the_roles(tmp_path, capsys):
    code = run(
        ["write", "Write to Jane Doe", "-o", str(tmp_path / "d"), "--drafter", "m"]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "critic" in err and "judge" in err and "reader" in err
    assert "--model" in err
    assert not (tmp_path / "d").exists()


def test_no_models_at_all_exits_2(tmp_path):
    assert run(["write", "Write to Jane Doe", "-o", str(tmp_path / "d")]) == 2


def test_missing_required_arguments_exit_2():
    with pytest.raises(SystemExit) as exc:
        run(["write", "task only"])  # --out is required
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        run([])
    assert exc.value.code == 2


def test_unknown_format_exits_2(tmp_path):
    argv = ["write", "t", "-o", str(tmp_path), "--model", "m", "--format", "pdf"]
    with pytest.raises(SystemExit) as exc:
        run(argv)  # not one of the formats argparse offers
    assert exc.value.code == 2


def test_resolve_roles_model_fills_the_gaps():
    parser = cli.build_parser()
    args = parser.parse_args(["continue", "folder", "--model", "base", "--critic", "big"])
    roles = cli.resolve_roles(args)
    assert roles == Roles(drafter="base", critic="big", judge="base", reader="base")


def test_resolve_roles_falls_back_to_recorded_models():
    parser = cli.build_parser()
    args = parser.parse_args(["check", "folder", "--judge", "j"])
    recorded = Roles.all("old")
    roles = cli.resolve_roles(args, fallback=recorded)
    assert roles.judge == "j" and roles.drafter == "old"


def test_genres_lists_all_seven_packs(capsys):
    assert run(["genres"]) == 0
    out = capsys.readouterr().out
    for name in ("email", "memo", "report", "article", "howto", "proposal", "grant"):
        assert name in out


def test_status_of_a_fresh_bundle(tmp_path, capsys):
    from andamentum.compose.bundle import Bundle

    bundle = Bundle.create(tmp_path / "doc.compose", title="Apollo")
    bundle.add_section(title="Body")
    assert run(["status", str(bundle.path)]) == 0
    out = capsys.readouterr().out
    assert "Apollo" in out and "Body" in out and "brief: missing" in out


def test_a_folder_that_is_not_a_document_exits_3(tmp_path, capsys):
    # Not a document folder: an input that cannot be read, exit 3.
    assert run(["status", str(tmp_path)]) == 3
    assert "error" in capsys.readouterr().err


async def test_write_refuses_provenance_off_for_grant_before_any_model_call(tmp_path):
    from andamentum.compose import write_document
    from andamentum.compose.render import ProvenanceRequired

    with pytest.raises(ProvenanceRequired):
        await write_document(
            "Write a grant",
            out=tmp_path / "g",
            roles=Roles.all("m"),
            genre="grant",
            provenance=False,
        )
    assert not (tmp_path / "g").exists()


async def test_write_validates_arguments_before_creating_anything(tmp_path):
    from andamentum.compose import write_document

    roles = Roles.all("m")
    with pytest.raises(ValueError, match="empty"):
        await write_document(" ", out=tmp_path / "a", roles=roles)
    with pytest.raises(ValueError, match="max_rounds"):
        await write_document("t", out=tmp_path / "a", roles=roles, max_rounds=0)
    with pytest.raises(ValueError, match="format"):
        await write_document("t", out=tmp_path / "a", roles=roles, formats=["pdf"])
    with pytest.raises(ValueError, match="unknown genre"):
        await write_document("t", out=tmp_path / "a", roles=roles, genre="poem")
    assert not (tmp_path / "a").exists()


def test_research_flags_need_research(tmp_path, capsys):
    base = ["write", "t", "-o", str(tmp_path / "d"), "--model", "m"]
    for flag, value in (
        ("--research-mode", ["web"]),
        ("--browser", []),
        ("--searxng-url", ["http://searx.test"]),
    ):
        assert run([*base, flag, *value]) == 2
        err = capsys.readouterr().err
        assert flag in err and "need --research" in err


def test_mailto_is_a_research_flag_only(tmp_path, capsys):
    base = ["write", "t", "-o", str(tmp_path / "d"), "--model", "m"]
    assert run([*base, "--mailto", "jane@example.org"]) == 2
    assert "--mailto" in capsys.readouterr().err


def test_write_puts_files_only_into_the_out_folder(tmp_path, monkeypatch):
    """A (stubbed) write creates files only inside the folder named by --out."""
    from andamentum.compose import api
    from andamentum.compose.schemas import ComposeResult

    async def fake_graph(deps):
        (deps.bundle_dir / "report.md").write_text("report", encoding="utf-8")
        return ComposeResult.model_construct(
            folder=deps.bundle_dir,
            outputs=[],
            goal_met=True,
            stop_reason="goal_met",
            rounds=[],
            open_questions=[],
            brief=None,
            usage={},
        )

    monkeypatch.setattr(api, "run_graph", fake_graph)
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    out = tmp_path / "elsewhere" / "doc"
    before = {p for p in tmp_path.rglob("*")}
    assert run(["write", "A memo", "-o", str(out), "--model", "m"]) == 0
    created = {p for p in tmp_path.rglob("*")} - before
    assert created and all(out == p or out in p.parents or p == out.parent for p in created)
    assert list(work.iterdir()) == []


def test_exit_one_when_the_goal_is_not_met(tmp_path, monkeypatch):
    from andamentum.compose import api
    from andamentum.compose.schemas import ComposeResult

    async def fake_graph(deps):
        return ComposeResult.model_construct(
            folder=deps.bundle_dir,
            outputs=[],
            goal_met=False,
            stop_reason="cap_reached",
            rounds=[],
            open_questions=[],
            brief=None,
            usage={},
        )

    monkeypatch.setattr(api, "run_graph", fake_graph)
    assert run(["write", "A memo", "-o", str(tmp_path / "d"), "--model", "m"]) == 1


def test_a_missing_source_exits_3(tmp_path, capsys):
    argv = ["write", "t", "-o", str(tmp_path / "d"), "--model", "m"]
    assert run([*argv, "--source", str(tmp_path / "nope.md")]) == 3
    assert "does not exist" in capsys.readouterr().err
    assert not (tmp_path / "d").exists()


def test_a_crash_exits_4(tmp_path, monkeypatch, capsys):
    from andamentum.compose import api

    async def boom(deps):
        raise RuntimeError("model unreachable")

    monkeypatch.setattr(api, "run_graph", boom)
    assert run(["write", "t", "-o", str(tmp_path / "d"), "--model", "m"]) == 4
    assert "error: RuntimeError: model unreachable" in capsys.readouterr().err
