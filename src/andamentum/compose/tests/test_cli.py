"""CLI: argument errors, role resolution, genres, status. No model is ever called."""

from __future__ import annotations

import pytest

from andamentum.compose import cli
from andamentum.compose.schemas import Roles


def run(argv: list[str]) -> int:
    return cli.main(argv)


def test_missing_role_model_exits_2_and_names_the_roles(tmp_path, capsys):
    code = run(
        ["write", "Write to Jane Doe", "--out", str(tmp_path / "d"), "--drafter", "m"]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "critic" in err and "judge" in err and "reader" in err
    assert "--model" in err
    assert not (tmp_path / "d").exists()


def test_no_models_at_all_exits_2(tmp_path):
    assert run(["write", "Write to Jane Doe", "--out", str(tmp_path / "d")]) == 2


def test_word_write_needs_only_the_drafter_and_the_judge(tmp_path, capsys):
    out = str(tmp_path / "paper.docx")
    assert run(["write", "t", "--out", out, "--drafter", "m"]) == 2
    err = capsys.readouterr().err
    assert "judge" in err and "critic" not in err and "reader" not in err


def test_flags_of_the_other_destination_are_refused(tmp_path, capsys):
    bundle = ["write", "t", "--out", str(tmp_path / "d"), "--model", "m"]
    assert run([*bundle, "--library", "refs.json"]) == 2
    assert "--library" in capsys.readouterr().err
    word = ["write", "t", "--out", str(tmp_path / "p.docx"), "--model", "m"]
    assert run([*word, "--format", "md"]) == 2
    assert "--format" in capsys.readouterr().err
    assert run([*word, "--offline", "--library", "refs.json"]) == 2
    assert "--offline" in capsys.readouterr().err


def test_missing_required_arguments_exit_2():
    with pytest.raises(SystemExit) as exc:
        run(["write", "task only"])  # --out is required
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        run([])
    assert exc.value.code == 2


def test_unknown_format_exits_2(tmp_path):
    argv = ["write", "t", "--out", str(tmp_path), "--model", "m", "--format", "pdf"]
    assert run(argv) == 2


def test_resolve_roles_model_fills_the_gaps():
    parser = cli.build_parser()
    args = parser.parse_args(["run", "bundle", "--model", "base", "--critic", "big"])
    roles = cli.resolve_roles(args, parser)
    assert roles == Roles(drafter="base", critic="big", judge="base", reader="base")


def test_resolve_roles_falls_back_to_recorded_models():
    parser = cli.build_parser()
    args = parser.parse_args(["check-bundle", "bundle", "--judge", "j"])
    recorded = Roles.all("old")
    roles = cli.resolve_roles(args, parser, fallback=recorded)
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


def test_a_crash_exits_3(tmp_path, capsys):
    # Not a bundle: opening it raises, which the CLI reports with exit 3.
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
    base = ["write", "t", "--out", str(tmp_path / "d"), "--model", "m"]
    for flag, value in (
        ("--research-mode", "web"),
        ("--browser-cdp-url", "http://127.0.0.1:9222"),
        ("--searxng-url", "http://searx.test"),
    ):
        assert run([*base, flag, value]) == 2
        err = capsys.readouterr().err
        assert flag in err and "need --research" in err


def test_research_needs_the_network(tmp_path, capsys):
    word = ["write", "t", "--out", str(tmp_path / "p.docx"), "--model", "m"]
    assert run([*word, "--research", "--offline"]) == 2
    assert "--offline" in capsys.readouterr().err


def test_a_bundle_accepts_mailto_only_for_research(tmp_path, capsys):
    base = ["write", "t", "--out", str(tmp_path / "d"), "--model", "m"]
    assert run([*base, "--mailto", "jane@example.org"]) == 2
    assert "--mailto" in capsys.readouterr().err
