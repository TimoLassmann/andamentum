"""CLI: argument errors, the flags, recorded settings, genres, status. No model is
ever called."""

from __future__ import annotations

import argparse

import pytest

from andamentum.compose import cli
from andamentum.compose.schemas import Roles


def run(argv: list[str]) -> int:
    return cli.main(argv)


def _flags(command: str) -> set[str]:
    parser = cli.build_parser()
    sub = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    ).choices[command]
    return {o for a in sub._actions for o in a.option_strings if o.startswith("--")}


# Mutation: add a flag back (a role model, --max-rounds, --research-mode, or
# --format on continue) and the matching set fails.
EXPECTED_FLAGS = {
    "write": {
        "--help", "--out", "--genre", "--source", "--confirm-own-sources",
        "--research", "--searxng-url", "--browser", "--mailto", "--model",
        "--format", "--no-provenance", "--json",
    },
    "continue": {
        "--help", "--confirm-own-sources", "--searxng-url", "--browser",
        "--mailto", "--model", "--json",
    },
    "check": {"--help", "--model", "--json"},
    "eval": {"--help", "--genre", "--out", "--model"},
}  # fmt: skip


@pytest.mark.parametrize("command", sorted(EXPECTED_FLAGS))
def test_the_model_commands_offer_exactly_these_flags(command):
    assert _flags(command) == EXPECTED_FLAGS[command]


@pytest.mark.parametrize(
    "argv",
    [
        ["write", "t", "-o", "d"],
        ["continue", "d"],
        ["check", "d"],
        ["eval", "-o", "d"],
    ],
)
def test_model_is_required(argv):
    with pytest.raises(SystemExit) as exc:
        run(argv)
    assert exc.value.code == 2


def test_research_always_takes_a_place_to_search():
    parser = cli.build_parser()
    base = ["write", "t", "-o", "d", "--model", "m"]
    assert parser.parse_args(base).research is None
    assert parser.parse_args([*base, "--research", "auto"]).research == "auto"
    # A value is required, so the flag cannot swallow the task by its position.
    with pytest.raises(SystemExit):
        parser.parse_args(["write", "-o", "d", "--model", "m", "--research", "t"])
    with pytest.raises(SystemExit):
        parser.parse_args([*base, "--research"])
    assert parser.parse_args([*base, "--research", "web"]).research == "web"
    with pytest.raises(SystemExit):
        parser.parse_args([*base, "--research", "library"])


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
        ("--browser", []),
        ("--searxng-url", ["http://searx.test"]),
    ):
        assert run([*base, flag, *value]) == 2
        err = capsys.readouterr().err
        assert flag in err and "only works with --research" in err


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
    assert created and all(
        out == p or out in p.parents or p == out.parent for p in created
    )
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


def _capture_graph(monkeypatch) -> list:
    """Replace the graph with a stub that records its deps and meets the goal."""
    from andamentum.compose import api
    from andamentum.compose.schemas import ComposeResult

    seen: list = []

    async def fake_graph(deps):
        seen.append(deps)
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
    return seen


def test_research_auto_leaves_the_place_to_the_genre(tmp_path, monkeypatch):
    seen = _capture_graph(monkeypatch)
    argv = ["write", "A memo", "-o", str(tmp_path / "a"), "--model", "m"]
    assert run([*argv, "--research", "auto"]) == 0
    assert seen[-1].research is not None and seen[-1].research.mode is None
    argv[3] = str(tmp_path / "b")
    assert run([*argv, "--research", "scholarly"]) == 0
    assert seen[-1].research.mode == "scholarly"
    argv[3] = str(tmp_path / "c")
    assert run(argv) == 0
    assert seen[-1].research is None


def test_write_records_its_settings_and_continue_repeats_them(tmp_path, monkeypatch):
    """Mutation: drop a setting from the record, or let continue use defaults
    instead of the record, and continue runs differently from write."""
    from andamentum.compose.api import recorded_settings

    seen = _capture_graph(monkeypatch)
    folder = tmp_path / "memo"
    argv = ["write", "A memo", "-o", str(folder), "--model", "m"]
    argv += ["--format", "md", "--format", "txt", "--no-provenance"]
    argv += ["--research", "scholarly", "--confirm-own-sources"]
    assert run(argv) == 0
    settings = recorded_settings(folder)
    assert settings.research and settings.research_mode == "scholarly"
    assert settings.formats == ["md", "txt"]
    assert settings.provenance is False and settings.confirm_own_sources is True

    assert run(["continue", str(folder), "--model", "other"]) == 0
    first, again = seen
    for deps in (first, again):
        assert deps.formats == ("md", "txt")
        assert deps.provenance is False
        assert deps.confirm_own_sources is True
        assert deps.research_requested and deps.research.mode == "scholarly"
    assert again.roles == Roles.all("other")


def test_continue_without_research_refuses_research_flags(
    tmp_path, monkeypatch, capsys
):
    _capture_graph(monkeypatch)
    folder = tmp_path / "memo"
    assert run(["write", "A memo", "-o", str(folder), "--model", "m"]) == 0
    assert run(["continue", str(folder), "--model", "m", "--browser"]) == 2
    err = capsys.readouterr().err
    assert "written without research" in err and "--browser" in err
    assert "--research" not in err  # continue has no such flag


def test_continue_takes_confirm_own_sources_when_write_did_not(tmp_path, monkeypatch):
    seen = _capture_graph(monkeypatch)
    folder = tmp_path / "memo"
    assert run(["write", "A memo", "-o", str(folder), "--model", "m"]) == 0
    assert seen[-1].confirm_own_sources is False
    argv = ["continue", str(folder), "--model", "m", "--confirm-own-sources"]
    assert run(argv) == 0
    assert seen[-1].confirm_own_sources is True


def test_continue_refuses_a_folder_without_recorded_settings(tmp_path, capsys):
    from andamentum.compose.bundle import Bundle

    folder = Bundle.create(tmp_path / "old", title="Apollo", metadata={"task": "t"})
    assert run(["continue", str(folder.path), "--model", "m"]) == 3
    err = capsys.readouterr().err
    assert err.startswith("error:") and "records no settings" in err
