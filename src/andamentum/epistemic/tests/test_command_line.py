"""The ``andamentum-epistemic`` command line and the ``ask`` / ``verify`` entry points."""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.epistemic import ask_verify, cli
from andamentum.epistemic.operations_runner import PipelineResult
from andamentum.epistemic.result_models import AskResult


def _search() -> list[str]:
    return ["--searxng-url", "http://searx.test"]


@pytest.mark.parametrize("command", ["ask", "verify"])
def test_a_model_and_an_embedding_model_are_required(command, capsys):
    base = [command, "text", *_search()]
    for argv in (
        base,  # neither
        [*base, "--model", "m"],  # no embedding model
        [*base, "--embedding-model", "e"],  # no model
    ):
        with pytest.raises(SystemExit) as raised:
            cli.main(argv)
        assert raised.value.code == 2
    capsys.readouterr()


@pytest.mark.parametrize("command", ["ask", "verify"])
def test_the_environment_does_not_supply_the_model(command, monkeypatch, capsys):
    """Mutation: make `--model` fall back to ANDAMENTUM_MAIN_LLM_MODEL (required=False,
    default from os.environ) and this fails, because exit 2 and the `--model` error vanish.
    Everything else is given, so only the model can be the missing piece."""
    monkeypatch.setenv("ANDAMENTUM_MAIN_LLM_MODEL", "anthropic:claude-haiku-4-5")
    with pytest.raises(SystemExit) as raised:
        cli.main([command, "text", "--embedding-model", "e", *_search()])
    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert "--model" in err and "--embedding-model" not in err.split("required:")[-1]


def test_ask_without_a_search_address_is_a_usage_error(capsys):
    code = cli.main(["ask", "q", "--model", "m", "--embedding-model", "e"])
    assert code == 2
    assert capsys.readouterr().err.count("error:") == 1


def test_inspect_of_an_empty_folder_is_an_input_error(tmp_path, capsys):
    assert cli.main(["inspect", str(tmp_path)]) == 3
    assert "error:" in capsys.readouterr().err


def test_no_command_is_a_usage_error(capsys):
    assert cli.main([]) == 2


def test_no_flag_names_a_hidden_location():
    flags = {
        option
        for action in cli.build_parser()._actions
        for option in action.option_strings
    }
    assert not flags & {"--db-dir", "--db", "--keep", "--name", "--output", "--output-dir"}


async def _fake_run(**kwargs):
    database = Path(kwargs["database"])
    database.parent.mkdir(parents=True, exist_ok=True)
    database.write_text("x")
    return PipelineResult(
        objective_id="obj_1", iterations=1, successful=1, failed=0, status="done"
    )


def test_the_command_writes_only_into_out_and_prints_json(
    tmp_path, monkeypatch, capsys
):
    seen = {}

    async def fake_handle_ask(text, **kwargs):
        seen.update(kwargs, text=text)
        return AskResult(success=True, question=text, database=str(kwargs["database"]))

    monkeypatch.setattr("andamentum.epistemic.cli_handlers.handle_ask", fake_handle_ask)
    out = tmp_path / "run"
    code = cli.main(
        [
            "ask",
            "Does X cause Y in adults?",
            "--model",
            "m",
            "--embedding-model",
            "e",
            *_search(),
            "--out",
            str(out),
            "--json",
        ]
    )
    assert code == 0
    assert seen["database"] == out / "epistemic.db"
    assert seen["report"] == out / "report.html"
    printed = capsys.readouterr().out
    assert AskResult.model_validate_json(printed).question == "Does X cause Y in adults?"


def test_without_out_the_command_names_no_database(monkeypatch, capsys):
    seen = {}

    async def fake_handle_ask(text, **kwargs):
        seen.update(kwargs)
        return AskResult(success=True, question=text)

    monkeypatch.setattr("andamentum.epistemic.cli_handlers.handle_ask", fake_handle_ask)
    code = cli.main(
        ["verify", "X causes Y", "--model", "m", "--embedding-model", "e", *_search()]
    )
    assert code == 0
    assert seen["database"] is None and seen["report"] is None
    assert seen["mode"] == "verify"
    capsys.readouterr()


async def test_ask_without_a_database_leaves_no_file_behind(monkeypatch):
    kept = []

    async def fake_run(**kwargs):
        kept.append(Path(kwargs["database"]))
        return await _fake_run(**kwargs)

    async def fake_open(database):
        raise RuntimeError("stop here")

    import andamentum.epistemic.operations_runner as runner

    monkeypatch.setattr(runner, "run_research_question", lambda **kw: fake_run(**kw))
    monkeypatch.setattr(ask_verify, "open_store", fake_open)
    with pytest.raises(RuntimeError, match="stop here"):
        await ask_verify.ask(
            "Does X cause Y in adults?",
            model="m",
            embedding_model="e",
            searxng_url="http://searx.test",
        )
    assert kept and not kept[0].exists()
    assert not kept[0].parent.exists()
