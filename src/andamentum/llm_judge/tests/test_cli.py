"""CLI tests for ``andamentum-llm-judge``.

No LLM calls: :func:`andamentum.llm_judge.cli.judge_score` /
``judge_compare`` are monkeypatched at the CLI module's namespace, and every
test asserts on what the CLI PASSED DOWN to them. That is the point — the
class of bug this file exists to prevent is the CLI silently handing the
judge the wrong text (see ``test_score_sends_the_positional_text_...``
below), which no library-level test can catch.
"""

from __future__ import annotations

import json

import pytest

from andamentum.llm_judge import cli
from andamentum.llm_judge.criteria import DEFAULT_CRITERIA
from andamentum.llm_judge.schemas import CompareResult, Criterion, ScoreResult

# ── Fakes ────────────────────────────────────────────────────────────────


def _fake_score_result() -> ScoreResult:
    return ScoreResult(
        per_criterion=[],
        overall="meets",
        confidence=0.9,
        doubt=0.1,
        needs_review=False,
        judges=None,
    )


def _fake_compare_result() -> CompareResult:
    return CompareResult(
        reasoning="because",
        winner="a",
        confidence=0.9,
        doubt=0.1,
        order_consistent=True,
        needs_review=False,
        judges=None,
    )


@pytest.fixture
def calls(monkeypatch):
    """Capture every argument the CLI hands to the library entry points."""
    captured: list[dict[str, object]] = []

    async def fake_judge_score(output, *, criteria, context, model):
        captured.append(
            {
                "fn": "score",
                "output": output,
                "criteria": criteria,
                "context": context,
                "model": model,
            }
        )
        return _fake_score_result()

    async def fake_judge_compare(output_a, output_b, *, criteria, context, model):
        captured.append(
            {
                "fn": "compare",
                "output_a": output_a,
                "output_b": output_b,
                "criteria": criteria,
                "context": context,
                "model": model,
            }
        )
        return _fake_compare_result()

    monkeypatch.setattr(cli, "judge_score", fake_judge_score)
    monkeypatch.setattr(cli, "judge_compare", fake_judge_compare)
    return captured


# ── The regression this module shipped without ───────────────────────────
#
# `score` declares a positional named `output`; the destination-file flag is
# `-o/--output`. Both once resolved to argparse dest `output`, so:
#   * without -o, the JSON was written to a FILE NAMED AFTER THE JUDGED TEXT
#     instead of stdout; and
#   * with -o, the judge was handed the FILE PATH as the text to judge and
#     returned a confident, well-formed verdict on the wrong input.
# The second is a silent-wrong-answer bug in a component whose entire job is
# to be a trustworthy signal. Both directions are pinned below.


def test_score_sends_the_positional_text_to_the_judge_not_the_output_path(
    calls, tmp_path, capsys
):
    out = tmp_path / "result.json"
    rc = cli.main(
        ["score", "Paris is the capital of France.", "--model", "m", "-o", str(out)]
    )
    assert rc == 0
    assert calls[0]["output"] == "Paris is the capital of France."
    assert json.loads(out.read_text())["overall"] == "meets"


def test_score_without_output_flag_writes_to_stdout_not_a_file_named_after_the_text(
    calls, tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    rc = cli.main(["score", "Paris is the capital of France.", "--model", "m"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["overall"] == "meets"
    assert calls[0]["output"] == "Paris is the capital of France."
    # Nothing may be written to disk on the stdout path.
    assert list(tmp_path.iterdir()) == []


# ── score ────────────────────────────────────────────────────────────────


def test_score_defaults_to_the_builtin_criteria_and_no_context(calls, capsys):
    assert cli.main(["score", "text", "--model", "m"]) == 0
    assert calls[0]["criteria"] == DEFAULT_CRITERIA
    assert calls[0]["context"] is None
    capsys.readouterr()


def test_score_passes_context_through(calls, capsys):
    assert cli.main(["score", "text", "--model", "m", "--context", "the task"]) == 0
    assert calls[0]["context"] == "the task"
    capsys.readouterr()


# ── compare ──────────────────────────────────────────────────────────────


def test_compare_passes_both_outputs_in_order(calls, capsys):
    assert cli.main(["compare", "first", "second", "--model", "m"]) == 0
    assert calls[0]["fn"] == "compare"
    assert calls[0]["output_a"] == "first"
    assert calls[0]["output_b"] == "second"
    assert json.loads(capsys.readouterr().out)["winner"] == "a"


def test_compare_honours_the_output_flag(calls, tmp_path, capsys):
    out = tmp_path / "nested" / "cmp.json"
    rc = cli.main(["compare", "first", "second", "--model", "m", "-o", str(out)])
    assert rc == 0
    # Parent directories are created for the caller.
    assert json.loads(out.read_text())["winner"] == "a"
    assert calls[0]["output_a"] == "first"


# ── model parsing: the fast/panel mode axis ──────────────────────────────


def test_single_model_is_a_bare_str_so_the_library_takes_the_fast_path(calls, capsys):
    assert cli.main(["score", "text", "--model", "m"]) == 0
    assert calls[0]["model"] == "m"
    assert isinstance(calls[0]["model"], str)
    capsys.readouterr()


def test_repeated_model_flags_become_a_list_so_the_library_takes_the_panel_path(
    calls, capsys
):
    assert cli.main(["score", "text", "--model", "m1", "--model", "m2"]) == 0
    assert calls[0]["model"] == ["m1", "m2"]
    capsys.readouterr()


def test_comma_separated_models_become_a_list(calls, capsys):
    assert cli.main(["score", "text", "--model", "m1,m2,m3"]) == 0
    assert calls[0]["model"] == ["m1", "m2", "m3"]
    capsys.readouterr()


def test_repeated_and_comma_separated_models_combine(calls, capsys):
    assert cli.main(["score", "text", "--model", "m1,m2", "--model", "m3"]) == 0
    assert calls[0]["model"] == ["m1", "m2", "m3"]
    capsys.readouterr()


def test_model_ids_containing_colons_survive_parsing(calls, capsys):
    """A model id is `provider:name[:tag]` — splitting must be on commas only."""
    assert cli.main(["score", "text", "--model", "ollama:gemma4:31b-nvfp4"]) == 0
    assert calls[0]["model"] == "ollama:gemma4:31b-nvfp4"
    capsys.readouterr()


def test_whitespace_around_comma_separated_models_is_stripped(calls, capsys):
    assert cli.main(["score", "text", "--model", " m1 , m2 "]) == 0
    assert calls[0]["model"] == ["m1", "m2"]
    capsys.readouterr()


def test_parse_models_rejects_a_value_that_resolves_to_nothing():
    with pytest.raises(ValueError):
        cli._parse_models([" , "])


# ── criteria loading ─────────────────────────────────────────────────────


def test_criteria_file_is_loaded_and_passed_through(calls, tmp_path, capsys):
    path = tmp_path / "criteria.json"
    path.write_text(
        json.dumps([{"name": "rigour", "description": "Is it rigorous?"}]),
        encoding="utf-8",
    )
    assert cli.main(["score", "text", "--model", "m", "--criteria", str(path)]) == 0
    assert calls[0]["criteria"] == [
        Criterion(name="rigour", description="Is it rigorous?")
    ]
    capsys.readouterr()


# ── failure surfaces: the documented exit codes ──────────────────────────


def test_missing_model_flag_is_an_argparse_error(calls):
    with pytest.raises(SystemExit) as exc:
        cli.main(["score", "text"])
    assert exc.value.code == 2  # argparse's own usage-error code


def test_unparseable_criteria_file_exits_2(calls, tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        cli.main(["score", "text", "--model", "m", "--criteria", str(path)])
    assert exc.value.code == 2
    assert "could not load criteria" in capsys.readouterr().err


def test_missing_criteria_file_exits_2(calls, tmp_path, capsys):
    missing = tmp_path / "nope.json"
    with pytest.raises(SystemExit) as exc:
        cli.main(["score", "text", "--model", "m", "--criteria", str(missing)])
    assert exc.value.code == 2
    assert "could not load criteria" in capsys.readouterr().err


def test_criteria_file_with_wrong_shape_exits_2(calls, tmp_path, capsys):
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps([{"nombre": "x"}]), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        cli.main(["score", "text", "--model", "m", "--criteria", str(path)])
    assert exc.value.code == 2
    capsys.readouterr()


def test_a_failing_judge_call_exits_3_and_reports_on_stderr(monkeypatch, capsys):
    async def boom(*args, **kwargs):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(cli, "judge_score", boom)
    with pytest.raises(SystemExit) as exc:
        cli.main(["score", "text", "--model", "m"])
    assert exc.value.code == 3
    err = capsys.readouterr().err
    assert "judge call failed" in err
    assert "RuntimeError" in err
    assert "model exploded" in err


def test_a_failing_judge_call_writes_no_output_file(monkeypatch, tmp_path):
    async def boom(*args, **kwargs):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(cli, "judge_score", boom)
    out = tmp_path / "result.json"
    with pytest.raises(SystemExit):
        cli.main(["score", "text", "--model", "m", "-o", str(out)])
    assert not out.exists()


def test_no_subcommand_is_an_argparse_error(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2


# ── serialisation ────────────────────────────────────────────────────────


def test_stdout_payload_is_the_full_result_schema(calls, capsys):
    assert cli.main(["compare", "a", "b", "--model", "m"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {
        "reasoning",
        "winner",
        "confidence",
        "doubt",
        "order_consistent",
        "needs_review",
        "judges",
    }


def test_written_file_is_utf8_and_round_trips(calls, tmp_path, capsys):
    out = tmp_path / "r.json"
    assert cli.main(["score", "héllo — ünicode", "--model", "m", "-o", str(out)]) == 0
    assert calls[0]["output"] == "héllo — ünicode"
    assert json.loads(out.read_text(encoding="utf-8"))["overall"] == "meets"
    capsys.readouterr()
