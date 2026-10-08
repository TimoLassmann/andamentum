"""The experimental notice: only on the commands that run the pipeline."""

from __future__ import annotations

from andamentum.epistemic import cli


def test_agents_prints_no_notice(capsys):
    assert cli.main(["agents"]) == 0
    assert "EXPERIMENTAL" not in capsys.readouterr().err


def test_the_notice_is_not_printed_before_a_usage_error(capsys):
    code = cli.main(["ask", "Is X true?", "--model", "m", "--embedding-model", "e"])
    captured = capsys.readouterr()
    assert code == 2
    assert "error:" in captured.err
    assert "EXPERIMENTAL" not in captured.err


def test_the_notice_has_no_colour_codes_when_stderr_is_not_a_terminal():
    assert "\033[" not in cli._experimental_banner()


def test_ask_has_a_description_and_dev_commands_are_marked():
    parser = cli.build_parser()
    import argparse

    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    assert sub.choices["ask"].description
    helps = {c.dest: c.help for c in sub._choices_actions}
    for name in ("run", "agents", "stage"):
        assert (helps[name] or "").startswith("(for development)")
