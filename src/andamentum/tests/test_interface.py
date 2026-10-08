"""Every tool follows the command-line rules in docs/interface.md.

These checks are mechanical: what a parser accepts, what ``--help`` loads,
which environment variables the code reads. They do not judge help text or
names; a person reads those.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path

import pytest

from andamentum.tools import TOOLS, Tool

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src" / "andamentum"

#: Libraries that take seconds and gigabytes to import; --help must not load them.
HEAVY = ("torch", "transformers", "sentence_transformers", "docling", "mlx")

#: Output flags other than -o/--out that the rules forbid.
FORBIDDEN_OUTPUT_FLAGS = {"--output", "--output-dir", "--out-dir", "--output-file", "--outdir"}

#: Environment variables the code may read: credentials and provider settings
#: that each provider documents, never andamentum configuration.
ALLOWED_ENV = {
    "NCBI_API_KEY",
    "OPENALEX_API_KEY",
    "OPENALEX_EMAIL",
    "SEMANTIC_SCHOLAR_API_KEY",
    "OLLAMA_BASE_URL",
    "AWS_PROFILE",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "DYLD_LIBRARY_PATH",
}

TOOL_IDS = [t.name for t in TOOLS]


def _scripts() -> dict[str, str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return dict(data["project"]["scripts"])


def test_every_command_is_a_listed_tool_and_back():
    scripts = _scripts()
    assert scripts.pop("andamentum") == "andamentum.cli:main"
    assert scripts == {t.command: f"{t.cli}:main" for t in TOOLS}


def _parsers(parser: argparse.ArgumentParser) -> Iterator[argparse.ArgumentParser]:
    """The parser and every subcommand parser below it."""
    yield parser
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                yield from _parsers(sub)


def _build_parser(tool: Tool) -> argparse.ArgumentParser:
    cli = importlib.import_module(tool.cli)
    assert hasattr(cli, "build_parser"), f"{tool.cli} has no build_parser()"
    parser = cli.build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    return parser


@pytest.mark.parametrize("name", TOOL_IDS)
def test_main_takes_argv(name):
    tool = next(t for t in TOOLS if t.name == name)
    cli = importlib.import_module(tool.cli)
    params = list(inspect.signature(cli.main).parameters)
    assert params[:1] == ["argv"], f"{tool.cli}.main must take argv first"


@pytest.mark.parametrize("name", TOOL_IDS)
def test_parser_follows_the_flag_rules(name):
    tool = next(t for t in TOOLS if t.name == name)
    parser = _build_parser(tool)
    top = {o for a in parser._actions for o in a.option_strings}
    assert "--version" in top, f"{tool.command} has no --version"
    model_flags: list[argparse.Action] = []
    for p in _parsers(parser):
        for action in p._actions:
            opts = set(action.option_strings)
            bad = opts & FORBIDDEN_OUTPUT_FLAGS
            assert not bad, f"{tool.command} {p.prog}: use -o/--out, not {sorted(bad)}"
            if "--out" in opts:
                assert "-o" in opts or p.prog.split()[-1] in _DIR_OUTPUT_OK, (
                    f"{p.prog}: --out should also be -o"
                )
            for opt in opts:
                if opt.endswith("-model"):
                    model_flags.append(action)
                    assert action.default is None, (
                        f"{p.prog} {opt} has a default ({action.default!r}); models have none"
                    )
    if tool.needs_model == "no":
        assert not model_flags, f"{tool.command} is listed as needing no model"
    elif tool.needs_model == "yes":
        assert any("--model" in a.option_strings for a in model_flags), (
            f"{tool.command} needs a model but has no --model"
        )
    else:  # "some": a model flag for the uses that need one, named by its job
        assert model_flags, f"{tool.command} needs a model for some uses but has no model flag"


#: Commands whose --out names a folder; -o is optional there because a
#: one-letter flag for a folder is easy to mistake for a file.
_DIR_OUTPUT_OK: set[str] = set()


@pytest.mark.parametrize("name", TOOL_IDS)
def test_help_and_version_are_cheap(name):
    tool = next(t for t in TOOLS if t.name == name)
    code = (
        "import sys\n"
        f"from {tool.cli} import main\n"
        "for argv in (['--help'], ['--version']):\n"
        "    try:\n"
        "        rc = main(argv)\n"
        "    except SystemExit as e:\n"
        "        rc = e.code\n"
        "    assert rc in (0, None), (argv, rc)\n"
        f"heavy = [m for m in {HEAVY!r} if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr[-2000:]


def test_the_umbrella_command_lists_and_forwards(capsys):
    from andamentum.cli import main

    assert main(["list"]) == 0
    listed = capsys.readouterr().out
    assert all(t.name in listed for t in TOOLS)
    assert main(["list", "--json"]) == 0
    assert main(["no-such-tool"]) == 2
    assert main(["chunker", "--version"]) in (0, None)


_ENV_READ = re.compile(
    r"""(?:environ(?:\.get)?\s*[\[(]\s*|getenv\s*\(\s*)["']([A-Z][A-Z0-9_]+)["']"""
)


def test_no_configuration_from_the_environment():
    """Only credentials and provider settings come from the environment."""
    found: dict[str, list[str]] = {}
    for path in SRC.rglob("*.py"):
        if "tests" in path.parts:
            continue
        text = path.read_text()
        for name in _ENV_READ.findall(text):
            if name not in ALLOWED_ENV:
                found.setdefault(name, []).append(str(path.relative_to(ROOT)))
        # An ANDAMENTUM_* name in quotes is an environment variable's name
        # (bare ANDAMENTUM_* identifiers are ordinary constants).
        for name in re.findall(r"""["'](ANDAMENTUM_[A-Z_]+)["']""", text):
            found.setdefault(name, []).append(str(path.relative_to(ROOT)))
    assert not found, f"configuration read from the environment: {found}"
