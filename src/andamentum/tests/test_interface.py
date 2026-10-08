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


@pytest.mark.parametrize("doc", ["README.md", "CLAUDE.md", "docs/index.md"])
def test_every_tool_is_documented(doc):
    text = (ROOT / doc).read_text()
    missing = [t.command for t in TOOLS if f"`{t.command}`" not in text]
    assert not missing, f"{doc} does not mention {missing}"


# --- --model is required wherever a tool needs a model --------------------------------

#: (tool, subcommand) pairs whose --model is legitimately optional although the
#: tool needs a model. Anything not listed must have ``required=True``, so a new
#: optional --model fails the test.
OPTIONAL_MODEL_OK = {
    # compose: --model fills all four roles at once; the per-role flags
    # (--drafter/--critic/--judge/--reader) are the alternative.
    ("compose", "write"),
    ("compose", "eval"),
    # compose continue/render/check take the models recorded in the folder, or
    # the per-role flags; check needs a model only for the optional numbers pass.
    ("compose", "continue"),
    ("compose", "render"),
    ("compose", "check"),
    # research mcp serves the store; the model is for the optional research tool.
    ("research", "mcp"),
}


def _subcommand(parser: argparse.ArgumentParser) -> str:
    return parser.prog.split()[1] if len(parser.prog.split()) > 1 else ""


@pytest.mark.parametrize("name", [t.name for t in TOOLS if t.needs_model == "yes"])
def test_a_model_tool_requires_model_on_every_command(name):
    """Mutation: change any ``--model`` of a "yes" tool to ``required=False`` (or give it a
    default) and the parser listed here fails, unless it is in OPTIONAL_MODEL_OK."""
    tool = next(t for t in TOOLS if t.name == name)
    for p in _parsers(_build_parser(tool)):
        for action in p._actions:
            if "--model" not in action.option_strings:
                continue
            if (name, _subcommand(p)) in OPTIONAL_MODEL_OK:
                continue
            assert action.required, f"{p.prog}: --model must be required"
            assert action.default is None


def test_the_allow_list_names_real_commands():
    """Mutation: rename a compose/research subcommand and a stale allow-list entry fails."""
    seen = set()
    for tool in TOOLS:
        if tool.needs_model != "yes":
            continue
        for p in _parsers(_build_parser(tool)):
            if any("--model" in a.option_strings for a in p._actions):
                seen.add((tool.name, _subcommand(p)))
    assert OPTIONAL_MODEL_OK <= seen


# --- --version, help vocabulary, usage errors --------------------------------------

@pytest.mark.parametrize("name", TOOL_IDS)
def test_version_prints_the_package_version(name, capsys):
    """Mutation: print a hard-coded or empty version and this fails."""
    import andamentum

    tool = next(t for t in TOOLS if t.name == name)
    cli = importlib.import_module(tool.cli)
    try:
        code = cli.main(["--version"])
    except SystemExit as exc:  # argparse exits; some mains return the code
        code = exc.code
    assert code in (0, None)
    out = capsys.readouterr().out
    assert tool.command in out and andamentum.__version__ in out


_INTERNAL_WORDS = re.compile(r"\b(harness|kernel|v2|v3)\b", re.IGNORECASE)

#: Tools whose help still shows internal words (tool name -> reason); fill in
#: only to make a known gap visible, never to hide a new one.
_HELP_WORDS_XFAIL: dict[str, str] = {}


@pytest.mark.parametrize("name", TOOL_IDS)
def test_help_uses_no_internal_words(name):
    """Mutation: put "harness" in any flag help or description and this fails."""
    if name in _HELP_WORDS_XFAIL:
        pytest.xfail(_HELP_WORDS_XFAIL[name])
    tool = next(t for t in TOOLS if t.name == name)
    hits = []
    for p in _parsers(_build_parser(tool)):
        hits += [(p.prog, m.group(0)) for m in _INTERNAL_WORDS.finditer(p.format_help())]
    assert not hits, hits


@pytest.mark.parametrize("name", TOOL_IDS)
def test_an_unknown_flag_is_a_usage_error(name):
    """Mutation: make main() catch SystemExit and return 1 / 4 and this fails."""
    tool = next(t for t in TOOLS if t.name == name)
    cli = importlib.import_module(tool.cli)
    try:
        code = cli.main(["--definitely-not-a-flag"])
    except SystemExit as exc:
        code = exc.code
    assert code == 2


# --- nothing is written into the current folder -------------------------------------

def test_no_model_free_tool_writes_into_the_current_folder(tmp_path, monkeypatch, capsys):
    """Each case runs from an empty folder with its outputs pointed at another one.
    Mutation: make a command write to a relative or default path (for example drop
    ``-o`` handling and write ``out.json`` / ``.cache`` beside the cwd) and the
    empty-folder assertion fails."""
    from andamentum.agentic_dialect.cli import main as dialect
    from andamentum.chunker.cli import main as chunker
    from andamentum.compose.cli import main as compose
    from andamentum.document_store.cli import main as docstore
    from andamentum.proofread.cli import main as proofread
    from andamentum.references.cli import main as references
    from andamentum.typeset.cli import main as typeset

    work = tmp_path / "work"
    work.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    doc = work / "in.md"
    doc.write_text("# Title\n\nA short synthetic paragraph about salt.\n\n## Part\n\nAnother sentence.\n")
    db = work / "s.db"
    monkeypatch.chdir(cwd)
    cases = [
        (chunker, ["--max-chars", "100", str(doc), "-o", str(work / "c.json")], {0}),
        (proofread, [str(doc), "-o", str(work / "p.txt")], {0, 1}),
        (proofread, [str(doc)], {0, 1}),
        (typeset, [str(doc)], {0}),
        (typeset, [str(doc), "-o", str(work / "t.html")], {0}),
        (dialect, ["laws"], {0}),
        (references, [str(doc), "--no-lookup", "-o", str(work / "r.md")], {0, 1}),
        (docstore, ["ingest", str(db), str(doc)], {0}),
        (docstore, ["list", str(db)], {0}),
        (compose, ["new", "-o", str(work / "newdoc"), "--title", "Pilot"], {0}),
    ]
    for main, argv, ok in cases:
        try:
            code = main(argv)
        except SystemExit as exc:
            code = exc.code
        capsys.readouterr()
        assert code in ok, (main.__module__, argv, code)
        assert list(cwd.iterdir()) == [], (main.__module__, argv)
    assert (work / "c.json").is_file() and (work / "newdoc").is_dir() and db.is_file()


@pytest.mark.parametrize("name", TOOL_IDS)
def test_the_listed_main_function_exists(name):
    """``andamentum list`` names each tool's Python entry point; it must resolve."""
    tool = next(t for t in TOOLS if t.name == name)
    module_name, _, rest = tool.main.partition(f"{tool.module}.")
    assert module_name == "", f"{tool.main} is not inside {tool.module}"
    obj = importlib.import_module(tool.module)
    for part in rest.split("."):
        obj = getattr(obj, part)
    assert callable(obj), tool.main
    top = rest.split(".")[0]
    assert top in getattr(importlib.import_module(tool.module), "__all__", ()), (
        f"{tool.main}: {top} is not exported in {tool.module}.__all__"
    )
