"""``andamentum`` — list the tools and run one of them.

    andamentum list              every tool, what it does, whether it needs a model
    andamentum list --json       the same as JSON
    andamentum <tool> [ARGS...]  run a tool; the same as andamentum-<tool> [ARGS...]
    andamentum help <tool>       the tool's own --help

This command only lists and forwards; each tool keeps its own command and
flags. Exit codes are the tool's own (see docs/interface.md); 2 for a usage
error here.
"""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Sequence

from . import __version__
from .tools import TOOLS, get_tool

_USAGE = """usage: andamentum list [--json]
       andamentum <tool> [ARGS...]
       andamentum help <tool>
       andamentum --version

Run 'andamentum list' to see the tools."""


def _list(as_json: bool) -> int:
    if as_json:
        rows = [
            {
                "name": t.name,
                "command": t.command,
                "summary": t.summary,
                "needs_model": t.needs_model,
                "python": t.main,
            }
            for t in TOOLS
        ]
        print(json.dumps(rows, indent=1))
        return 0
    width = max(len(t.name) for t in TOOLS)
    model = {"no": "", "yes": "  [model]", "some": "  [model for some uses]"}
    for t in TOOLS:
        print(f"{t.name:<{width}}  {t.summary}{model[t.needs_model]}")
    print("\nRun 'andamentum <tool> --help' for a tool's options.")
    return 0


def _run(name: str, argv: list[str]) -> int:
    try:
        tool = get_tool(name)
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 2
    cli = importlib.import_module(tool.cli)
    code = cli.main(argv)
    return int(code or 0)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(_USAGE)
        return 0 if args else 2
    first, rest = args[0], args[1:]
    if first == "--version":
        print(f"andamentum {__version__}")
        return 0
    if first == "list":
        if rest not in ([], ["--json"]):
            print("error: usage: andamentum list [--json]", file=sys.stderr)
            return 2
        return _list(as_json=rest == ["--json"])
    if first == "help":
        if len(rest) != 1:
            print("error: usage: andamentum help <tool>", file=sys.stderr)
            return 2
        return _run(rest[0], ["--help"])
    return _run(first, rest)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
