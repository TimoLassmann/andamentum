"""Command-line adapter over the dialect's public functions.

``andamentum-agentic-dialect laws | law <id> | roles | role <role> | checklist |
skeleton | check <path>``. The CLI is a thin wrapper - every command maps to one
public function. ``--json`` prints the listing commands and ``check`` as JSON.

Exit codes:
    0 - success (for ``check``: the portable gates pass)
    1 - ``check`` found violations (they are still printed)
    2 - usage error, including an unknown law id or role name
    3 - ``check``'s path does not exist or cannot be read
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from ._laws import checklist, law, laws
from ._roles import ROLES, for_role, roles
from .checks import check_code
from .doc import skeleton
from .errors import DialectError

_EXIT_CODES = """\
exit codes:
  0  success (for check: the portable gates pass)
  1  check found violations (they are still printed)
  2  usage error, including an unknown law id or role name
  3  the path given to check does not exist or cannot be read
"""

_DESCRIPTION = """\
The agentic dialect: the house style for agentic graph systems, as numbered
laws, role briefs and checks you can run over code.

A law is named by an id such as L7 (L1 to L9; list them with `laws`). A role
is the kind of step you are writing: {roles}. `role <name>` prints the
laws and checklist that job needs, sized to paste into a prompt.
"""


def _dump(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    from andamentum import __version__

    parser = argparse.ArgumentParser(
        prog="andamentum-agentic-dialect",
        description=_DESCRIPTION.format(roles=", ".join(ROLES)),
        epilog=_EXIT_CODES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"andamentum-agentic-dialect {__version__}",
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    def _json_flag(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--json", action="store_true", help="print the result as JSON")

    p = sub.add_parser("laws", help="list all laws (id, name, one-sentence statement)")
    _json_flag(p)
    p = sub.add_parser("law", help="show one law with its checklist")
    p.add_argument("id", metavar="ID", help="law id, L1 to L9 (e.g. L7); see `laws`")
    _json_flag(p)
    p = sub.add_parser("roles", help="list the roles you can ask a brief for")
    _json_flag(p)
    p = sub.add_parser("role", help="print the prompt-slice for one role")
    p.add_argument(
        "role",
        metavar="ROLE",
        help=f"kind of step you are writing: {', '.join(ROLES)}; see `roles`",
    )
    _json_flag(p)
    p = sub.add_parser(
        "checklist", help="print the pre-commit checklist, one item per line"
    )
    _json_flag(p)
    sub.add_parser("skeleton", help="print the runnable copy-paste skeleton")
    p = sub.add_parser(
        "check",
        help="run the portable checks over a file or folder of Python code",
        epilog=_EXIT_CODES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "path", metavar="PATH", help="a Python file, or a folder searched recursively"
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="also run the opt-in checks (the L7 swallowed-exception check)",
    )
    _json_flag(p)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "laws":
        if args.json:
            _dump([lw.model_dump() for lw in laws()])
            return 0
        for lw in laws():
            print(f"{lw.id} — {lw.name}  [{lw.tier}]")
            print(f"    {lw.statement}")
        return 0
    if args.cmd == "law":
        try:
            lw = law(args.id)
        except DialectError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        if args.json:
            _dump(lw.model_dump())
            return 0
        print(f"{lw.id} — {lw.name}  [{lw.tier}]")
        print(lw.statement)
        for item in lw.checklist:
            print(f"  - {item}")
        return 0
    if args.cmd == "roles":
        if args.json:
            _dump(list(roles()))
            return 0
        for r in roles():
            print(r)
        return 0
    if args.cmd == "role":
        try:
            text = for_role(args.role)
        except DialectError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        if args.json:
            _dump({"role": args.role.lower(), "text": text})
        else:
            print(text)
        return 0
    if args.cmd == "checklist":
        if args.json:
            _dump([{"item": item, "law": lid} for item, lid in checklist()])
            return 0
        for item, lid in checklist():
            print(f"- {item} ({lid})")
        return 0
    if args.cmd == "skeleton":
        print(skeleton())
        return 0
    if args.cmd == "check":
        if not Path(args.path).exists():
            print(f"error: path not found: {args.path}", file=sys.stderr)
            return 3
        try:
            violations = check_code(args.path, strict=args.strict)
        except (OSError, UnicodeDecodeError) as e:
            print(f"error: cannot read {args.path}: {e}", file=sys.stderr)
            return 3
        if args.json:
            _dump(
                {
                    "ok": not violations,
                    "violations": [v.model_dump() for v in violations],
                }
            )
            return 1 if violations else 0
        for v in violations:
            print(f"{v.file}:{v.line}: [{v.law or v.code}] {v.message}")
        if violations:
            print(f"\n{len(violations)} violation(s).", file=sys.stderr)
            return 1
        print("ok — portable gates pass (review-only laws unchecked).")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
