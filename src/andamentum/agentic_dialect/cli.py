"""Command-line adapter over the dialect's public functions.

``andamentum-agentic-dialect laws | law <id> | roles | role <role> | checklist |
skeleton | check <path>``. The CLI is a thin wrapper — every command maps to one
public function.
"""

from __future__ import annotations

import argparse
import sys

from ._laws import checklist, law, laws
from ._roles import for_role, roles
from .checks import check_code
from .doc import skeleton


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="andamentum-agentic-dialect",
        description="The agentic dialect: conventions, role briefs, and conformance checks.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("laws", help="list all laws")
    p_law = sub.add_parser("law", help="show one law by id")
    p_law.add_argument("id")
    sub.add_parser("roles", help="list briefable roles")
    p_role = sub.add_parser("role", help="the prompt-slice for a role")
    p_role.add_argument("role")
    sub.add_parser("checklist", help="the greppable pre-commit checklist")
    sub.add_parser("skeleton", help="the runnable copy-paste skeleton")
    p_check = sub.add_parser("check", help="run the portable gates over a path")
    p_check.add_argument("path")
    p_check.add_argument(
        "--strict",
        action="store_true",
        help="also run the opt-in gates (the L7 swallowed-exception check)",
    )

    args = parser.parse_args(argv)

    if args.cmd == "laws":
        for lw in laws():
            print(f"{lw.id} — {lw.name}  [{lw.tier}]")
            print(f"    {lw.statement}")
        return 0
    if args.cmd == "law":
        try:
            lw = law(args.id)
        except KeyError as e:
            print(e, file=sys.stderr)
            return 2
        print(f"{lw.id} — {lw.name}  [{lw.tier}]")
        print(lw.statement)
        for item in lw.checklist:
            print(f"  - {item}")
        return 0
    if args.cmd == "roles":
        for r in roles():
            print(r)
        return 0
    if args.cmd == "role":
        try:
            print(for_role(args.role))
        except KeyError as e:
            print(e, file=sys.stderr)
            return 2
        return 0
    if args.cmd == "checklist":
        for item, lid in checklist():
            print(f"- {item} ({lid})")
        return 0
    if args.cmd == "skeleton":
        print(skeleton())
        return 0
    if args.cmd == "check":
        violations = check_code(args.path, strict=args.strict)
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
