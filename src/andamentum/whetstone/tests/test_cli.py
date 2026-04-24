"""CLI smoke tests — help text and agents listing work without LLM access."""

import subprocess
import sys


def test_help_succeeds():
    r = subprocess.run(
        [sys.executable, "-m", "andamentum.whetstone.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    assert "whetstone" in r.stdout.lower() or "whetstone" in r.stderr.lower()
    assert "--task" in r.stdout
    # The self-review framing must be visible in help text.
    assert "peer-review" in r.stdout.lower() or "own draft" in r.stdout.lower()


def test_agents_subcommand():
    r = subprocess.run(
        [sys.executable, "-m", "andamentum.whetstone.cli", "agents"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    # At least a couple of well-known agents should appear.
    assert "unified_editor" in r.stdout
    assert "expert_reviewer" in r.stdout


def test_cli_accepts_consistency_task():
    from andamentum.whetstone.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["doc.md", "--task", "consistency"])
    assert args.task == "consistency"


def test_cli_accepts_checklist_task():
    from andamentum.whetstone.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["doc.md", "--task", "checklist"])
    assert args.task == "checklist"


def test_cli_guidelines_flag_parses():
    from andamentum.whetstone.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["doc.md", "--task", "checklist", "--guidelines", "hello"])
    assert args.guidelines == "hello"


def test_cli_resolve_guidelines_inline():
    from andamentum.whetstone.cli import _resolve_guidelines

    assert _resolve_guidelines("some text") == "some text"
    assert _resolve_guidelines(None) is None
    assert _resolve_guidelines("") is None


def test_cli_resolve_guidelines_from_file(tmp_path):
    from andamentum.whetstone.cli import _resolve_guidelines

    p = tmp_path / "guidelines.txt"
    p.write_text("rules here", encoding="utf-8")
    assert _resolve_guidelines(f"@{p}") == "rules here"
