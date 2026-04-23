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
