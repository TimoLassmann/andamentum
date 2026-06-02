#!/usr/bin/env bash
#
# ci-local.sh — run the GitHub Actions test workflow locally before pushing.
#
# Two modes:
#
#   ./scripts/ci-local.sh quick        # fast, no Docker: runs the exact
#                                       # commands from .github/workflows/test.yml
#                                       # natively with your local uv. Catches
#                                       # lock drift, lint, format, pyright, tests.
#
#   ./scripts/ci-local.sh act [JOB]    # full fidelity: runs the workflow inside
#                                       # a Linux container via `act`, the same
#                                       # way GitHub does. JOB is optional
#                                       # (test | examples | typecheck); default
#                                       # runs all ubuntu jobs.
#
# Notes:
#   * `act` runs Linux containers only — it validates the ubuntu-latest matrix
#     legs, NOT the macos-latest ones. Those only run on GitHub.
#   * First `act` run pulls a multi-GB runner image (catthehacker/ubuntu).
#   * `quick` mode mirrors the steps but runs on YOUR OS/Python, so it is a
#     close-but-not-identical check. Use `act` for a faithful reproduction.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORKFLOW=".github/workflows/test.yml"
MODE="${1:-quick}"

die() { echo "error: $*" >&2; exit 1; }
group() { echo; echo "=== $* ==="; }

run_quick() {
    command -v uv >/dev/null || die "uv not found on PATH"

    group "Install dependencies (locked)  [uv sync --locked --extra dev]"
    # This is the step that broke CI: a uv-version/lockfile mismatch fails here.
    uv sync --locked --extra dev

    group "Lint  [ruff check]"
    uv run ruff check

    group "Format check  [ruff format --check]"
    uv run ruff format --check

    group "Tests  [pytest, default markers]"
    uv run pytest

    group "Pyright  [non-blocking, mirrors CI]"
    uv run pyright || true

    echo
    echo "quick checks passed (note: ran on local OS/Python, not the CI matrix)."
}

run_act() {
    command -v act >/dev/null || die "act not found — install with: brew install act"
    docker info >/dev/null 2>&1 || die "Docker is not running — start Docker Desktop first"

    local job="${1:-}"
    local -a args=(
        -W "$WORKFLOW"
        # Constrain the matrix to the Linux legs; act cannot run macos-latest.
        --matrix os:ubuntu-latest
    )
    [ -n "$job" ] && args+=(-j "$job")

    group "act ${job:+(job: $job) }— ubuntu jobs only"
    echo "(macos-latest matrix legs are skipped; they only run on GitHub)"
    act "${args[@]}"
}

case "$MODE" in
    quick) run_quick ;;
    act)   run_act "${2:-}" ;;
    -h|--help|help)
        sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        ;;
    *) die "unknown mode '$MODE' — use 'quick', 'act [job]', or 'help'" ;;
esac
