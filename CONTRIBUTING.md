# Contributing to andamentum

Thanks for your interest in andamentum. This document covers the
essentials for working on the codebase.

## Development setup

andamentum uses [uv](https://github.com/astral-sh/uv) for everything
Python-related. Do not use a bare `python` / `pip`.

```bash
uv sync --extra dev          # install with dev dependencies
uv sync --extra dev --extra html-articles --extra pdf   # everything, for the full test surface
```

## The green gate

Before opening a pull request, run all three and confirm they pass:

```bash
uv run ruff check
uv run ruff format --check
uv run pyright
uv run pytest
```

- **ruff** must be clean (both `check` and `format --check`).
- **pyright** errors are tolerated only in test files (pre-existing
  pydantic-graph generic-variance and dict-form-fixture noise); shipped
  library code under `src/andamentum/` must be type-clean.
- **pytest** must be fully green. Tests live next to the code they test
  (`src/andamentum/<module>/tests/`), not in a top-level `tests/` dir.
  The `ollama`, `benchmark`, `cloud`, and `live_provider` markers are
  deselected by default; run them explicitly (e.g. `uv run pytest -m benchmark`).

## Architectural conventions

These are enforced in review — please read them before making changes:

- **Explicit model argument, no hidden defaults.** Every public function
  that calls an LLM takes `model=` as a keyword-only argument. There is
  no shared config module and no ambient default.
- **No silent failures.** A crash beats a silently-wrong answer. Don't
  swallow exceptions; if you must tolerate one, narrow the `except` and
  log it.
- **No environment-variable configuration in new code.** Configuration
  flows top-down from the CLI / caller through explicit keyword arguments.
- **Module layering is strict.** Each sub-module documents which siblings
  it may depend on (see `CLAUDE.md`). `core` is the shared base and
  depends on no sub-module.
- **Public API lives in `__init__.py`** via an explicit `__all__`.

`CLAUDE.md` is the detailed map of the codebase, its modules, and the
conventions above — it's the best starting point for understanding how
the pieces fit together.

## Responsible use

andamentum ships in-code responsible-use protections (fetch gating,
confidentiality tripwires, AI-provenance watermarking). Please read
[`RESPONSIBLE_USE.md`](./RESPONSIBLE_USE.md) and do not weaken these
protections without discussion.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/)
(`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`, …) with an optional
module scope, e.g. `fix(whetstone): …`.

## License

By contributing you agree that your contributions are licensed under the
project's [MIT License](./LICENSE).
