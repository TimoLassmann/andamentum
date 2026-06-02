# Running CI locally

The GitHub Actions workflow (`.github/workflows/test.yml`) can be reproduced on
your machine before pushing, so a red build is caught locally instead of on
GitHub. The wrapper is `scripts/ci-local.sh`.

## Quick check (fast, no Docker)

```bash
./scripts/ci-local.sh quick
```

Runs the exact step sequence from the workflow — `uv sync --locked --extra dev`,
`ruff check`, `ruff format --check`, `pytest`, `pyright` — natively with your
local toolchain. This is the fastest way to catch the common failures:

- **lock drift** (`uv sync --locked` fails) — the failure that took the 0.3.0
  release red. It surfaces in seconds here.
- lint / format violations
- test regressions

Because it runs on *your* OS and Python, it is a close mirror, not a byte-exact
reproduction of the CI matrix.

## Faithful check (Docker, via `act`)

```bash
./scripts/ci-local.sh act            # all ubuntu jobs (test matrix, examples, pyright)
./scripts/ci-local.sh act typecheck  # a single job
```

[`act`](https://github.com/nektos/act) executes the workflow inside a Linux
container that closely matches the github-hosted `ubuntu-latest` runner — same
`setup-uv`, same `uv sync --locked`, same steps. Configuration lives in
`.actrc` (runner image + `linux/amd64` architecture for Apple Silicon).

Requirements and caveats:

- **Docker must be running** (Docker Desktop on macOS).
- **First run pulls a multi-GB runner image** (`catthehacker/ubuntu:act-latest`).
- **Linux only.** `act` cannot run the `macos-latest` matrix legs — those only
  run on GitHub. The wrapper constrains the matrix to `os:ubuntu-latest`.
- On Apple Silicon the amd64 container runs under emulation, so the full test
  suite is noticeably slower than native.

Install `act` with `brew install act`.

## Why CI broke on the 0.3.0 release

CI pinned `uv` to `0.5.x` while the `uv.lock` was produced by a local `uv`
0.11.x. The newer uv writes the lockfile at `revision = 3` with a relative
`exclude-newer` window (`exclude-newer-span = "P28D"`, from
`exclude-newer = "28 days"` in `pyproject.toml`). The old `0.5.x` uv could not
parse either, ignored the lockfile, and failed `uv sync --locked`. The fix was
to align the CI pin to `0.11.x`. `./scripts/ci-local.sh quick` would have caught
it before the push.
