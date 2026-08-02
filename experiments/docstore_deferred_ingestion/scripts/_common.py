"""Shared harness scaffolding — and the isolation guard that runs at import.

WHY THE GUARD IS AT IMPORT TIME
--------------------------------
Two failure modes would silently invalidate this entire experiment, and both are
invisible in the output if you only check them by convention:

1. A rule that forgets ``DOCUMENT_STORE_DIR`` falls back to
   ``~/.local/share/document-store/`` — i.e. the user's real 'brain' store. That
   is a *destructive* mistake, not a measurement error.
2. A rule that forgets ``OLLAMA_BASE_URL`` still runs to completion. Verified on
   this machine: ``extract_chunk_metadata`` catches the pydantic-ai provider
   error, logs a warning and RETURNS DEFAULTS — 0.47 s with empty topics instead
   of 10.85 s with populated ones. Every drain would then report
   ``documents_enriched`` while writing no LLM metadata at all.

So importing this module raises unless both are set, the store directory
resolves INSIDE the experiment directory, and (when a database name is passed to
:func:`require_db_name`) the name carries the mandatory ``dfr_`` prefix.

Belt and braces: the Snakefile also exports both variables in its ENV prefix.
Neither mechanism is relied on alone.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

EXP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EXP_DIR.parent.parent

DATA = EXP_DIR / "data"
PDFS = DATA / "pdfs"
MARKDOWN = DATA / "markdown"
RESULTS = EXP_DIR / "results"
LEDGERS = RESULTS / "ledgers"
SNAPSHOTS = RESULTS / "snapshots"
EVENTS = RESULTS / "events"
FIGURES = EXP_DIR / "figures"
LOGS = EXP_DIR / "logs"
DBS = EXP_DIR / "dbs"
RUNS = EXP_DIR / "runs"

REGISTRY_PATH = DATA / "REGISTRY.json"
PROVENANCE_PATH = RESULTS / "provenance.json"
PREREG_PATH = RESULTS / "preregistration.json"

#: Every database in this experiment must start with this. ``lifecycle.py``'s
#: EPHEMERAL_PREFIXES = ("ask_", "test_", "varfolders", "tmp") silently
#: redirects such names into <DOCUMENT_STORE_DIR>/.ephemeral/, where a
#: fingerprint script looking for dbs/<name>.db would find nothing and — written
#: carelessly — report a clean state.
DB_PREFIX = "dfr_"

SCHEMA_VERSION = "andamentum.experiment.docstore_deferred/1"


class IsolationError(RuntimeError):
    """The process is not safely isolated from the user's real document store."""


def _assert_isolated() -> Path:
    """Verify the environment before anything can touch a database. Raises."""
    raw = os.environ.get("DOCUMENT_STORE_DIR")
    if not raw:
        raise IsolationError(
            "DOCUMENT_STORE_DIR is not set. Without it the document store writes "
            f"to ~/.local/share/document-store/ — the user's REAL databases. "
            f"Set it to {EXP_DIR / 'dbs'} (the Snakefile does this for every rule)."
        )
    store_dir = Path(raw).resolve()
    try:
        store_dir.relative_to(EXP_DIR)
    except ValueError as exc:
        raise IsolationError(
            f"DOCUMENT_STORE_DIR={store_dir} resolves OUTSIDE the experiment "
            f"directory {EXP_DIR}. Refusing to run: this experiment must never "
            "write to a store it does not own."
        ) from exc

    if not os.environ.get("OLLAMA_BASE_URL"):
        raise IsolationError(
            "OLLAMA_BASE_URL is not set. pydantic-ai's Ollama provider then fails, "
            "extraction.py CATCHES that failure and returns empty defaults, and "
            "every drain reports success while writing no LLM metadata. "
            "Export OLLAMA_BASE_URL=http://localhost:11434/v1."
        )
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir


STORE_DIR = _assert_isolated()


def require_db_name(name: str) -> str:
    """Return ``name`` if it is a legal experiment database name, else raise."""
    if not name.startswith(DB_PREFIX):
        raise IsolationError(
            f"Database name {name!r} must start with {DB_PREFIX!r}. Names beginning "
            "ask_/test_/tmp/varfolders are silently redirected into an .ephemeral/ "
            "subdirectory by lifecycle.py, which would make every fingerprint wrong."
        )
    return name


def db_file(name: str) -> Path:
    """On-disk path of an experiment database (mirrors ``lifecycle.get_db_path``)."""
    return STORE_DIR / f"{require_db_name(name)}.db"


def require_cli_binary() -> Path:
    """The venv console script, or a loud failure.

    Lives here so the CHEAPEST rule can check it seconds into the run. The CLI
    arms must signal the console script DIRECTLY: spawning ``uv run
    andamentum-docstore`` and signalling puts a signal-forwarding wrapper in the
    process group, which turns one operator Ctrl-C into two deliveries and sends
    the handler down its force-exit branch — so the arm measures the forced path
    while believing it measured the cooperative one.
    """
    cli_bin = REPO_ROOT / ".venv" / "bin" / "andamentum-docstore"
    if not cli_bin.exists():
        raise FileNotFoundError(
            f"{cli_bin} is missing — the SIGTERM arms must signal the CLI process "
            "itself, not a `uv run` wrapper that would double-deliver the signal. "
            "Run `uv sync --extra dev --extra benchmark` first."
        )
    return cli_bin


def drop_database(name: str) -> None:
    """Delete a database and its WAL/SHM siblings — the clean-slate primitive."""
    base = db_file(name)
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(base) + suffix)
        if candidate.exists():
            candidate.unlink()


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def load_config() -> dict[str, Any]:
    """Read config.yaml. Fails loud if it is missing or malformed."""
    with (EXP_DIR / "config.yaml").open() as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml did not parse to a mapping")
    return cfg


CONFIG = load_config()
LLM_MODEL: str = CONFIG["models"]["llm"]
EMBEDDING_MODEL: str = CONFIG["models"]["embedding"]


# --------------------------------------------------------------------------
# Provenance linkage + JSON output
# --------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    """sha256 of a file's bytes, streamed."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """sha256 of a string's UTF-8 bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    """Timezone-aware UTC ISO-8601 timestamp."""
    return datetime.now(timezone.utc).isoformat()


def git_short_sha() -> str:
    """Short HEAD sha, or 'nogit' when git is unavailable (never raises)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "nogit"
    return out.stdout.strip() or "nogit"


def provenance_ref() -> dict[str, Any]:
    """The {run_id, file, sha256} stamp every result JSON carries.

    Read from results/provenance.json when it exists so that every artefact in a
    run points at the same immutable provenance record. Before that file exists
    (i.e. inside the provenance rule itself) a self-describing placeholder is
    returned rather than a fabricated hash.
    """
    if PROVENANCE_PATH.exists():
        with PROVENANCE_PATH.open() as fh:
            prov = json.load(fh)
        return {
            "run_id": prov.get("run_id", "unknown"),
            "file": str(PROVENANCE_PATH.relative_to(EXP_DIR)),
            "sha256": sha256_file(PROVENANCE_PATH),
        }
    return {
        "run_id": f"{utc_now()}-{git_short_sha()}",
        "file": None,
        "sha256": None,
        "note": "provenance.json did not exist when this artefact was written",
    }


def run_id() -> str:
    """The run identifier: '<UTC ISO8601>-<git short sha>' from provenance.json."""
    return str(provenance_ref()["run_id"])


def write_json(path: str | Path, payload: dict[str, Any], *, schema: str) -> Path:
    """Write a result JSON with the standard FAIR envelope. Creates parents.

    Every artefact carries its schema id, the run it belongs to, when it was
    written, and a hash-pinned pointer back to the provenance record — so a file
    found on its own years later is still interpretable.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema": schema,
        "schema_version": SCHEMA_VERSION,
        "written_at": utc_now(),
        "provenance_ref": provenance_ref(),
        **payload,
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(envelope, fh, indent=2, sort_keys=False, default=str)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(target)
    return target


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON artefact. Fails loud when absent — never returns a default."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(
            f"{target} is missing. This is an input the rule depends on; "
            "producing zeros instead would be a silent failure."
        )
    with target.open() as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# Verdict-carrying failure: a failed claim must still leave evidence
# --------------------------------------------------------------------------


class ClaimFailure(AssertionError):
    """A pre-registered assertion failed. Carries the observation that flipped it."""


class ClaimRecorder:
    """Collects per-hypothesis checks so a FAILED rule still writes its JSON.

    The five lineages are chained by artificial DAG edges (to serialise Ollama),
    so one failing rule blocks everything downstream. That is only acceptable if
    the failing rule leaves an inspectable artefact — hence: record every check,
    write the JSON with an explicit ``verdict``, and only THEN exit non-zero.
    """

    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(
        self,
        hypothesis: str,
        passed: bool,
        *,
        measured: Any,
        expected: Any,
        detail: str = "",
    ) -> bool:
        """Record one pre-registered check. Returns ``passed`` unchanged."""
        self.checks.append(
            {
                "hypothesis": hypothesis,
                "passed": bool(passed),
                "measured": measured,
                "expected": expected,
                "detail": detail,
            }
        )
        return bool(passed)

    def observe(self, name: str, value: Any, *, detail: str = "") -> None:
        """Record a measured observation that carries NO pass/fail threshold."""
        self.checks.append(
            {
                "hypothesis": name,
                "passed": None,
                "measured": value,
                "expected": None,
                "detail": detail or "observation only — no pre-registered threshold",
            }
        )

    @property
    def verdict(self) -> str:
        """PASS only when every thresholded check passed."""
        graded = [c for c in self.checks if c["passed"] is not None]
        if not graded:
            return "OBSERVATION_ONLY"
        return "PASS" if all(c["passed"] for c in graded) else "FAIL"

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [c for c in self.checks if c["passed"] is False]

    def payload(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "checks": self.checks}

    def raise_if_failed(self) -> None:
        """Exit non-zero AFTER the caller has written its artefact."""
        if self.failures:
            lines = [
                f"  - {c['hypothesis']}: measured={c['measured']!r} "
                f"expected={c['expected']!r} {c['detail']}"
                for c in self.failures
            ]
            raise ClaimFailure(
                "Pre-registered check(s) failed:\n" + "\n".join(lines)
            )


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------


class Stopwatch:
    """Context manager recording a monotonic duration into a dict.

    ``time.monotonic`` is authoritative for every published timing in this
    experiment; Snakemake's ``benchmark:`` wall-clock is a cross-check only (its
    memory/IO columns are literally 0 on this macOS host).
    """

    def __init__(self, sink: dict[str, float], key: str) -> None:
        self.sink = sink
        self.key = key
        self.seconds = 0.0

    def __enter__(self) -> "Stopwatch":
        self._t0 = time.monotonic()
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.monotonic() - self._t0
        self.sink[self.key] = self.seconds
