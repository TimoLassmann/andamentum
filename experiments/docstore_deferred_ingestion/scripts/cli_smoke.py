"""The CLI surface itself: argument parsing, output shape, exit codes.

The CLI is part of the feature under test, and until this rule existed exactly
ONE of its six subcommands had ever been executed against a real database
(``process-pending``, by claim (d)'s SIGTERM arm). The library equivalents are
well covered, so this is a coverage gap rather than a false claim — but it means
five subcommands' argument parsing, output formatting and exit codes had never
run at all.

Every subcommand here is exercised through the venv CONSOLE SCRIPT, not through
``uv run`` and not through an in-process import, because the exit code and the
stdout a user actually sees are the things under test.

CHEAP BY CONSTRUCTION. Everything except the final ``process-pending --max-docs
1`` is LLM-free:

    ingest --defer      returns before _preflight is ever called (that is H-a1)
    ingest-source       registers a pending_source row; no conversion
    status              one GROUP BY
    retry-failed        one UPDATE over failed rows

The single ``process-pending`` invocation drains ONE truncated document so the
drain-and-report path is exercised end to end, including ``--max-docs`` and
``--max-seconds``, which claim (d) drives through the library only.

Run:  uv run python -m experiments.docstore_deferred_ingestion.scripts.cli_smoke
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from typing import Any

from . import _common as C
from .fingerprint import snapshot

SCHEMA = "andamentum.experiment.docstore_deferred.cli_smoke/1"

SMOKE_DIR = C.EXP_DIR / "data" / "cli_smoke"


def run_cli(*args: str, timeout: float = 900.0) -> dict[str, Any]:
    """One CLI invocation, recorded verbatim. Never raises on a non-zero exit.

    A non-zero exit is DATA here — the assertions below decide what it means —
    so the record is written either way and the caller fails loud on the value.
    """
    cli_bin = C.require_cli_binary()
    proc = subprocess.run(
        [str(cli_bin), *args],
        cwd=str(C.REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=os.environ.copy(),
    )
    return {
        "argv": ["andamentum-docstore", *args],
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr_tail": (proc.stderr or "")[-2000:],
    }


def parse_status(stdout: str) -> dict[str, int]:
    """The `status` subcommand's own output, parsed back into numbers.

    Parsing what the user is shown — rather than re-querying the database — is
    the point: this checks the FORMATTING, which nothing else in the experiment
    touches.
    """
    out: dict[str, int] = {}
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value.isdigit():
            out[key] = int(value)
    return out


async def prepare() -> dict[str, Any]:
    """Materialise the smoke fixtures. No CLI, no LLM — just files on disk."""
    cfg = C.CONFIG
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    baseline = C.read_json(C.RESULTS / "conversion_baseline.json")
    truncate_at = int(cfg["lineages"]["pause_markdown_chars"])

    markdown = SMOKE_DIR / "note.md"
    markdown.write_text(
        "# CLI smoke note\n\n"
        + (C.EXP_DIR / baseline["conversions"][0]["markdown_path"]).read_text()[
            :truncate_at
        ]
    )
    registry = {p["arxiv_id"]: p for p in C.read_json(C.REGISTRY_PATH)["papers"]}
    good_pdf = C.EXP_DIR / registry[cfg["small_paper"]]["path"]
    # A source that genuinely fails, so `retry-failed` has something to requeue.
    broken = SMOKE_DIR / "broken.pdf"
    broken.write_bytes(b"not a pdf, 40 bytes of noise for the CLI\n")
    return {
        "markdown": str(markdown),
        "markdown_chars": len(markdown.read_text()),
        "good_pdf": str(good_pdf),
        "broken_pdf": str(broken),
    }


def run() -> tuple[dict[str, Any], C.ClaimRecorder]:
    rec = C.ClaimRecorder()
    cfg = C.CONFIG
    database = C.require_db_name(cfg["databases"]["cli"])
    db_path = C.db_file(database)
    C.drop_database(database)

    fixtures = asyncio.run(prepare())
    invocations: list[dict[str, Any]] = []
    payload: dict[str, Any] = {"database": database, "fixtures": fixtures}

    def step(name: str, *args: str, timeout: float = 900.0) -> dict[str, Any]:
        result = run_cli(*args, timeout=timeout)
        result["step"] = name
        invocations.append(result)
        print(f"{name}: exit {result['returncode']}", flush=True)
        return result

    # --- ingest --defer : the LLM-free capture path ------------------------
    ingest = step("ingest --defer", "ingest", database, fixtures["markdown"], "--defer")
    rec.check(
        "CLI/ingest_defer_exit",
        ingest["returncode"] == 0,
        measured=ingest["returncode"],
        expected=0,
    )
    rec.check(
        "CLI/ingest_defer_output",
        ingest["stdout"].startswith("queued: "),
        measured=ingest["stdout"].strip()[:80],
        expected="'queued: <doc_id>'",
        detail="the word distinguishes a deferred capture from an immediate ingest",
    )

    # --- ingest-source : queue a PDF without converting it -----------------
    source = step("ingest-source", "ingest-source", database, fixtures["good_pdf"])
    rec.check(
        "CLI/ingest_source_exit",
        source["returncode"] == 0,
        measured=source["returncode"],
        expected=0,
    )
    broken = step("ingest-source (broken)", "ingest-source", database, fixtures["broken_pdf"])
    rec.check(
        "CLI/ingest_source_broken_exit",
        broken["returncode"] == 0,
        measured=broken["returncode"],
        expected=0,
        detail="queueing a source never converts it, so a broken file queues cleanly",
    )

    # --- status : the formatting nothing else exercises --------------------
    status = step("status", "status", database)
    parsed = parse_status(status["stdout"])
    payload["status_parsed"] = parsed
    db_counts = snapshot(db_path, label="cli_smoke_after_queue")["status_counts"]
    payload["status_database_counts"] = db_counts
    rec.check(
        "CLI/status_exit", status["returncode"] == 0, measured=status["returncode"], expected=0
    )
    rec.check(
        "CLI/status_matches_database",
        parsed.get("pending_source") == db_counts["pending_source"]
        and parsed.get("pending_enrich") == db_counts["pending_enrich"]
        and parsed.get("complete") == db_counts["complete"]
        and parsed.get("failed") == db_counts["failed"],
        measured=parsed,
        expected=db_counts,
        detail="what the operator is shown must be what the database holds",
    )
    rec.check(
        "CLI/status_total_pending",
        parsed.get("total pending") == db_counts["pending_source"] + db_counts["pending_enrich"],
        measured=parsed.get("total pending"),
        expected=db_counts["pending_source"] + db_counts["pending_enrich"],
    )

    # --- process-pending --max-docs 1 : the drain path, capped -------------
    # The ONE model-touching step. --max-docs is driven through the library by
    # claim (d); this is the only place its CLI flag is parsed.
    drain = step(
        "process-pending --max-docs 1",
        "process-pending",
        database,
        "--model",
        C.LLM_MODEL,
        "--embedding-model",
        C.EMBEDDING_MODEL,
        "--max-docs",
        "1",
        timeout=float(cfg["timeouts"]["cli_pause_exit_seconds"]),
    )
    rec.check(
        "CLI/process_pending_exit",
        drain["returncode"] == 0,
        measured=drain["returncode"],
        expected=0,
    )
    rec.check(
        "CLI/process_pending_paused_message",
        "Paused before the queue was empty" in drain["stdout"],
        measured=drain["stdout"][-400:],
        expected="the pause line, because --max-docs 1 leaves 2 documents queued",
        detail="--max-docs is a cooperative stop and must be reported as one",
    )
    for field in ("converted", "enriched", "failed", "remaining"):
        rec.check(
            f"CLI/process_pending_reports_{field}",
            f"{field} " in drain["stdout"],
            measured=field in drain["stdout"],
            expected=True,
            detail="the report block a cron job's log actually captures",
        )

    # --- process-pending --max-seconds : the other cap ---------------------
    budget = step(
        "process-pending --max-seconds 1",
        "process-pending",
        database,
        "--model",
        C.LLM_MODEL,
        "--embedding-model",
        C.EMBEDDING_MODEL,
        "--max-seconds",
        "1",
        timeout=float(cfg["timeouts"]["cli_pause_exit_seconds"]),
    )
    rec.check(
        "CLI/max_seconds_exit",
        budget["returncode"] == 0,
        measured=budget["returncode"],
        expected=0,
        detail=(
            "the budget is checked BEFORE each document, so a 1-second budget still "
            "processes exactly one and exits normally"
        ),
    )

    # --- retry-failed : requeue whatever the broken source left behind -----
    after_drains = snapshot(db_path, label="cli_smoke_after_drains")
    payload["failed_before_retry"] = after_drains["status_counts"]["failed"]
    retry = step("retry-failed", "retry-failed", database)
    rec.check(
        "CLI/retry_failed_exit",
        retry["returncode"] == 0,
        measured=retry["returncode"],
        expected=0,
    )
    rec.check(
        "CLI/retry_failed_output",
        "requeued " in retry["stdout"] and "failed document(s)" in retry["stdout"],
        measured=retry["stdout"].strip()[:80],
        expected="'requeued N failed document(s)'",
    )
    after_retry = snapshot(db_path, label="cli_smoke_after_retry")
    payload["snapshot_after_retry"] = after_retry
    rec.check(
        "CLI/retry_failed_cleared",
        after_retry["status_counts"]["failed"] == 0,
        measured=after_retry["status_counts"]["failed"],
        expected=0,
        detail="every failed row goes back to a pending stage",
    )

    payload["invocations"] = [
        {**inv, "stdout": inv["stdout"][-2000:]} for inv in invocations
    ]
    payload["n_invocations"] = len(invocations)
    payload["subcommands_exercised"] = sorted(
        {inv["argv"][1] for inv in invocations}
    )
    return payload, rec


def main() -> int:
    payload: dict[str, Any] = {}
    rec = C.ClaimRecorder()
    try:
        payload, rec = run()
    finally:
        C.write_json(C.RESULTS / "cli_smoke.json", {**payload, **rec.payload()}, schema=SCHEMA)
    print(f"verdict: {rec.verdict}")
    rec.raise_if_failed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
