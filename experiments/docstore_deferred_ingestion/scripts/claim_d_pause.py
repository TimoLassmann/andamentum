"""Claim (d): every cooperative pause stops BETWEEN documents and loses nothing.

Queue: 1 source (the smallest PDF) + 4 TRUNCATED markdown documents (~6000 chars,
about 2 chunks each).

SCOPE NOTE, stated rather than buried: pause semantics do not depend on document
size. The one size-sensitive number — the ``max_seconds`` overrun bound — is
expressed RELATIVE to the longest document observed in that same drain, so
truncation cannot flatter it. Truncating saves roughly 12 minutes and costs
nothing this claim needs. The one number that DOES transfer to an operator — the
SIGTERM-to-exit latency used to size a shutdown grace period — is therefore
measured a second time on an UNTRUNCATED document (arm 5).

  arm 1  max_docs=1        exactly 1 processed, stopped_early, and the item
                           processed was the SOURCE (H-d3 — list_pending orders
                           sources first with an explicit CASE)
  arm 2  max_seconds=T     records the overrun, plus the explicit observation that
                           a drain always processes at least one document however
                           small T is (the budget is checked BEFORE each document)
  arm 3  should_continue   flips False after the first on_progress; exactly 1 more.
                           The queue is TOPPED UP FIRST — see the note on that arm.
  arm 4  CLI + ONE SIGTERM the real signal handler on the real path: exit code 0,
                           "Paused before the queue was empty" on stdout, the
                           in-flight document committed as `complete`
  arm 5  the same, on an UNTRUNCATED in-flight document: the SIGTERM->exit latency
                           a launchd/cron user actually needs

DISCARDED WORK IS MEASURED, NOT ASSERTED. An earlier version of this script wrote
``pause_discarded_seconds = 0.0`` as a literal with the comment "0 by
CONSTRUCTION", and the pre-registration scored that constant against ``== 0``.
An assertion that compares a constant to itself cannot fail — including under a
regression that moved the stop checks into the middle of a document. Two real
measurements replace it:

  * ``mid_stage_documents`` over EACH ARM'S OWN post-drain snapshot (not the
    snapshot after the final unrestricted drain, which would have repaired
    exactly the damage the metric is looking for): a row that is ``failed``, or
    that carries chunk / chunk-embedding rows while not being ``complete``, is
    enrichment work committed and stranded.
  * ``post_commit_seconds`` per arm — wall-clock between the last committed
    document and the drain's return — expressed as a FRACTION of that same
    drain's longest single-document cost. A stop between documents leaves loop
    teardown; a stop inside one would leave the remainder of a document.

Run:  uv run python -m experiments.docstore_deferred_ingestion.scripts.claim_d_pause
"""

from __future__ import annotations

import asyncio
import itertools
import os
import signal
import subprocess
import time
from typing import Any

from . import _common as C
from .drain import instrumented_drain, mid_stage_documents
from .events import EventLog
from .fingerprint import snapshot
from .instrument import (
    counting_convert_fn,
    http_recorder,
    kill_process_group,
    truncate_ledger,
)
from .provenance import ollama_ps

SCHEMA = "andamentum.experiment.docstore_deferred.claim_d/1"

CLI_LOG = C.LOGS / "pause_cli.log"
CLI_FULL_LOG = C.LOGS / "pause_cli_fullsize.log"


def per_document_intervals(arm: dict[str, Any]) -> list[float]:
    """Per-document seconds inside one drain, from tick index 1 onwards.

    Index 0 is excluded on purpose: its predecessor is the drain's ``t0``, so the
    interval carries ``process_pending``'s preflight (and any conversion) rather
    than that document's own cost.
    """
    ticks = arm.get("progress") or []
    return [
        ticks[i]["monotonic_s"] - ticks[i - 1]["monotonic_s"]
        for i in range(1, len(ticks))
    ]


async def setup_queue(database: str) -> dict[str, Any]:
    """Queue 1 source + 4 truncated markdown documents into a clean database."""
    from andamentum.document_store import ingest
    from andamentum.document_store.pipeline import ingest_source

    cfg = C.CONFIG
    C.drop_database(database)

    registry = {p["arxiv_id"]: p for p in C.read_json(C.REGISTRY_PATH)["papers"]}
    baseline = C.read_json(C.RESULTS / "conversion_baseline.json")
    truncate_at = int(cfg["lineages"]["pause_markdown_chars"])

    small_id = cfg["small_paper"]
    source_doc_id = await ingest_source(
        database,
        str(C.EXP_DIR / registry[small_id]["path"]),
        title=None,
        metadata={"arxiv_id": small_id, "arm": "source"},
        process="defer",
    )

    markdown_docs = []
    for conv in baseline["conversions"][: int(cfg["lineages"]["pause_markdown_documents"])]:
        text = (C.EXP_DIR / conv["markdown_path"]).read_text()[:truncate_at]
        doc_id = await ingest(
            database,
            text,
            source=f"arxiv:{conv['arxiv_id']}",
            metadata={"arxiv_id": conv["arxiv_id"], "arm": "markdown"},
            model=None,
            embedding_model=None,
            process="defer",
        )
        markdown_docs.append(
            {"arxiv_id": conv["arxiv_id"], "doc_id": doc_id, "chars": len(text)}
        )

    return {
        "source_doc_id": source_doc_id,
        "source_arxiv_id": small_id,
        "markdown_docs": markdown_docs,
        "truncated_to_chars": truncate_at,
    }


async def setup_fullsize_queue(database: str) -> dict[str, Any]:
    """Arm 5's queue: truncated, FULL SIZE, truncated — in that order.

    ``list_pending`` orders ``pending_source`` first, then ``created_date ASC``.
    Every row here is markdown (no sources), so insertion order IS drain order and
    the full-size paper is deterministically the second document — the one in
    flight when the SIGTERM lands.
    """
    from andamentum.document_store import ingest

    cfg = C.CONFIG
    C.drop_database(database)
    baseline = C.read_json(C.RESULTS / "conversion_baseline.json")
    truncate_at = int(cfg["lineages"]["pause_markdown_chars"])

    # The SMALLEST full paper: still 25-40 chunks, but the cheapest such arm.
    full_conv = min(baseline["conversions"], key=lambda c: c["markdown_chars"])
    others = [c for c in baseline["conversions"] if c["arxiv_id"] != full_conv["arxiv_id"]]

    plan = [
        ("truncated_leader", others[0], truncate_at),
        ("fullsize_target", full_conv, None),
        ("truncated_tail", others[1], truncate_at),
    ]
    queued: list[dict[str, Any]] = []
    for role, conv, limit in plan:
        text = (C.EXP_DIR / conv["markdown_path"]).read_text()
        text = text[:limit] if limit else text
        doc_id = await ingest(
            database,
            f"# {role} {conv['arxiv_id']}\n\n{text}",
            source=f"arxiv:{conv['arxiv_id']}",
            metadata={"arxiv_id": conv["arxiv_id"], "arm": role},
            model=None,
            embedding_model=None,
            process="defer",
        )
        queued.append(
            {
                "role": role,
                "arxiv_id": conv["arxiv_id"],
                "doc_id": doc_id,
                "chars": len(text),
                "truncated": limit is not None,
            }
        )
    return {
        "database": database,
        "documents": queued,
        "why_three": (
            "the supervisor signals on the FIRST progress line and process_pending "
            "checks should_continue at the TOP of the next iteration, microseconds "
            "later — so document 2 is always already in flight and the pause can first "
            "take effect at the check before document 3"
        ),
    }


#: Distinguishes one ``top_up_queue`` call from the next. See the dedup note below.
_TOP_UP_CALLS = itertools.count()


async def top_up_queue(database: str, *, minimum_pending: int) -> dict[str, Any]:
    """Ensure at least ``minimum_pending`` cheap documents are queued. LLM-free.

    THE CALL COUNTER IS LOAD-BEARING. ``ingest`` deduplicates on a content hash,
    so two calls that build the same text produce the same hash and the second
    returns the EXISTING doc_id instead of queueing anything. This function is
    invoked twice (before arm 3 and before arm 4); with only ``index`` in the
    prefix, call 2's document 0 was byte-identical to call 1's document 0 and
    silently collapsed onto it. The CLI arm then ran against a 2-document queue,
    drained it to completion, and measured an exhausted queue rather than a
    pause — exactly the defect this experiment exists to have fixed, reappearing
    one arm to the left. Measured: '[1/2] ... remaining: 0' and no 'Paused' line.

    And because a silent shortfall is what made that possible, the post-condition
    is now CHECKED rather than assumed: if the queue is still short after the
    ingests, this raises instead of letting a later arm quietly measure nothing.
    """
    from andamentum.document_store import ingest, pending_status

    call = next(_TOP_UP_CALLS)
    status = await pending_status(database)
    needed = max(0, minimum_pending - status.pending)
    baseline = C.read_json(C.RESULTS / "conversion_baseline.json")
    truncate_at = int(C.CONFIG["lineages"]["pause_markdown_chars"])

    added: list[str] = []
    for index in range(needed):
        conv = baseline["conversions"][index % len(baseline["conversions"])]
        text = (C.EXP_DIR / conv["markdown_path"]).read_text()[:truncate_at]
        # The (call, index) pair makes every top-up document's content unique
        # across the whole rule, so neither the original documents nor an earlier
        # top-up can absorb it.
        added.append(
            await ingest(
                database,
                f"# top-up call {call} doc {index} {conv['arxiv_id']}\n\n{text}",
                source=f"arxiv:{conv['arxiv_id']}",
                metadata={"arxiv_id": conv["arxiv_id"], "arm": "top_up", "call": call},
                model=None,
                embedding_model=None,
                process="defer",
            )
        )

    after = await pending_status(database)
    if after.pending < minimum_pending:
        raise RuntimeError(
            f"top_up_queue({database!r}, minimum_pending={minimum_pending}) ended with "
            f"only {after.pending} pending ({status.pending} before, {len(added)} "
            "ingests issued). The most likely cause is content-hash dedup collapsing a "
            "top-up onto an existing row, which would leave the next arm measuring an "
            "exhausted queue instead of a pause. Failing loud rather than measuring "
            "nothing."
        )
    return {
        "call": call,
        "pending_before": status.pending,
        "pending_after": after.pending,
        "minimum_required": minimum_pending,
        "documents_added": added,
        "why": (
            "arms 1-3 consume an unpredictable number of documents (arm 2 stops on a "
            "wall-clock budget), and the CLI arm needs work left to pause after"
        ),
    }


def in_flight_document(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """The document a CLI arm committed while the SIGTERM was in flight.

    Its SIZE is what makes ``sigterm_to_exit_seconds`` interpretable: the same
    latency on a 2-chunk truncation and on a 37-chunk paper are different
    operational facts, and publishing the first as guidance for the second
    understates a shutdown grace period several-fold.
    """
    was_complete = {
        r["doc_uuid"] for r in before["documents"] if r["ingest_status"] == "complete"
    }
    newly = [
        r
        for r in after["documents"]
        if r["ingest_status"] == "complete" and r["doc_uuid"] not in was_complete
    ]
    if not newly:
        return {"doc_uuid": None, "note": "no document reached complete during this arm"}
    last = max(newly, key=lambda r: (r["ingest_updated_at"] or "", r["doc_uuid"]))
    return {
        "doc_uuid": last["doc_uuid"],
        "dc_title": last["dc_title"],
        "markdown_chars": last["markdown_chars"],
        "n_chunks": last["n_chunks"],
        "n_documents_committed_during_arm": len(newly),
    }


def run_cli_sigterm(database: str, *, log_file: Any = None) -> dict[str, Any]:
    """The real CLI, one SIGTERM, measured to exit.

    Verified in cli.py: ``_install_pause_handler`` flips a flag on the FIRST
    SIGINT/SIGTERM and force-exits 130 on the second. This arm sends exactly one,
    so a paused drain must be a NORMAL outcome (exit 0), not an error.

    ONE SIGNAL MEANS ONE DELIVERY — WHY THE CONSOLE SCRIPT IS INVOKED DIRECTLY.
    Spawning ``uv run andamentum-docstore`` and then ``killpg``-ing the group
    delivers SIGTERM TWICE to the CLI: once directly, and once forwarded by the
    ``uv`` wrapper, which is itself in the group. The handler then takes its
    "second signal — user means it" branch and force-exits 130, so the arm
    measured the FORCE path while believing it measured the cooperative one.
    (Observed exactly that: stderr carried both "SIGTERM received — finishing the
    current document" and "Forced exit — in-flight document not committed".)
    Running the venv console script directly removes the intermediary, so the
    process signalled is the process under test and one signal is one delivery.
    """
    C.LOGS.mkdir(parents=True, exist_ok=True)
    log_path = log_file or CLI_LOG
    cli_bin = C.require_cli_binary()
    cmd = [
        str(cli_bin), "process-pending", database,
        "--model", C.LLM_MODEL,
        "--embedding-model", C.EMBEDDING_MODEL,
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(C.REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
        env=os.environ.copy(),
    )
    try:

        stdout_lines: list[str] = []
        first_progress_at: float | None = None
        deadline = time.monotonic() + float(C.CONFIG["timeouts"]["cli_pause_exit_seconds"])

        assert proc.stdout is not None
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue
            stdout_lines.append(line.rstrip("\n"))
            # The CLI prints "[done/total] title" after each committed document.
            if line.startswith("[") and "/" in line:
                first_progress_at = time.monotonic()
                break

        if first_progress_at is None:
            kill_process_group(proc)
            raise TimeoutError(
                "The CLI never printed a '[N/M] title' progress line, so there was no "
                "committed document to pause after. Failing loud rather than signalling "
                "into an unknown state."
            )

        sent_at = time.monotonic()
        # send_signal, NOT killpg: the hypothesis is about ONE SIGTERM reaching
        # the CLI. Signalling the group would hit every process in it, and any
        # signal-forwarding parent turns one operator Ctrl-C into two deliveries
        # and a forced exit. Cleanup still uses the group (see the finally).
        proc.send_signal(signal.SIGTERM)
        try:
            rest_out, rest_err = proc.communicate(timeout=deadline - time.monotonic())
        except subprocess.TimeoutExpired:
            kill_process_group(proc)
            rest_out, rest_err = proc.communicate()
            raise TimeoutError("CLI did not exit after a single SIGTERM within the budget")
        exited_at = time.monotonic()

        stdout_text = "\n".join(stdout_lines) + "\n" + (rest_out or "")
        log_path.write_text(stdout_text + "\n--- STDERR ---\n" + (rest_err or ""))

        return {
            "returncode": proc.returncode,
            "sigterm_to_exit_seconds": exited_at - sent_at,
            "first_progress_to_sigterm_seconds": sent_at - first_progress_at,
            "stdout": stdout_text,
            "stderr": rest_err,
            "paused_message_present": "Paused before the queue was empty" in stdout_text,
            "log_file": str(log_path.relative_to(C.EXP_DIR)),
        }
    finally:
        # Guarantee the CLI's whole process group is gone even if this arm
        # raises mid-measurement. A leaked drain keeps mutating the database and
        # keeps calling Ollama, corrupting every later rule silently.
        kill_process_group(proc)


async def run() -> tuple[dict[str, Any], C.ClaimRecorder]:
    from andamentum.document_store.pipeline import harvest_convert_fn

    rec = C.ClaimRecorder()
    cfg = C.CONFIG
    pause_db = C.require_db_name(cfg["databases"]["pause"])
    db_path = C.db_file(pause_db)
    root_url = cfg["models"]["ollama_root_url"]
    log = EventLog(C.EVENTS / "claim_d.jsonl", rule="claim_d_pause", run_id=C.run_id())

    convert_ledger = truncate_ledger(C.LEDGERS / "d_convert.jsonl")
    convert_fn = counting_convert_fn(convert_ledger, harvest_convert_fn())

    queue = await setup_queue(pause_db)
    payload: dict[str, Any] = {
        "database": pause_db,
        "queue": queue,
        "scope_note": (
            "pause semantics do not depend on document size; the one size-sensitive "
            "number (the max_seconds overrun bound) is expressed relative to the longest "
            "document observed in that same drain"
        ),
        "ollama_ps_before": ollama_ps(root_url),
    }

    arms: dict[str, Any] = {}
    with http_recorder(C.LEDGERS / "d_http.jsonl") as ledger:
        # --- arm 1: max_docs=1 -------------------------------------------
        arm1 = await instrumented_drain(
            database=pause_db, db_path=db_path, label="d_arm1_max_docs",
            model=C.LLM_MODEL, embedding_model=C.EMBEDDING_MODEL,
            convert_fn=convert_fn, convert_ledger=convert_ledger,
            max_docs=1, event_log=log, http_ledger=ledger, snapshot_dir=C.SNAPSHOTS,
        )
        arms["arm1_max_docs"] = arm1
        processed_1 = arm1["report"]["documents_enriched"] + arm1["report"]["documents_failed"]
        rec.check(
            "H-d/arm1_count", processed_1 == 1, measured=processed_1, expected=1,
        )
        rec.check(
            "H-d/arm1_stopped_early",
            arm1["report"]["stopped_early"] is True,
            measured=arm1["report"]["stopped_early"], expected=True,
        )
        rec.check(
            "H-d/arm1_remaining",
            arm1["report"]["remaining"] == 4,
            measured=arm1["report"]["remaining"], expected=4,
        )
        first_is_source = arm1["report"]["documents_converted"] == 1
        payload["first_processed_is_source"] = first_is_source
        rec.check(
            "H-d3", first_is_source,
            measured=arm1["report"]["documents_converted"], expected=1,
            detail=(
                "list_pending orders pending_source before pending_enrich with an "
                "explicit CASE, so a partial run leaves the CHEAP work behind"
            ),
        )

        # --- arm 2: max_seconds ------------------------------------------
        # T is sized from the observed short-document cost: one document's worth,
        # so the drain is guaranteed to overrun by at most one more.
        observed = [p["monotonic_s"] for p in arm1["progress"]]
        budget = max(5.0, (observed[0] if observed else 30.0) * 0.5)
        arm2 = await instrumented_drain(
            database=pause_db, db_path=db_path, label="d_arm2_max_seconds",
            model=C.LLM_MODEL, embedding_model=C.EMBEDDING_MODEL,
            convert_fn=convert_fn, convert_ledger=convert_ledger,
            max_seconds=budget, event_log=log, http_ledger=ledger,
            snapshot_dir=C.SNAPSHOTS,
        )
        arms["arm2_max_seconds"] = arm2
        per_doc = [
            t["monotonic_s"] - (arm2["progress"][i - 1]["monotonic_s"] if i else 0.0)
            for i, t in enumerate(arm2["progress"])
        ]
        longest = max(per_doc) if per_doc else 0.0
        overrun = max(0.0, arm2["elapsed_seconds"] - budget)
        payload["max_seconds"] = {
            "budget_seconds": budget,
            "elapsed_seconds": arm2["elapsed_seconds"],
            "overrun_seconds": overrun,
            "longest_document_seconds": longest,
            "documents_processed": len(per_doc),
            "always_at_least_one": (
                "the budget is checked BEFORE each document, so a drain always processes "
                "at least one document however small T is"
            ),
        }
        rec.check(
            "H-d2",
            overrun <= longest + 1.0,
            measured=overrun,
            expected=f"<= longest single-document cost ({longest:.1f}s) + 1s slack",
            detail="an overrun larger than one document would mean a mid-document stop",
        )
        rec.observe(
            "H-d2/at_least_one",
            len(per_doc) >= 1,
            detail="a drain always processes at least one document however small T is",
        )

        # --- arm 3: should_continue --------------------------------------
        # The flag is driven by the event log's own tick count, which
        # instrumented_drain appends to as each document commits — so the pause
        # is genuinely caller-driven rather than a pre-decided document count.
        #
        # TOP UP FIRST. Arms 1 and 2 consume an unpredictable number of the 5
        # queued documents (arm 2 stops on a wall-clock budget, so its count
        # depends on machine speed). An earlier version ran this arm against
        # whatever was left, which on this host was exactly ONE document: the
        # drain enriched it, the loop ended because the queue was empty, and the
        # arm recorded `stopped_early: false, remaining: 0`. `should_continue`
        # never caused a stop, yet the assertion (`processed <= 1`) passed —
        # an exhausted queue satisfies it just as happily as a real pause. Two
        # pending documents is the smallest queue in which this mechanism is
        # observable at all: one commits, the flag flips, the check before the
        # second one stops the drain with work still queued.
        topped_up_arm3 = await top_up_queue(pause_db, minimum_pending=2)
        payload["arm3_top_up"] = topped_up_arm3

        state = {"ticks": 0}

        def count_tick(done: int, total: int, title: str) -> None:
            state["ticks"] += 1

        arm3 = await instrumented_drain(
            database=pause_db, db_path=db_path, label="d_arm3_should_continue",
            model=C.LLM_MODEL, embedding_model=C.EMBEDDING_MODEL,
            convert_fn=convert_fn, convert_ledger=convert_ledger,
            should_continue=lambda: state["ticks"] < 1,
            progress_hook=count_tick,
            event_log=log, http_ledger=ledger, snapshot_dir=C.SNAPSHOTS,
        )
        arms["arm3_should_continue"] = arm3
        payload["arm3_ticks_before_pause"] = state["ticks"]
        processed_3 = arm3["report"]["documents_enriched"] + arm3["report"]["documents_failed"]
        # THREE CONJUNCTS, none of which a natural loop exit can satisfy. `== 1`
        # (not `<= 1`) plus `stopped_early` plus work left behind is what
        # distinguishes "should_continue stopped it" from "it ran out of work".
        rec.check(
            "H-d/arm3_count",
            processed_3 == 1,
            measured=processed_3,
            expected=1,
            detail=(
                "should_continue is checked BETWEEN documents; the flag flips after the "
                "first commits"
            ),
        )
        rec.check(
            "H-d/arm3_stopped_early",
            arm3["report"]["stopped_early"] is True,
            measured=arm3["report"]["stopped_early"],
            expected=True,
            detail=(
                "an exhausted queue also ends the loop; only stopped_early distinguishes "
                "a caller-driven pause from running out of work"
            ),
        )
        rec.check(
            "H-d/arm3_remaining",
            arm3["report"]["remaining"] >= 1,
            measured=arm3["report"]["remaining"],
            expected=">= 1",
            detail="a pause must leave the unprocessed work queued, not consume it",
        )
        payload["arm3_http"] = arm3["http_delta_ollama_requests"]

    # --- arm 4: the real CLI + ONE SIGTERM --------------------------------
    # TOP-UP, stated rather than hidden: arms 1-3 consume an unpredictable number
    # of the 5 queued documents (arm 2 stops on a wall-clock budget, so its count
    # depends on machine speed). Topping up with cheap deferred markdown is free:
    # ingest(process="defer") makes no model call at all (that is H-a1, measured
    # separately).
    #
    # WHY THREE, NOT TWO. The supervisor sends SIGTERM when it sees the FIRST
    # "[n/m]" progress line, and ``process_pending`` checks ``should_continue()``
    # at the TOP of each iteration — microseconds after the ``on_progress`` call
    # that ends the previous one. No signal can be delivered inside that window,
    # so document 2 is always already in flight when the flag flips. The pause can
    # therefore first take effect at the check before document 3.
    #
    # With a 2-document queue the drain correctly runs to completion:
    # ``stopped_early`` is False, ``remaining`` is 0, and the CLI correctly prints
    # no "Paused" line — the arm was measuring an exhausted queue, not a pause.
    # Three pending documents is the smallest queue in which a cooperative stop is
    # observable at all. (Measured: a 2-document queue produced exit 0, enriched 2,
    # remaining 0, and no pause message.)
    topped_up = await top_up_queue(pause_db, minimum_pending=3)
    payload["arm4_top_up"] = topped_up

    before_cli = snapshot(db_path, label="d_arm4_pre")
    cli = run_cli_sigterm(pause_db)
    after_cli = snapshot(db_path, label="d_arm4_post")
    cli_in_flight = in_flight_document(before_cli, after_cli)
    arms["arm4_cli_sigterm"] = {
        "cli": {k: v for k, v in cli.items() if k not in ("stdout", "stderr")},
        "stdout_tail": cli["stdout"][-3000:],
        "stderr_tail": (cli["stderr"] or "")[-2000:],
        "in_flight_document": cli_in_flight,
        "snapshot_pre": before_cli,
        "snapshot_post": after_cli,
    }
    payload["sigterm_to_exit_seconds_truncated"] = cli["sigterm_to_exit_seconds"]
    payload["sigterm_truncated_in_flight"] = cli_in_flight
    rec.check(
        "H-d/arm4_exit_code",
        cli["returncode"] == 0,
        measured=cli["returncode"],
        expected=0,
        detail="a paused drain is a NORMAL outcome, not an error",
    )
    rec.check(
        "H-d/arm4_message",
        cli["paused_message_present"],
        measured=cli["paused_message_present"],
        expected=True,
        detail="'Paused before the queue was empty' is the operator-visible signal",
    )
    completed_delta = (
        after_cli["status_counts"]["complete"] - before_cli["status_counts"]["complete"]
    )
    rec.check(
        "H-d/arm4_committed",
        completed_delta >= 1,
        measured=completed_delta,
        expected=">= 1",
        detail="the in-flight document must be committed as complete, not abandoned",
    )

    # --- arm 5: the SAME SIGTERM, on an UNTRUNCATED in-flight document ------
    #
    # WHY A SECOND CLI ARM. `sigterm_to_exit_seconds` is the one number here an
    # operator acts on — it sizes a launchd/cron shutdown grace period. Arm 4
    # measures it on 6000-char truncations (~2 chunks) while the real corpus is
    # 39k-70k chars (25-43 chunks), so publishing arm 4's number as guidance
    # understates the grace period several-fold. This arm therefore builds its own
    # small queue in its own database, ordered so that the document IN FLIGHT when
    # the signal lands is a FULL-SIZE paper:
    #
    #   doc 1  truncated  -> commits quickly, emits the "[1/3]" line we signal on
    #   doc 2  FULL SIZE  -> in flight when the flag flips; the drain must finish it
    #   doc 3  truncated  -> never started; its existence is what makes the pause
    #                        observable (with a 2-document queue the drain
    #                        correctly runs to completion and prints no "Paused")
    full_db = C.require_db_name(cfg["databases"]["pause_full"])
    full_db_path = C.db_file(full_db)
    queue_full = await setup_fullsize_queue(full_db)
    payload["arm5_queue"] = queue_full

    before_full = snapshot(full_db_path, label="d_arm5_pre")
    cli_full = run_cli_sigterm(full_db, log_file=CLI_FULL_LOG)
    after_full = snapshot(full_db_path, label="d_arm5_post")
    full_in_flight = in_flight_document(before_full, after_full)
    arms["arm5_cli_sigterm_fullsize"] = {
        "cli": {k: v for k, v in cli_full.items() if k not in ("stdout", "stderr")},
        "stdout_tail": cli_full["stdout"][-3000:],
        "stderr_tail": (cli_full["stderr"] or "")[-2000:],
        "in_flight_document": full_in_flight,
        "snapshot_pre": before_full,
        "snapshot_post": after_full,
    }
    # THE headline operational number, and it comes from this arm, not arm 4.
    payload["sigterm_to_exit_seconds"] = cli_full["sigterm_to_exit_seconds"]
    payload["sigterm_in_flight_document"] = full_in_flight
    payload["sigterm_seconds_per_chunk"] = (
        cli_full["sigterm_to_exit_seconds"] / full_in_flight["n_chunks"]
        if full_in_flight.get("n_chunks")
        else None
    )
    payload["sigterm_measurement_note"] = (
        "sigterm_to_exit_seconds is measured with an UNTRUNCATED paper in flight "
        f"({full_in_flight.get('markdown_chars')} chars, "
        f"{full_in_flight.get('n_chunks')} chunks). "
        "sigterm_to_exit_seconds_truncated is the same measurement on a 6000-char "
        "truncation and is reported only for contrast — it is NOT operational guidance."
    )
    rec.check(
        "H-d/arm5_exit_code",
        cli_full["returncode"] == 0,
        measured=cli_full["returncode"],
        expected=0,
        detail="a paused drain is a NORMAL outcome, at any document size",
    )
    rec.check(
        "H-d/arm5_message",
        cli_full["paused_message_present"],
        measured=cli_full["paused_message_present"],
        expected=True,
    )
    rec.check(
        "H-d/arm5_fullsize_in_flight",
        bool(full_in_flight.get("n_chunks"))
        and full_in_flight["markdown_chars"] > int(cfg["lineages"]["pause_markdown_chars"]),
        measured=full_in_flight,
        expected=(
            f"an in-flight document larger than the "
            f"{cfg['lineages']['pause_markdown_chars']}-char truncation"
        ),
        detail=(
            "the whole point of this arm is that the committed document was NOT the "
            "cheap truncated one; without this the number is arm 4 again"
        ),
    )

    # --- the final unrestricted drain -------------------------------------
    with http_recorder(C.LEDGERS / "d_final_http.jsonl") as final_ledger:
        final = await instrumented_drain(
            database=pause_db, db_path=db_path, label="d_final",
            model=C.LLM_MODEL, embedding_model=C.EMBEDDING_MODEL,
            convert_fn=convert_fn, convert_ledger=convert_ledger,
            event_log=log, http_ledger=final_ledger, snapshot_dir=C.SNAPSHOTS,
        )
    arms["final_drain"] = final

    # --- discarded work: MEASURED per arm, on each arm's OWN snapshot -------
    #
    # The snapshot AFTER the final unrestricted drain cannot answer this question:
    # that drain repairs precisely the state the metric is hunting for. A document
    # genuinely stranded by a pause would be picked up and completed, and the list
    # would come back empty for the wrong reason. So each paused arm is judged on
    # the database as IT left it.
    midstage_by_arm: dict[str, list[dict[str, Any]]] = {
        "arm1_max_docs": mid_stage_documents(arm1["snapshot_post"]),
        "arm2_max_seconds": mid_stage_documents(arm2["snapshot_post"]),
        "arm3_should_continue": mid_stage_documents(arm3["snapshot_post"]),
        "arm4_cli_sigterm": mid_stage_documents(after_cli),
        "arm5_cli_sigterm_fullsize": mid_stage_documents(after_full),
    }
    stranded = [
        {"arm": arm, **row} for arm, rows in midstage_by_arm.items() for row in rows
    ]
    payload["pause_midstage_by_arm"] = midstage_by_arm
    payload["pause_midstage_documents"] = stranded
    rec.check(
        "H-d1",
        not stranded,
        measured=stranded,
        expected=[],
        detail=(
            "per-arm, on that arm's own post-drain snapshot: no row is 'failed' and no "
            "non-complete row carries chunk or chunk-embedding rows"
        ),
    )

    # Wall-clock committed after the last document, per library arm, as a fraction
    # of that same drain's longest single-document cost. Loop teardown is a small
    # fraction; the remainder of a document would be a large one.
    discarded: dict[str, Any] = {}
    fractions: list[float] = []
    pooled_longest = max(
        (
            value
            for arm in (arm1, arm2, arm3, final)
            for value in per_document_intervals(arm)
        ),
        default=0.0,
    )
    for name, arm in (
        ("arm1_max_docs", arm1),
        ("arm2_max_seconds", arm2),
        ("arm3_should_continue", arm3),
    ):
        intervals = per_document_intervals(arm)
        longest = max(intervals) if intervals else pooled_longest
        post_commit = arm.get("post_commit_seconds")
        fraction = (post_commit / longest) if (post_commit is not None and longest) else None
        if fraction is not None:
            fractions.append(fraction)
        discarded[name] = {
            "post_commit_seconds": post_commit,
            "longest_document_seconds": longest,
            "fraction_of_one_document": fraction,
        }
    payload["pause_discarded_by_arm"] = discarded
    payload["pause_discarded_seconds"] = sum(
        d["post_commit_seconds"] or 0.0 for d in discarded.values()
    )
    payload["pause_discarded_document_fraction"] = max(fractions) if fractions else None
    payload["pause_longest_document_seconds"] = pooled_longest
    payload["discarded_work_rationale"] = (
        "MEASURED, not asserted. pause_discarded_seconds sums the wall-clock each "
        "library arm spent after its last committed document; the fraction expresses "
        "the worst of those against that drain's own longest single-document cost. A "
        "stop inside a document would leave that document's remainder here. The CLI "
        "arms are judged structurally instead (mid_stage_documents on their own "
        "snapshots), because their in-flight document is deliberately allowed to finish "
        "— that is what a cooperative SIGTERM means."
    )
    rec.check(
        "H-d1/discarded_fraction",
        payload["pause_discarded_document_fraction"] is not None
        and payload["pause_discarded_document_fraction"] < 0.5,
        measured=payload["pause_discarded_document_fraction"],
        expected="< 0.5 of one document",
        detail=(
            "post-commit wall-clock is loop teardown, not the remainder of an "
            "interrupted document"
        ),
    )

    # Separately, and under its own name: everything DOES eventually complete once
    # the queue is drained without restriction. That is a different claim from
    # "no pause stranded anything", and conflating the two is what let the
    # mid-stage check be answered by the repair rather than by the pause.
    final_snapshot = final["snapshot_post"]
    incomplete_after_final = [
        r["doc_uuid"]
        for r in final_snapshot["documents"]
        if r["ingest_status"] != "complete"
    ]
    payload["incomplete_after_final_drain"] = incomplete_after_final
    rec.check(
        "H-d/eventually_complete",
        not incomplete_after_final,
        measured=incomplete_after_final,
        expected=[],
        detail="after five pauses, an unrestricted drain finishes every queued document",
    )

    mismatches = sum(
        arm["reconciliation"]["n_mismatches"]
        for arm in (arm1, arm2, arm3, final)
    )
    payload["report_db_mismatches"] = mismatches
    rec.check("H-x/claim_d", mismatches == 0, measured=mismatches, expected=0)

    payload["arms"] = arms
    payload["ollama_ps_after"] = ollama_ps(root_url)
    payload["converter_calls_total"] = sum(
        arm["converter_calls"] for arm in (arm1, arm2, arm3, final)
    )
    rec.observe(
        "converter_calls_total",
        payload["converter_calls_total"],
        detail=(
            "exactly 1 expected: the single queued source, converted once in arm 1. The "
            "CLI arm's conversions are not counted here because the CLI supplies its own "
            "harvest converter — that arm is measured through the database instead"
        ),
    )
    return payload, rec


def main() -> int:
    payload: dict[str, Any] = {}
    rec = C.ClaimRecorder()
    try:
        payload, rec = asyncio.run(run())
    finally:
        C.write_json(
            C.RESULTS / "claim_d_pause.json", {**payload, **rec.payload()}, schema=SCHEMA
        )
    print(f"verdict: {rec.verdict}")
    rec.raise_if_failed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
