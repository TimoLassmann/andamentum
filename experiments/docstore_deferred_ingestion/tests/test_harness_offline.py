"""Offline tests of the instruments themselves. No Ollama, no network, no store.

An instrument you have not tested is not evidence. These cover the four places a
silent harness bug would corrupt a headline number:

  * the converter ledger (H-c1's entire falsifier is a line count)
  * the logical fingerprint (H-b's "it really did nothing")
  * the recall metric (claim (e)'s only quantity)
  * the prereg/analyze contract (a threshold that analyze cannot resolve would
    silently become NOT_MEASURED rather than FAIL)
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from experiments.docstore_deferred_ingestion.scripts import corpus, instrument
from experiments.docstore_deferred_ingestion.scripts.analyze import (
    COMPARATORS,
    RESULT_FILES,
    resolve_metrics,
    score,
)
from experiments.docstore_deferred_ingestion.scripts.fingerprint import (
    MissingDatabaseError,
    document_rows,
    logical_fingerprint,
    status_counts,
)
from experiments.docstore_deferred_ingestion.scripts.prereg import HYPOTHESES, SCORING_RULES
from experiments.docstore_deferred_ingestion.scripts.retrieval import aggregate, score_probe


# ---------------------------------------------------------------------------
# Converter ledger
# ---------------------------------------------------------------------------


def test_ledger_records_one_line_per_invocation(tmp_path: Path) -> None:
    ledger = tmp_path / "convert.jsonl"

    async def fake_convert(source: str) -> str:
        return f"# {source}\n\nbody"

    wrapped = instrument.counting_convert_fn(ledger, fake_convert)
    asyncio.run(wrapped("a.pdf"))
    asyncio.run(wrapped("b.pdf"))

    records = instrument.read_jsonl(ledger)
    assert len(records) == 2
    assert instrument.ledger_sources(ledger) == ["a.pdf", "b.pdf"]
    assert instrument.duplicate_sources(ledger) == []
    assert all(r["sha256"] and r["chars"] > 0 for r in records)


def test_ledger_detects_a_repeated_conversion(tmp_path: Path) -> None:
    """The exact falsifier for H-c1: the same source converted twice."""
    ledger = tmp_path / "convert.jsonl"

    async def fake_convert(source: str) -> str:
        return "x"

    wrapped = instrument.counting_convert_fn(ledger, fake_convert)
    asyncio.run(wrapped("a.pdf"))
    asyncio.run(wrapped("a.pdf"))
    assert instrument.duplicate_sources(ledger) == ["a.pdf"]


def test_ledger_records_a_failed_conversion(tmp_path: Path) -> None:
    """A failed conversion is still an INVOCATION and must be counted."""
    ledger = tmp_path / "convert.jsonl"

    async def broken(source: str) -> str:
        raise FileNotFoundError(source)

    wrapped = instrument.counting_convert_fn(ledger, broken)
    with pytest.raises(FileNotFoundError):
        asyncio.run(wrapped("gone.pdf"))
    records = instrument.read_jsonl(ledger)
    assert len(records) == 1
    assert records[0]["error_type"] == "FileNotFoundError"


def test_truncate_ledger_empties_it(tmp_path: Path) -> None:
    ledger = tmp_path / "l.jsonl"
    instrument.append_jsonl(ledger, {"a": 1})
    instrument.truncate_ledger(ledger)
    assert instrument.read_jsonl(ledger) == []


def test_read_jsonl_raises_on_a_corrupt_line(tmp_path: Path) -> None:
    """A corrupt ledger must fail loud, never be silently partially parsed."""
    ledger = tmp_path / "l.jsonl"
    ledger.write_text('{"a": 1}\nnot json\n')
    with pytest.raises(ValueError):
        instrument.read_jsonl(ledger)


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def _make_db(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    """Minimal documents table — enough for the fingerprint's projection."""
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_uuid TEXT UNIQUE NOT NULL,
            file_path TEXT UNIQUE NOT NULL,
            dc_title TEXT, markdown_content TEXT, file_hash TEXT,
            source_file_path TEXT, created_date TEXT NOT NULL,
            updated_date TEXT NOT NULL, metadata TEXT DEFAULT '{}',
            ingest_status TEXT, ingest_error TEXT, ingest_updated_at TEXT,
            deleted_at TEXT
        )
        """
    )
    for uuid, status, content, metadata in rows:
        conn.execute(
            "INSERT INTO documents (doc_uuid, file_path, dc_title, markdown_content, "
            "file_hash, created_date, updated_date, metadata, ingest_status) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (uuid, f"f://{uuid}", uuid, content, "h", "t", "t", metadata, status),
        )
    conn.commit()
    conn.close()


def test_fingerprint_is_stable_and_meaning_sensitive(tmp_path: Path) -> None:
    db = tmp_path / "dfr_x.db"
    _make_db(
        db,
        [
            ("u1", "pending_enrich", "hello", '{"source":"s","title":"t"}'),
            ("u2", "complete", "world", '{"source":"s","title":"t"}'),
        ],
    )
    rows_a = document_rows(db)
    rows_b = document_rows(db)
    assert logical_fingerprint(rows_a) == logical_fingerprint(rows_b)

    # A status change MUST move the fingerprint — that is H-b's whole mechanism.
    moved = [dict(r) for r in rows_a]
    moved[0]["ingest_status"] = "complete"
    assert logical_fingerprint(moved) != logical_fingerprint(rows_a)


def test_fingerprint_ignores_row_order(tmp_path: Path) -> None:
    db = tmp_path / "dfr_y.db"
    _make_db(db, [("b", "complete", "x", "{}"), ("a", "complete", "y", "{}")])
    rows = document_rows(db)
    assert logical_fingerprint(rows) == logical_fingerprint(list(reversed(rows)))


def test_status_counts_always_reports_all_four_keys(tmp_path: Path) -> None:
    db = tmp_path / "dfr_z.db"
    _make_db(db, [("a", "complete", "x", "{}")])
    counts = status_counts(document_rows(db))
    assert set(counts) == {"pending_source", "pending_enrich", "complete", "failed"}
    assert counts["complete"] == 1


def test_missing_database_fails_loud_rather_than_reporting_zeros(tmp_path: Path) -> None:
    """A zeroed fingerprint and a clean database look identical — refuse to conflate."""
    with pytest.raises(MissingDatabaseError):
        document_rows(tmp_path / "absent.db")


# ---------------------------------------------------------------------------
# Retrieval metric
# ---------------------------------------------------------------------------


def test_score_probe_ranks() -> None:
    assert score_probe(["a", "b", "c"], "a")["recall_at_1"] == 1.0
    assert score_probe(["a", "b", "c"], "c")["recall_at_1"] == 0.0
    assert score_probe(["a", "b", "c"], "c")["recall_at_3"] == 1.0
    assert score_probe(["a", "b", "c", "d"], "d")["recall_at_3"] == 0.0
    assert score_probe([], "a")["reciprocal_rank"] == 0.0
    assert score_probe(["a", "b"], "b")["reciprocal_rank"] == 0.5


def test_score_probe_accepts_any_row_holding_the_right_paper() -> None:
    """The same paper can legally exist under two rows (markdown + queued source)."""
    assert score_probe(["x", "b"], {"a", "b"})["recall_at_3"] == 1.0
    assert score_probe(["b", "a"], {"a", "b"})["recall_at_1"] == 1.0
    assert score_probe(["x", "y"], {"a", "b"})["recall_at_3"] == 0.0


def test_acceptable_doc_ids_includes_the_source_probe_row() -> None:
    """Otherwise claim (e) would fail for a bookkeeping reason, not a retrieval one."""
    from experiments.docstore_deferred_ingestion.scripts.retrieval import (
        acceptable_doc_ids,
    )

    claim_a = {
        "defer_records": [
            {"short": "adam", "doc_id": "md-adam"},
            {"short": "bert", "doc_id": "md-bert"},
        ],
        "source_probe": {"arxiv_id": "1412.6980v9", "doc_id": "src-adam"},
    }
    mapping = acceptable_doc_ids(claim_a)
    assert set(mapping["adam"]) == {"md-adam", "src-adam"}
    assert mapping["bert"] == ["md-bert"]


def test_aggregate_averages_across_probes() -> None:
    per_probe = [
        {"signals": {"s": score_probe(["a"], "a")}},
        {"signals": {"s": score_probe(["b"], "a")}},
    ]
    agg = aggregate(per_probe, "s")
    assert agg["recall_at_1"] == 0.5
    assert agg["n_probes"] == 2


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def test_corpus_is_well_formed() -> None:
    corpus.validate_corpus()


def test_every_arxiv_id_is_version_pinned() -> None:
    """An unversioned URL silently serves the latest revision — pinning is the point."""
    for paper in corpus.PAPERS:
        assert "v" in paper.arxiv_id.rsplit(".", 1)[-1], paper.arxiv_id


def test_probes_avoid_the_obvious_keyword() -> None:
    """A probe containing the model's name would measure string matching, not semantics."""
    banned = {
        "attention": ("transformer",),
        "bert": ("bert",),
        "resnet": ("resnet", "residual network"),
        "adam": ("adam",),
    }
    for probe in corpus.PROBE_QUERIES:
        for term in banned[probe.expect_short]:
            assert term not in probe.query.lower(), (probe.expect_short, term)


# ---------------------------------------------------------------------------
# The prereg / analyze contract
# ---------------------------------------------------------------------------


def test_every_pre_registered_metric_is_resolvable() -> None:
    """A threshold analyze cannot resolve would silently become NOT_MEASURED."""
    resolvable = set(resolve_metrics({}))
    missing = []
    for hypothesis in HYPOTHESES:
        for key in ("threshold", "threshold_secondary", "threshold_tertiary"):
            threshold = hypothesis.get(key)
            if threshold and threshold["metric"] not in resolvable:
                missing.append((hypothesis["id"], threshold["metric"]))
    assert not missing, f"analyze.resolve_metrics has no entry for: {missing}"


def test_every_threshold_uses_a_known_comparator() -> None:
    for hypothesis in HYPOTHESES:
        for key in ("threshold", "threshold_secondary", "threshold_tertiary"):
            threshold = hypothesis.get(key)
            if threshold:
                assert threshold["op"] in COMPARATORS, (hypothesis["id"], threshold["op"])


def test_scoring_marks_an_absent_measurement_not_measured() -> None:
    """A missing artefact must never score as PASS by defaulting to zero."""
    hypothesis = {
        "id": "H-test",
        "claim": "t",
        "statement": "s",
        "threshold": {"metric": "defer_ollama_requests", "op": "==", "value": 0},
        "falsifier": "f",
    }
    rows = score(hypothesis, {"defer_ollama_requests": None}, {"usable": False}, SCORING_RULES)
    assert rows[0]["verdict"] == "NOT_MEASURED"


def test_scoring_pass_and_fail() -> None:
    hypothesis = {
        "id": "H-test",
        "claim": "t",
        "statement": "s",
        "threshold": {"metric": "defer_ollama_requests", "op": "==", "value": 0},
        "falsifier": "f",
    }
    assert (
        score(hypothesis, {"defer_ollama_requests": 0}, {"usable": False}, SCORING_RULES)[0]["verdict"]
        == "PASS"
    )
    assert (
        score(hypothesis, {"defer_ollama_requests": 3}, {"usable": False}, SCORING_RULES)[0]["verdict"]
        == "FAIL"
    )


def test_a_ratio_inside_the_noise_band_is_inconclusive_not_pass() -> None:
    """The band must SOFTEN a marginal pass, never harden a fail into a pass."""
    hypothesis = {
        "id": "H-a3",
        "claim": "a",
        "statement": "s",
        "threshold": {"metric": "defer_now_ratio", "op": "<", "value": 0.02},
        "falsifier": "f",
    }
    band = {"usable": True, "relative_spread": 0.30}
    row = score(hypothesis, {"defer_now_ratio": 0.019}, band, SCORING_RULES)[0]
    assert row["verdict"] == "INCONCLUSIVE"
    # A ratio far below the threshold is a clean PASS, not softened.
    clean = score(hypothesis, {"defer_now_ratio": 0.0001}, band, SCORING_RULES)[0]
    assert clean["verdict"] == "PASS"


def test_count_metrics_are_never_softened() -> None:
    """Counts are exact; only wall-clock ratios are subject to the band."""
    hypothesis = {
        "id": "H-a1",
        "claim": "a",
        "statement": "s",
        "threshold": {"metric": "defer_ollama_requests", "op": "==", "value": 0},
        "falsifier": "f",
    }
    row = score(
        hypothesis,
        {"defer_ollama_requests": 1},
        {"usable": True, "relative_spread": 5.0},
        SCORING_RULES,
    )[0]
    assert row["verdict"] == "FAIL"


def test_result_files_map_covers_every_claim_rule() -> None:
    expected = {
        "gate_llm.json",
        "claim_a_defer.json",
        "claim_e_pre.json",
        "claim_b_drain.json",
        "claim_e_post.json",
        "micro_stages.json",
        "claim_c_kill.json",
        "claim_c_resume.json",
        "claim_d_pause.json",
        "claim_f_fail.json",
        "drain_overhead.json",
    }
    assert set(RESULT_FILES.values()) == expected


# ---------------------------------------------------------------------------
# Claim recorder — a FAILED rule must still leave an artefact
# ---------------------------------------------------------------------------


def test_claim_recorder_writes_a_verdict_before_raising() -> None:
    from experiments.docstore_deferred_ingestion.scripts._common import (
        ClaimFailure,
        ClaimRecorder,
    )

    rec = ClaimRecorder()
    rec.check("H-x", False, measured=1, expected=0)
    payload = rec.payload()
    assert payload["verdict"] == "FAIL"
    assert json.dumps(payload)  # the artefact must be serialisable
    with pytest.raises(ClaimFailure):
        rec.raise_if_failed()


def test_observation_only_recorder_is_not_a_failure() -> None:
    from experiments.docstore_deferred_ingestion.scripts._common import ClaimRecorder

    rec = ClaimRecorder()
    rec.observe("something", 42)
    assert rec.verdict == "OBSERVATION_ONLY"
    rec.raise_if_failed()


# ---------------------------------------------------------------------------
# Database-name safety
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["test_drain", "tmp_x", "ask_thing", "varfolders_y", "brain"])
def test_illegal_database_names_are_refused(name: str) -> None:
    """lifecycle.EPHEMERAL_PREFIXES silently redirects these; 'brain' is the user's real store."""
    from experiments.docstore_deferred_ingestion.scripts._common import (
        IsolationError,
        require_db_name,
    )

    with pytest.raises(IsolationError):
        require_db_name(name)


def test_legal_database_names_are_accepted() -> None:
    from experiments.docstore_deferred_ingestion.scripts._common import require_db_name

    for name in ("dfr_main", "dfr_kill", "dfr_empty"):
        assert require_db_name(name) == name


# ---------------------------------------------------------------------------
# The fixes for edition 1's unfalsifiable checks
#
# Each test below pins a defect an independent review found: a metric that could
# not fail, a bar that was a literal, an attribution computed the wrong way. They
# are offline on purpose — they must hold regardless of what any run measures.
# ---------------------------------------------------------------------------


def test_per_path_concurrency_is_not_the_global_depth() -> None:
    """The bug: a global counter grouped by path, published as per-endpoint.

    Edition 1 reported `/v1/chat/completions: 7` for a code path bounded at
    Semaphore(5), because concurrent /api/embeddings calls inflated the global
    reading sampled at a chat request's start.
    """
    ledger = instrument.HttpLedger(path=Path("x"))
    # 1 chat in flight; 5 embeddings concurrent with it -> global depth reaches 6.
    ledger.records = [
        {"path": "/v1/chat/completions", "global_depth_at_start": 6, "in_flight_at_start_path": 1},
        {"path": "/api/embeddings", "global_depth_at_start": 6, "in_flight_at_start_path": 5},
    ]
    assert ledger.max_in_flight == 6, "the global figure is still available"
    assert ledger.max_in_flight_by_path() == {
        "/v1/chat/completions": 1,
        "/api/embeddings": 5,
    }
    assert ledger.max_in_flight_for("/chat/completions") == 1


def test_latency_summaries_are_available_per_endpoint() -> None:
    """Pooling a 2 ms embed with a 20 s chat describes neither distribution."""
    ledger = instrument.HttpLedger(path=Path("x"))
    ledger.records = [
        {"path": "/v1/chat/completions", "ts_start": 0.0, "ts_end": 20.0},
        {"path": "/api/embeddings", "ts_start": 0.0, "ts_end": 0.002},
    ]
    by_path = ledger.latency_seconds_by_path()
    assert by_path["/v1/chat/completions"]["max"] == 20.0
    assert by_path["/api/embeddings"]["max"] == 0.002
    assert ledger.latencies("/chat/completions") == [20.0]


def test_attribution_refuses_the_first_tick() -> None:
    """Tick 0's interval starts at the drain's t0 and carries the preflight.

    Edition 1's title-prefix match landed on tick 0 of the wrong row, so the
    "enrichment seconds" it produced contained a Docling conversion.
    """
    from experiments.docstore_deferred_ingestion.scripts.drain import (
        attribute_document_seconds,
    )

    drains = [
        {
            "label": "d1",
            "progress": [
                {"doc_uuid": "first", "monotonic_s": 265.0},
                {"doc_uuid": "second", "monotonic_s": 545.0},
            ],
        }
    ]
    refused = attribute_document_seconds(drains, "first")
    assert refused["seconds"] is None and "refused" in refused["note"]

    attributed = attribute_document_seconds(drains, "second")
    assert attributed["seconds"] == pytest.approx(280.0)
    assert attributed["tick_index"] == 1

    absent = attribute_document_seconds(drains, "nowhere")
    assert absent["seconds"] is None


def test_mid_stage_detects_stranded_enrichment() -> None:
    """The metric must be able to FAIL. Edition 1's could not.

    A row carrying chunk rows while not `complete` is enrichment work committed
    and then stranded — exactly the discarded work claim (d) says cannot exist.
    """
    from experiments.docstore_deferred_ingestion.scripts.drain import mid_stage_documents

    clean = {
        "documents": [
            {"doc_uuid": "a", "ingest_status": "complete", "n_chunks": 12, "n_chunk_embeddings": 12},
            {"doc_uuid": "b", "ingest_status": "pending_enrich", "n_chunks": 0, "n_chunk_embeddings": 0},
        ]
    }
    assert mid_stage_documents(clean) == []

    stranded = {
        "documents": [
            {"doc_uuid": "c", "ingest_status": "pending_enrich", "n_chunks": 7, "n_chunk_embeddings": 7},
        ]
    }
    assert [row["doc_uuid"] for row in mid_stage_documents(stranded)] == ["c"]

    failed = {
        "documents": [
            {"doc_uuid": "d", "ingest_status": "failed", "n_chunks": 0, "n_chunk_embeddings": 0},
        ]
    }
    assert [row["reason"] for row in mid_stage_documents(failed)] == ["failed"]


def test_reconcile_covers_all_six_report_fields() -> None:
    """H-x names six fields; edition 1's reconcile() checked four."""
    from experiments.docstore_deferred_ingestion.scripts.drain import reconcile

    pre = {"status_counts": {"pending_source": 2, "pending_enrich": 1, "complete": 0, "failed": 0}}
    post = {"status_counts": {"pending_source": 0, "pending_enrich": 0, "complete": 3, "failed": 0}}
    report = {
        "documents_converted": 2,
        "documents_enriched": 3,
        "documents_failed": 0,
        "documents_skipped": 0,
        "remaining": 0,
        "stopped_early": False,
    }
    result = reconcile(
        report,
        pre,
        post,
        ledger_delta=[{"source": "a", "error_type": None}, {"source": "b", "error_type": None}],
        convert_fn_supplied=True,
    )
    assert set(result["fields_checked"]) == {
        "documents_converted",
        "documents_enriched",
        "documents_failed",
        "documents_skipped",
        "remaining",
        "stopped_early",
    }
    assert result["n_mismatches"] == 0


def test_reconcile_catches_a_skipped_count_that_should_be_zero() -> None:
    from experiments.docstore_deferred_ingestion.scripts.drain import reconcile

    pre = {"status_counts": {"pending_source": 1, "pending_enrich": 0, "complete": 0, "failed": 0}}
    post = {"status_counts": {"pending_source": 1, "pending_enrich": 0, "complete": 0, "failed": 0}}
    report = {
        "documents_converted": 0,
        "documents_enriched": 0,
        "documents_failed": 0,
        # A converter WAS supplied, so nothing may be reported skipped.
        "documents_skipped": 1,
        "remaining": 1,
        "stopped_early": False,
    }
    result = reconcile(report, pre, post, ledger_delta=[], convert_fn_supplied=True)
    assert [m["field"] for m in result["mismatches"]] == ["documents_skipped"]


def test_stopped_early_without_queued_work_is_a_mismatch() -> None:
    """stopped_early implies at least one document is still queued."""
    from experiments.docstore_deferred_ingestion.scripts.drain import reconcile

    counts = {"pending_source": 0, "pending_enrich": 0, "complete": 1, "failed": 0}
    report = {
        "documents_converted": 0,
        "documents_enriched": 1,
        "documents_failed": 0,
        "documents_skipped": 0,
        "remaining": 0,
        "stopped_early": True,
    }
    result = reconcile(
        report,
        {"status_counts": {**counts, "pending_enrich": 1, "complete": 0}},
        {"status_counts": counts},
        ledger_delta=[],
        convert_fn_supplied=True,
    )
    assert [m["field"] for m in result["mismatches"]] == ["stopped_early"]


def test_scoring_rules_are_pre_registered_not_invented_by_the_analyzer() -> None:
    """The softening rule must live in the register, with the thresholds."""
    assert set(SCORING_RULES["ratio_metrics"]) == {"defer_now_ratio", "concurrency_speedup"}
    # An absolute duration of an LLM-free path is not softenable by an LLM band.
    assert "defer_median_seconds" not in SCORING_RULES["ratio_metrics"]
    for metric in SCORING_RULES["ratio_metrics"]:
        assert metric in SCORING_RULES["bands"], metric
        assert SCORING_RULES["bands"][metric]["band_source"]


def test_concurrency_speedup_is_judged_against_one_not_its_threshold() -> None:
    """A 1.01 speedup inside a 9% replicate spread is INCONCLUSIVE, not a result."""
    hypothesis = {
        "id": "H-m1",
        "claim": "cross-cutting",
        "statement": "s",
        "threshold": {"metric": "concurrency_speedup", "op": "<=", "value": 1.3},
        "falsifier": "f",
    }
    band = {"usable": True, "relative_spread": 0.2, "in_pipeline_replicate_spread": 0.094}
    row = score(hypothesis, {"concurrency_speedup": 1.01}, band, SCORING_RULES)[0]
    assert row["verdict"] == "INCONCLUSIVE"
    assert row["band_reference"] == 1.0

    # A genuine fan-out gain is far enough from 1.0 to be reported as measured.
    real = score(hypothesis, {"concurrency_speedup": 2.5}, band, SCORING_RULES)[0]
    assert real["verdict"] == "FAIL"


def test_every_glob_artefact_has_a_documented_schema() -> None:
    """101 of 122 manifest entries carried schema: null, ledgers included."""
    from experiments.docstore_deferred_ingestion.scripts.manifest import _spec_for

    for relative in (
        "results/ledgers/kill_convert.jsonl",
        "results/ledgers/b_http.jsonl",
        "results/snapshots/d_arm1_max_docs_pre.json",
        "results/events/claim_d.jsonl",
        "figures/figures_status.json",
    ):
        spec = _spec_for(relative)
        assert spec.get("schema"), relative
        assert spec.get("produced_by"), relative


def test_deviations_name_the_hypotheses_they_could_have_moved() -> None:
    from experiments.docstore_deferred_ingestion.scripts.deviations import DEVIATIONS

    ids = {h["id"] for h in HYPOTHESES}
    assert DEVIATIONS, "an empty amendment trail is not a trail"
    for row in DEVIATIONS:
        assert row["what"] and row["why"] and row["could_have_moved_the_verdict"]
        for affected in row["affects"]:
            # Free-text scope notes are allowed; hypothesis-shaped ids must exist.
            if affected.startswith("H-") and " " not in affected:
                assert affected in ids, affected
