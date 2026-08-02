"""Pre-registration: every hypothesis, its numeric threshold, and its falsifier.

Written BEFORE any measurement exists, and ``analyze.py`` may only compare
against this file — it may not invent a threshold. That is the entire point: a
threshold chosen after seeing the number is not a threshold, it is a description.

Run:  uv run python -m experiments.docstore_deferred_ingestion.scripts.prereg
"""

from __future__ import annotations

from typing import Any

from . import _common as C

SCHEMA = "andamentum.experiment.docstore_deferred.preregistration/1"


HYPOTHESES: list[dict[str, Any]] = [
    {
        "id": "H-G",
        "claim": "enrichment gate",
        "statement": (
            "With OLLAMA_BASE_URL exported, one extract_chunk_metadata call on a fixed "
            "~4.3k-char string returns a non-empty topics list."
        ),
        "threshold": {"metric": "gate_topics_count", "op": ">=", "value": 1},
        "falsifier": "the gate call returns empty metadata with the variable set",
        "blocking": True,
        "why": (
            "extraction.py swallows provider errors and returns defaults (measured: "
            "0.47 s + empty topics unset vs 10.85 s + 2 topics set). Without this gate "
            "every downstream drain would report documents_enriched while writing empty "
            "LLM metadata, and the whole experiment would silently measure a degraded path."
        ),
    },
    {
        "id": "H-a1",
        "claim": "a",
        "statement": (
            "Across the whole ingest(process='defer') arm the in-process httpx ledger "
            "records ZERO requests to 127.0.0.1:11434 / localhost:11434."
        ),
        "threshold": {"metric": "defer_ollama_requests", "op": "==", "value": 0},
        "falsifier": ">= 1 request to the Ollama host during the defer arm",
        "why": (
            "The API permits model=None, embedding_model=None here (the defer branch "
            "returns before _preflight is ever called), so any request at all is a defect."
        ),
    },
    {
        "id": "H-a2",
        "claim": "a",
        "statement": (
            "Every deferred doc has 0 chunks, 0 chunk embeddings, no doc-embedding row, "
            "ingest_status='pending_enrich', and a metadata key set exactly "
            "{source, title} union the caller's own keys."
        ),
        "threshold": {"metric": "defer_state_violations", "op": "==", "value": 0},
        "falsifier": "any extra metadata key, or any chunk/embedding row",
        "why": (
            "Deliberately redundant with H-a1: transport evidence is machine-local and "
            "perishable, persisted state is re-checkable offline from the archived "
            "fingerprint."
        ),
    },
    {
        "id": "H-a3",
        "claim": "a",
        "statement": (
            "Median per-document ingest(process='defer') wall time < 1.0 s on a ~50k-char "
            "paper, and t_defer / t_now < 0.02 against the process='now' control."
        ),
        "threshold": {"metric": "defer_median_seconds", "op": "<", "value": 1.0},
        "threshold_secondary": {"metric": "defer_now_ratio", "op": "<", "value": 0.02},
        "falsifier": "median >= 1.0 s, or ratio >= 0.02",
        "why": (
            "Stated as a ratio because absolute wall-clock is not portable and must "
            "never be published as a reproducible constant. The ratio is computed on "
            "ONE document present on both sides — the control ingests that same "
            "markdown — so it is not a median-over-four divided by a single run."
        ),
    },
    {
        "id": "H-a4",
        "claim": "a",
        "statement": (
            "Every deferred document is FTS5-retrievable by a rare token from its own "
            "body BEFORE any drain runs, and its stored title equals the first non-empty "
            "line of the content stripped of heading marks."
        ),
        "threshold": {"metric": "defer_fts_misses", "op": "==", "value": 0},
        "falsifier": "any miss on either the FTS probe or the title",
    },
    {
        "id": "H-a5",
        "claim": "a",
        "statement": (
            "After pipeline.ingest_source(..., process='defer') returns, an FTS5 probe "
            "for a term unambiguously present in the PDF body returns 0 hits."
        ),
        "threshold": {"metric": "source_fts_hits", "op": "==", "value": 0},
        "falsifier": "the probe returns > 0 hits",
        "why": (
            "register_pending_source writes markdown_content=''. This is a SCOPE "
            "CORRECTION to the module docstring, not a bug report: 'searchable "
            "immediately' holds for ingest() and is over-general for ingest_source()."
        ),
    },
    {
        "id": "H-b",
        "claim": "b",
        "statement": (
            "Draining a 4-document markdown queue as max_docs=2 then unrestricted yields "
            "exactly 4 completions, pending==0, failed==0; and a THIRD drain returns "
            "documents_enriched==0, leaves the logical fingerprint byte-identical, and "
            "costs at most 2 Ollama requests (the per-process preflight: 1 embed + 1 chat)."
        ),
        "threshold": {"metric": "drain3_enriched", "op": "==", "value": 0},
        "threshold_secondary": {"metric": "drain3_ollama_requests", "op": "<=", "value": 2},
        "threshold_tertiary": {"metric": "fingerprint_moved", "op": "==", "value": 0},
        "falsifier": "remaining>0, failed>0, a moved fingerprint, or >2 requests in drain #3",
        "why": (
            "The fingerprint clause distinguishes 'the report says it did nothing' from "
            "'it did nothing'."
        ),
    },
    {
        "id": "H-c1",
        "claim": "c",
        "statement": (
            "After a SIGKILL delivered once document #1's conversion has committed AND "
            "its row reads pending_enrich, a fresh-process resume drain leaves EXACTLY 3 "
            "converter-ledger entries for 3 sources, no duplicate source string, and "
            "documents_converted==2 in the resume run."
        ),
        "threshold": {"metric": "ledger_entries", "op": "==", "value": 3},
        "threshold_secondary": {"metric": "ledger_duplicates", "op": "==", "value": 0},
        "threshold_tertiary": {"metric": "resume_converted", "op": "==", "value": 2},
        "falsifier": "any duplicate ledger entry, or a converted count of 3",
    },
    {
        "id": "H-c2",
        "claim": "c",
        "statement": (
            "sha256 of document #1's markdown_content read immediately after the SIGKILL "
            "is byte-identical to the sha256 read after the resume drain completes."
        ),
        "threshold": {"metric": "markdown_sha_stable", "op": "==", "value": 1},
        "falsifier": "any difference between the two hashes",
        "why": (
            "Strictly stronger than H-c1: a call counter cannot distinguish 'not "
            "re-converted' from 're-converted to the same string'. Compared WITHIN one "
            "run only — Docling output is not byte-stable across versions or machines."
        ),
    },
    {
        "id": "H-c3",
        "claim": "c",
        "statement": (
            "After the SIGKILL, PRAGMA integrity_check and quick_check both return 'ok', "
            "and the resume drain opens the database through the library's normal path "
            "with no manual repair and no deletion of any -wal/-shm file."
        ),
        "threshold": {"metric": "sqlite_ok", "op": "==", "value": 1},
        "falsifier": "corruption, a wedged lock, or a resume needing intervention",
    },
    {
        "id": "H-c4",
        "claim": "c",
        "statement": (
            "The supervisor records whether it EVER observes a row with non-empty "
            "markdown_content while ingest_status is still pending_source."
        ),
        "threshold": None,
        "observation_only": True,
        "falsifier": None,
        "why": (
            "_convert_document commits markdown via store.update(...) and only THEN calls "
            "set_ingest_status(PENDING_ENRICH) — two transactions. Observing that window "
            "is a reportable FINDING (such a document WOULD be re-converted). Observing "
            "NONE bounds nothing: the two commits are microseconds apart and the poller "
            "samples every 0.25 s, so a null result is what a non-existent window and a "
            "sub-millisecond window both look like. It is recorded as an observation "
            "precisely because it cannot support an inference either way."
        ),
    },
    {
        "id": "H-d1",
        "claim": "d",
        "statement": (
            "Pauses via max_docs, max_seconds, should_continue and SIGTERM to the real CLI "
            "(on both a truncated and a full-size in-flight document) all stop BETWEEN "
            "documents: on EACH ARM'S OWN post-drain snapshot no row is 'failed' and no "
            "non-complete row carries chunk or chunk-embedding rows; and the wall-clock "
            "each library arm spends after its last commit is under half of that same "
            "drain's longest single-document cost."
        ),
        "threshold": {"metric": "pause_midstage_documents", "op": "==", "value": 0},
        "threshold_secondary": {
            "metric": "pause_discarded_document_fraction",
            "op": "<",
            "value": 0.5,
        },
        "falsifier": (
            "any document left failed or partially enriched by a pause, or post-commit "
            "wall-clock amounting to a substantial fraction of a document"
        ),
        "why": (
            "All three stop conditions are checked at the TOP of the loop iteration and a "
            "source is converted AND enriched inside a single iteration — which is exactly "
            "why claim (c) needs a SIGKILL and cannot use a cooperative stop. BOTH metrics "
            "are measurements. A previous edition of this experiment registered "
            "'discarded work is exactly 0 seconds' against a hard-coded 0.0 literal, an "
            "assertion no behaviour of process_pending could falsify — including a "
            "regression that moved the stop checks into the middle of a document."
        ),
    },
    {
        "id": "H-d5",
        "claim": "d",
        "statement": (
            "The should_continue arm genuinely PAUSES rather than exhausting its queue: "
            "the drain reports stopped_early=True with work still queued after processing "
            "exactly one document."
        ),
        "threshold": {"metric": "should_continue_stopped_early", "op": "==", "value": 1},
        "threshold_secondary": {"metric": "should_continue_remaining", "op": ">=", "value": 1},
        "falsifier": "stopped_early=False, or remaining==0 — an exhausted queue, not a pause",
        "why": (
            "A previous edition ran this arm against whatever arms 1-2 left behind, which "
            "was exactly one document: the drain enriched it, the loop ended naturally, "
            "and the assertion (processed <= 1) passed without should_continue ever having "
            "caused a stop. The queue is now topped up first, and stopped_early is scored."
        ),
    },
    {
        "id": "H-d2",
        "claim": "d",
        "statement": (
            "Elapsed wall-clock of a drain with max_seconds=T satisfies "
            "T <= elapsed <= T + (longest single-document processing time in that drain)."
        ),
        "threshold": {"metric": "max_seconds_overrun_documents", "op": "<=", "value": 1},
        "falsifier": "an overrun exceeding one document's cost, or a stop mid-document",
        "why": (
            "The budget is checked BEFORE each document, so a drain always processes at "
            "least one document however small T is — recorded as an explicit observation."
        ),
    },
    {
        "id": "H-d3",
        "claim": "d",
        "statement": (
            "In a mixed queue of 1 source + 4 markdown documents, a max_docs=1 drain "
            "processes the SOURCE."
        ),
        "threshold": {"metric": "first_processed_is_source", "op": "==", "value": 1},
        "falsifier": "the first processed item is pending_enrich",
        "why": "list_pending orders sources first with an explicit CASE, not alphabetically.",
    },
    {
        "id": "H-d4",
        "claim": "d",
        "statement": (
            "After the SIGKILL the only work repeated on resume is the enrichment of the "
            "single in-flight document: EXACTLY one row is pending_enrich when the resume "
            "starts, and the repeated enrichment seconds (killed_at minus the converter "
            "ledger's last ts_end) are strictly less than that document's measured "
            "in-drain conversion time."
        ),
        "threshold": {
            "metric": "documents_pending_enrich_at_resume",
            "op": "==",
            "value": 1,
        },
        "threshold_secondary": {
            "metric": "kill_repeated_enrichment_over_conversion",
            "op": "<",
            "value": 1.0,
        },
        "falsifier": (
            "more or fewer than one document mid-stage at resume, or repeated enrichment "
            "costing more than the conversion the checkpoint preserved"
        ),
        "why": (
            "The seconds clause was previously in the statement with NO metric attached, "
            "so the row read PASS on the row-count proxy alone. It is now scored against "
            "a value measured on this lineage, not borrowed from another."
        ),
    },
    {
        "id": "H-e1",
        "claim": "e",
        "statement": (
            "On 8 paraphrase probes, chunk-embedding retrieval achieves recall@1 == 1.0 "
            "AFTER the drain (chance level 1/N over an N-document pool); and the full "
            "4-signal RRF stack, which scores ABOVE zero before the drain because the "
            "keyword index already exists, rises to recall@3 == 1.0 after it."
        ),
        "threshold": {"metric": "post_chunk_recall_at_1", "op": ">=", "value": 1.0},
        "threshold_secondary": {
            "metric": "pre_unified_recall_at_3",
            "op": "<",
            "value": 0.5,
        },
        "threshold_tertiary": {
            "metric": "post_unified_recall_at_3",
            "op": ">=",
            "value": 1.0,
        },
        "falsifier": "post-drain recall@1 < 1.0, or a unified stack that did not rise",
        "why": (
            "recall@1 not recall@3, and the UNIFIED stack not the chunk table, because "
            "the obvious pair is uninformative. `pre_chunk_recall_at_3 == 0` is "
            "guaranteed by H-a2's own assertion that deferred documents have zero "
            "chunks — semantic_search cannot return a row that does not exist — so it "
            "adds nothing and is demoted to an observation. Chance recall@3 over this "
            "candidate pool is 3/N, which recall@3 == 1.0 barely beats; recall@1 "
            "against 1/N is the statement with content. STATED LIMITATION, carried into "
            "claim_e_post and the report: 4 topically disjoint papers with no hard "
            "negatives rules out an empty or scrambled index, and does not measure "
            "retrieval quality. All numbers come from the deterministic retrieval "
            "functions with a precomputed query embedding; public.search() is a recorded "
            "smoke call only, because its LLM query planner varies run to run."
        ),
    },
    {
        "id": "H-e2",
        "claim": "e",
        "statement": (
            "Every completed document has chunks > 0, exactly one chunk-embedding row per "
            "chunk, and a non-empty LLM-extracted metadata key set."
        ),
        "threshold": {"metric": "enrichment_structure_violations", "op": "==", "value": 0},
        "falsifier": (
            "any completed document with zero chunks, a chunk/embedding count mismatch, "
            "or empty LLM metadata"
        ),
        "why": (
            "Doc-level embeddings are explicitly NOT asserted — _embed_doc_level swallows "
            "an oversized-input failure with an INFO log — so the doc-embedding skip rate "
            "is reported as a measured observation instead."
        ),
    },
    {
        "id": "H-f1",
        "claim": "f",
        "statement": (
            "A queue of 1 good PDF plus 4 genuinely broken sources ends with complete==1, "
            "failed==4, a non-empty ingest_error naming the exception TYPE on every failed "
            "row, and 4 corresponding strings in ProcessReport.failures."
        ),
        "threshold": {"metric": "failed_count", "op": "==", "value": 4},
        "threshold_secondary": {"metric": "errors_without_type_name", "op": "==", "value": 0},
        "falsifier": "a swallowed failure, an aborted drain, or an error string with no type name",
    },
    {
        "id": "H-f2",
        "claim": "f",
        "statement": (
            "retry_failed() returns 4 and sends all four back to pending_source; a document "
            "force-marked failed while its markdown is present is requeued to pending_enrich, "
            "and the following drain does NOT invoke the converter."
        ),
        "threshold": {"metric": "retry_wrong_stage", "op": "==", "value": 0},
        "threshold_secondary": {
            "metric": "stage_test_converter_calls_good_doc",
            "op": "==",
            "value": 0,
        },
        "falsifier": (
            "a wrong-stage requeue, or any re-invocation of the converter on the "
            "MARKDOWN-BEARING document"
        ),
        "why": (
            "An in-process cross-check of the same checkpoint guarantee H-c1/H-c2 test "
            "against a process kill. SCOPED TO THE MARKDOWN-BEARING DOCUMENT on purpose: "
            "retry_failed requeues every failed row, and three sources are still "
            "genuinely broken at that point. They hold no markdown, so returning them to "
            "pending_source and re-converting them is the SAME rule the hypothesis rests "
            "on, correctly applied. The unscoped total is reported separately as "
            "stage_test_converter_calls_total. (A previous edition registered the "
            "unscoped name, watched it measure 3 against an expected 0, and then "
            "re-pointed the name at the scoped field in the analyzer — so claims.json "
            "and claim_f_fail.json published different values under one identifier. "
            "The rename is recorded in results/deviations.json.)"
        ),
    },
    {
        "id": "H-x",
        "claim": "cross-cutting",
        "statement": (
            "For every drain — including the resume drain that runs in its own "
            "subprocess — all six ProcessReport fields reconcile against the database and "
            "the converter ledger: documents_converted against error-free ledger entries, "
            "documents_enriched / documents_failed against the complete / failed deltas, "
            "remaining against post-drain pending rows, documents_skipped against 0 when a "
            "converter was supplied and against the pre-drain pending_source count when it "
            "was not, and stopped_early against the implication that at least one document "
            "is still queued."
        ),
        "threshold": {"metric": "report_db_mismatches", "op": "==", "value": 0},
        "falsifier": "any mismatch",
        "why": (
            "The report is what a cron job logs and a UI shows; if it disagrees with the "
            "database, every operational claim built on it is unsafe. The statement names "
            "six fields because reconcile() now checks six; it previously named six and "
            "checked four, which turns a PASS into a claim nobody tested."
        ),
    },
    {
        "id": "H-m1",
        "claim": "cross-cutting",
        "statement": (
            "Enriching one document strictly sequentially costs no more than 1.3x its "
            "in-pipeline cost under the shipped Semaphore(_INGEST_CONCURRENCY=5) fan-out "
            "— i.e. the fan-out buys little against an Ollama that serialises same-model "
            "calls. The ratio is attributed by doc_uuid, never by title, and is reported "
            "INCONCLUSIVE when it sits within the measured replicate spread of 1.0."
        ),
        "threshold": {"metric": "concurrency_speedup", "op": "<=", "value": 1.3},
        "falsifier": "a speedup above 1.3, which would mean the fan-out is doing real work",
        "why": (
            "Registered so it is SCORED. It previously reached the report as an "
            "unqualified cross-cutting fact on n=1, computed against the wrong database "
            "row (two rows share this paper's title) with a Docling conversion and the "
            "drain preflight inside its denominator. The run contains a free replicate — "
            "the same content enriched twice in the same lineage — and the softening rule "
            "below uses it."
        ),
    },
]

#: HOW A MEASUREMENT BECOMES A VERDICT. Registered HERE, with the thresholds, and
#: read by analyze.py from this file — so the analyzer can no more invent a
#: scoring rule than it can invent a threshold. The INCONCLUSIVE rule previously
#: lived only in analyze.py and was rewritten after observing which way it scored.
SCORING_RULES: dict[str, Any] = {
    "comparison": "measured <op> value, exactly as registered above",
    "ratio_metrics": ["defer_now_ratio", "concurrency_speedup"],
    "ratio_metrics_note": (
        "Only dimensionless RATIOS may be softened. Counts are exact. "
        "defer_median_seconds was removed from this list: it is an absolute duration "
        "of a code path this same run proves makes zero Ollama requests (H-a1), so an "
        "LLM-latency band has nothing to say about it."
    ),
    "separation_statistic": "abs(log(measured / reference))",
    "separation_note": (
        "A log-ratio, not |measured-threshold|/threshold. The latter SATURATES at 1.0 "
        "as measured falls far below threshold, while a coefficient of variation is "
        "unbounded — so any CV above 1.0 made the softening fire hardest on the "
        "STRONGEST evidence. Both sides are relative quantities; compare them in log "
        "space."
    ),
    "bands": {
        "defer_now_ratio": {
            "reference": "the registered threshold",
            "band_source": "chat_latency_relative_spread",
            "band_definition": (
                "coefficient of variation of per-request latency on "
                "/v1/chat/completions ONLY, from the raw per-request JSONL ledgers. "
                "Pooling embedding calls (tens of milliseconds) with chat calls (tens "
                "of seconds) measures workload heterogeneity, not run-to-run variance."
            ),
        },
        "concurrency_speedup": {
            "reference": 1.0,
            "band_source": "in_pipeline_replicate_spread",
            "band_definition": (
                "relative range of the SAME content enriched twice inside claim_b's "
                "drains — the run's only true repeated measurement. A speedup within "
                "this of 1.0 is reported 'not distinguishable from 1.0', never as a "
                "measured effect."
            ),
        },
    },
    "verdicts": ["PASS", "FAIL", "INCONCLUSIVE", "NOT_MEASURED", "OBSERVATION"],
}


def build() -> dict[str, Any]:
    """Assemble the pre-registration payload."""
    cfg = C.CONFIG
    return {
        "title": (
            "Deferred ingestion on the real path: a pre-registered, ledger-instrumented "
            "validation of the document_store drain"
        ),
        "registered_at": C.utc_now(),
        "models": cfg["models"],
        "corpus": cfg["corpus"],
        "lineages": cfg["lineages"],
        "n_hypotheses": len(HYPOTHESES),
        "hypotheses": HYPOTHESES,
        "scoring_rules": SCORING_RULES,
        "not_reproducible": [
            "LLM-generated metadata CONTENT (nondeterministic even at temperature 0)",
            "wall-clock timings (machine- and load-dependent; ratios are published instead)",
            "Docling markdown bytes across docling versions or machines",
        ],
        "deterministic_claims": [
            "queue state transitions",
            "conversion bytes within a fixed docling version, within one run",
            "chunk boundaries for a fixed input and chunker configuration",
        ],
    }


def main() -> int:
    C.write_json(C.PREREG_PATH, build(), schema=SCHEMA)
    print(f"wrote {C.PREREG_PATH} ({len(HYPOTHESES)} hypotheses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
