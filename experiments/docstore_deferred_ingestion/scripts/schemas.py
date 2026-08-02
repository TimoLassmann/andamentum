"""ONE definition of every artefact's shape — the validator and the human doc.

``results/SCHEMAS.json`` (machine, used by ``validate_results``) and
``results/SCHEMAS.md`` (human table: field, type, units, meaning) are BOTH
generated from the dictionary below. Generating the documentation and the
enforcement from one definition is what stops the docs drifting from what is
actually checked.

Run:  uv run python -m experiments.docstore_deferred_ingestion.scripts.schemas
"""

from __future__ import annotations

from typing import Any

from . import _common as C

SCHEMA = "andamentum.experiment.docstore_deferred.schemas/1"


def field(name: str, type_: str, units: str, meaning: str) -> dict[str, str]:
    return {"name": name, "type": type_, "units": units, "meaning": meaning}


ENVELOPE_FIELDS = [
    field("schema", "string", "-", "the artefact's schema id"),
    field("schema_version", "string", "-", "the experiment-wide schema generation"),
    field("written_at", "ISO-8601 UTC", "-", "when this file was written"),
    field(
        "provenance_ref",
        "object",
        "-",
        "{run_id, file, sha256} pointing at results/provenance.json",
    ),
]

ARTEFACTS: dict[str, dict[str, Any]] = {
    "results/preregistration.json": {
        "schema": "andamentum.experiment.docstore_deferred.preregistration/1",
        "produced_by": "prereg",
        "description": (
            "Every hypothesis with its numeric threshold and falsifier, written before "
            "any measurement exists. analyze.py may only compare against this file."
        ),
        "required": ["hypotheses", "n_hypotheses"],
        "fields": [
            field("hypotheses[].id", "string", "-", "hypothesis id, e.g. H-c2"),
            field("hypotheses[].claim", "string", "-", "which lettered claim it belongs to"),
            field("hypotheses[].threshold", "object|null", "-", "{metric, op, value}"),
            field("hypotheses[].falsifier", "string|null", "-", "what would flip it"),
        ],
    },
    "results/provenance.json": {
        "schema": "andamentum.experiment.provenance/1",
        "produced_by": "provenance",
        "description": "Run identity: git, versions, model digests, machine, allowlisted env.",
        "required": ["run_id", "git", "machine", "packages", "ollama"],
        "fields": [
            field("run_id", "string", "-", "'<UTC ISO8601>-<git short sha>'"),
            field(
                "ollama.pinned_models.<name>.digest",
                "string",
                "-",
                "THE load-bearing pin: the tag is mutable, the digest is not",
            ),
            field("git.dirty", "bool", "-", "whether the working tree had uncommitted changes"),
            field(
                "environment_allowlisted",
                "object",
                "-",
                "only DOCUMENT_STORE_DIR / OLLAMA_BASE_URL / TZ / LANG — never the full environ",
            ),
        ],
    },
    "data/REGISTRY.json": {
        "schema": "andamentum.experiment.docstore_deferred.registry/1",
        "produced_by": "fetch_pdfs",
        "description": "Per-paper input provenance. Committed; data/pdfs/ is gitignored.",
        "required": ["papers", "registry_version"],
        "fields": [
            field("papers[].arxiv_id", "string", "-", "VERSIONED id, e.g. 1706.03762v7"),
            field("papers[].sha256", "hex string", "-", "content hash of the downloaded PDF"),
            field("papers[].bytes", "int", "bytes", "file size"),
            field("papers[].final_url", "string", "-", "URL after redirects"),
            field("papers[].etag", "string|null", "-", "upstream ETag, when offered"),
            field("papers[].download_seconds", "float", "s", "wall-clock of the fetch"),
        ],
    },
    "results/data_integrity.json": {
        "schema": "andamentum.experiment.docstore_deferred.data_integrity/1",
        "produced_by": "verify_registry",
        "description": "Re-hash of every registered PDF. Exits 1 on drift.",
        "required": ["verdict", "checks", "n_failed"],
        "fields": [
            field("checks[].ok", "bool", "-", "sha256 and size both match the registry"),
        ],
    },
    "results/conversion_baseline.json": {
        "schema": "andamentum.experiment.docstore_deferred.conversion_baseline/1",
        "produced_by": "convert_reference",
        "description": (
            "Standalone harvest.extract per PDF: the markdown corpus, and the per-PDF "
            "conversion cost that prices the checkpoint."
        ),
        "required": ["conversions", "docling_version"],
        "fields": [
            field("conversions[].conversion_seconds", "float", "s", "harvest.extract wall-clock"),
            field("conversions[].markdown_chars", "int", "characters", "extracted length"),
            field(
                "conversions[].markdown_sha256",
                "hex string",
                "-",
                "valid WITHIN a docling version only — never compared across runs",
            ),
        ],
    },
    "results/gate_llm.json": {
        "schema": "andamentum.experiment.docstore_deferred.gate_llm/1",
        "produced_by": "gate_llm",
        "description": "H-G. The blocking precondition that the LLM really enriches.",
        "required": ["verdict", "topics", "topics_count", "extract_chunk_metadata_seconds"],
        "fields": [
            field("topics_count", "int", "count", "non-empty is the pass condition"),
            field(
                "extract_chunk_metadata_seconds",
                "float",
                "s",
                "a sub-second value is the signature of the swallowed-provider-error path",
            ),
            field("resolved_ollama_base_url", "string", "-", "what the process actually saw"),
        ],
    },
    "results/claim_a_defer.json": {
        "schema": "andamentum.experiment.docstore_deferred.claim_a/1",
        "produced_by": "claim_a_defer",
        "description": "H-a1..H-a5: defer is LLM-free, fast, findable — but not for sources.",
        "required": ["verdict", "defer_records", "defer_ollama_requests", "defer_now_ratio"],
        "fields": [
            field("defer_ollama_requests", "int", "count", "MUST be 0 (H-a1)"),
            field("defer_median_seconds", "float", "s", "median per-document defer cost"),
            field("defer_now_ratio", "float", "ratio", "t_defer / t_now — the portable claim"),
            field("source_probe.self_hits", "int", "count", "MUST be 0 (H-a5)"),
        ],
    },
    "results/claim_e_pre.json": {
        "schema": "andamentum.experiment.docstore_deferred.claim_e_pre/1",
        "produced_by": "claim_e_pre",
        "description": "Claim (e) control arm + the auto_repair tripwire.",
        "required": ["verdict", "probes", "pre_chunk_recall_at_3"],
        "fields": [
            field("pre_chunk_recall_at_3", "float", "0-1", "must be 0 — there are no chunks yet"),
            field("fingerprint_moved", "bool", "-", "tripwire: search()'s auto_repair must not drain"),
        ],
    },
    "results/claim_b_drain.json": {
        "schema": "andamentum.experiment.docstore_deferred.claim_b/1",
        "produced_by": "claim_b_drain",
        "description": "H-b: resumable, and a no-op drain is genuinely a no-op.",
        "required": ["verdict", "drains", "drain3_enriched", "drain3_ollama_requests"],
        "fields": [
            field("drains[].fingerprint_pre", "hex string", "-", "logical fingerprint before"),
            field("drains[].fingerprint_moved", "bool", "-", "did the meaning change"),
            field("drains[].reconciliation.n_mismatches", "int", "count", "H-x, must be 0"),
            field("drain3_ollama_requests", "int", "count", "the no-op drain's request count"),
        ],
    },
    "results/claim_e_post.json": {
        "schema": "andamentum.experiment.docstore_deferred.claim_e_post/1",
        "produced_by": "claim_e_post",
        "description": "H-e1/H-e2: the drain created semantic reachability.",
        "required": ["verdict", "probes", "post_chunk_recall_at_3", "per_document"],
        "fields": [
            field("post_chunk_recall_at_3", "float", "0-1", "chunk-embedding recall@3 after"),
            field(
                "doc_embedding_skip_rate",
                "float",
                "0-1",
                "MEASURED not asserted — _embed_doc_level swallows oversized input",
            ),
            field("public_search_smoke", "object", "-", "RECORDED, never scored (LLM planner)"),
        ],
    },
    "results/micro_stages.json": {
        "schema": "andamentum.experiment.docstore_deferred.micro_stages/1",
        "produced_by": "micro_stages",
        "description": "Sequential attribution of enrichment, and the fan-out ratio.",
        "required": ["verdict", "timings_seconds", "sequential_total_seconds"],
        "fields": [
            field("timings_seconds.<stage>", "float", "s", "per-component sequential cost"),
            field(
                "concurrency_speedup",
                "float|null",
                "ratio",
                "sequential total / in-pipeline seconds; ~1.0 means Semaphore(5) is decoration",
            ),
            field("per_chunk_seconds", "float[]", "s", "the unit of run-to-run variance"),
        ],
    },
    "results/claim_c_kill.json": {
        "schema": "andamentum.experiment.docstore_deferred.claim_c_kill/1",
        "produced_by": "claim_c_kill",
        "description": "The SIGKILL, the gate that opened it, and the post-kill truth.",
        "required": ["verdict", "pre_kill_markdown_sha256", "sqlite_integrity", "hc4_window"],
        "fields": [
            field("pre_kill_markdown_sha256", "hex string", "-", "H-c2's before value"),
            field("sqlite_integrity.integrity_ok", "bool", "-", "H-c3"),
            field("hc4_window.observed", "bool", "-", "the two-transaction window, if seen"),
        ],
    },
    "results/claim_c_resume.json": {
        "schema": "andamentum.experiment.docstore_deferred.claim_c_resume/1",
        "produced_by": "claim_c_resume",
        "description": "H-c1/H-c2/H-c3/H-d4 and the checkpoint's price in seconds.",
        "required": ["verdict", "ledger_entries", "ledger_duplicates", "markdown_sha_stable"],
        "fields": [
            field("ledger_entries", "int", "count", "MUST equal the number of sources"),
            field("ledger_duplicates", "string[]", "-", "MUST be empty"),
            field("markdown_sha_stable", "bool", "-", "H-c2, within one run only"),
            field(
                "checkpoint_savings_seconds",
                "float",
                "s",
                "conversion seconds not repeated, taken from THIS lineage's own "
                "converter-ledger entry. INCLUDES one-time Docling initialisation "
                "(that conversion was first in its process); the measured warm cost "
                "and docling_init_seconds sit in checkpoint_savings_basis so the "
                "marginal saving is legible",
            ),
            field(
                "repeated_enrichment_seconds",
                "float",
                "s",
                "MEASURED on this lineage: killed_at minus the converter ledger's last "
                "ts_end — the enrichment the resume must redo. Previously this field "
                "copied micro_stages' sequential total, a different paper in a "
                "different database",
            ),
            field(
                "repeated_enrichment_over_conversion",
                "float",
                "ratio",
                "H-d4's second threshold: repeated enrichment must cost less than the "
                "conversion the checkpoint preserved",
            ),
            field(
                "reconciliation.n_mismatches",
                "int",
                "count",
                "H-x for the subprocess resume drain, which previously bypassed it",
            ),
        ],
    },
    "results/claim_d_pause.json": {
        "schema": "andamentum.experiment.docstore_deferred.claim_d/1",
        "produced_by": "claim_d_pause",
        "description": "Four pause mechanisms, all stopping between documents.",
        "required": ["verdict", "arms", "pause_midstage_documents"],
        "fields": [
            field(
                "pause_midstage_documents",
                "object[]",
                "-",
                "MUST be empty (H-d1). Union over EACH ARM'S OWN post-drain snapshot "
                "of rows that are 'failed' or carry chunk/chunk-embedding rows while "
                "not 'complete'. Computed per arm on purpose: the snapshot after the "
                "final unrestricted drain would have repaired exactly this",
            ),
            field(
                "pause_discarded_seconds",
                "float",
                "s",
                "MEASURED: wall-clock each library arm spent after its last committed "
                "document, summed. Not a constant",
            ),
            field(
                "pause_discarded_document_fraction",
                "float",
                "ratio",
                "the worst arm's post-commit seconds over that drain's own longest "
                "single-document cost — H-d1's scored secondary threshold",
            ),
            field("max_seconds.overrun_seconds", "float", "s", "H-d2"),
            field(
                "sigterm_to_exit_seconds",
                "float",
                "s",
                "ARM 5: measured with an UNTRUNCATED document in flight. THIS is the "
                "number that sizes a launchd/cron shutdown grace period",
            ),
            field(
                "sigterm_to_exit_seconds_truncated",
                "float",
                "s",
                "ARM 4: the same measurement on a 6000-char truncation (~2 chunks). "
                "Reported for contrast ONLY — NOT operational guidance",
            ),
            field(
                "sigterm_in_flight_document",
                "object",
                "-",
                "size of the document arm 5 committed while the signal was in flight, "
                "which is what makes the latency interpretable",
            ),
        ],
    },
    "results/claim_f_fail.json": {
        "schema": "andamentum.experiment.docstore_deferred.claim_f/1",
        "produced_by": "claim_f_fail",
        "description": "H-f1/H-f2: failures recorded and diagnosable; retry routes by stage.",
        "required": ["verdict", "failure_fidelity", "retry_returned", "stage_test"],
        "fields": [
            field("failure_fidelity[].ingest_error", "string", "-", "verbatim error column"),
            field("failure_fidelity[].exception_type", "string", "-", "what a UI can match on"),
            field(
                "stage_test_converter_calls",
                "int",
                "count",
                "ALL conversions in the stage-test drain, including the still-broken "
                "sources that correctly return to pending_source",
            ),
            field(
                "stage_test_converter_calls_good_doc",
                "int",
                "count",
                "conversions of the MARKDOWN-BEARING document — MUST be 0 (H-f2)",
            ),
        ],
    },
    "results/drain_overhead.json": {
        "schema": "andamentum.experiment.docstore_deferred.drain_overhead/1",
        "produced_by": "drain_overhead",
        "description": "The fixed preflight tax a cron-driven drain pays on every wake-up.",
        "required": ["verdict", "median_seconds", "median_ollama_requests"],
        "fields": [
            field("median_seconds", "float", "s", "across three FRESH processes"),
            field("median_ollama_requests", "float", "count", "expected 2: 1 embed + 1 chat"),
        ],
    },
    "results/claims.json": {
        "schema": "andamentum.experiment.docstore_deferred.claims/1",
        "produced_by": "analyze",
        "description": "One row per hypothesis: threshold, measured value, verdict.",
        "required": ["rows", "n_pass", "n_fail", "n_inconclusive"],
        "fields": [
            field("rows[].id", "string", "-", "hypothesis id"),
            field("rows[].threshold", "object|null", "-", "READ FROM preregistration.json"),
            field("rows[].measured", "any", "-", "the observed value"),
            field("rows[].verdict", "string", "-", "PASS | FAIL | INCONCLUSIVE | OBSERVATION"),
            field("rows[].falsifier", "string|null", "-", "what would have flipped it"),
        ],
    },
    "results/validation.json": {
        "schema": "andamentum.experiment.docstore_deferred.validation/1",
        "produced_by": "validate_results",
        "description": "Schema + referential-integrity validation of every results file.",
        "required": ["verdict", "violations"],
        "fields": [field("violations", "string[]", "-", "empty means valid; any entry exits 1")],
    },
    "results/events.jsonl": {
        "schema": "andamentum.experiment.docstore_deferred.events/1",
        "produced_by": "merge_events",
        "description": (
            "Every OBSERVED queue transition, concatenated in rule order. "
            "TRAP: documents_skipped uses `continue` WITHOUT incrementing `done`, so "
            "on_progress never fires for a skipped source and `done` can end below `total`."
        ),
        "required": [],
        "fields": [
            field("ts_utc", "float", "s since epoch", "wall-clock of the observation"),
            field("monotonic_s", "float", "s", "seconds since the rule started"),
            field("observer", "string", "-", "on_progress | poll | report"),
            field("from_status/to_status", "string|null", "-", "the transition observed"),
        ],
    },
    "results/timings.csv": {
        "schema": "andamentum.experiment.docstore_deferred.timings/1",
        "produced_by": "analyze",
        "description": "Per-rule cost roll-up.",
        "required": [],
        "fields": [
            field("rule", "string", "-", "the producing rule"),
            field(
                "benchmark_wall_seconds",
                "float",
                "s",
                "process wall time from Snakemake's benchmark:. Populated for EVERY "
                "rule and one single quantity, so it is the only column safe to plot "
                "on one axis. Its memory/IO columns are literally 0 on this macOS host "
                "and are NOT measurements",
            ),
            field(
                "in_script_component_seconds",
                "float|empty",
                "s",
                "in-script time.monotonic, populated only where an artefact carries "
                "one. NOT commensurable across rules — for micro_stages it is a SUM OF "
                "COMPONENTS, elsewhere a single drain — which is why it is not plotted",
            ),
            field(
                "in_script_component_meaning",
                "string",
                "-",
                "which of those two the value is, stated per row",
            ),
            field("ollama_requests", "int", "count", "from the http ledger"),
            field("ollama_seconds", "float", "s", "summed request latency"),
            field(
                "max_in_flight_global",
                "int",
                "count",
                "whole-process depth over all endpoints; per-endpoint peaks live in "
                "each artefact's http.max_in_flight_by_path",
            ),
        ],
    },
    "results/per_document.csv": {
        "schema": "andamentum.experiment.docstore_deferred.per_document/1",
        "produced_by": "analyze",
        "description": "One row per document in the main lineage after the drain.",
        "required": [],
        "fields": [
            field("doc_uuid", "string", "-", "the document"),
            field("n_chunks", "int", "count", "chunks written by enrichment"),
            field("n_chunk_embeddings", "int", "count", "must equal n_chunks"),
            field("has_doc_embedding", "bool", "-", "measured, not asserted"),
            field("llm_metadata_populated", "bool", "-", "persistent trace of the H-G failure"),
        ],
    },
    "results/MANIFEST.json": {
        "schema": "andamentum.experiment.docstore_deferred.manifest/1",
        "produced_by": "manifest",
        "description": "Every artefact with sha256, bytes, producing rule and schema.",
        "required": ["artefacts"],
        "fields": [
            field("artefacts[].sha256", "hex string", "-", "content hash"),
            field("artefacts[].producing_rule", "string", "-", "which rule wrote it"),
            field(
                "harness_sha256",
                "hex string",
                "-",
                "one digest over Snakefile + config.yaml + every scripts/*.py",
            ),
        ],
    },
    "results/deviations.json": {
        "schema": "andamentum.experiment.docstore_deferred.deviations/1",
        "produced_by": "deviations",
        "description": (
            "Every change to a hypothesis, metric, threshold or scoring rule made "
            "AFTER a measurement was observed. A pre-registration that can be edited "
            "after the fact with no amendment trail provides the appearance of the "
            "guarantee rather than the guarantee."
        ),
        "required": ["deviations", "n_deviations"],
        "fields": [
            field("deviations[].what", "string", "-", "the change"),
            field("deviations[].why", "string", "-", "what made it necessary"),
            field("deviations[].affects", "string[]", "-", "hypothesis ids it could move"),
            field(
                "deviations[].could_have_moved_the_verdict",
                "string",
                "-",
                "the honest direction: harder, easier, or neutral",
            ),
        ],
    },
    "results/cli_smoke.json": {
        "schema": "andamentum.experiment.docstore_deferred.cli_smoke/1",
        "produced_by": "cli_smoke",
        "description": (
            "Every andamentum-docstore subcommand run through the venv console script: "
            "exit code, stdout shape, and `status` output parsed back and compared with "
            "the database."
        ),
        "required": ["verdict", "invocations", "subcommands_exercised"],
        "fields": [
            field("invocations[].argv", "string[]", "-", "the exact command line"),
            field("invocations[].returncode", "int", "-", "process exit code"),
            field("status_parsed", "object", "counts", "the `status` output, parsed"),
            field(
                "status_database_counts",
                "object",
                "counts",
                "the same counts read straight from sqlite — must agree",
            ),
        ],
    },
}

#: GLOB-KEYED artefacts. Everything above is a fixed path; the files below are
#: written one-per-drain or one-per-rule and were previously undocumented and
#: unvalidated — 101 of 122 manifest entries carried `schema: null`, including
#: exactly the ledgers and snapshots the README calls load-bearing.
ARTEFACT_GLOBS: dict[str, dict[str, Any]] = {
    "results/ledgers/*_convert.jsonl": {
        "schema": "andamentum.experiment.docstore_deferred.converter_ledger/1",
        "produced_by": "claim_b_drain / claim_c_kill / claim_c_resume / claim_d_pause / claim_f_fail",
        "format": "JSONL, one object per line, append-only, fsynced before return",
        "description": (
            "ONE LINE PER convert_fn INVOCATION. The load-bearing instrument for the "
            "checkpoint claim: 'conversion is not repeated' reduces to a line count a "
            "reader can verify by opening a text file. fsynced before the wrapper "
            "returns, which is what makes it valid evidence under a SIGKILL."
        ),
        "required": [],
        "fields": [
            field("ts_start", "float", "s since epoch", "when the conversion began"),
            field("ts_end", "float", "s since epoch", "when it returned or raised"),
            field("seconds", "float", "s", "monotonic duration"),
            field("source", "string", "-", "the path or URL converted; DUPLICATES FALSIFY H-c1"),
            field("pid", "int", "-", "which process converted it — kill vs resume"),
            field("chars", "int", "characters", "length of the markdown produced; 0 on error"),
            field("sha256", "hex string|null", "-", "hash of the markdown; H-c2 compares it"),
            field("error_type", "string|null", "-", "exception class name, or null on success"),
            field("error_message", "string|null", "-", "first 500 chars of the message"),
        ],
    },
    "results/ledgers/*_http.jsonl": {
        "schema": "andamentum.experiment.docstore_deferred.http_ledger/1",
        "produced_by": "every measurement rule (one ledger per arm)",
        "format": "JSONL, one object per httpx request, appended in completion order",
        "description": (
            "Every HTTP request the recording process made, with in-flight depth at "
            "start. H-a1 (defer is LLM-free) is the emptiness of one of these files."
        ),
        "required": [],
        "fields": [
            field("ts_start / ts_end", "float", "s since epoch", "request start / completion"),
            field("method", "string", "-", "HTTP method"),
            field("host", "string", "-", "netloc; '11434' identifies Ollama"),
            field("path", "string", "-", "/v1/chat/completions | /api/embeddings | /api/embed"),
            field(
                "global_depth_at_start",
                "int",
                "count",
                "requests open across ALL endpoints when this one started — an upper "
                "bound on any single endpoint, never an attribution",
            ),
            field(
                "in_flight_at_start_path",
                "int",
                "count",
                "requests open ON THIS PATH when this one started — the per-endpoint "
                "concurrency figure",
            ),
            field("status", "int|null", "-", "HTTP status, or null if the request raised"),
            field("error_type", "string|null", "-", "exception class name on failure"),
        ],
    },
    "results/snapshots/*.json": {
        "schema": "andamentum.experiment.docstore_deferred.snapshot/1",
        "produced_by": "drain.instrumented_drain (pre and post, one pair per drain)",
        "format": "JSON with the standard envelope",
        "description": (
            "The complete read-only observation of one database at one instant: the "
            "logical fingerprint, status counts, and a full row dump. Every headline "
            "count in this experiment is re-derivable from these offline."
        ),
        "required": ["logical_fingerprint", "status_counts", "documents"],
        "fields": [
            field("label", "string", "-", "which drain and which side (pre / post)"),
            field(
                "logical_fingerprint",
                "hex string",
                "-",
                "sha256 over the MEANING of every row (uuid|status|file_hash|"
                "markdown_len|n_chunks|has_doc_embedding|n_chunk_embeddings|"
                "metadata_keys). The file hash is not used: WAL checkpoints move bytes "
                "without changing meaning",
            ),
            field("status_counts", "object", "counts", "pending_source / pending_enrich / complete / failed"),
            field("documents[].ingest_status", "string", "-", "the row's queue state"),
            field("documents[].markdown_sha256", "hex string", "-", "H-c2's byte-identity check"),
            field("documents[].n_chunks", "int", "count", "chunk rows written by enrichment"),
            field("documents[].n_chunk_embeddings", "int", "count", "must equal n_chunks when complete"),
            field("documents[].llm_metadata_populated", "bool", "-", "the H-G failure's persistent trace"),
            field("file_advisory", "object", "-", "byte-level facts, ADVISORY — nothing asserts on them"),
        ],
    },
    "results/events/*.jsonl": {
        "schema": "andamentum.experiment.docstore_deferred.events/1",
        "produced_by": "each claim rule (one file per rule)",
        "format": "JSONL, one observed transition per line",
        "description": (
            "The per-rule event log merged by `merge_events` into results/events.jsonl "
            "and events.csv. Same field set as that merged file."
        ),
        "required": [],
        "fields": [
            field("ts_utc", "float", "s since epoch", "wall-clock of the observation"),
            field("monotonic_s", "float", "s", "seconds since the rule started"),
            field("observer", "string", "-", "on_progress | poll | report"),
            field("from_status / to_status", "string|null", "-", "the transition observed"),
        ],
    },
    "results/events.csv": {
        "schema": "andamentum.experiment.docstore_deferred.events_csv/1",
        "produced_by": "merge_events",
        "format": "CSV with a header row",
        "description": "The flat rendering of results/events.jsonl, same fields.",
        "required": [],
        "fields": [field("(see results/events.jsonl)", "-", "-", "identical field set")],
    },
    "results/working_tree.patch": {
        "schema": "andamentum.experiment.docstore_deferred.working_tree_patch/1",
        "produced_by": "provenance",
        "format": "unified diff (`git diff HEAD`)",
        "description": (
            "WRITTEN ONLY WHEN THE TREE IS DIRTY, and REQUIRED whenever git.dirty is "
            "true. A recorded sha with dirty=true and no diff names code that cannot "
            "be recovered — the previous edition of this run was measured against an "
            "uncommitted one-line fix without which it crashes."
        ),
        "required": [],
        "fields": [
            field(
                "(whole file)",
                "text",
                "-",
                "hashed into provenance.json as git.working_tree_patch_sha256",
            )
        ],
    },
    "figures/figures_status.json": {
        "schema": "andamentum.experiment.docstore_deferred.figures_status/1",
        "produced_by": "figures",
        "format": "JSON with the standard envelope",
        "description": (
            "Per-figure success/failure. Consulted by build_report, which prints a "
            "visible FIGURE COULD NOT BE DRAWN line rather than embedding a "
            "placeholder as though it were a result."
        ),
        "required": ["figures"],
        "fields": [
            field("figures[].figure", "string", "-", "the PNG filename"),
            field("figures[].ok", "bool", "-", "false means the file is a placeholder"),
            field("figures[].error", "string|null", "-", "type and message when not ok"),
        ],
    },
}


def to_markdown() -> str:
    lines = [
        "# Artefact schemas",
        "",
        "Generated by `scripts/schemas.py` — the SAME definition that",
        "`validate_results` enforces. Editing this file by hand will be overwritten;",
        "edit `scripts/schemas.py` instead.",
        "",
        "## Common envelope",
        "",
        "Every JSON artefact written by `_common.write_json` carries:",
        "",
        "| field | type | units | meaning |",
        "|---|---|---|---|",
    ]
    for f in ENVELOPE_FIELDS:
        lines.append(f"| `{f['name']}` | {f['type']} | {f['units']} | {f['meaning']} |")
    lines.append("")
    lines.append("## Authoritative timings")
    lines.append("")
    lines.append(
        "In-script `time.monotonic()` deltas are AUTHORITATIVE. Snakemake's "
        "`benchmark:` wall-clock is used for cross-checking only, and its memory / IO "
        "columns are literally `0` on this macOS host — reporting them as measurements "
        "would be fabrication."
    )
    lines.append("")

    def _section(path: str, spec: dict[str, Any]) -> list[str]:
        out = [
            f"## `{path}`",
            "",
            f"- **schema**: `{spec['schema']}`",
            f"- **produced by**: `{spec['produced_by']}`",
            f"- **required keys**: {', '.join(f'`{k}`' for k in spec['required']) or '(none)'}",
        ]
        if spec.get("format"):
            out.append(f"- **format**: {spec['format']}")
        out += [
            "",
            spec["description"],
            "",
            "| field | type | units | meaning |",
            "|---|---|---|---|",
        ]
        for f in spec["fields"]:
            out.append(f"| `{f['name']}` | {f['type']} | {f['units']} | {f['meaning']} |")
        out.append("")
        return out

    for path, spec in ARTEFACTS.items():
        lines += _section(path, spec)

    lines += [
        "# Glob-keyed artefacts",
        "",
        "Written one-per-drain or one-per-rule rather than at a fixed path. Two",
        "structurally different row shapes share `results/ledgers/`, which is why they",
        "are documented as separate globs: `*_convert.jsonl` records converter",
        "invocations, `*_http.jsonl` records HTTP requests.",
        "",
    ]
    for pattern, spec in ARTEFACT_GLOBS.items():
        lines += _section(pattern, spec)
    return "\n".join(lines) + "\n"


def main() -> int:
    C.write_json(
        C.RESULTS / "SCHEMAS.json",
        {
            "envelope": ENVELOPE_FIELDS,
            "artefacts": ARTEFACTS,
            "artefact_globs": ARTEFACT_GLOBS,
        },
        schema=SCHEMA,
    )
    (C.RESULTS / "SCHEMAS.md").write_text(to_markdown())
    print(
        f"wrote {C.RESULTS / 'SCHEMAS.json'} and {C.RESULTS / 'SCHEMAS.md'} "
        f"({len(ARTEFACTS)} paths + {len(ARTEFACT_GLOBS)} globs)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
