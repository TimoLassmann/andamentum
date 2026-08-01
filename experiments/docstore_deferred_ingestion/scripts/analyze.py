"""Score every hypothesis against its PRE-REGISTERED threshold. Never against a new one.

Reads ``results/preregistration.json`` for the thresholds and the measurement
artefacts for the values. A threshold that does not appear in the pre-registration
cannot be scored here — that is the point.

INCONCLUSIVE IS A REAL VERDICT. When an effect sits inside the measured noise
band it is reported as INCONCLUSIVE rather than rounded up to PASS. The band comes
from the per-call LLM latency distribution accumulated across the whole run (dozens
of free samples) — the per-call latency IS the unit of variance for every
wall-clock ratio in this experiment.

Also emits the cross-cutting observations: max concurrent Ollama in-flight, the
doc-embedding skip rate, the preflight tax, and the report-vs-database
reconciliation (H-x).

Run:  uv run python -m experiments.docstore_deferred_ingestion.scripts.analyze
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Any, Callable

from . import _common as C

SCHEMA = "andamentum.experiment.docstore_deferred.claims/1"

#: Endpoint the LLM-latency band is estimated from. Chat completions ONLY —
#: see prereg's SCORING_RULES for why pooling endpoints is not a variance estimate.
CHAT_PATH_FRAGMENT = "/chat/completions"

RESULT_FILES = {
    "gate": "gate_llm.json",
    "a": "claim_a_defer.json",
    "e_pre": "claim_e_pre.json",
    "b": "claim_b_drain.json",
    "e_post": "claim_e_post.json",
    "micro": "micro_stages.json",
    "c_kill": "claim_c_kill.json",
    "c_resume": "claim_c_resume.json",
    "d": "claim_d_pause.json",
    "f": "claim_f_fail.json",
    "overhead": "drain_overhead.json",
}


def load_all() -> dict[str, dict[str, Any]]:
    """Load every measurement artefact that exists. Missing ones are recorded, not faked."""
    loaded: dict[str, dict[str, Any]] = {}
    for key, name in RESULT_FILES.items():
        path = C.RESULTS / name
        if path.exists():
            loaded[key] = C.read_json(path)
    return loaded


def _safe(fn: Callable[[], Any]) -> Any:
    """Return fn() or None — a metric whose source artefact is absent is None, not 0."""
    try:
        return fn()
    except (KeyError, IndexError, TypeError, AttributeError):
        return None


def resolve_metrics(data: dict[str, Any]) -> dict[str, Any]:
    """Map every pre-registered metric name to its measured value."""
    a = data.get("a", {})
    b = data.get("b", {})
    d = data.get("d", {})
    f = data.get("f", {})
    gate = data.get("gate", {})
    e_pre = data.get("e_pre", {})
    e_post = data.get("e_post", {})
    c_kill = data.get("c_kill", {})
    c_res = data.get("c_resume", {})
    micro = data.get("micro", {})

    overrun = _safe(lambda: d["max_seconds"]["overrun_seconds"])
    longest = _safe(lambda: d["max_seconds"]["longest_document_seconds"])
    overrun_documents = (
        (overrun / longest) if (overrun is not None and longest) else (0.0 if overrun == 0 else None)
    )

    return {
        "gate_topics_count": gate.get("topics_count"),
        "defer_ollama_requests": a.get("defer_ollama_requests"),
        "defer_state_violations": _safe(lambda: len(a["defer_state_violations"])),
        "defer_median_seconds": a.get("defer_median_seconds"),
        "defer_now_ratio": a.get("defer_now_ratio"),
        "defer_fts_misses": _safe(lambda: len(a["defer_fts_misses"])),
        "source_fts_hits": _safe(lambda: a["source_probe"]["self_hits"]),
        "drain3_enriched": b.get("drain3_enriched"),
        "drain3_ollama_requests": b.get("drain3_ollama_requests"),
        "fingerprint_moved": _safe(lambda: int(bool(b["drain3_fingerprint_moved"]))),
        "ledger_entries": c_res.get("ledger_entries"),
        "ledger_duplicates": _safe(lambda: len(c_res["ledger_duplicates"])),
        "resume_converted": _safe(lambda: c_res["resume_report"]["documents_converted"]),
        "markdown_sha_stable": _safe(lambda: int(bool(c_res["markdown_sha_stable"]))),
        "sqlite_ok": _safe(lambda: int(bool(c_res["sqlite_ok"]))),
        # MEASURED per arm, on each arm's OWN post-drain snapshot — not on the
        # snapshot after the final unrestricted drain, which repairs exactly the
        # state this metric exists to detect.
        "pause_midstage_documents": _safe(lambda: len(d["pause_midstage_documents"])),
        "pause_discarded_document_fraction": d.get("pause_discarded_document_fraction"),
        "pause_discarded_seconds": d.get("pause_discarded_seconds"),
        # H-d5: the should_continue arm must have PAUSED, not run out of work.
        "should_continue_stopped_early": _safe(
            lambda: int(bool(d["arms"]["arm3_should_continue"]["report"]["stopped_early"]))
        ),
        "should_continue_remaining": _safe(
            lambda: d["arms"]["arm3_should_continue"]["report"]["remaining"]
        ),
        "max_seconds_overrun_documents": overrun_documents,
        "first_processed_is_source": _safe(lambda: int(bool(d["first_processed_is_source"]))),
        # Named for what it counts: rows in a state at resume start. It is a proxy
        # for "stages repeated", and calling it kill_repeated_stages claimed to be
        # the thing it stands in for.
        "documents_pending_enrich_at_resume": _safe(
            lambda: c_res["snapshot_pre"]["status_counts"]["pending_enrich"]
        ),
        "kill_repeated_enrichment_over_conversion": c_res.get(
            "repeated_enrichment_over_conversion"
        ),
        "post_chunk_recall_at_1": e_post.get("post_chunk_recall_at_1"),
        "post_chunk_recall_at_3": e_post.get("post_chunk_recall_at_3"),
        "post_chunk_mrr": e_post.get("post_chunk_mrr"),
        "pre_chunk_recall_at_3": e_pre.get("pre_chunk_recall_at_3"),
        "pre_unified_recall_at_3": e_post.get("pre_unified_recall_at_3"),
        "post_unified_recall_at_3": e_post.get("post_unified_recall_at_3"),
        "enrichment_structure_violations": _safe(
            lambda: len(e_post["enrichment_structure_violations"])
        ),
        "failed_count": f.get("failed_count"),
        "errors_without_type_name": _safe(lambda: len(f["errors_without_type_name"])),
        "retry_wrong_stage": _safe(lambda: len(f["retry_wrong_stage"])),
        # ONE NAME, ONE VALUE. H-f2 registers the SCOPED name, and the artefact's
        # field of that name is what is read. The unscoped total keeps its own
        # name and is reported separately, so the two can never disagree under one
        # identifier the way they did when the analyzer silently re-pointed the
        # registered name at the other field.
        "stage_test_converter_calls_good_doc": f.get("stage_test_converter_calls_good_doc"),
        "stage_test_converter_calls_total": f.get("stage_test_converter_calls"),
        "concurrency_speedup": micro.get("concurrency_speedup"),
        # None, NOT 0, when no drain artefact reported it: summing over an empty
        # set would score H-x as PASS on the strength of no evidence at all.
        # c_resume is included now that its subprocess drain is reconciled too.
        "report_db_mismatches": (
            sum(
                x["report_db_mismatches"]
                for x in (b, d, f, c_res)
                if "report_db_mismatches" in x
            )
            if any("report_db_mismatches" in x for x in (b, d, f, c_res))
            else None
        ),
        # kill-lineage observation, referenced by H-c4 (no threshold)
        "hc4_window_observed": _safe(lambda: int(bool(c_kill["hc4_window"]["observed"]))),
    }


COMPARATORS: dict[str, Callable[[Any, Any], bool]] = {
    "==": lambda m, v: m == v,
    "<": lambda m, v: m < v,
    "<=": lambda m, v: m <= v,
    ">": lambda m, v: m > v,
    ">=": lambda m, v: m >= v,
}


def chat_latency_samples() -> tuple[list[float], list[str]]:
    """Every /v1/chat/completions request latency in the run, from the raw ledgers.

    ONE ENDPOINT. An earlier version seeded this pool with the per-chunk LLM
    timings and then extended it with the min / median / max of each rule's HTTP
    summary — summaries that pool ALL endpoints, so a 1.5 ms embedding call and a
    96 s chat call entered the same sample set. A standard deviation over the
    order statistics of a bimodal mixture estimates nothing; the resulting
    coefficient of variation described workload heterogeneity, not run-to-run
    variance, and 12 of its 37 "samples" were not independent draws at all.

    Reading the per-request JSONL directly gives real samples of one distribution.
    """
    from .instrument import read_jsonl

    samples: list[float] = []
    sources: list[str] = []
    if not C.LEDGERS.exists():
        return samples, sources
    for path in sorted(C.LEDGERS.glob("*_http.jsonl")):
        found = 0
        for record in read_jsonl(path):
            if CHAT_PATH_FRAGMENT not in str(record.get("path")):
                continue
            if record.get("ts_end") is None or record.get("ts_start") is None:
                continue
            samples.append(record["ts_end"] - record["ts_start"])
            found += 1
        if found:
            sources.append(f"{path.name}:{found}")
    return samples, sources


def noise_band(data: dict[str, Any]) -> dict[str, Any]:
    """Relative spread of per-request LLM latency, one endpoint, raw samples."""
    samples, sources = chat_latency_samples()
    micro = data.get("micro", {})
    replicate_spread = micro.get("in_pipeline_replicate_spread")

    band: dict[str, Any] = {
        "chat_path_fragment": CHAT_PATH_FRAGMENT,
        "n_samples": len(samples),
        "ledger_sources": sources,
        "in_pipeline_replicate_spread": replicate_spread,
        "in_pipeline_replicate_seconds": micro.get("in_pipeline_replicate_seconds"),
        "source": (
            "raw per-request latencies on /v1/chat/completions across every ledger in "
            "the run — one endpoint, one distribution. The replicate spread beside it "
            "comes from the SAME content enriched twice inside claim_b, and is the band "
            "concurrency_speedup is judged against."
        ),
    }
    if len(samples) < 3:
        band.update({"relative_spread": None, "usable": False})
        return band
    mean = statistics.mean(samples)
    spread = statistics.pstdev(samples) / mean if mean else None
    band.update(
        {
            "mean_seconds": mean,
            "stdev_seconds": statistics.pstdev(samples),
            "min_seconds": min(samples),
            "max_seconds": max(samples),
            "relative_spread": spread,
            "usable": spread is not None,
        }
    )
    return band


def _band_for(
    metric: str, rules: dict[str, Any], band: dict[str, Any], threshold_value: Any
) -> tuple[float | None, Any, str]:
    """(spread, reference, source-name) for one metric, PER THE PRE-REGISTRATION.

    Returns ``(None, ...)`` when the metric is not registered as softenable or the
    named band has no usable value — in which case the threshold comparison stands
    unmodified. The analyzer chooses nothing here; ``prereg.SCORING_RULES`` does.
    """
    if metric not in set(rules.get("ratio_metrics") or []):
        return None, None, "not a registered ratio metric"
    spec = (rules.get("bands") or {}).get(metric) or {}
    source = str(spec.get("band_source") or "")
    if source == "chat_latency_relative_spread":
        spread = band.get("relative_spread") if band.get("usable") else None
    elif source == "in_pipeline_replicate_spread":
        spread = band.get("in_pipeline_replicate_spread")
    else:
        return None, None, f"unknown band_source {source!r}"
    reference = spec.get("reference")
    if reference == "the registered threshold" or reference is None:
        reference = threshold_value
    return (float(spread) if spread else None), reference, source


def score(
    hypothesis: dict[str, Any],
    metrics: dict[str, Any],
    band: dict[str, Any],
    rules: dict[str, Any],
) -> list[dict[str, Any]]:
    """One row per threshold on a hypothesis (primary, secondary, tertiary)."""
    rows: list[dict[str, Any]] = []
    thresholds = [
        hypothesis.get("threshold"),
        hypothesis.get("threshold_secondary"),
        hypothesis.get("threshold_tertiary"),
    ]
    thresholds = [t for t in thresholds if t]

    if not thresholds:
        rows.append(
            {
                "id": hypothesis["id"],
                "claim": hypothesis["claim"],
                "metric": None,
                "threshold": None,
                "measured": metrics.get("hc4_window_observed"),
                "verdict": "OBSERVATION",
                "falsifier": hypothesis.get("falsifier"),
                "statement": hypothesis["statement"],
                "note": "observation only — no pre-registered threshold",
            }
        )
        return rows

    for threshold in thresholds:
        metric = threshold["metric"]
        measured = metrics.get(metric)
        row: dict[str, Any] = {
            "id": hypothesis["id"],
            "claim": hypothesis["claim"],
            "metric": metric,
            "threshold": threshold,
            "measured": measured,
            "falsifier": hypothesis.get("falsifier"),
            "statement": hypothesis["statement"],
            "note": "",
        }
        if measured is None:
            row["verdict"] = "NOT_MEASURED"
            row["note"] = "the producing rule did not write this value"
            rows.append(row)
            continue

        comparator = COMPARATORS[threshold["op"]]
        passed = bool(comparator(measured, threshold["value"]))
        row["verdict"] = "PASS" if passed else "FAIL"

        # SOFTENING, ENTIRELY PER THE PRE-REGISTRATION. Which metrics are
        # softenable, what each is measured against, and where its band comes from
        # are read from prereg's SCORING_RULES — the analyzer can no more invent a
        # scoring rule than it can invent a threshold. The separation statistic is
        # a log-ratio because both sides are relative quantities; the obvious
        # |measured-threshold|/threshold saturates at 1.0 as measured falls far
        # below threshold, so any coefficient of variation above 1.0 made the
        # softening fire hardest on the strongest evidence.
        spread, reference, band_source = _band_for(
            metric, rules, band, threshold["value"]
        )
        row["band_source"] = band_source
        if spread:
            row["band_reference"] = reference
            row["band_relative_spread"] = spread
            if measured > 0 and isinstance(reference, (int, float)) and reference > 0:
                separation = abs(math.log(measured / reference))
            else:
                # measured == 0 against a positive reference is an unambiguous
                # pass; no finite band can reach across it.
                separation = math.inf
            row["separation_log_ratio"] = None if separation == math.inf else separation
            if separation < spread:
                row["verdict"] = "INCONCLUSIVE"
                row["note"] = (
                    f"not distinguishable from {reference} at the measured spread "
                    f"(|ln(measured/{reference})| {separation:.3f} < {band_source} "
                    f"{spread:.3f}); NOT rounded up to PASS"
                )
        rows.append(row)
    return rows


def timings_csv(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-rule cost roll-up, from Snakemake's benchmark files.

    ``benchmark_wall_seconds`` is populated for EVERY rule and is one quantity —
    process wall time — so it is the only column safe to plot on a single axis.
    The old ``wall_seconds`` column was populated for just two rules (whichever
    artefact happened to carry ``elapsed_seconds`` / ``resume_seconds`` /
    ``sequential_total_seconds``) and mixed a SUM OF COMPONENTS with a PROCESS
    WALL TIME; fig1 plotted those two bars against each other under one label and
    silently omitted the largest cost in the run. It is renamed
    ``in_script_component_seconds`` so it can never again be mistaken for the
    former.
    """
    bench_dir = C.EXP_DIR / "bench"
    rows: list[dict[str, Any]] = []
    for key, filename in RESULT_FILES.items():
        payload = data.get(key)
        if payload is None:
            continue
        rule = Path(filename).stem
        http = payload.get("http") or {}
        bench_file = bench_dir / f"{rule}.tsv"
        bench_wall = ""
        if bench_file.exists():
            lines = bench_file.read_text().splitlines()
            if len(lines) >= 2:
                bench_wall = lines[1].split("\t")[0]
        component = (
            payload.get("elapsed_seconds")
            or payload.get("resume_seconds")
            or payload.get("sequential_total_seconds")
            or ""
        )
        rows.append(
            {
                "rule": rule,
                "benchmark_wall_seconds": bench_wall,
                "in_script_component_seconds": component,
                "in_script_component_meaning": (
                    "sum of timed components"
                    if payload.get("sequential_total_seconds")
                    else ("one drain / subprocess" if component else "")
                ),
                "ollama_requests": http.get("requests_total", ""),
                "ollama_seconds": (http.get("latency_seconds") or {}).get("sum", ""),
                "max_in_flight_global": http.get("max_in_flight", ""),
            }
        )
    return rows


def main() -> int:
    prereg = C.read_json(C.PREREG_PATH)
    data = load_all()
    metrics = resolve_metrics(data)
    band = noise_band(data)

    rules = prereg.get("scoring_rules") or {}
    if not rules:
        raise ValueError(
            f"{C.PREREG_PATH} carries no scoring_rules block. The softening rule must "
            "be pre-registered alongside the thresholds; an analyzer free to define "
            "one after seeing the numbers provides the appearance of the guarantee "
            "rather than the guarantee."
        )

    rows: list[dict[str, Any]] = []
    for hypothesis in prereg["hypotheses"]:
        rows.extend(score(hypothesis, metrics, band, rules))

    # Metrics measured and reported but carrying NO pre-registered threshold. They
    # appear on the scoreboard as OBSERVATION rows so a reader sees them without
    # them inflating the PASS count.
    for name, why in (
        (
            "stage_test_converter_calls_total",
            "ALL conversions in the stage-test drain, including the three still-broken "
            "sources that hold no markdown and are therefore correctly re-converted. "
            "H-f2 scores stage_test_converter_calls_good_doc; this is the unscoped "
            "companion, reported so the two values are never confused.",
        ),
        (
            "pause_discarded_seconds",
            "raw wall-clock the library pause arms spent after their last commit, "
            "summed. Scored as a fraction of one document under H-d1.",
        ),
        (
            "pre_chunk_recall_at_3",
            "0 by definition — H-a2 asserts deferred documents have zero chunks. "
            "Demoted from a scored control to a sanity note.",
        ),
        ("post_chunk_recall_at_3", "reported beside recall@1; chance level is 3/N."),
        ("post_chunk_mrr", "mean reciprocal rank over the same 8 probes."),
    ):
        rows.append(
            {
                "id": f"OBS/{name}",
                "claim": "observation",
                "metric": name,
                "threshold": None,
                "measured": metrics.get(name),
                "verdict": "OBSERVATION",
                "falsifier": None,
                "statement": why,
                "note": "measured and reported; no pre-registered threshold",
            }
        )

    counts = {
        "n_pass": len([r for r in rows if r["verdict"] == "PASS"]),
        "n_fail": len([r for r in rows if r["verdict"] == "FAIL"]),
        "n_inconclusive": len([r for r in rows if r["verdict"] == "INCONCLUSIVE"]),
        "n_not_measured": len([r for r in rows if r["verdict"] == "NOT_MEASURED"]),
        "n_observation": len([r for r in rows if r["verdict"] == "OBSERVATION"]),
    }

    # PER-ENDPOINT, because the aggregate is unattributable. Every payload's HTTP
    # summary now carries a per-path peak from a per-path in-flight counter, so the
    # fan-out can be attributed to the endpoint that produced it instead of being
    # blamed on whichever semaphore was nearest to hand. (The previous edition
    # published a global peak of 8 next to a hard-coded note naming
    # Semaphore(_INGEST_CONCURRENCY=5) — a number its own explanation forbids.)
    peaks_by_path: dict[str, int] = {}
    for payload in data.values():
        for path, peak in ((payload.get("http") or {}).get("max_in_flight_by_path") or {}).items():
            peaks_by_path[path] = max(peaks_by_path.get(path, 0), int(peak or 0))
    top_endpoint = max(peaks_by_path.items(), key=lambda kv: kv[1], default=(None, 0))

    cross_cutting = {
        "max_concurrent_ollama_in_flight": max(
            (
                (payload.get("http") or {}).get("max_in_flight", 0) or 0
                for payload in data.values()
            ),
            default=0,
        ),
        "max_in_flight_by_path": peaks_by_path,
        "max_in_flight_top_endpoint": top_endpoint[0],
        "max_in_flight_top_peak": top_endpoint[1],
        "max_in_flight_note": (
            f"The deepest fan-out is on {top_endpoint[0]} at {top_endpoint[1]} concurrent "
            "requests, counted by a per-endpoint in-flight counter. The global figure "
            "beside it is the whole-process depth over all endpoints at once and is an "
            "upper bound on any single one. Two distinct fan-outs exist and must not be "
            "conflated: core.embeddings.make_embedder gathers /api/embeddings under "
            "Semaphore(8) inside the chunker's embedding_fn, and _run_phase2 gathers "
            "chunk-metadata extraction on /v1/chat/completions under "
            "Semaphore(_INGEST_CONCURRENCY=5). Both are PRE-EXISTING library behaviour, "
            "reported not changed — and both contradict this project's "
            "one-inference-at-a-time rule."
        ),
        "doc_embedding_skip_rate": _safe(lambda: data["e_post"]["doc_embedding_skip_rate"]),
        "preflight_tax_seconds": _safe(lambda: data["overhead"]["median_seconds"]),
        "preflight_tax_seconds_range": _safe(
            lambda: [
                data["overhead"]["min_seconds"],
                data["overhead"]["median_seconds"],
                data["overhead"]["max_seconds"],
            ]
        ),
        "preflight_tax_cold_seconds": _safe(
            lambda: data["overhead"]["runs"][0]["elapsed_seconds"]
        ),
        "preflight_tax_requests": _safe(lambda: data["overhead"]["median_ollama_requests"]),
        "report_db_mismatches": metrics.get("report_db_mismatches"),
        "checkpoint_savings_seconds": _safe(
            lambda: data["c_resume"]["checkpoint_savings_seconds"]
        ),
        "checkpoint_savings_basis": _safe(
            lambda: data["c_resume"]["checkpoint_savings_basis"]
        ),
        "docling_init_seconds": _safe(lambda: data["c_resume"]["checkpoint_savings_basis"]["docling_init_seconds"]),
        "repeated_enrichment_seconds": _safe(
            lambda: data["c_resume"]["repeated_enrichment_seconds"]
        ),
        "concurrency_speedup": _safe(lambda: data["micro"]["concurrency_speedup"]),
        "concurrency_speedup_replicate_spread": _safe(
            lambda: data["micro"]["in_pipeline_replicate_spread"]
        ),
        "hc4_window_observed": bool(metrics.get("hc4_window_observed")),
        # From arm 5 — an UNTRUNCATED in-flight document. Arm 4's truncated figure
        # is carried beside it so the two can never be mistaken for each other.
        "sigterm_to_exit_seconds": _safe(lambda: data["d"]["sigterm_to_exit_seconds"]),
        "sigterm_to_exit_seconds_truncated": _safe(
            lambda: data["d"]["sigterm_to_exit_seconds_truncated"]
        ),
        "sigterm_in_flight_document": _safe(
            lambda: data["d"]["sigterm_in_flight_document"]
        ),
        "sigterm_seconds_per_chunk": _safe(lambda: data["d"]["sigterm_seconds_per_chunk"]),
        "retrieval_power": _safe(lambda: data["e_post"]["retrieval_power"]),
    }

    C.write_json(
        C.RESULTS / "claims.json",
        {
            "measured_metrics": metrics,
            "noise_band": band,
            "cross_cutting": cross_cutting,
            "artefacts_present": sorted(data),
            "artefacts_missing": sorted(set(RESULT_FILES) - set(data)),
            "rows": rows,
            **counts,
        },
        schema=SCHEMA,
    )

    # --- timings.csv ------------------------------------------------------
    timing_rows = timings_csv(data)
    with (C.RESULTS / "timings.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rule",
                "benchmark_wall_seconds",
                "in_script_component_seconds",
                "in_script_component_meaning",
                "ollama_requests",
                "ollama_seconds",
                "max_in_flight_global",
            ],
        )
        writer.writeheader()
        writer.writerows(timing_rows)

    # --- per_document.csv -------------------------------------------------
    per_doc = _safe(lambda: data["e_post"]["per_document"]) or []
    with (C.RESULTS / "per_document.csv").open("w", newline="") as fh:
        fieldnames = [
            "doc_uuid",
            "dc_title",
            "markdown_chars",
            "n_chunks",
            "n_chunk_embeddings",
            "has_doc_embedding",
            "llm_metadata_populated",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_doc)

    print(
        f"claims: {counts['n_pass']} PASS, {counts['n_fail']} FAIL, "
        f"{counts['n_inconclusive']} INCONCLUSIVE, "
        f"{counts['n_not_measured']} NOT_MEASURED"
    )
    for row in rows:
        if row["verdict"] in ("FAIL", "NOT_MEASURED"):
            print(f"  {row['verdict']:14s} {row['id']:6s} {row['metric']} = {row['measured']!r}")
    # analyze does NOT exit non-zero on a FAIL: the individual claim rule already
    # did that. This rule's job is to render the scoreboard, including for a run
    # that failed, so a failed claim leaves an inspectable artefact.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
