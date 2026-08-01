"""Render report.md from the measured artefacts. Facts only — no interpretation invented here.

Every number in the report is read from a results JSON; nothing is recomputed, so
the report cannot disagree with the evidence it cites.

Run:  uv run python -m experiments.docstore_deferred_ingestion.scripts.build_report
"""

from __future__ import annotations

from typing import Any

from . import _common as C


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) or "—"
    return str(value)


def main() -> int:
    claims = C.read_json(C.RESULTS / "claims.json")
    prov = C.read_json(C.PROVENANCE_PATH)
    prereg = C.read_json(C.PREREG_PATH)

    lines: list[str] = [
        "# Deferred ingestion on the real path",
        "",
        "A pre-registered, ledger-instrumented validation of the `document_store` drain",
        "against real arXiv PDFs and a local Ollama.",
        "",
        f"- **run_id**: `{prov['run_id']}`",
        f"- **git**: `{prov['git']['head_short']}` on `{prov['git']['branch']}`"
        + (" (working tree DIRTY)" if prov["git"]["dirty"] else ""),
        f"- **host**: {prov['machine']['platform']}",
        "",
        "## Models (pinned by digest, not by tag)",
        "",
        "| role | tag | digest | quantisation |",
        "|---|---|---|---|",
    ]
    for name, info in prov["ollama"]["pinned_models"].items():
        lines.append(
            f"| — | `{name}` | `{str(info['digest'])[:20]}` | "
            f"{info.get('quantization_level') or '—'} |"
        )

    lines += [
        "",
        "## Scoreboard",
        "",
        f"{claims['n_pass']} PASS · {claims['n_fail']} FAIL · "
        f"{claims['n_inconclusive']} INCONCLUSIVE · {claims['n_not_measured']} NOT MEASURED",
        "",
        "| id | claim | metric | threshold | measured | verdict | note |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in claims["rows"]:
        threshold = row["threshold"]
        threshold_text = (
            f"`{threshold['op']} {threshold['value']}`" if threshold else "observation"
        )
        note = str(row.get("note") or "").replace("|", "/")
        lines.append(
            f"| `{row['id']}` | {row['claim']} | `{row['metric'] or '—'}` | "
            f"{threshold_text} | {_fmt(row['measured'])} | **{row['verdict']}** | "
            f"{note or '—'} |"
        )

    cross = claims["cross_cutting"]
    in_flight = cross.get("sigterm_in_flight_document") or {}
    power = cross.get("retrieval_power") or {}
    basis = cross.get("checkpoint_savings_basis") or {}
    band = claims["noise_band"]
    tax_range = cross.get("preflight_tax_seconds_range") or []

    lines += [
        "",
        "## Cross-cutting observations",
        "",
        "| observation | value | why it matters |",
        "|---|---|---|",
        f"| max concurrent Ollama requests in flight (GLOBAL) | "
        f"{_fmt(cross['max_concurrent_ollama_in_flight'])} | whole-process depth over all "
        "endpoints at once — an upper bound on any single one, not an attribution |",
    ]
    for path, peak in sorted(
        (cross.get("max_in_flight_by_path") or {}).items(), key=lambda kv: -kv[1]
    ):
        lines.append(
            f"| &nbsp;&nbsp;peak on `{path}` | {peak} | counted by a per-endpoint "
            "in-flight counter |"
        )
    lines += [
        f"| deepest fan-out attributed to | `{cross.get('max_in_flight_top_endpoint')}` | "
        + cross["max_in_flight_note"].replace("|", "/")
        + " |",
        f"| doc-embedding skip rate | {_fmt(cross['doc_embedding_skip_rate'])} | "
        "`_embed_doc_level` swallows an oversized-input failure with an INFO log, so one "
        "of the four RRF signals can go dark without an error |",
        f"| preflight tax, min / median / max | "
        f"{' / '.join(_fmt(v) for v in tax_range) or '—'} s over "
        f"{_fmt(cross['preflight_tax_requests'])} requests | what a cron-driven drain pays "
        f"on every wake-up. The COLD run (the one a cron wake-up actually resembles) is "
        f"{_fmt(cross.get('preflight_tax_cold_seconds'))} s; all three ran back-to-back "
        "against an already-warm Ollama, so none is a true cold start |",
        f"| checkpoint savings | {_fmt(cross['checkpoint_savings_seconds'])} s | conversion "
        "seconds not repeated after the hard kill, from this lineage's own converter "
        f"ledger. INCLUDES one-time Docling initialisation "
        f"({_fmt(basis.get('docling_init_seconds'))} s, measured by converting one PDF "
        f"twice); the MARGINAL saving is the warm cost, "
        f"{_fmt(basis.get('docling_warm_conversion_seconds'))} s |",
        f"| enrichment discarded by the SIGKILL | "
        f"{_fmt(cross.get('repeated_enrichment_seconds'))} s | measured as killed_at minus "
        "the converter ledger's last ts_end — the work the resume redoes |",
        f"| concurrency speedup | {_fmt(cross['concurrency_speedup'])} | sequential total "
        "over in-pipeline seconds for the SAME doc_uuid. Read against the replicate spread "
        f"of {_fmt(cross.get('concurrency_speedup_replicate_spread'))} (the same content "
        "enriched twice in one lineage) — see H-m1's verdict, which is INCONCLUSIVE when "
        "the effect is inside it |",
        f"| report-vs-database mismatches (H-x) | {_fmt(cross['report_db_mismatches'])} | "
        "all six ProcessReport fields, across every drain including the subprocess resume |",
        f"| two-transaction window observed (H-c4) | {_fmt(cross['hc4_window_observed'])} | "
        "`_convert_document` commits markdown then flips status separately. A NULL result "
        "bounds nothing: the two commits are microseconds apart and the poller samples "
        "every 0.25 s |",
        f"| SIGTERM to exit | {_fmt(cross['sigterm_to_exit_seconds'])} s | size a launchd/cron "
        f"shutdown grace period with this. Measured with an UNTRUNCATED document in flight "
        f"({_fmt(in_flight.get('markdown_chars'))} chars, {_fmt(in_flight.get('n_chunks'))} "
        f"chunks) = {_fmt(cross.get('sigterm_seconds_per_chunk'))} s/chunk |",
        f"| &nbsp;&nbsp;the same, on a 6000-char truncation | "
        f"{_fmt(cross.get('sigterm_to_exit_seconds_truncated'))} s | CONTRAST ONLY, not "
        "guidance — ~2 chunks against the corpus's 25-43 |",
        "",
        "## Noise band and how a measurement becomes a verdict",
        "",
        f"Relative spread of per-request latency on `{band.get('chat_path_fragment')}`: "
        f"{_fmt(band.get('relative_spread'))} over {band.get('n_samples')} raw samples "
        "(one endpoint, one distribution — read from the per-request JSONL ledgers, not "
        "from pooled order statistics).",
        "",
        f"Replicate spread of in-pipeline enrichment: "
        f"{_fmt(band.get('in_pipeline_replicate_spread'))}, from the same content enriched "
        "twice inside claim (b) — the run's only true repeated measurement.",
        "",
        "Which metrics may be softened to INCONCLUSIVE, what each is compared against, "
        "and where its band comes from are all registered in "
        "`results/preregistration.json` under `scoring_rules`; the analyzer reads them "
        "from there and can no more invent a scoring rule than it can invent a threshold. "
        "Post-hoc changes are itemised in `results/deviations.json`.",
        "",
        "## Statistical power, stated plainly",
        "",
        f"- Retrieval probes discriminate between {_fmt(power.get('n_candidate_documents'))} "
        f"candidate documents, so chance recall@1 is {_fmt(power.get('chance_recall_at_1'))} "
        f"and chance recall@3 is {_fmt(power.get('chance_recall_at_3'))}. "
        + str(power.get("limitation", "")),
        "- Every timing claim is n=1: one kill, one resume, one pause per mechanism, one "
        "drain sequence. The exact count- and state-based hypotheses (H-a1, H-a2, H-b, "
        "H-c1, H-c2, H-c3, H-f1, H-x) are discrete facts that need no n; the derived "
        "TIMINGS carry no uncertainty except where a replicate exists.",
        "",
        "## What is NOT reproducible here",
        "",
    ]
    for item in prereg["not_reproducible"]:
        lines.append(f"- {item}")
    lines += ["", "Exactly three things are claimed deterministic:", ""]
    for item in prereg["deterministic_claims"]:
        lines.append(f"- {item}")

    # FIGURE STATUS IS CONSULTED, not assumed. Every figure is drawn inside its
    # own try/except that writes a legible placeholder PNG on failure — legible
    # enough that a report embedding it unconditionally would present a failure as
    # a result. The status file exists precisely to be read here.
    figure_status = {
        entry["figure"]: entry
        for entry in (
            C.read_json(C.FIGURES / "figures_status.json")["figures"]
            if (C.FIGURES / "figures_status.json").exists()
            else []
        )
    }
    lines += ["", "## Figures", ""]
    for name, caption in (
        (
            "fig1_stage_timeline.png",
            "Process wall time per rule (Snakemake `benchmark:`) — one quantity, every rule.",
        ),
        (
            "fig2_retrieval_recall_pre_vs_post.png",
            "recall@1 by signal, before and after the drain, with the chance level plotted "
            "as its own bar.",
        ),
        (
            "fig3_stop_cost_by_mode.png",
            "Discarded work per stop mechanism. EVERY bar is measured: the library arms "
            "from their own post-commit wall-clock, the SIGKILL from `killed_at` minus the "
            "converter ledger's last `ts_end`. The CLI arms are deliberately absent — "
            "their in-flight document is allowed to finish, so 'discarded seconds' is not "
            "the quantity that describes them.",
        ),
        (
            "fig4_enrichment_attribution.png",
            "Sequential component breakdown for one paper. Note that `extract_units` "
            "Stage 2 calls the embedding model, so 'chunking' is not pure CPU.",
        ),
    ):
        entry = figure_status.get(name)
        if entry is not None and not entry.get("ok"):
            lines += [
                f"> **FIGURE COULD NOT BE DRAWN — `{name}`**: {entry.get('error')}",
                "> The embedded PNG is a placeholder, not a result.",
                "",
            ]
        lines += [f"![{name}](figures/{name})", "", f"*{caption}*", ""]

    lines += [
        "## Provenance and integrity",
        "",
        f"- **harness sha256**: `{str(prov.get('harness', {}).get('harness_sha256'))[:20]}` "
        f"over {prov.get('harness', {}).get('n_files')} files — the Snakefile, config.yaml "
        "and every `scripts/*.py`. One value identifies the measuring apparatus.",
        "- `results/provenance.json` — git, versions, model digests, allowlisted env, and "
        "`results/working_tree.patch` whenever the tree is dirty (required by "
        "`validate_results`, so a run cannot record a sha that names code nobody has)",
        "- `results/preregistration.json` — hypotheses, thresholds AND scoring rules",
        "- `results/deviations.json` — every post-hoc change, with the direction it could "
        "have moved a verdict",
        "- `data/REGISTRY.json` — per-paper sha256 + URL provenance",
        "- `results/validation.json` — schema + referential integrity (exits 1 on violation)",
        "- `results/MANIFEST.json` — sha256 of every artefact AND of every harness script; "
        "re-check it with `uv run python -m "
        "experiments.docstore_deferred_ingestion.scripts.manifest --verify`",
        "",
    ]

    (C.EXP_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {C.EXP_DIR / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
