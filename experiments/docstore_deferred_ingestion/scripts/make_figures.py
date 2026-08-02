"""Four figures via andamentum.figures. Never blocks the core results.

Every figure is drawn inside its own try/except that RECORDS the failure into
``figures/figures_status.json`` and still produces a placeholder file, because a
plotting problem must not be able to hide a measured result. The failure is
reported, not swallowed silently.

  fig1  stage timeline               — where the seconds go per rule
  fig2  retrieval recall pre vs post — the capability the drain created
  fig3  stop cost by mode            — discarded seconds per pause mechanism
  fig4  enrichment attribution       — the sequential component breakdown

Run:  uv run python -m experiments.docstore_deferred_ingestion.scripts.make_figures
"""

from __future__ import annotations

from typing import Any, Callable

from . import _common as C


def _placeholder(path: str, message: str) -> None:
    """Write a 1-line PNG-substitute so the DAG has its declared output."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig1_stage_timeline() -> None:
    """Process wall time per rule, from Snakemake's benchmark files.

    ONE QUANTITY, EVERY RULE. The previous version read the old ``wall_seconds``
    column, which analyze populated for exactly two rules — and for those two it
    held incommensurable things: a SUM OF TIMED COMPONENTS for micro_stages and a
    PROCESS WALL TIME for claim_c_resume, drawn against each other under a single
    axis label. It also omitted claim_b_drain, the largest cost in the run at
    ~1900 s, because that artefact happens not to carry an ``elapsed_seconds``
    key. ``benchmark_wall_seconds`` is populated for every rule and means one
    thing.
    """
    import csv

    from andamentum.figures import figure

    rows = []
    with (C.RESULTS / "timings.csv").open() as fh:
        for record in csv.DictReader(fh):
            value = (record.get("benchmark_wall_seconds") or "").strip()
            if not value:
                continue
            rows.append({"rule": record["rule"], "seconds": float(value)})
    if not rows:
        raise ValueError(
            "timings.csv has no usable benchmark_wall_seconds values — Snakemake's "
            "bench/*.tsv files are missing, so per-rule cost cannot be plotted"
        )
    figure(
        rows,
        kind="bar",
        x="rule",
        y="seconds",
        title="Where the wall-clock goes, per rule (Snakemake process wall time)",
        y_label="seconds (process wall time)",
        style="npg",
        output=str(C.FIGURES / "fig1_stage_timeline.png"),
    )


def fig2_recall() -> None:
    """recall@1, not recall@3 — plus the chance line the number must beat.

    Over a 5-document candidate pool, chance recall@3 is 3/5: a bar reaching 1.0
    against a 0.6 baseline reads as strong evidence and is not. recall@1 against
    1/5 is the contrast with content, so it is what is plotted.
    """
    from andamentum.figures import figure

    post_payload = C.read_json(C.RESULTS / "claim_e_post.json")
    pre = C.read_json(C.RESULTS / "claim_e_pre.json")["probes"]["aggregate"]
    post = post_payload["probes"]["aggregate"]
    power = post_payload.get("retrieval_power") or {}
    chance = power.get("chance_recall_at_1")

    rows = []
    for signal in ("fts5", "chunk_embeddings", "unified_rrf"):
        rows.append(
            {"signal": signal, "phase": "pre-drain", "recall_at_1": pre[signal]["recall_at_1"]}
        )
        rows.append(
            {"signal": signal, "phase": "post-drain", "recall_at_1": post[signal]["recall_at_1"]}
        )
    if chance:
        rows.append({"signal": "chance", "phase": "pre-drain", "recall_at_1": chance})
        rows.append({"signal": "chance", "phase": "post-drain", "recall_at_1": chance})
    figure(
        rows,
        kind="bar",
        x="signal",
        y="recall_at_1",
        group="phase",
        title=(
            "Retrieval recall@1 on 8 paraphrase probes, before and after the drain "
            f"({power.get('n_candidate_documents')} candidate documents)"
        ),
        y_label="recall@1 (chance shown as its own bar)",
        style="npg",
        output=str(C.FIGURES / "fig2_retrieval_recall_pre_vs_post.png"),
    )


def fig3_stop_cost() -> None:
    """Discarded work per stop mechanism — EVERY bar a measurement.

    The previous version hard-coded 0.0 for three of five bars in the plotting
    script, read a hard-coded constant for the fourth, and drew the fifth from
    ``micro_stages.sequential_total_seconds`` — a sequential micro-benchmark of a
    DIFFERENT paper in a DIFFERENT database, plotted as the SIGKILL's discarded
    work on this one. The true value was already derivable from committed
    artefacts and is ~4,800x smaller than what was drawn.

    Now: the three library arms read their own measured post-commit wall-clock,
    and the SIGKILL bar reads ``killed_at - last_conversion.ts_end`` from the kill
    lineage's own ledger. The CLI arms are absent by design — their in-flight
    document is deliberately allowed to finish, so "discarded seconds" is not the
    quantity that describes them; they are judged structurally instead.
    """
    from andamentum.figures import figure

    d = C.read_json(C.RESULTS / "claim_d_pause.json")
    k = C.read_json(C.RESULTS / "claim_c_kill.json")
    by_arm = d.get("pause_discarded_by_arm") or {}

    rows = []
    for arm, label in (
        ("arm1_max_docs", "max_docs"),
        ("arm2_max_seconds", "max_seconds"),
        ("arm3_should_continue", "should_continue"),
    ):
        entry = by_arm.get(arm) or {}
        if entry.get("post_commit_seconds") is None:
            raise ValueError(
                f"claim_d_pause.json has no measured post_commit_seconds for {arm} — "
                "refusing to draw a bar the harness did not measure"
            )
        rows.append(
            {"mode": label, "discarded_seconds": float(entry["post_commit_seconds"])}
        )

    kill_discarded = k.get("discarded_enrichment_seconds")
    if kill_discarded is None:
        raise ValueError(
            "claim_c_kill.json has no discarded_enrichment_seconds — the SIGKILL bar "
            "must come from this lineage's own ledger, never borrowed from another"
        )
    rows.append({"mode": "SIGKILL", "discarded_seconds": float(kill_discarded)})

    figure(
        rows,
        kind="bar",
        x="mode",
        y="discarded_seconds",
        title="Work discarded by each stop mechanism (all bars measured)",
        y_label="seconds of work thrown away",
        style="npg",
        output=str(C.FIGURES / "fig3_stop_cost_by_mode.png"),
    )


def fig4_attribution() -> None:
    from andamentum.figures import figure

    micro = C.read_json(C.RESULTS / "micro_stages.json")
    rows = [
        {"stage": stage, "seconds": seconds}
        for stage, seconds in micro["timings_seconds"].items()
    ]
    figure(
        rows,
        kind="bar",
        x="stage",
        y="seconds",
        title="Enrichment attribution for one paper, measured strictly sequentially",
        y_label="seconds",
        style="npg",
        output=str(C.FIGURES / "fig4_enrichment_attribution.png"),
    )


FIGURES: list[tuple[str, Callable[[], None]]] = [
    ("fig1_stage_timeline.png", fig1_stage_timeline),
    ("fig2_retrieval_recall_pre_vs_post.png", fig2_recall),
    ("fig3_stop_cost_by_mode.png", fig3_stop_cost),
    ("fig4_enrichment_attribution.png", fig4_attribution),
]


def main() -> int:
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    status: list[dict[str, Any]] = []
    for name, builder in FIGURES:
        target = C.FIGURES / name
        try:
            builder()
            status.append({"figure": name, "ok": True, "error": None})
            print(f"ok   {name}")
        except Exception as exc:  # noqa: BLE001 — recorded, never hidden
            error = f"{type(exc).__name__}: {exc}"
            _placeholder(str(target), f"{name}\ncould not be drawn:\n{error}")
            status.append({"figure": name, "ok": False, "error": error})
            print(f"FAIL {name}: {error}")
    # Through write_json, so this artefact carries the same schema / written_at /
    # provenance_ref envelope SCHEMAS.md declares EVERY json artefact carries. It
    # was previously a bare json.dumps and therefore exempt from the discipline
    # the experiment sets for itself — and from validate_results entirely.
    C.write_json(
        C.FIGURES / "figures_status.json",
        {
            "n_figures": len(status),
            "n_failed": len([s for s in status if not s["ok"]]),
            "figures": status,
            "consumed_by": (
                "build_report, which prints a visible 'FIGURE COULD NOT BE DRAWN' line "
                "for any entry with ok=false rather than embedding a placeholder PNG as "
                "though it were a result"
            ),
        },
        schema="andamentum.experiment.docstore_deferred.figures_status/1",
    )
    # Deliberately exit 0: figure generation must never block the core results.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
