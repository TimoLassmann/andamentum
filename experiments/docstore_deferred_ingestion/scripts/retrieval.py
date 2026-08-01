"""Retrieval probing: three signals, one precomputed query embedding, no LLM.

"Semantically searchable" must be a behavioural claim, not a row count. So each
of the 8 paraphrase probes runs against the SAME database in three modes:

  (i)   ``search.search_fts5``           — keyword only
  (ii)  ``chunks_search.semantic_search`` — chunk embeddings (+ BM25 re-score)
  (iii) ``search.search_unified``         — the full 4-signal RRF stack

Splitting the signals separates "embeddings exist" from "embeddings help".

The query embedding is computed ONCE per probe and passed in, so the metric never
depends on an LLM. ``public.search()`` runs an LLM query planner whose output
varies run to run; using it for the metric would make claim (e) non-reproducible,
so it is exercised as a recorded smoke call only and is never scored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .corpus import PROBE_QUERIES, Probe


def _rank_of(doc_ids: Sequence[str], targets: set[str]) -> int | None:
    """1-based rank of the first acceptable doc_id, or None when none appear."""
    for index, doc_id in enumerate(doc_ids, start=1):
        if doc_id in targets:
            return index
    return None


def score_probe(
    ranked_doc_ids: Sequence[str], target_doc_id: str | set[str]
) -> dict[str, Any]:
    """recall@1, recall@3 and reciprocal rank for one probe against one signal.

    The target is a SET of acceptable doc_ids, because the same paper can legally
    be present under more than one row — claim (a) arm 2 queues one PDF as a
    ``pending_source`` alongside the markdown ingest of the same paper, and the
    drain completes both. The hypothesis is about retrieving the correct PAPER,
    so retrieving either row is a hit; scoring against a single row would fail
    claim (e) for a bookkeeping reason rather than a retrieval one.
    """
    targets = {target_doc_id} if isinstance(target_doc_id, str) else set(target_doc_id)
    rank = _rank_of(ranked_doc_ids, targets)
    return {
        "rank": rank,
        "recall_at_1": 1.0 if rank == 1 else 0.0,
        "recall_at_3": 1.0 if (rank is not None and rank <= 3) else 0.0,
        "reciprocal_rank": (1.0 / rank) if rank else 0.0,
        "n_returned": len(ranked_doc_ids),
    }


def aggregate(per_probe: list[dict[str, Any]], signal: str) -> dict[str, Any]:
    """Mean recall@1 / recall@3 / MRR across probes for one signal."""
    scored = [p["signals"][signal] for p in per_probe]
    n = len(scored) or 1
    return {
        "n_probes": len(scored),
        "recall_at_1": sum(s["recall_at_1"] for s in scored) / n,
        "recall_at_3": sum(s["recall_at_3"] for s in scored) / n,
        "mrr": sum(s["reciprocal_rank"] for s in scored) / n,
    }


def acceptable_doc_ids(claim_a: dict[str, Any]) -> dict[str, list[str]]:
    """short name -> every doc_uuid that legitimately holds that paper.

    Claim (a) arm 2 queues one PDF as a ``pending_source`` in the SAME database as
    the markdown ingest of that paper, and the drain completes both. Both rows are
    correct answers for a probe about that paper, so both are acceptable targets.
    """
    from .corpus import BY_ID

    mapping: dict[str, list[str]] = {}
    for record in claim_a["defer_records"]:
        mapping.setdefault(record["short"], []).append(record["doc_id"])

    probe = claim_a.get("source_probe") or {}
    source_arxiv = probe.get("arxiv_id")
    source_doc_id = probe.get("doc_id")
    if source_arxiv and source_doc_id and source_arxiv in BY_ID:
        mapping.setdefault(BY_ID[source_arxiv].short, []).append(source_doc_id)
    return mapping


async def run_probes(
    db_path: Path,
    doc_id_by_short: dict[str, str | list[str]],
    *,
    embedding_model: str,
    limit: int = 10,
    probes: Sequence[Probe] = PROBE_QUERIES,
) -> dict[str, Any]:
    """Run every probe through all three signals. Returns per-probe + aggregate.

    ``doc_id_by_short`` maps a paper's short name to the doc_uuid(s) it was
    ingested as, so scoring never depends on titles or on ingest order. A list
    means the same paper is present under more than one row (see
    :func:`score_probe`).
    """
    from andamentum.document_store.chunks_search import semantic_search
    from andamentum.document_store.embeddings import EmbeddingService
    from andamentum.document_store.search import search_fts5, search_unified

    embed_svc = EmbeddingService(model=embedding_model)
    per_probe: list[dict[str, Any]] = []
    try:
        for probe in probes:
            raw = doc_id_by_short.get(probe.expect_short)
            if not raw:
                raise KeyError(
                    f"probe expects paper {probe.expect_short!r} but no doc_id was "
                    "supplied for it — scoring against a missing target would silently "
                    "report zero recall for the wrong reason"
                )
            target = {raw} if isinstance(raw, str) else set(raw)
            # ONE embedding call per probe, reused across signals (ii) and (iii)
            # so the two measure retrieval, not embedding variance.
            query_embedding = await embed_svc.embed_text(probe.query, text_type="query")

            fts_hits = await search_fts5(str(db_path), probe.query, limit)
            fts_ids = [doc_id for doc_id, _ in fts_hits]

            chunk_hits = semantic_search(
                probe.query,
                query_embedding,
                limit=limit,
                db_path=db_path,
            )
            # De-duplicate to document order, preserving best-chunk rank.
            chunk_ids: list[str] = []
            for hit in chunk_hits:
                if hit.doc_id not in chunk_ids:
                    chunk_ids.append(hit.doc_id)

            unified_hits = await search_unified(
                str(db_path), probe.query, limit, query_embedding=query_embedding
            )
            unified_ids = [hit.doc_id for hit in unified_hits]

            per_probe.append(
                {
                    "query": probe.query,
                    "expect_short": probe.expect_short,
                    "expect_doc_ids": sorted(target),
                    "rationale": probe.rationale,
                    "signals": {
                        "fts5": score_probe(fts_ids, target),
                        "chunk_embeddings": score_probe(chunk_ids, target),
                        "unified_rrf": score_probe(unified_ids, target),
                    },
                }
            )
    finally:
        await embed_svc.close()

    # THE CHANCE BASELINE, published beside the score rather than left to the
    # reader. recall@3 over a 5-document candidate pool has a chance level of 3/5;
    # an 8/8 result against 4 topically disjoint famous papers is a real but WEAK
    # discrimination, and reporting 1.0 without the baseline invites it to be read
    # as strong evidence about the index. recall@1 and MRR are the informative
    # headline here, which is why all three are aggregated.
    n_candidates = _n_active_documents(db_path)
    return {
        "per_probe": per_probe,
        "aggregate": {
            signal: aggregate(per_probe, signal)
            for signal in ("fts5", "chunk_embeddings", "unified_rrf")
        },
        "power": {
            "n_probes": len(per_probe),
            "n_candidate_documents": n_candidates,
            "chance_recall_at_1": (1.0 / n_candidates) if n_candidates else None,
            "chance_recall_at_3": (min(3, n_candidates) / n_candidates) if n_candidates else None,
            "limitation": (
                "the probes discriminate between 4 topically disjoint papers with no "
                "hard negatives, so this measures 'the index is not empty and not "
                "scrambled', NOT 'retrieval quality is high'. Read recall@1 and MRR "
                "against the chance baselines above, not recall@3 against 1.0."
            ),
        },
    }


def _n_active_documents(db_path: Path) -> int:
    """Size of the candidate pool a probe is discriminating within."""
    import sqlite3

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
    try:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM documents WHERE deleted_at IS NULL"
            ).fetchone()[0]
        )
    finally:
        conn.close()
