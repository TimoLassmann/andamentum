"""Benchmark: does the FTS query fix let the store discriminate near-miss identifiers?

Tests the change in `document_store/fts_query.py` — phrase-quoting punctuated
tokens before they hit FTS5 MATCH — against the real `documents_fts` index that
the store's triggers populate. No LLM, no embeddings: this isolates the keyword
layer, which is the only thing the fix touches.

Three things are measured, old (raw query, today's behaviour) vs new (prepared):

  1. Robustness  — raw identifier queries raise `fts5: syntax error` and the
     keyword signal silently drops out; prepared queries execute.
  2. Discrimination — for an exact identifier query, the doc carrying that exact
     identifier ranks #1 and its near-miss siblings (same prose, one token
     different) are excluded.
  3. No prose regression — ordinary prose queries return byte-identical result
     lists old vs new (prose passes through `prepare_fts_query` unchanged).

Run:  uv run python benchmarks/document_store/identifier_discrimination.py
"""
from __future__ import annotations

import asyncio
import sqlite3
import tempfile

from andamentum.document_store.api import DocumentStore
from andamentum.document_store.fts_query import prepare_fts_query

# --- corpus: identifier families (same prose, one token differs) + prose docs ---

FAMILIES: dict[str, tuple[str, list[str]]] = {
    "hgvs_coding": (
        "A recurrent {id} substitution was identified in the index case and "
        "confirmed by Sanger sequencing of the proband and both parents.",
        ["c.1234G>A", "c.1234G>T", "c.1234G>C", "c.1235G>A", "c.76A>T"],
    ),
    "hgvs_protein": (
        "The {id} change alters a conserved residue and co-segregates with the "
        "phenotype across three affected generations of the family.",
        ["p.Arg502Trp", "p.Arg502Gln", "p.Gly12Asp", "p.Gly12Val"],
    ),
    "accession": (
        "Transcript {id} was used as the reference sequence for variant "
        "annotation throughout this study of the cohort.",
        ["NM_000256.3", "NM_000256.4", "NM_000257.3"],
    ),
    "doi": (
        "See the companion analysis published as {id} for the full cohort "
        "description and the extended supplementary methods.",
        ["10.1038/nature12373", "10.1038/nature12374"],
    ),
}

PROSE_DOCS: list[tuple[str, str]] = [
    ("mitosis", "Mitosis proceeds through prophase, metaphase, anaphase and telophase, "
                "segregating replicated chromosomes into two daughter nuclei."),
    ("ml", "Gradient descent optimises model parameters by iteratively stepping "
           "against the gradient of a differentiable loss function."),
    ("history", "The printing press transformed the circulation of ideas across "
                "early modern Europe and accelerated literacy."),
    ("ecology", "Keystone predators regulate community structure by limiting the "
                "abundance of dominant competitors in an ecosystem."),
]

PROSE_QUERIES = ["sanger sequencing proband", "conserved residue phenotype",
                 "reference transcript annotation", "gradient descent loss",
                 "printing press literacy"]


def query_fts(db_path: str, match_query: str) -> list[str] | str:
    """Run a MATCH against documents_fts. Returns ranked doc titles, or the
    error type name if FTS5 rejects the query (today's raw-query failure)."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT d.dc_title FROM documents_fts fts
            JOIN documents d ON fts.rowid = d.id
            WHERE documents_fts MATCH ? AND d.deleted_at IS NULL
            ORDER BY rank
            """,
            (match_query,),
        ).fetchall()
        return [r[0] for r in rows]
    except sqlite3.OperationalError as e:
        return f"ERROR: {e}"
    finally:
        conn.close()


async def main() -> None:
    tmp = tempfile.mkdtemp(prefix="idbench_")
    store = DocumentStore(database_name="idbench", db_dir=tmp)
    await store.initialize()

    # title == unique id (or prose key) so results are self-identifying
    identifier_docs: dict[str, str] = {}  # id -> family
    for fam, (template, ids) in FAMILIES.items():
        for ident in ids:
            await store.register_document(title=ident, content=template.format(id=ident))
            identifier_docs[ident] = fam
    for title, body in PROSE_DOCS:
        await store.register_document(title=title, content=body)

    db_path = str(store.db_path)

    # ---- 1 & 2: robustness + discrimination on exact identifier queries ----
    print("=" * 78)
    print("IDENTIFIER QUERIES  (raw = today, prepared = fixed)")
    print("=" * 78)
    n = 0
    raw_crashes = 0
    top1 = 0
    leaked = 0
    for ident, fam in identifier_docs.items():
        siblings = {i for i in FAMILIES[fam][1] if i != ident}
        raw = query_fts(db_path, ident)
        prep = query_fts(db_path, prepare_fts_query(ident))
        n += 1
        if isinstance(raw, str):
            raw_crashes += 1
        ok_top1 = isinstance(prep, list) and prep[:1] == [ident]
        leak = 0 if isinstance(prep, str) else len(set(prep) & siblings)
        top1 += ok_top1
        leaked += leak
        raw_str = raw if isinstance(raw, str) else f"{len(raw)} docs {raw[:3]}"
        prep_str = prep if isinstance(prep, str) else f"{prep}"
        flag = "OK " if ok_top1 and leak == 0 else "!! "
        print(f"{flag}{ident:22}")
        print(f"     raw     -> {raw_str[:60]}")
        print(f"     prepared-> {prep_str}")

    # ---- 3: prose no-regression ----
    print("\n" + "=" * 78)
    print("PROSE QUERIES  (must be identical raw vs prepared)")
    print("=" * 78)
    prose_identical = 0
    for q in PROSE_QUERIES:
        raw = query_fts(db_path, q)
        prep = query_fts(db_path, prepare_fts_query(q))
        same = raw == prep
        prose_identical += same
        print(f"{'OK ' if same else '!! '}{q:34} raw==prepared: {same}  -> {raw}")

    # ---- summary ----
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"identifier queries              : {n}")
    print(f"  raw query crashed (old bug)   : {raw_crashes}/{n}")
    print(f"  prepared: correct doc rank #1 : {top1}/{n}")
    print(f"  prepared: near-miss leakage    : {leaked} (want 0)")
    print(f"prose queries identical raw==prep: {prose_identical}/{len(PROSE_QUERIES)}")
    ok = (top1 == n and leaked == 0 and prose_identical == len(PROSE_QUERIES))
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    asyncio.run(main())
