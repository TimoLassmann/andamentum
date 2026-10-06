"""LitSearch: realistic literature-search queries with gold papers.

597 queries about recent ML and NLP papers, each with the corpus ids of the
papers that answer it. The Hugging Face dataset viewer serves the queries as
JSON. The gold papers are Semantic Scholar corpus ids; ``download_litsearch``
resolves the ids of the chosen subset to titles and identifiers through the
Semantic Scholar batch endpoint (the 1.2 GB parquet corpus is not needed: v2
searches the open literature, not the LitSearch corpus). Everything is written
under the directory the caller names; ``load_litsearch`` then reads it offline.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, JsonValue

from andamentum.references import (
    Work,
    normalise_arxiv_id,
    normalise_doi,
    normalise_pmid,
    normalise_s2_id,
)

from .download import (
    LITSEARCH_CARD,
    DatasetMissing,
    DownloadError,
    Downloader,
    save_file,
)
from .subsets import select_subset

HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HF_DATASET = "princeton-nlp/LitSearch"
HF_PAGE_SIZE = 100
S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_BATCH_SIZE = 100
S2_FIELDS = "title,year,externalIds"
ACL_ANTHOLOGY = "https://aclanthology.org/"
QUERIES_FILE = "litsearch/queries.jsonl"
GOLD_FILE = "litsearch/gold.jsonl"
DEFAULT_SUBSET = 30
DEFAULT_SEED = 0


class LitSearchQuery(BaseModel):
    query_id: str
    query: str = Field(min_length=1)
    query_set: str
    specificity: int
    quality: int
    corpus_ids: list[int] = Field(min_length=1)


class GoldRecord(BaseModel):
    """What the Semantic Scholar batch endpoint said about one gold corpus id."""

    corpus_id: int
    resolved: bool
    title: str = ""
    year: int | None = None
    paper_id: str = ""
    doi: str = ""
    arxiv: str = ""
    acl: str = ""
    pubmed: str = ""

    def to_work(self) -> Work | None:
        """The gold paper as a ``Work``, or None when it could not be resolved."""
        if not self.resolved:
            return None
        return Work(
            title=self.title,
            year=self.year,
            s2_id=normalise_s2_id(self.paper_id),
            doi=normalise_doi(self.doi),
            arxiv_id=normalise_arxiv_id(self.arxiv),
            pmid=normalise_pmid(self.pubmed),
            url=f"{ACL_ANTHOLOGY}{self.acl}/" if self.acl else None,
        )


class LitSearchItem(BaseModel):
    """A query and its gold papers (those that could be resolved)."""

    query: LitSearchQuery
    gold: list[Work] = Field(min_length=1)


class LitSearchSet(BaseModel):
    items: list[LitSearchItem]
    n_queries_total: int
    n_subset: int = Field(description="Queries chosen before any was dropped.")
    n_unresolved_gold: int = Field(
        description="Gold papers that could not be resolved."
    )
    n_dropped_queries: int = Field(
        description="Chosen queries left with no gold paper."
    )
    seed: int


class LitSearchDownload(BaseModel):
    n_queries_total: int
    n_subset: int
    n_gold_fetched: int


def _jsonl(records: list[BaseModel]) -> bytes:
    return "".join(r.model_dump_json() + "\n" for r in records).encode("utf-8")


def _int(value: JsonValue, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DownloadError(HF_ROWS_URL, f"{name} is {value!r}, not an integer")
    return value


def _text(value: JsonValue, name: str) -> str:
    if not isinstance(value, str):
        raise DownloadError(HF_ROWS_URL, f"{name} is {value!r}, not a string")
    return value


async def _fetch_queries(downloader: Downloader) -> list[LitSearchQuery]:
    queries: list[LitSearchQuery] = []
    total: int | None = None
    offset = 0
    while total is None or offset < total:
        payload = json.loads(
            await downloader.get(
                HF_ROWS_URL,
                params={
                    "dataset": HF_DATASET,
                    "config": "query",
                    "split": "full",
                    "offset": str(offset),
                    "length": str(HF_PAGE_SIZE),
                },
            )
        )
        total = _int(payload["num_rows_total"], "num_rows_total")
        rows = payload["rows"]
        if not rows:
            raise DownloadError(HF_ROWS_URL, f"no rows at offset {offset} of {total}")
        for entry in rows:
            row = entry["row"]
            ids = row["corpusids"]
            queries.append(
                LitSearchQuery(
                    query_id=f"LS{_int(entry['row_idx'], 'row_idx'):04d}",
                    query=_text(row["query"], "query"),
                    query_set=_text(row["query_set"], "query_set"),
                    specificity=_int(row["specificity"], "specificity"),
                    quality=_int(row["quality"], "quality"),
                    corpus_ids=[_int(i, "corpusids") for i in ids],
                )
            )
        offset += len(rows)
    if len(queries) != total:
        raise DownloadError(HF_ROWS_URL, f"read {len(queries)} of {total} rows")
    return queries


def _gold_record(corpus_id: int, found: JsonValue) -> GoldRecord:
    if not isinstance(found, dict):
        return GoldRecord(corpus_id=corpus_id, resolved=False)
    ids = found.get("externalIds")
    external = ids if isinstance(ids, dict) else {}
    title = found.get("title")
    year = found.get("year")
    paper_id = found.get("paperId")

    def known(key: str) -> str:
        value = external.get(key)
        return value if isinstance(value, str) else ""

    if not isinstance(title, str) or not title.strip():
        return GoldRecord(corpus_id=corpus_id, resolved=False)
    return GoldRecord(
        corpus_id=corpus_id,
        resolved=True,
        title=title,
        year=year if isinstance(year, int) and not isinstance(year, bool) else None,
        paper_id=paper_id if isinstance(paper_id, str) else "",
        doi=known("DOI"),
        arxiv=known("ArXiv"),
        acl=known("ACL"),
        pubmed=known("PubMed"),
    )


async def _resolve_gold(
    downloader: Downloader, corpus_ids: list[int], *, api_key: str | None
) -> list[GoldRecord]:
    headers = {"x-api-key": api_key} if api_key else None
    records: list[GoldRecord] = []
    for start in range(0, len(corpus_ids), S2_BATCH_SIZE):
        chunk = corpus_ids[start : start + S2_BATCH_SIZE]
        answer = json.loads(
            await downloader.post_json(
                S2_BATCH_URL,
                {"ids": [f"CorpusId:{i}" for i in chunk]},
                params={"fields": S2_FIELDS},
                headers=headers,
            )
        )
        if not isinstance(answer, list) or len(answer) != len(chunk):
            raise DownloadError(
                S2_BATCH_URL, f"expected {len(chunk)} answers, got {answer!r:.200}"
            )
        records.extend(_gold_record(i, found) for i, found in zip(chunk, answer))
    return records


def _read_gold(path: Path) -> dict[int, GoldRecord]:
    if not path.exists():
        return {}
    records = [
        GoldRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {r.corpus_id: r for r in records}


def _read_queries(root: Path) -> list[LitSearchQuery]:
    path = root / QUERIES_FILE
    if not path.exists():
        raise DatasetMissing(
            f"LitSearch is not in {root}; run: uv run python -m "
            f"andamentum.deep_research.v2.bench download litsearch --dir {root}"
        )
    return [
        LitSearchQuery.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


async def download_litsearch(
    root: Path,
    *,
    downloader: Downloader,
    n: int = DEFAULT_SUBSET,
    seed: int = DEFAULT_SEED,
    s2_api_key: str | None = None,
) -> LitSearchDownload:
    """Fetch the queries and the gold papers of the first *n* queries of the subset.

    Re-running with a larger *n* resolves only the gold papers not yet held.
    """
    queries = await _fetch_queries(downloader)
    save_file(
        root,
        QUERIES_FILE,
        _jsonl(list(queries)),
        url=f"{HF_ROWS_URL}?dataset={HF_DATASET}&config=query",
        card=LITSEARCH_CARD,
    )
    chosen = set(select_subset([q.query_id for q in queries], n, seed=seed))
    wanted = sorted({c for q in queries if q.query_id in chosen for c in q.corpus_ids})
    held = _read_gold(root / GOLD_FILE)
    missing = [c for c in wanted if c not in held]
    if missing:
        held.update(
            {
                r.corpus_id: r
                for r in await _resolve_gold(downloader, missing, api_key=s2_api_key)
            }
        )
        save_file(
            root,
            GOLD_FILE,
            _jsonl(list(held.values())),
            url=S2_BATCH_URL,
            card=LITSEARCH_CARD,
        )
    return LitSearchDownload(
        n_queries_total=len(queries), n_subset=len(chosen), n_gold_fetched=len(missing)
    )


def load_litsearch(
    root: Path, *, n: int = DEFAULT_SUBSET, seed: int = DEFAULT_SEED, skip: int = 0
) -> LitSearchSet:
    """The deterministic subset of *n* queries with their resolved gold papers.

    With *skip* the subset is the seeded order's positions ``skip + 1`` to
    ``skip + n`` (``select_subset(ids, skip + n)[skip:]``): the queries a
    smaller subset of the same seed did not hold. Offline. Raises
    ``DatasetMissing`` when the queries, or the gold papers of a chosen query,
    were not downloaded.
    """
    queries = _read_queries(root)
    by_id = {q.query_id: q for q in queries}
    chosen = select_subset(list(by_id), skip + n, seed=seed)[skip:]
    held = _read_gold(root / GOLD_FILE)
    items: list[LitSearchItem] = []
    unresolved = 0
    dropped = 0
    for query_id in chosen:
        query = by_id[query_id]
        absent = [c for c in query.corpus_ids if c not in held]
        if absent:
            raise DatasetMissing(
                f"the gold papers of {query_id} are not in {root}; run: uv run "
                f"python -m andamentum.deep_research.v2.bench download litsearch "
                f"--dir {root} --n {skip + n} --seed {seed}"
            )
        gold = [w for c in query.corpus_ids if (w := held[c].to_work()) is not None]
        unresolved += len(query.corpus_ids) - len(gold)
        if not gold:
            dropped += 1
            continue
        items.append(LitSearchItem(query=query, gold=gold))
    return LitSearchSet(
        items=items,
        n_queries_total=len(queries),
        n_subset=len(chosen),
        n_unresolved_gold=unresolved,
        n_dropped_queries=dropped,
        seed=seed,
    )
