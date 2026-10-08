"""LitSearch against its own corpus: queries, gold papers and a local corpus folder.

The LitSearch dataset (Ajith et al., EMNLP 2024) holds 597 queries and a corpus
of 64,183 papers, each query naming the corpus papers that answer it. Here the
benchmark runs on that corpus alone: a retrieval system is given the corpus as
its only source, and a gold paper is the corpus record with that id. No network
is used by a run, so the searches are deterministic.

Layout of the dataset folder (all of it local evaluation data, never committed):

- ``query/full-00000-of-00001.parquet``: the Hugging Face file, byte for byte.
- ``queries.jsonl``: the 597 queries, one ``LitSearchQuery`` per line (derived).
- ``corpus.jsonl``: title, abstract and citation ids of every paper, one
  ``sources.local_corpus.CorpusRecord`` per line (derived from the
  ``corpus_clean`` configuration, whose files also hold each paper's full text:
  only the four wanted columns are read, one column chunk at a time, by HTTP
  range requests).
- ``corpus.sqlite``: the BM25 and citation index of the corpus (derived).
- ``MANIFEST.json`` and ``PROVENANCE.md``: where every file came from, with
  sha256, and on what terms it may be used.

The dataset revision is pinned (``HF_REVISION``): a download of another
revision is a different dataset and must be a deliberate change here.
"""

from __future__ import annotations

import hashlib
import io
from collections import OrderedDict
from collections.abc import Callable, Iterator
from datetime import date
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from andamentum.references import Work

from ..sources.local_corpus import (
    CORPUS_FILE,
    INDEX_FILE,
    CorpusRecord,
    IndexSummary,
    build_index,
    work_of,
)
from .subsets import select_subset
from .download import (
    DatasetCard,
    DatasetMissing,
    DownloadError,
    save_file,
    write_atomic,
)

DEFAULT_SEED = 0
DEFAULT_SUBSET = 30


class LitSearchQuery(BaseModel):
    query_id: str
    query: str = Field(min_length=1)
    query_set: str
    specificity: int
    quality: int
    corpus_ids: list[int] = Field(min_length=1)


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


HF_DATASET = "princeton-nlp/LitSearch"
HF_REVISION = "9573fb284a1026c998df47024b888a163f0f0e25"
HF_URL = f"https://huggingface.co/datasets/{HF_DATASET}"
HF_API_TREE = f"https://huggingface.co/api/datasets/{HF_DATASET}/tree/{HF_REVISION}"
HF_RESOLVE = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{HF_REVISION}"
QUERY_PARQUET = "query/full-00000-of-00001.parquet"
CORPUS_PARQUETS: tuple[str, ...] = tuple(
    f"corpus_clean/full-{i:05d}-of-00006.parquet" for i in range(6)
)
QUERIES_FILE = "queries.jsonl"
PROVENANCE_FILE = "PROVENANCE.md"
FETCH_FILE = "download.json"
CORPUS_COLUMNS = ["corpusid", "title", "abstract", "citations"]
BLOCK_BYTES = 4 * 1024 * 1024
BATCH_ROWS = 2000
CORPUS_CITATION = (
    "Ajith, Xia, Chevalier, Goyal, Chen, Gao. LitSearch: A Retrieval Benchmark for "
    "Scientific Literature Search. Empirical Methods in Natural Language Processing "
    "(EMNLP), 2024."
)
USAGE_RULE = "local evaluation only, do not redistribute"

S2ORC_PAGE = "https://github.com/allenai/s2orc"
S2ORC_README = "https://raw.githubusercontent.com/allenai/s2orc/master/README.md"
# Verbatim from the README of allenai/s2orc (the corpus' official page), read
# 2026-10-07; the corpus_s2orc configuration of the dataset is "the same set of
# 64183 documents expressed in the Semantic Scholar Open Research Corpus (S2ORC)
# schema" (dataset card).
S2ORC_LICENCE_QUOTES: tuple[str, ...] = (
    "S2ORC is released under the [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/).  "
    "By using S2ORC, you agree to the terms in the license.",
    "S2ORC is currently released through the [Semantic Scholar Public API]"
    "(https://www.semanticscholar.org/product/api) under the "
    "[ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/). By using S2ORC, you "
    "are agreeing to its usage terms.",
    "Original S2ORC was released under a non-commercial license. The current S2ORC is "
    "released under an ODC-By 1.0 license. We ask that users take care to double-check "
    "whether their intended usage of S2ORC and its underlying contents is permissible "
    "under this license.",
)
S2ORC_CITATION = (
    "Lo, Wang, Neumann, Kinney, Weld. S2ORC: The Semantic Scholar Open Research "
    "Corpus. Proceedings of the 58th Annual Meeting of the Association for "
    "Computational Linguistics, 2020. https://doi.org/10.18653/v1/2020.acl-main.447"
)
CODE_REPOSITORY = "https://github.com/princeton-nlp/LitSearch"

LITSEARCH_LOCAL_CARD = DatasetCard(
    key="litsearch-local",
    title="LitSearch (queries and corpus_clean) for evaluation against its own corpus",
    source=HF_URL,
    licence="no licence declared for the dataset; code MIT; corpus derives from S2ORC (ODC-By 1.0)",
    licence_evidence=(
        f"the Hugging Face dataset card declares no licence; the code repository "
        f"{CODE_REPOSITORY} is MIT (GitHub licence API, 2026-10-07); S2ORC, the "
        f"corpus' source, is released under ODC-By 1.0 per {S2ORC_PAGE}"
    ),
    terms=(
        "Queries and corpus_clean (titles, abstracts and citation ids) only; the "
        "full texts of corpus_clean and the corpus_s2orc configuration are not "
        f"downloaded. Usage rule: {USAGE_RULE}. The folder is never committed."
    ),
    citation=CORPUS_CITATION,
)


class RemoteFile(BaseModel):
    """One file of the pinned revision, as the Hub's tree API lists it."""

    path: str
    size: int
    sha256: str = Field(description="The LFS object id: the sha256 of the file.")


class LitSearchLocalDownload(BaseModel):
    n_queries: int
    n_gold_ids: int
    n_gold_in_corpus: int
    index: IndexSummary
    bytes_fetched: int = Field(description="Bytes read from the network by the corpus.")
    corpus_files: list[RemoteFile]
    query_file: RemoteFile


# -- reading parquet over HTTP range requests -----------------------------------


class RangeFile(io.RawIOBase):
    """A read-only seekable file over an HTTP resource that accepts range requests.

    pyarrow reads a parquet file through ``read``/``seek``; with a column
    selection it asks only for the footer and the chosen column chunks. Reads are
    served from blocks of ``BLOCK_BYTES`` kept in a small LRU, so many small
    reads inside one column chunk make one request. ``bytes_fetched`` counts
    what came over the network.
    """

    def __init__(
        self,
        client: httpx.Client,
        url: str,
        *,
        size: int,
        block_bytes: int = BLOCK_BYTES,
        keep_blocks: int = 8,
    ) -> None:
        super().__init__()
        self._client = client
        self._url = url
        self._size = size
        self._block = block_bytes
        self._keep = keep_blocks
        self._blocks: OrderedDict[int, bytes] = OrderedDict()
        self._pos = 0
        self.bytes_fetched = 0
        self.requests = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(f"bad whence {whence}")
        if self._pos < 0:
            raise ValueError("negative seek position")
        return self._pos

    def _fetch_block(self, index: int) -> bytes:
        cached = self._blocks.get(index)
        if cached is not None:
            self._blocks.move_to_end(index)
            return cached
        start = index * self._block
        end = min(start + self._block, self._size) - 1
        response = self._client.get(
            self._url, headers={"Range": f"bytes={start}-{end}"}
        )
        if response.status_code != 206 or len(response.content) != end - start + 1:
            raise DownloadError(
                self._url,
                f"range {start}-{end}: HTTP {response.status_code}, "
                f"{len(response.content)} bytes (expected 206 and {end - start + 1})",
                status=response.status_code,
            )
        self.requests += 1
        self.bytes_fetched += len(response.content)
        self._blocks[index] = response.content
        while len(self._blocks) > self._keep:
            self._blocks.popitem(last=False)
        return response.content

    def read(self, size: int = -1) -> bytes:
        if self._pos >= self._size:
            return b""
        end = (
            self._size
            if size is None or size < 0
            else min(self._pos + size, self._size)
        )
        parts: list[bytes] = []
        position = self._pos
        while position < end:
            index = position // self._block
            block = self._fetch_block(index)
            offset = position - index * self._block
            take = min(len(block) - offset, end - position)
            parts.append(block[offset : offset + take])
            position += take
        self._pos = end
        return b"".join(parts)

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def _pyarrow() -> Any:
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-not-found,unused-ignore]
    except ImportError as error:
        raise ImportError(
            "reading the LitSearch parquet files needs pyarrow: install the "
            "bench-data extra (uv sync --extra bench-data)"
        ) from error
    return parquet


def list_remote(client: httpx.Client, folder: str) -> list[RemoteFile]:
    """The files under *folder* of the pinned revision with their sizes and sha256."""
    response = client.get(f"{HF_API_TREE}/{folder}")
    if response.status_code != 200:
        raise DownloadError(
            f"{HF_API_TREE}/{folder}",
            f"HTTP {response.status_code}",
            status=response.status_code,
        )
    files: list[RemoteFile] = []
    for entry in response.json():
        if entry.get("type") != "file":
            continue
        lfs = entry.get("lfs") or {}
        files.append(
            RemoteFile(
                path=entry["path"],
                size=int(entry["size"]),
                sha256=str(lfs.get("oid", "")),
            )
        )
    return sorted(files, key=lambda f: f.path)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def corpus_rows(
    client: httpx.Client, remote: RemoteFile, *, block_bytes: int = BLOCK_BYTES
) -> Iterator[tuple[CorpusRecord, int]]:
    """The records of one ``corpus_clean`` file (four columns read), each with
    the bytes fetched so far for the file."""
    parquet = _pyarrow()
    file = RangeFile(
        client, f"{HF_RESOLVE}/{remote.path}", size=remote.size, block_bytes=block_bytes
    )
    reader = parquet.ParquetFile(file)
    for batch in reader.iter_batches(batch_size=BATCH_ROWS, columns=CORPUS_COLUMNS):
        for row in batch.to_pylist():
            if row["corpusid"] is None:
                raise DownloadError(remote.path, "a record has no corpusid")
            yield (
                CorpusRecord(
                    corpus_id=str(row["corpusid"]),
                    title=_text(row["title"]),
                    abstract=_text(row["abstract"]),
                    citations=[str(c) for c in (row["citations"] or [])],
                ),
                file.bytes_fetched,
            )


# -- queries --------------------------------------------------------------------


def queries_from_parquet(data: bytes) -> list[LitSearchQuery]:
    """The queries of the ``query`` configuration, ids ``LS<row>`` as the other
    LitSearch loader names them (so a query keeps its id across the two)."""
    parquet = _pyarrow()
    table = parquet.read_table(io.BytesIO(data))
    rows = table.to_pylist()
    return [
        LitSearchQuery(
            query_id=f"LS{index:04d}",
            query=row["query"],
            query_set=row["query_set"],
            specificity=row["specificity"],
            quality=row["quality"],
            corpus_ids=list(row["corpusids"]),
        )
        for index, row in enumerate(rows)
    ]


def _jsonl(records: list[LitSearchQuery]) -> bytes:
    return "".join(r.model_dump_json() + "\n" for r in records).encode("utf-8")


# -- download -------------------------------------------------------------------


def download_litsearch_local(
    root: Path,
    *,
    client: httpx.Client,
    today: date,
    progress: Callable[[str], None] = lambda _line: None,
    force: bool = False,
    block_bytes: int = BLOCK_BYTES,
) -> LitSearchLocalDownload:
    """Fetch the queries and the title/abstract/citation columns of the corpus of
    the pinned revision into *root*, build the index, write ``PROVENANCE.md``.

    A folder that already holds the corpus is not fetched again unless *force*;
    the index and the provenance are rebuilt from what is there. Raises
    ``DownloadError`` when the revision does not hold exactly the expected files
    or a file differs from the hash the Hub lists.
    """
    root.mkdir(parents=True, exist_ok=True)
    query_remote = list_remote(client, "query")
    corpus_remote = list_remote(client, "corpus_clean")
    if [f.path for f in query_remote] != [QUERY_PARQUET] or [
        f.path for f in corpus_remote
    ] != list(CORPUS_PARQUETS):
        raise DownloadError(
            HF_API_TREE,
            "the pinned revision does not hold the expected files: "
            f"{[f.path for f in query_remote]} {[f.path for f in corpus_remote]}",
        )
    query_file = query_remote[0]
    query_path = root / QUERY_PARQUET
    if force or not query_path.exists():
        progress(f"queries: {query_file.path} ({query_file.size} bytes)")
        response = client.get(f"{HF_RESOLVE}/{query_file.path}")
        if response.status_code != 200:
            raise DownloadError(
                f"{HF_RESOLVE}/{query_file.path}",
                f"HTTP {response.status_code}",
                status=response.status_code,
            )
        if _sha256(response.content) != query_file.sha256:
            raise DownloadError(
                query_file.path, "sha256 differs from the hash the Hub lists"
            )
        save_file(
            root,
            QUERY_PARQUET,
            response.content,
            url=f"{HF_RESOLVE}/{query_file.path}",
            card=LITSEARCH_LOCAL_CARD,
        )
    queries = queries_from_parquet(query_path.read_bytes())
    save_file(
        root,
        QUERIES_FILE,
        _jsonl(queries),
        url=f"{HF_RESOLVE}/{query_file.path} (rows as LS<row>)",
        card=LITSEARCH_LOCAL_CARD,
    )

    corpus_path = root / CORPUS_FILE
    fetched = 0
    if force or not corpus_path.exists():
        partial = corpus_path.with_name(corpus_path.name + ".partial")
        seen: set[str] = set()
        with partial.open("w", encoding="utf-8") as out:
            for remote in corpus_remote:
                progress(f"corpus: {remote.path} ({remote.size} bytes in the file)")
                last = 0
                for record, bytes_so_far in corpus_rows(
                    client, remote, block_bytes=block_bytes
                ):
                    if record.corpus_id in seen:
                        raise DownloadError(
                            remote.path, f"duplicate corpus id {record.corpus_id}"
                        )
                    seen.add(record.corpus_id)
                    out.write(record.model_dump_json() + "\n")
                    last = bytes_so_far
                fetched += last
        partial.replace(corpus_path)
        write_atomic(
            root / FETCH_FILE,
            (
                FetchRecord(
                    date=today.isoformat(), revision=HF_REVISION, bytes_fetched=fetched
                ).model_dump_json(indent=2)
                + "\n"
            ).encode("utf-8"),
        )
    summary = build_index(root)
    in_corpus = _gold_in_corpus(root, queries)
    gold_ids = {c for q in queries for c in q.corpus_ids}
    result = LitSearchLocalDownload(
        n_queries=len(queries),
        n_gold_ids=len(gold_ids),
        n_gold_in_corpus=len(in_corpus),
        index=summary,
        bytes_fetched=fetched,
        corpus_files=corpus_remote,
        query_file=query_file,
    )
    _write_provenance(root, result, today=today)
    return result


def _count_untitled(root: Path) -> tuple[int, int]:
    """(records without a title, records with neither title nor abstract)."""
    import sqlite3

    conn = sqlite3.connect(f"{(root / INDEX_FILE).as_uri()}?mode=ro", uri=True)
    try:
        untitled = conn.execute("SELECT count(*) FROM docs WHERE title = ''").fetchone()
        empty = conn.execute(
            "SELECT count(*) FROM docs WHERE title = '' AND abstract = ''"
        ).fetchone()
        return int(untitled[0]), int(empty[0])
    finally:
        conn.close()


def _gold_in_corpus(root: Path, queries: list[LitSearchQuery]) -> set[int]:
    ids = {c for q in queries for c in q.corpus_ids}
    found = records_by_id(root, ids)
    return {int(i) for i in found}


class FetchRecord(BaseModel):
    """What the fetch that wrote ``corpus.jsonl`` recorded of itself."""

    date: str
    revision: str
    bytes_fetched: int


def _read_fetch_record(root: Path) -> FetchRecord | None:
    """The record of the fetch that wrote ``corpus.jsonl``, or None for a folder
    that has none."""
    path = root / FETCH_FILE
    if not path.exists():
        return None
    record = FetchRecord.model_validate_json(path.read_text(encoding="utf-8"))
    if record.revision != HF_REVISION:
        raise DownloadError(
            str(path),
            f"the folder was fetched at revision {record.revision}, not the "
            f"pinned {HF_REVISION}: fetch again with --force",
        )
    return record


def _write_provenance(
    root: Path, result: LitSearchLocalDownload, *, today: date
) -> None:
    fetch = _read_fetch_record(root)
    fetched_on = today.isoformat() if fetch is None else fetch.date
    fetched_bytes = None if fetch is None else fetch.bytes_fetched
    read_note = (
        "the number of bytes read was not recorded"
        if fetched_bytes is None
        else f"about {fetched_bytes / 1e6:,.0f} MB of the "
        f"{sum(f.size for f in result.corpus_files) / 1e6:,.0f} MB of the six files"
    )

    def row(name: str) -> str:
        path = root / name
        return f"| `{name}` | {path.stat().st_size:,} | `{_file_sha256(path)}` |"

    fetch_row = row(FETCH_FILE) if fetch is not None else ""

    index = result.index
    untitled, empty = _count_untitled(root)
    sources = "\n".join(
        f"| `{f.path}` | {f.size:,} | `{f.sha256}` |"
        for f in [result.query_file, *result.corpus_files]
    )
    quotes = "\n".join(f"> {q}\n" for q in S2ORC_LICENCE_QUOTES)
    text = f"""# LitSearch (local evaluation copy): provenance

Local evaluation data. **{USAGE_RULE.capitalize()}.** This folder is ignored by git
(`data/*` in the repository `.gitignore`) and must never be committed, published or
attached to a release. Results computed from it may be reported as counts and scores;
the records themselves stay here.

## Source

- Dataset: {HF_DATASET} (queries and `corpus_clean`)
- URL: {HF_URL}
- Dataset revision (pinned, the commit of the Hub repository): `{HF_REVISION}`
- Download date: {fetched_on}
- Not downloaded: the `corpus_s2orc` configuration, and the `full_paper` text column of
  `corpus_clean` (the Hub stores it in the same parquet files as the other columns, so
  only the four wanted columns were read, by HTTP range requests; {read_note}).

Source files at that revision (size and sha256 as the Hub lists them; the query file was
downloaded whole and checked against the hash, the corpus files were read by column):

| file | bytes | sha256 (Hub LFS object id) |
|---|---|---|
{sources}

## Files in this folder

| file | bytes | sha256 |
|---|---|---|
{row(QUERY_PARQUET)}
{row(QUERIES_FILE)}
{row(CORPUS_FILE)}
{row(INDEX_FILE)}
{fetch_row}

- `{QUERY_PARQUET}`: the Hub file, unchanged.
- `{QUERIES_FILE}`: derived. The {result.n_queries} queries, ids `LS<row index>` (the same
  ids the earlier LitSearch loader uses), gold corpus ids as integers.
- `{CORPUS_FILE}`: derived. {index.n_records:,} records (`corpus_id`, `title`, `abstract`,
  `citations`) in the order of the six source files; {index.n_with_abstract:,} have an
  abstract. {untitled:,} records have no title and {empty:,} of them no abstract either
  (the Hub holds them so: they stay empty here, are in the index, cannot be found by a
  search and can be reached only through citation links).
- `{INDEX_FILE}`: derived, and rebuildable from `{CORPUS_FILE}` (SQLite
  {index.sqlite_version}, FTS5 BM25 over title and abstract with the `porter unicode61`
  tokenizer; {index.n_citation_edges:,} citation edges between records of the corpus,
  {index.n_citation_edges_outside:,} cited ids outside it dropped). Its hash is that of the
  file built on the download date; the contents, not the bytes, are what is reproducible.
- `{FETCH_FILE}`: the date, revision and bytes read of the fetch that wrote
  `{CORPUS_FILE}` (a rebuild of the index or of this file keeps them).
- `MANIFEST.json`: the downloader's own record (URL, size, sha256, licence card).
- Gold papers: {result.n_gold_ids} distinct corpus ids over all queries,
  {result.n_gold_in_corpus} of them in `{CORPUS_FILE}`.

## Licence, as declared

- The Hub dataset card ({HF_URL}) declares **no licence** (its metadata has no
  `license` field).
- The code repository ({CODE_REPOSITORY}) is **MIT** (GitHub licence API, 2026-10-07).
  That covers the code, not the data.
- The corpus derives from S2ORC: the dataset card says `corpus_s2orc` holds "the same set of
  64183 documents expressed in the Semantic Scholar Open Research Corpus (S2ORC) schema".
  S2ORC's official page is {S2ORC_PAGE}; its README ({S2ORC_README}, read 2026-10-07)
  says, verbatim:

{quotes}
- The papers themselves keep their authors' and publishers' rights; only titles,
  abstracts and citation ids are held here.
- Nothing above grants a right to redistribute this folder. Until the position of the
  dataset authors is clear, the folder is used for local evaluation only.

## Citation requested by the authors

{CORPUS_CITATION}

```bibtex
@inproceedings{{ajith2024litsearch,
   title={{LitSearch: A Retrieval Benchmark for Scientific Literature Search}},
   author={{Ajith, Anirudh and Xia, Mengzhou and Chevalier, Alexis and Goyal, Tanya and Chen, Danqi and Gao, Tianyu}},
   booktitle={{Empirical Methods in Natural Language Processing (EMNLP)}},
   year={{2024}}
}}
```

The S2ORC authors ask for this citation when S2ORC is used: {S2ORC_CITATION}

## Rebuild

`uv run python -m andamentum.deep_research.v2.bench download litsearch-local --dir <this folder>`
fetches what is missing and rebuilds the index and this file (`--force` fetches again).
Needs the `bench-data` extra (pyarrow).
"""
    write_atomic(root / PROVENANCE_FILE, text.encode("utf-8"))


# -- loading --------------------------------------------------------------------


def records_by_id(
    root: Path, corpus_ids: set[int] | set[str]
) -> dict[str, CorpusRecord]:
    """The corpus records with these ids (as strings), read from the index."""
    import sqlite3

    index = root / INDEX_FILE
    if not index.is_file():
        raise DatasetMissing(
            f"LitSearch (local) is not in {root}; run: uv run python -m "
            f"andamentum.deep_research.v2.bench download litsearch-local --dir {root}"
        )
    wanted = sorted(str(i) for i in corpus_ids)
    found: dict[str, CorpusRecord] = {}
    conn = sqlite3.connect(f"{index.as_uri()}?mode=ro", uri=True)
    try:
        for start in range(0, len(wanted), 500):
            chunk = wanted[start : start + 500]
            marks = ",".join("?" * len(chunk))
            for corpus_id, title, abstract in conn.execute(
                f"SELECT corpus_id, title, abstract FROM docs WHERE corpus_id IN ({marks})",
                chunk,
            ):
                found[corpus_id] = CorpusRecord(
                    corpus_id=corpus_id, title=title, abstract=abstract
                )
    finally:
        conn.close()
    return found


def read_queries(root: Path) -> list[LitSearchQuery]:
    path = root / QUERIES_FILE
    if not path.exists():
        raise DatasetMissing(
            f"LitSearch (local) is not in {root}; run: uv run python -m "
            f"andamentum.deep_research.v2.bench download litsearch-local --dir {root}"
        )
    return [
        LitSearchQuery.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]


def load_litsearch_local(
    root: Path, *, n: int, seed: int = DEFAULT_SEED, skip: int = 0
) -> LitSearchSet:
    """The deterministic subset of queries with their gold corpus records.

    The items are the seeded order's positions ``skip + 1`` to ``skip + n``
    (``select_subset(ids, skip + n)[skip:]``, as ``load_litsearch``). A gold id that
    is not in the corpus is counted as unresolved; a query left with no gold paper
    drops out and is counted. Offline.
    """
    queries = read_queries(root)
    by_id = {q.query_id: q for q in queries}
    chosen = select_subset(list(by_id), skip + n, seed=seed)[skip:]
    wanted = {c for query_id in chosen for c in by_id[query_id].corpus_ids}
    records = records_by_id(root, wanted)
    items: list[LitSearchItem] = []
    unresolved = 0
    dropped = 0
    for query_id in chosen:
        query = by_id[query_id]
        gold = [work_of(records[str(c)]) for c in query.corpus_ids if str(c) in records]
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


def excluded_positions(*, upto: int, seed: int = DEFAULT_SEED) -> list[str]:
    """The ids of the first *upto* positions of the seeded order of all 597
    queries (ids ``LS0000`` to ``LS0596``): what an experiment that starts at
    ``skip: upto`` leaves out. Needs no data."""
    return select_subset([f"LS{i:04d}" for i in range(597)], upto, seed=seed)


__all__ = [
    "CORPUS_PARQUETS",
    "DEFAULT_SEED",
    "DEFAULT_SUBSET",
    "HF_REVISION",
    "LitSearchItem",
    "LitSearchQuery",
    "LitSearchSet",
    "LITSEARCH_LOCAL_CARD",
    "LitSearchLocalDownload",
    "QUERY_PARQUET",
    "RangeFile",
    "download_litsearch_local",
    "excluded_positions",
    "load_litsearch_local",
    "read_queries",
    "records_by_id",
]
