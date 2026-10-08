"""LitSearch against its own corpus: the download, the loader, the excluded queries.

The Hugging Face Hub is a mock transport serving tiny invented parquet files in
the real layout (a query file and six ``corpus_clean`` files whose rows also hold
a large ``full_paper`` column), with the tree API and range requests. Nothing is
fetched from the network and no real record is used.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from ..bench import litsearch_local as ll  # noqa: E402
from ..bench.download import DatasetMissing, DownloadError  # noqa: E402
from ..bench.litsearch_local import (  # noqa: E402
    CORPUS_PARQUETS,
    HF_REVISION,
    QUERY_PARQUET,
    RangeFile,
    download_litsearch_local,
    excluded_positions,
    load_litsearch_local,
    read_queries,
)
from ..bench.metrics import work_matches  # noqa: E402
from ..bench.subsets import select_subset  # noqa: E402
from ..sources.local_corpus import CORPUS_FILE, INDEX_FILE, iter_records  # noqa: E402
from ..sources import LocalCorpusSource  # noqa: E402

N_QUERIES = 12
ROWS_PER_FILE = 4


def _full_paper(corpus_id: int) -> str:
    """200 kB of incompressible text per row, which must never be fetched."""
    return random.Random(corpus_id).randbytes(100_000).hex()


def _corpus_rows(file_number: int) -> list[dict[str, Any]]:
    rows = []
    for n in range(ROWS_PER_FILE):
        corpus_id = 100 + file_number * ROWS_PER_FILE + n
        rows.append(
            {
                "corpusid": corpus_id,
                # one record without a title and one with neither title nor abstract
                "title": None
                if corpus_id == 101
                else f"Invented paper number {corpus_id}",
                "abstract": None
                if corpus_id in (101, 102)
                else f"Abstract of invented paper {corpus_id} about topic {corpus_id % 5}.",
                "citations": [corpus_id - 1, 999_999] if corpus_id > 100 else [],
                "full_paper": _full_paper(corpus_id),
            }
        )
    return rows


def _parquet(rows: list[dict[str, Any]], *, row_group_size: int = 2) -> bytes:
    sink = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), sink, row_group_size=row_group_size)
    return sink.getvalue()


def _query_rows() -> list[dict[str, Any]]:
    return [
        {
            "query_set": "inline_acl",
            "query": f"Which invented paper studies topic {i % 5}?",
            "specificity": i % 2,
            "quality": 2,
            "corpusids": [100 + (i % 24)]
            + ([100 + ((i + 7) % 24)] if i % 3 == 0 else []),
        }
        for i in range(N_QUERIES)
    ]


class FakeHub:
    """The Hub: tree API, whole-file and range requests."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {QUERY_PARQUET: _parquet(_query_rows())}
        for number, path in enumerate(CORPUS_PARQUETS):
            self.files[path] = _parquet(_corpus_rows(number))
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        tree = f"/api/datasets/{ll.HF_DATASET}/tree/{HF_REVISION}/"
        if request.url.path.startswith(tree):
            folder = request.url.path.removeprefix(tree)
            return httpx.Response(
                200,
                json=[
                    {
                        "type": "file",
                        "path": path,
                        "size": len(data),
                        "lfs": {"oid": hashlib.sha256(data).hexdigest()},
                    }
                    for path, data in sorted(self.files.items())
                    if path.startswith(folder + "/")
                ],
            )
        prefix = f"/datasets/{ll.HF_DATASET}/resolve/{HF_REVISION}/"
        if request.url.path.startswith(prefix):
            data = self.files[request.url.path.removeprefix(prefix)]
            span = request.headers.get("range")
            if span is None:
                return httpx.Response(200, content=data)
            found = re.fullmatch(r"bytes=(\d+)-(\d+)", span)
            assert found, url
            start, end = int(found.group(1)), int(found.group(2))
            return httpx.Response(206, content=data[start : end + 1])
        return httpx.Response(404, text=f"no route for {url}")

    def client(self) -> httpx.Client:
        return httpx.Client(
            transport=httpx.MockTransport(self.handler), follow_redirects=True
        )


@pytest.fixture
def hub() -> FakeHub:
    return FakeHub()


@pytest.fixture
def downloaded(tmp_path: Path, hub: FakeHub) -> tuple[Path, ll.LitSearchLocalDownload]:
    root = tmp_path / "litsearch"
    with hub.client() as client:
        result = download_litsearch_local(
            root,
            client=client,
            today=__import__("datetime").date(2026, 10, 7),
            block_bytes=4096,
        )
    return root, result


# -- the range file -----------------------------------------------------------------


def test_a_range_file_reads_like_a_file_by_range_requests() -> None:
    data = bytes(range(256)) * 100
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.headers["range"])
        start, end = map(
            int, request.headers["range"].removeprefix("bytes=").split("-")
        )
        return httpx.Response(206, content=data[start : end + 1])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        file = RangeFile(
            client,
            "https://hub.test/f",
            size=len(data),
            block_bytes=1000,
            keep_blocks=100,
        )
        assert file.read(10) == data[:10]
        assert file.tell() == 10
        file.seek(-5, 2)
        assert file.read() == data[-5:]
        file.seek(2500)
        assert file.read(1000) == data[2500:3500]
        assert file.read(10**9) == data[3500:]
        assert file.read(5) == b""
        file.seek(5)
        assert file.read(3) == data[5:8]  # served from a block already held
    assert requests.count("bytes=0-999") == 1
    assert file.bytes_fetched == sum(
        int(r.split("-")[1]) - int(r.split("=")[1].split("-")[0]) + 1 for r in requests
    )


def test_a_range_file_refuses_a_server_that_ignores_ranges() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 50)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        file = RangeFile(client, "https://hub.test/f", size=50, block_bytes=10)
        with pytest.raises(DownloadError, match="expected 206"):
            file.read(5)


# -- the download -------------------------------------------------------------------


def test_the_download_reads_only_the_wanted_columns(
    downloaded: tuple[Path, ll.LitSearchLocalDownload], hub: FakeHub
) -> None:
    root, result = downloaded
    total = sum(len(hub.files[p]) for p in CORPUS_PARQUETS)
    assert total > 1_000_000  # the files hold the large column
    assert 0 < result.bytes_fetched < total / 8
    records = list(iter_records(root))
    assert [r.corpus_id for r in records] == [
        str(100 + n) for n in range(6 * ROWS_PER_FILE)
    ]
    by_id = {r.corpus_id: r for r in records}
    assert by_id["100"].title == "Invented paper number 100"
    assert by_id["101"].title == "" and by_id["101"].abstract == ""  # not invented
    assert by_id["102"].abstract == "" and by_id["102"].title
    assert by_id["103"].citations == ["102", "999999"]
    assert (root / CORPUS_FILE).stat().st_size < 10_000  # no full text in the corpus


def test_the_download_writes_the_queries_the_index_and_the_provenance(
    downloaded: tuple[Path, ll.LitSearchLocalDownload], hub: FakeHub
) -> None:
    root, result = downloaded
    assert result.n_queries == N_QUERIES
    assert result.index.n_records == 6 * ROWS_PER_FILE
    assert result.index.n_with_abstract == 6 * ROWS_PER_FILE - 2
    assert result.n_gold_in_corpus == result.n_gold_ids  # every gold id is in 100-123
    # the query file is the Hub's bytes
    assert (root / QUERY_PARQUET).read_bytes() == hub.files[QUERY_PARQUET]
    queries = read_queries(root)
    assert [q.query_id for q in queries] == [f"LS{i:04d}" for i in range(N_QUERIES)]
    assert queries[0].query.startswith("Which invented paper studies topic 0")
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    assert {f["path"] for f in manifest["files"]} == {QUERY_PARQUET, "queries.jsonl"}

    text = (root / "PROVENANCE.md").read_text(encoding="utf-8")
    assert HF_REVISION in text and "2026-10-07" in text
    assert "local evaluation only, do not redistribute" in text.lower()
    assert hashlib.sha256((root / CORPUS_FILE).read_bytes()).hexdigest() in text
    assert hashlib.sha256((root / INDEX_FILE).read_bytes()).hexdigest() in text
    for path in (QUERY_PARQUET, *CORPUS_PARQUETS):
        assert f"`{path}`" in text
        assert hashlib.sha256(hub.files[path]).hexdigest() in text
    for quote in ll.S2ORC_LICENCE_QUOTES:
        assert quote in text
    assert "ODC-By 1.0" in text and "MIT" in text and "no licence" in text
    assert "Ajith, Xia, Chevalier, Goyal, Chen, Gao" in text and "EMNLP" in text
    assert (
        "1 of them no abstract" in text.replace("\n", " ")
        or "no abstract either" in text
    )
    assert "corpus_s2orc" in text  # said to be not downloaded


def test_the_download_leaves_a_working_corpus_for_the_source(
    downloaded: tuple[Path, ll.LitSearchLocalDownload],
) -> None:
    root, _ = downloaded
    source = LocalCorpusSource(corpus_dir=root)
    assert source.summary.n_records == 6 * ROWS_PER_FILE


def test_a_second_download_reuses_the_files_and_force_fetches_again(
    tmp_path: Path, hub: FakeHub
) -> None:
    root = tmp_path / "litsearch"
    from datetime import date

    with hub.client() as client:
        download_litsearch_local(
            root, client=client, today=date(2026, 10, 7), block_bytes=4096
        )
        before = len(hub.requests)
        again = download_litsearch_local(
            root, client=client, today=date(2026, 10, 8), block_bytes=4096
        )
        assert again.bytes_fetched == 0
        # the rebuilt provenance keeps the date and bytes of the fetch
        kept = (root / "PROVENANCE.md").read_text(encoding="utf-8")
        assert "Download date: 2026-10-07" in kept and "2026-10-08" not in kept
        assert "bytes read was not recorded" not in kept
        # only the two tree listings were asked
        assert len(hub.requests) - before == 2
        forced = download_litsearch_local(
            root, client=client, today=date(2026, 10, 9), force=True, block_bytes=4096
        )
        assert forced.bytes_fetched > 0
    text = (root / "PROVENANCE.md").read_text(encoding="utf-8")
    assert "Download date: 2026-10-09" in text


def test_a_revision_with_other_files_is_refused(tmp_path: Path, hub: FakeHub) -> None:
    del hub.files[CORPUS_PARQUETS[-1]]
    with hub.client() as client, pytest.raises(DownloadError, match="expected files"):
        download_litsearch_local(
            tmp_path / "x",
            client=client,
            today=__import__("datetime").date(2026, 10, 7),
        )


def test_a_query_file_that_differs_from_the_listed_hash_is_refused(
    tmp_path: Path, hub: FakeHub
) -> None:
    real = hub.handler

    def tampered(request: httpx.Request) -> httpx.Response:
        response = real(request)
        if request.url.path.endswith(QUERY_PARQUET) and "range" not in request.headers:
            return httpx.Response(200, content=response.content + b"!")
        return response

    client = httpx.Client(transport=httpx.MockTransport(tampered))
    with client, pytest.raises(DownloadError, match="sha256 differs"):
        download_litsearch_local(
            tmp_path / "x",
            client=client,
            today=__import__("datetime").date(2026, 10, 7),
        )


# -- loading ------------------------------------------------------------------------


def test_the_loader_gives_the_seeded_subset_with_gold_corpus_records(
    downloaded: tuple[Path, ll.LitSearchLocalDownload],
) -> None:
    root, _ = downloaded
    data = load_litsearch_local(root, n=5, seed=0, skip=3)
    ids = [f"LS{i:04d}" for i in range(N_QUERIES)]
    expected = select_subset(ids, 8, seed=0)[3:]
    assert [i.query.query_id for i in data.items] == expected
    assert (
        data.n_subset == 5
        and data.n_dropped_queries == 0
        and data.n_unresolved_gold == 0
    )
    assert data.n_queries_total == N_QUERIES
    for item in data.items:
        assert [w.local_id for w in item.gold] == [
            f"local:{c}" for c in item.query.corpus_ids
        ]
    first = data.items[0]
    assert (
        first.gold[0].title.startswith("Invented paper number")
        or first.gold[0].title == ""
    )
    # a run's result holding the record matches it by the corpus id, whatever its title
    retrieved = first.gold[0].model_copy(update={"title": "Another title"})
    assert work_matches(retrieved, first.gold[0])


def test_a_gold_id_outside_the_corpus_is_counted_not_invented(tmp_path: Path) -> None:
    from ..bench.litsearch_local import LitSearchQuery
    from ..sources.local_corpus import CorpusRecord, build_index

    root = tmp_path / "c"
    root.mkdir()
    (root / CORPUS_FILE).write_text(
        CorpusRecord(
            corpus_id="1001", title="Invented cache paper", abstract="About a cache."
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    build_index(root)
    queries = [
        LitSearchQuery(
            query_id="LS0000",
            query="q0",
            query_set="s",
            specificity=0,
            quality=2,
            corpus_ids=[1001, 424242],
        ),
        LitSearchQuery(
            query_id="LS0001",
            query="q1",
            query_set="s",
            specificity=0,
            quality=2,
            corpus_ids=[777777],
        ),
    ]
    (root / "queries.jsonl").write_text(
        "".join(q.model_dump_json() + "\n" for q in queries), encoding="utf-8"
    )
    data = load_litsearch_local(root, n=2, seed=0)
    assert [i.query.query_id for i in data.items] == ["LS0000"]
    assert [w.local_id for w in data.items[0].gold] == ["local:1001"]
    assert data.n_unresolved_gold == 2 and data.n_dropped_queries == 1


def test_a_missing_dataset_names_the_command(tmp_path: Path) -> None:
    with pytest.raises(DatasetMissing, match="download litsearch-local --dir"):
        load_litsearch_local(tmp_path, n=3)


def test_the_excluded_positions_are_a_prefix_of_the_seeded_order() -> None:
    ids = [f"LS{i:04d}" for i in range(597)]
    assert excluded_positions(upto=60) == select_subset(ids, 60, seed=0)
    assert excluded_positions(upto=30) == excluded_positions(upto=60)[:30]
    # the items of a set that skips 60 are the next ones and never an excluded one
    chosen = select_subset(ids, 140, seed=0)[60:]
    assert len(chosen) == 80 and not set(chosen) & set(excluded_positions(upto=60))
