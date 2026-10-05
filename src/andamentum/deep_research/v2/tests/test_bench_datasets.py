"""Downloaders (over a mock transport) and loaders, on invented data."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from andamentum.deep_research.v2.bench.download import (
    LITSEARCH_CARD,
    SYNERGY_CARD,
    DatasetMissing,
    Downloader,
    DownloadError,
    read_manifest,
    save_file,
)
from andamentum.deep_research.v2.bench.litsearch import (
    GOLD_FILE,
    HF_ROWS_URL,
    S2_BATCH_URL,
    download_litsearch,
    load_litsearch,
)
from andamentum.deep_research.v2.bench.subsets import select_subset
from andamentum.deep_research.v2.bench.synergy import (
    RAW_URL,
    abstract_text,
    download_synergy,
    load_review,
    sample_records,
)

from .bench_fixtures import (
    N_EXCLUDED,
    N_INCLUDED,
    REVIEW_KEY,
    REVIEW_TITLE,
    inverted,
    litsearch_rows,
    make_synergy_dir,
    s2_paper,
)


async def no_sleep(_seconds: float) -> None:
    return None


def downloader(
    handler: httpx.MockTransport | None = None, *, retries: int = 3
) -> tuple[Downloader, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=handler)
    return Downloader(client, retries=retries, sleep=no_sleep), client


# -- subsets -----------------------------------------------------------------------


def test_a_subset_does_not_depend_on_input_order_and_extends_a_smaller_one() -> None:
    ids = [f"Q{i}" for i in range(50)]
    forward = select_subset(ids, 10, seed=0)
    assert forward == select_subset(list(reversed(ids)), 10, seed=0)
    assert select_subset(ids, 5, seed=0) == forward[:5]
    assert select_subset(ids, 10, seed=1) != forward
    assert len(select_subset(ids, 500, seed=0)) == 50


def test_a_subset_refuses_duplicates_and_a_negative_size() -> None:
    with pytest.raises(ValueError, match="unique"):
        select_subset(["a", "a"], 1, seed=0)
    with pytest.raises(ValueError, match="negative"):
        select_subset(["a"], -1, seed=0)


# -- downloader and manifest -----------------------------------------------------------


async def test_a_transient_status_is_retried_then_succeeds() -> None:
    statuses = iter([429, 503, 200])

    def handle(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        return httpx.Response(status, content=b"payload" if status == 200 else b"")

    dl, client = downloader(httpx.MockTransport(handle))
    assert await dl.get("https://example.org/f") == b"payload"
    await client.aclose()


async def test_retries_run_out_with_the_last_reason() -> None:
    dl, client = downloader(
        httpx.MockTransport(lambda r: httpx.Response(503)), retries=2
    )
    with pytest.raises(DownloadError, match="HTTP 503 after 3 attempts"):
        await dl.get("https://example.org/f")
    await client.aclose()


async def test_a_client_error_is_not_retried_and_404_can_be_optional() -> None:
    calls: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(404)

    dl, client = downloader(httpx.MockTransport(handle))
    assert await dl.get_optional("https://example.org/missing") is None
    with pytest.raises(DownloadError) as caught:
        await dl.get("https://example.org/missing")
    assert caught.value.status == 404
    assert len(calls) == 2
    await client.aclose()


def test_save_file_records_checksum_licence_and_replaces_by_path(
    tmp_path: Path,
) -> None:
    save_file(tmp_path, "a/x.txt", b"one", url="https://e.org/x", card=SYNERGY_CARD)
    save_file(tmp_path, "a/x.txt", b"two!", url="https://e.org/x", card=SYNERGY_CARD)
    save_file(tmp_path, "b/y.txt", b"y", url="https://e.org/y", card=LITSEARCH_CARD)
    manifest = read_manifest(tmp_path)
    assert (tmp_path / "a" / "x.txt").read_bytes() == b"two!"
    assert [f.path for f in manifest.files] == ["a/x.txt", "b/y.txt"]
    assert manifest.files[0].bytes == 4
    assert {c.key: c.licence for c in manifest.cards} == {
        "synergy": "CC0-1.0",
        "litsearch": "MIT",
    }
    assert not list(tmp_path.rglob("*.part"))


# -- litsearch ---------------------------------------------------------------------------


def litsearch_handler(seen: dict[str, list[object]]) -> httpx.MockTransport:
    rows = litsearch_rows()

    def handle(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith(HF_ROWS_URL):
            offset = int(request.url.params["offset"])
            seen.setdefault("offsets", []).append(offset)
            page = rows[offset : offset + 2]  # the server pages by two
            return httpx.Response(200, json={"rows": page, "num_rows_total": len(rows)})
        assert url.startswith(S2_BATCH_URL)
        ids = json.loads(request.content)["ids"]
        seen.setdefault("batches", []).append(ids)
        seen.setdefault("keys", []).append(request.headers.get("x-api-key"))
        return httpx.Response(
            200, json=[s2_paper(int(i.removeprefix("CorpusId:"))) for i in ids]
        )

    return httpx.MockTransport(handle)


async def test_litsearch_download_pages_the_queries_and_resolves_the_subset_gold(
    tmp_path: Path,
) -> None:
    seen: dict[str, list[object]] = {}
    dl, client = downloader(litsearch_handler(seen))
    done = await download_litsearch(
        tmp_path, downloader=dl, n=5, seed=0, s2_api_key="secret"
    )
    await client.aclose()
    assert seen["offsets"] == [0, 2, 4]
    assert done.n_queries_total == 5 and done.n_subset == 5
    assert done.n_gold_fetched == 6  # 9000-9004 and 9100
    assert seen["keys"] == ["secret"]
    manifest = read_manifest(tmp_path)
    assert {f.path for f in manifest.files} == {"litsearch/queries.jsonl", GOLD_FILE}
    assert manifest.cards[0].licence == "MIT"


async def test_litsearch_load_keeps_resolved_gold_and_drops_queries_without_any(
    tmp_path: Path,
) -> None:
    dl, client = downloader(litsearch_handler({}))
    await download_litsearch(tmp_path, downloader=dl, n=5, seed=0)
    await client.aclose()
    data = load_litsearch(tmp_path, n=5, seed=0)
    assert data.n_queries_total == 5
    assert data.n_unresolved_gold == 1  # 9003 is unknown to Semantic Scholar
    assert data.n_dropped_queries == 1  # the query whose only gold paper is 9003
    assert len(data.items) == 4
    first = next(i for i in data.items if i.query.query_id == "LS0000")
    assert len(first.gold) == 2
    work = first.gold[0]
    assert work.doi == "10.5555/gold.9000"
    assert work.title == "An invented paper about topic 9000"
    assert next(i for i in data.items if i.query.query_id == "LS0001").gold[
        0
    ].arxiv_id == ("2101.09001")


async def test_a_second_download_fetches_only_the_gold_it_lacks(tmp_path: Path) -> None:
    seen: dict[str, list[object]] = {}
    dl, client = downloader(litsearch_handler(seen))
    await download_litsearch(tmp_path, downloader=dl, n=2, seed=0)
    first_batches = len(seen["batches"])
    await download_litsearch(tmp_path, downloader=dl, n=2, seed=0)
    assert len(seen["batches"]) == first_batches  # nothing missing, nothing asked
    await download_litsearch(tmp_path, downloader=dl, n=5, seed=0)
    await client.aclose()
    assert len(seen["batches"]) == first_batches + 1


async def test_a_short_batch_answer_fails_loudly(tmp_path: Path) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith(HF_ROWS_URL):
            return httpx.Response(
                200, json={"rows": litsearch_rows()[:1], "num_rows_total": 1}
            )
        return httpx.Response(200, json=[])

    dl, client = downloader(httpx.MockTransport(handle))
    with pytest.raises(DownloadError, match="expected 2 answers"):
        await download_litsearch(tmp_path, downloader=dl, n=1, seed=0)
    await client.aclose()


def test_litsearch_load_names_the_command_when_nothing_is_downloaded(
    tmp_path: Path,
) -> None:
    with pytest.raises(DatasetMissing, match="download litsearch --dir"):
        load_litsearch(tmp_path)


async def test_litsearch_load_asks_for_a_larger_download_when_gold_is_missing(
    tmp_path: Path,
) -> None:
    dl, client = downloader(litsearch_handler({}))
    await download_litsearch(tmp_path, downloader=dl, n=1, seed=0)
    await client.aclose()
    with pytest.raises(DatasetMissing, match="--n 5"):
        load_litsearch(tmp_path, n=5, seed=0)


# -- synergy -------------------------------------------------------------------------------


def test_abstract_text_puts_the_words_back_in_order() -> None:
    assert abstract_text(inverted("salt was measured twice in salt kilns")) == (
        "salt was measured twice in salt kilns"
    )
    assert abstract_text(None) == ""


def test_load_review_joins_labels_to_works_across_zip_parts(tmp_path: Path) -> None:
    make_synergy_dir(tmp_path)
    review = load_review(tmp_path, REVIEW_KEY, seed=3)
    assert review.question == REVIEW_TITLE
    assert review.n_records_total == N_INCLUDED + N_EXCLUDED
    assert review.n_included_total == N_INCLUDED
    first = review.records[0]
    assert first.record_id == "W1000"
    assert first.included and first.title.startswith("SaltLog")
    assert first.venue == "Journal of Invented Results"
    assert first.doi == "10.5555/invented.0"
    assert first.abstract == ""  # record 0 has no abstract; the record is kept
    assert (
        review.records[1].abstract
        == "SaltLog was measured in kiln 1 of the Apollo plant"
    )
    work = first.to_work()
    assert work.work_id == "W1000" and work.openalex_id == "W1000"


def test_load_review_refuses_a_bad_label_and_a_label_without_a_work(
    tmp_path: Path,
) -> None:
    folder = make_synergy_dir(tmp_path)
    labels = (folder / "labels.csv").read_text()
    (folder / "labels.csv").write_text(labels.replace(",1\n", ",maybe\n", 1))
    with pytest.raises(ValueError, match="not 0 or 1"):
        load_review(tmp_path, REVIEW_KEY)
    (folder / "labels.csv").write_text(labels + "https://openalex.org/W999999,,,0\n")
    with pytest.raises(ValueError, match="no work for labelled record"):
        load_review(tmp_path, REVIEW_KEY)


def test_load_review_names_the_command_for_a_missing_review(tmp_path: Path) -> None:
    with pytest.raises(DatasetMissing, match=f"--reviews {REVIEW_KEY}"):
        load_review(tmp_path, REVIEW_KEY)
    with pytest.raises(ValueError, match="not a SYNERGY review key"):
        load_review(tmp_path, "../escape")


def test_the_sample_keeps_prevalence_with_a_floor_and_is_deterministic(
    tmp_path: Path,
) -> None:
    make_synergy_dir(tmp_path)
    records = load_review(tmp_path, REVIEW_KEY).records
    sample = sample_records(records, 10, seed=0, min_included=1)
    assert len(sample) == 10
    assert sum(r.included for r in sample) == 2  # 4/20 of 10
    assert sample == sample_records(records, 10, seed=0, min_included=1)
    assert [r.record_id for r in sample] == sorted(
        (r.record_id for r in sample), key=[x.record_id for x in records].index
    )
    floored = sample_records(records, 10, seed=0, min_included=3)
    assert sum(r.included for r in floored) == 3
    everything = sample_records(records, 10, seed=0, min_included=100)
    assert sum(r.included for r in everything) == N_INCLUDED  # never more than exist
    assert sample_records(records, None, seed=0) == records
    assert sample_records(records, 500, seed=0) == records


async def test_synergy_download_fetches_each_part_until_a_404(tmp_path: Path) -> None:
    requested: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path.rsplit("/", 1)[-1])
        if request.url.path.endswith("works_3.zip"):
            return httpx.Response(404)
        return httpx.Response(200, content=b"bytes of " + request.url.path.encode())

    dl, client = downloader(httpx.MockTransport(handle))
    written = await download_synergy(tmp_path, downloader=dl, reviews=(REVIEW_KEY,))
    await client.aclose()
    assert written == [
        f"synergy/{REVIEW_KEY}/labels.csv",
        f"synergy/{REVIEW_KEY}/metadata_publication.json",
        f"synergy/{REVIEW_KEY}/CITATION.txt",
        f"synergy/{REVIEW_KEY}/works_1.zip",
        f"synergy/{REVIEW_KEY}/works_2.zip",
    ]
    assert requested[-1] == "works_3.zip"
    assert read_manifest(tmp_path).cards[0].licence == "CC0-1.0"
    assert RAW_URL.format(key="A_1", name="labels.csv").endswith("v1.0/A_1/labels.csv")


async def test_synergy_download_of_an_unknown_review_fails_loudly(
    tmp_path: Path,
) -> None:
    dl, client = downloader(httpx.MockTransport(lambda r: httpx.Response(404)))
    with pytest.raises(DownloadError) as caught:
        await download_synergy(tmp_path, downloader=dl, reviews=("Nobody_1999",))
    assert caught.value.status == 404
    with pytest.raises(ValueError, match="not a SYNERGY review key"):
        await download_synergy(tmp_path, downloader=dl, reviews=("bad key",))
    await client.aclose()
