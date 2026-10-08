"""The literature layer: library readers, Zotero (read only), Crossref search, resolve."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from andamentum.compose.harness.literature import (
    LiteratureError,
    Sources,
    Unverified,
    VerifiedWork,
    Work,
    ZoteroLibrary,
    find_works,
    read_library,
    resolve,
    search_crossref,
)
from andamentum.references import to_csl
from andamentum.compose.harness.literature.search import rank
from andamentum.references import ReferenceHttp

SALT = {
    "id": "saltlog2021",
    "type": "article-journal",
    "title": "Daily salt logging lowers reported sodium intake",
    "container-title": "Journal of Invented Results",
    "author": [
        {"family": "Smith", "given": "Jane"},
        {"family": "Jones", "given": "Kim"},
    ],
    "issued": {"date-parts": [[2021]]},
    "DOI": "10.5555/saltlog.001",
}
DIARY = {
    "id": "diary2019",
    "type": "article-journal",
    "title": "Meal diaries in invented cohorts",
    "container-title": "Invented Nutrition Review",
    "author": [{"family": "Lee", "given": "Min"}],
    "issued": {"date-parts": [[2019]]},
    "DOI": "10.5555/saltlog.002",
}
RETRACTED = {
    "id": "gone2018",
    "title": "Sodium and sleep in invented adults",
    "container-title": "Journal of Invented Results",
    "author": [{"family": "Gone", "given": "Rae"}],
    "issued": {"date-parts": [[2018]]},
    "DOI": "10.5555/saltlog.003",
}
CORRECTED = {
    "id": "fixed2017",
    "title": "Potassium swaps in invented kitchens",
    "author": [{"family": "Fixed", "given": "Ola"}],
    "issued": {"date-parts": [[2017]]},
    "DOI": "10.5555/saltlog.004",
}

NOTICES = {
    RETRACTED["DOI"]: [
        {
            "type": "retraction",
            "DOI": "10.5555/saltlog.003r",
            "source": "retraction-watch",
        }
    ],
    CORRECTED["DOI"]: [
        {"type": "correction", "DOI": "10.5555/saltlog.004c", "source": "publisher"}
    ],
}


def _record(csl: dict) -> dict:
    record = {
        "DOI": csl["DOI"],
        "title": [csl["title"]],
        "author": csl["author"],
        "issued": csl["issued"],
        "type": "journal-article",
    }
    if csl.get("container-title"):
        record["container-title"] = [csl["container-title"]]
    if csl["DOI"] in NOTICES:
        record["updated-by"] = NOTICES[csl["DOI"]]
    return record


class Crossref:
    """A fake Crossref: DOI fetches and searches over four invented works."""

    def __init__(self, *, fail_with: int | None = None) -> None:
        self.works = {w["DOI"]: w for w in (SALT, DIARY, RETRACTED, CORRECTED)}
        self.requests: list[str] = []
        self.fail_with = fail_with

    def transport(self) -> httpx.MockTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            self.requests.append(str(request.url))
            if self.fail_with:
                return httpx.Response(self.fail_with)
            assert request.url.host == "api.crossref.org"
            path = request.url.path
            if path.startswith("/works/"):
                doi = path[len("/works/") :]
                found = self.works.get(doi)
                if found is None:
                    return httpx.Response(404)
                return httpx.Response(200, json={"message": _record(found)})
            query = " ".join(
                request.url.params.get(k, "") for k in request.url.params
            ).lower()
            rows = int(request.url.params.get("rows", "10"))
            hits = [
                _record(w)
                for w in self.works.values()
                if any(
                    word in query
                    for word in w["title"].lower().split()
                    if len(word) > 5
                )
            ]
            return httpx.Response(200, json={"message": {"items": hits[:rows]}})

        return httpx.MockTransport(handle)

    def http(self) -> ReferenceHttp:
        return ReferenceHttp(client=httpx.AsyncClient(transport=self.transport()))


# ------------------------------------------------------------------ library files


def test_csl_json_library_list_and_items_object(tmp_path: Path):
    as_list = tmp_path / "refs.json"
    as_list.write_text(json.dumps([SALT, {"title": "No id here"}]))
    works = read_library(as_list)
    assert [w.source for w in works] == ["library", "library"]
    assert works[0].doi == "10.5555/saltlog.001"
    assert works[0].authors == ["Smith", "Jones"] and works[0].year == 2021
    assert works[1].csl["id"] == "item2"  # an item without an id gets one
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"items": [DIARY]}))
    assert read_library(wrapped)[0].title == DIARY["title"]


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("bad.json", "{not json", "not valid JSON"),
        ("odd.json", json.dumps({"a": 1}), "not a CSL JSON list"),
        ("empty.json", json.dumps([{"id": "x"}]), "neither a title nor a DOI"),
        ("refs.txt", "x", "unknown library format"),
        ("open.bib", "@article{k, title = {Never closed", "never closed"),
    ],
)
def test_a_library_that_cannot_be_read_says_why(
    tmp_path: Path, name: str, content: str, message: str
):
    path = tmp_path / name
    path.write_text(content)
    with pytest.raises(LiteratureError, match=message):
        read_library(path)


def test_a_missing_library_file_is_a_literature_error(tmp_path: Path):
    with pytest.raises(LiteratureError, match="cannot read the library"):
        read_library(tmp_path / "absent.json")


BIB = r"""
@comment{ignored}
@string{jir = "Journal of Invented Results"}
@article{smith2021,
  author  = {Smith, Jane and Jones, Kim and {World Salt Consortium}},
  title   = {Daily {S}alt Logging in {M}\"uller's Clinics},
  journal = {Journal of Invented Results},
  year    = 2021,
  volume  = {12},
  number  = "3",
  pages   = {45--67},
  doi     = {https://doi.org/10.5555/saltlog.001},
  keywords = {salt, logging},
}
@inproceedings{lee2019,
  author = {Min Lee and Ola {van der Berg}},
  title = {Diaries},
  booktitle = {Proceedings of Invented Nutrition},
  year = {2019},
}
@book{book1, title = {A Book}, author = {Roe, R.}, publisher = {Invented Press}, year = {2010}}
"""


def test_bibtex_library_reads_entries_names_accents_and_pages(tmp_path: Path):
    path = tmp_path / "refs.bib"
    path.write_text(BIB, encoding="utf-8")
    first, second, third = read_library(path)
    assert first.csl["id"] == "smith2021" and first.csl["type"] == "article-journal"
    assert first.title == "Daily Salt Logging in Müller's Clinics"
    assert first.csl["author"] == [
        {"family": "Smith", "given": "Jane"},
        {"family": "Jones", "given": "Kim"},
        {"literal": "World Salt Consortium"},
    ]
    assert first.csl["page"] == "45–67" and first.csl["issue"] == "3"
    assert first.doi == "10.5555/saltlog.001" and first.year == 2021
    assert first.csl["container-title"] == "Journal of Invented Results"
    assert second.csl["type"] == "paper-conference"
    assert second.csl["author"] == [
        {"family": "Lee", "given": "Min"},
        {"family": "van der Berg", "given": "Ola"},
    ]
    assert second.csl["container-title"] == "Proceedings of Invented Nutrition"
    assert third.csl["type"] == "book" and third.csl["publisher"] == "Invented Press"


# ----------------------------------------------------------------------- ranking


def test_rank_orders_by_shared_content_words_and_needs_two_of_them():
    works = [Work(csl=c, source="library") for c in (DIARY, SALT, RETRACTED)]
    claim = "Daily logging of salt reduced the sodium intake that people reported"
    ranked = rank(works, claim, limit=5)
    assert ranked[0].csl["id"] == "saltlog2021"
    assert rank(works, "quantum chromodynamics lattice", limit=5) == []
    assert rank(works, "the and of", limit=5) == []
    assert [w.csl["id"] for w in rank(works, "meal diaries", limit=1)] == ["diary2019"]


# ------------------------------------------------------------------------ Zotero


def _zotero_entry(
    csl: dict,
    key: str,
    *,
    item_type: str = "journalArticle",
    csl_id: str | None = None,
    library: dict | None = None,
) -> dict:
    entry: dict = {
        "key": key,
        "data": {"itemType": item_type, "title": csl["title"]},
        "csljson": {**csl, "id": csl_id or key},
    }
    if library is not None:
        entry["library"] = library
    return entry


class FakeZotero:
    def __init__(self, entries: list[dict], *, status: int = 200) -> None:
        self.entries = entries
        self.status = status
        self.requests: list[httpx.Request] = []

    def client(self) -> httpx.AsyncClient:
        def handle(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if self.status != 200:
                return httpx.Response(self.status, text="denied")
            start = int(request.url.params.get("start", "0"))
            limit = int(request.url.params.get("limit", "100"))
            return httpx.Response(200, json=self.entries[start : start + limit])

        return httpx.AsyncClient(transport=httpx.MockTransport(handle))


async def test_zotero_reads_items_with_their_uri_and_skips_attachments_and_notes():
    fake = FakeZotero(
        [
            _zotero_entry(SALT, "KEY1", csl_id="http://zotero.org/users/42/items/KEY1"),
            _zotero_entry(DIARY, "KEY2"),
            _zotero_entry(
                {**DIARY, "title": "Invented group study of meals"},
                "KEY5",
                library={"type": "group", "id": 77},
            ),
            _zotero_entry(
                {**SALT, "title": "Invented synced study of salt"},
                "KEY6",
                library={"type": "user", "id": 42},
            ),
            _zotero_entry(RETRACTED, "KEY3", item_type="attachment"),
            _zotero_entry(CORRECTED, "KEY4", item_type="note"),
        ]
    )
    library = ZoteroLibrary(client=fake.client())
    works = await library.works()
    assert [w.csl["title"] for w in works][:2] == [SALT["title"], DIARY["title"]]
    assert len(works) == 4
    assert works[0].uri == "http://zotero.org/users/42/items/KEY1"
    # No library and no URI in the answer: the work carries none, never an invented one.
    assert works[1].uri is None
    assert works[2].uri == "http://zotero.org/groups/77/items/KEY5"
    assert works[3].uri == "http://zotero.org/users/42/items/KEY6"
    assert all(w.source == "zotero" for w in works)
    found = await library.search("daily salt logging sodium", limit=3)
    assert [w.csl["title"] for w in found] == [SALT["title"]]
    # Read once and kept; every request was a GET to the local API.
    assert len(fake.requests) == 1
    assert {r.method for r in fake.requests} == {"GET"}
    assert fake.requests[0].url.host == "127.0.0.1"
    assert "write" not in " ".join(dir(ZoteroLibrary)).lower()


async def test_zotero_pages_through_a_large_library():
    entries = [
        _zotero_entry(
            {**SALT, "title": f"Invented study number {i} of salt"}, f"K{i:04d}"
        )
        for i in range(230)
    ]
    fake = FakeZotero(entries)
    works = await ZoteroLibrary(client=fake.client()).works()
    assert len(works) == 230 and len(fake.requests) == 3


async def test_zotero_that_is_unreachable_or_refusing_raises():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    down = httpx.AsyncClient(transport=httpx.MockTransport(refuse))
    with pytest.raises(LiteratureError, match="not reachable"):
        await ZoteroLibrary(client=down).works()
    with pytest.raises(LiteratureError, match="refused the request"):
        await ZoteroLibrary(client=FakeZotero([], status=403).client()).works()
    with pytest.raises(LiteratureError, match="answered 500"):
        await ZoteroLibrary(client=FakeZotero([], status=500).client()).works()


async def test_a_zotero_item_without_csl_json_is_an_error_not_a_skip():
    fake = FakeZotero([{"key": "K", "data": {"itemType": "book"}}])
    with pytest.raises(LiteratureError, match="without CSL JSON"):
        await ZoteroLibrary(client=fake.client()).works()


async def test_a_library_over_the_cap_raises_rather_than_truncating():
    entries = [
        _zotero_entry({**SALT, "title": f"Study {i} of invented salt"}, f"K{i}")
        for i in range(250)
    ]
    fake = FakeZotero(entries)
    with pytest.raises(LiteratureError, match="more than 200 items"):
        await ZoteroLibrary(client=fake.client(), max_items=200).works()


# ---------------------------------------------------------------------- Crossref


def test_crossref_record_becomes_csl_json():
    csl = to_csl(
        {
            "DOI": "10.5555/x",
            "type": "proceedings-article",
            "title": ["A  title\nwith space"],
            "container-title": ["Proc. Invented"],
            "author": [
                {"family": "Roe", "given": "R."},
                {"name": "The Invented Group"},
            ],
            "issued": {"date-parts": [[2020, 5]]},
            "volume": "3",
            "page": "1-9",
            "ignored": "x",
        }
    )
    assert csl == {
        "id": "10.5555/x",
        "type": "paper-conference",
        "DOI": "10.5555/x",
        "title": "A title with space",
        "container-title": "Proc. Invented",
        "author": [{"family": "Roe", "given": "R."}, {"literal": "The Invented Group"}],
        "issued": {"date-parts": [[2020, 5]]},
        "volume": "3",
        "page": "1-9",
    }


async def test_crossref_search_returns_works_and_sends_only_the_query():
    fake = Crossref()
    works = await search_crossref("daily salt logging", http=fake.http(), limit=5)
    assert [w.doi for w in works] == ["10.5555/saltlog.001"]
    assert works[0].source == "crossref" and works[0].uri is None
    (url,) = fake.requests
    assert "query.bibliographic=daily+salt+logging" in url and "rows=5" in url


async def test_crossref_search_failure_raises_not_empty():
    with pytest.raises(LiteratureError, match="Crossref search failed"):
        await search_crossref("salt", http=Crossref(fail_with=400).http(), limit=5)


# ------------------------------------------------------------------- find_works


async def test_find_works_orders_sources_dedupes_and_asks_crossref_last(tmp_path: Path):
    library = tmp_path / "refs.json"
    library.write_text(json.dumps([SALT]))
    fake = Crossref()
    found = await find_works(
        "daily salt logging meal diaries cohorts",
        sources=Sources(library=library, crossref=fake.http()),
        limit=5,
    )
    # The library's SALT comes first and is not repeated from Crossref.
    assert [w.doi for w in found] == ["10.5555/saltlog.001", "10.5555/saltlog.002"]
    assert [w.source for w in found] == ["library", "crossref"]


async def test_find_works_skips_crossref_when_the_library_fills_the_limit(
    tmp_path: Path,
):
    library = tmp_path / "refs.json"
    library.write_text(json.dumps([SALT, DIARY]))
    fake = Crossref()
    found = await find_works(
        "daily salt logging sodium meal diaries",
        sources=Sources(library=library, crossref=fake.http()),
        limit=2,
    )
    assert len(found) == 2 and fake.requests == []


async def test_find_works_puts_zotero_between_the_library_and_crossref(tmp_path: Path):
    zotero = ZoteroLibrary(client=FakeZotero([_zotero_entry(DIARY, "K1")]).client())
    found = await find_works(
        "meal diaries cohorts", sources=Sources(zotero=zotero), limit=3
    )
    assert [w.source for w in found] == ["zotero"]


async def test_find_works_with_no_source_finds_nothing_and_rejects_a_bad_limit():
    assert await find_works("anything at all", sources=Sources(), limit=3) == []
    with pytest.raises(ValueError):
        await find_works("x", sources=Sources(), limit=0)


# ---------------------------------------------------------------------- resolve


def _work(csl: dict, source: str = "library") -> Work:
    return Work(csl=csl, source=source)  # type: ignore[arg-type]


async def test_a_real_unretracted_work_is_verified():
    fake = Crossref()
    result = await resolve(_work(SALT), http=fake.http())
    assert isinstance(result, VerifiedWork)
    assert result.work.doi == "10.5555/saltlog.001" and result.notices == []


async def test_a_fabricated_doi_is_unverified_not_found():
    fake = Crossref()
    invented = {**SALT, "DOI": "10.5555/saltlog.nowhere"}
    result = await resolve(_work(invented), http=fake.http())
    assert isinstance(result, Unverified) and result.kind == "not_found"
    assert "10.5555/saltlog.nowhere" in result.reason


async def test_a_doi_that_belongs_to_another_work_is_unverified():
    fake = Crossref()
    wrong = {**SALT, "DOI": DIARY["DOI"]}
    result = await resolve(_work(wrong), http=fake.http())
    assert isinstance(result, Unverified) and result.kind == "details_differ"


async def test_a_retracted_work_is_never_verified():
    fake = Crossref()
    result = await resolve(_work(RETRACTED), http=fake.http())
    assert isinstance(result, Unverified) and result.kind == "retracted"
    assert result.notices[0].kind == "retraction"
    assert "retracted" in result.reason


async def test_a_corrected_work_is_verified_with_its_notice():
    fake = Crossref()
    result = await resolve(_work(CORRECTED), http=fake.http())
    assert isinstance(result, VerifiedWork)
    assert [n.kind for n in result.notices] == ["correction"]


async def test_a_work_with_no_doi_is_found_by_title_and_gets_its_doi():
    fake = Crossref()
    without = {k: v for k, v in SALT.items() if k != "DOI"}
    result = await resolve(_work(without, "zotero"), http=fake.http())
    assert isinstance(result, VerifiedWork)
    assert result.work.doi == "10.5555/saltlog.001"
    assert result.work.csl["DOI"] == "10.5555/saltlog.001"
    assert result.work.source == "zotero"


async def test_a_work_nobody_has_heard_of_is_unverified_by_title():
    fake = Crossref()
    ghost = {
        "id": "ghost",
        "title": "Imaginary glaciers beneath invented volcanoes",
        "author": [{"family": "Nobody", "given": "N."}],
        "issued": {"date-parts": [[2020]]},
    }
    result = await resolve(_work(ghost), http=fake.http())
    assert isinstance(result, Unverified) and result.kind == "not_found"


async def test_a_lookup_that_cannot_run_is_unchecked_never_a_pass():
    result = await resolve(_work(SALT), http=Crossref(fail_with=400).http())
    assert isinstance(result, Unverified) and result.kind == "unchecked"
    assert "could not be checked" in result.reason


async def test_a_work_with_neither_doi_nor_title_is_unchecked():
    result = await resolve(
        _work({"id": "x", "author": [{"family": "Roe"}]}), http=Crossref().http()
    )
    assert isinstance(result, Unverified) and result.kind == "unchecked"


async def test_resolve_uses_the_http_cache_so_a_second_resolve_asks_nothing():
    class Memory:
        def __init__(self) -> None:
            self.items: dict[str, bytes] = {}

        def get(self, key: str) -> bytes | None:
            return self.items.get(key)

        def put(self, key: str, value: bytes) -> None:
            self.items[key] = value

    fake = Crossref()
    http = ReferenceHttp(
        client=httpx.AsyncClient(transport=fake.transport()), cache=Memory()
    )
    await resolve(_work(SALT), http=http)
    first = len(fake.requests)
    await resolve(_work(SALT), http=http)
    assert first >= 1 and len(fake.requests) == first


def test_a_work_makes_a_citation_item_that_keeps_the_zotero_uri():
    work = Work(csl=SALT, source="zotero", uri="http://zotero.org/users/1/items/A")
    item = work.citation_item(locator="3", prefix="see")
    assert (item.csl, item.uri, item.locator, item.prefix) == (
        SALT,
        work.uri,
        "3",
        "see",
    )
    assert _work(SALT).citation_item().uri is None


async def test_find_works_drops_a_crossref_record_that_shares_one_query_word():
    class Unrelated(Crossref):
        def __init__(self) -> None:
            super().__init__()
            self.works = {RETRACTED["DOI"]: RETRACTED}

    title_words = [w for w in RETRACTED["title"].lower().split() if len(w) > 5]
    query = f"{title_words[0]} unrelated gardening roses"
    found = await find_works(
        query, sources=Sources(crossref=Unrelated().http()), limit=5
    )
    assert found == []
