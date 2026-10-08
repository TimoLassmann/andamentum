"""The local corpus source: BM25 search, citation links and abstracts from a folder.

Offline by construction (the source has no network), over the invented corpus of
``local_world``. Also the ``local_id`` identifier it introduced: normalisation,
identity, dumps, the store column and its migration, expansion eligibility.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from andamentum.references import Work, merge_works, works_compatible
from andamentum.references.works import normalise_local_id

from .. import store as store_module
from ..sources import (
    ExpandingSource,
    LocalCorpusSource,
    Source,
    SourceConfig,
    SourceError,
    TextSource,
    build_sources,
)
from ..sources.local_corpus import (
    CORPUS_FILE,
    INDEX_FILE,
    CorpusRecord,
    build_index,
    iter_records,
    match_expression,
    read_summary,
    work_of,
)
from ..search import is_scholarly
from ..store import SCHEMA_VERSION, ResearchStore
from .local_world import (
    APOLLO_A,
    APOLLO_B,
    EMPTY,
    FOLLOW,
    GARDEN,
    NO_ABSTRACT,
    SALTLOG_D,
    records,
    write_corpus,
)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return write_corpus(tmp_path / "corpus")


def local(corpus_id: str) -> Work:
    return Work(local_id=f"local:{corpus_id}")


# -- the identifier -----------------------------------------------------------------


def test_a_local_id_needs_its_prefix_and_a_plain_token() -> None:
    assert normalise_local_id("local:252715594") == "local:252715594"
    assert normalise_local_id(" local:a-1.b_2 ") == "local:a-1.b_2"
    assert normalise_local_id("252715594") is None
    assert normalise_local_id("local:") is None
    assert normalise_local_id("local:has space") is None
    assert normalise_local_id(None) is None
    with pytest.raises(ValidationError, match="not a valid local_id"):
        Work(local_id="252715594")


def test_a_work_dump_has_a_local_id_only_when_it_has_one() -> None:
    """Every other work's dump, and any digest of it, is what it was before the field."""
    assert "local_id" not in Work(title="x").model_dump(mode="json")
    assert "local_id" not in Work(title="x").model_dump()
    assert '"local_id"' not in Work(title="x").model_dump_json()
    dumped = local("7").model_dump(mode="json")
    assert dumped["local_id"] == "local:7"
    assert Work.model_validate(dumped) == local("7")


def test_a_local_id_is_a_hard_identity_key() -> None:
    a, b = local("7"), Work(local_id="local:7", title="Another title")
    assert a.hard_keys() == [("local", "local:7")]
    assert set(a.hard_keys()) & set(b.hard_keys())
    assert works_compatible(a, b)
    assert not works_compatible(a, local("8"))
    merged = merge_works(Work(title="T", doi="10.5555/x"), b)
    assert merged.local_id == "local:7" and merged.doi == "10.5555/x"


def test_a_local_record_can_seed_citation_expansion() -> None:
    assert is_scholarly(local("7"))
    assert not is_scholarly(Work(title="A page"))


# -- the store ----------------------------------------------------------------------


def test_the_store_keeps_and_finds_a_work_by_its_local_id() -> None:
    with ResearchStore.in_memory() as store:
        stored = store.upsert_work(Work(local_id="local:7", title="Seven"))
        assert stored.local_id == "local:7"
        again = store.upsert_work(Work(local_id="local:7", doi="10.5555/seven"))
        assert again.work_id == stored.work_id and again.doi == "10.5555/seven"
        other = store.upsert_work(Work(local_id="local:8", title="Eight"))
        assert other.work_id != stored.work_id
        found = store.find_work(local("7"))
        assert found is not None and found.work_id == stored.work_id


def test_a_store_of_version_three_gains_the_local_id_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "old.db"
    monkeypatch.setattr(store_module, "SCHEMA_VERSION", 3)
    with ResearchStore(path) as old:
        assert old.schema_version() == 3
    monkeypatch.undo()
    raw = sqlite3.connect(path)
    raw.execute(
        "INSERT INTO works (work_id, doi, title, first_seen, last_checked) "
        "VALUES ('w_old', '10.5555/kept', 'Kept', 0, 0)"
    )
    raw.commit()
    raw.close()
    with ResearchStore(path) as store:
        assert store.schema_version() == SCHEMA_VERSION
        assert store.find_work(Work(doi="10.5555/kept")) is not None
        stored = store.upsert_work(Work(local_id="local:7", title="Seven"))
        assert stored.local_id == "local:7"


# -- building the index -------------------------------------------------------------


def test_the_index_counts_what_the_corpus_holds(tmp_path: Path) -> None:
    folder = write_corpus(tmp_path / "c", index=False)
    summary = build_index(folder)
    everything = records()
    assert summary.n_records == len(everything)
    assert summary.n_with_abstract == sum(1 for r in everything if r.abstract)
    # edges inside the corpus only: 1006->1001, 1002->1001, 1005->1004, 2003->1001
    assert summary.n_citation_edges == 4
    assert summary.n_citation_edges_outside == 1  # 1002 -> 9999
    assert summary.corpus_bytes == (folder / CORPUS_FILE).stat().st_size
    assert read_summary(folder) == summary
    assert [r.corpus_id for r in iter_records(folder)] == [
        r.corpus_id for r in everything
    ]


def test_a_duplicate_corpus_id_stops_the_build(tmp_path: Path) -> None:
    folder = tmp_path / "dup"
    folder.mkdir()
    line = CorpusRecord(corpus_id="1", title="One").model_dump_json() + "\n"
    (folder / CORPUS_FILE).write_text(line * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate corpus id 1"):
        build_index(folder)
    assert not (folder / INDEX_FILE).exists()
    assert not (folder / (INDEX_FILE + ".partial")).exists()


def test_a_bad_record_names_its_line(tmp_path: Path) -> None:
    folder = tmp_path / "bad"
    folder.mkdir()
    (folder / CORPUS_FILE).write_text(
        CorpusRecord(corpus_id="1", title="One").model_dump_json()
        + '\n{"corpus_id": "2", "surprise": 1}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="corpus.jsonl:2"):
        build_index(folder)


def test_a_missing_or_stale_index_is_a_config_error(tmp_path: Path) -> None:
    folder = write_corpus(tmp_path / "c", index=False)
    with pytest.raises(SourceError, match="no corpus.sqlite") as missing:
        LocalCorpusSource(corpus_dir=folder)
    assert missing.value.kind == "config"
    build_index(folder)
    LocalCorpusSource(corpus_dir=folder)
    with (folder / CORPUS_FILE).open("a", encoding="utf-8") as handle:
        handle.write(
            CorpusRecord(corpus_id="3000", title="Late").model_dump_json() + "\n"
        )
    with pytest.raises(SourceError, match="another corpus.jsonl") as stale:
        LocalCorpusSource(corpus_dir=folder)
    assert stale.value.kind == "config"
    with pytest.raises(SourceError, match="no corpus.jsonl"):
        LocalCorpusSource(corpus_dir=tmp_path / "nowhere")


def test_a_file_that_is_not_an_index_is_refused(tmp_path: Path) -> None:
    folder = write_corpus(tmp_path / "c", index=False)
    (folder / INDEX_FILE).write_bytes(b"not a database at all")
    with pytest.raises(SourceError, match="not a corpus index"):
        LocalCorpusSource(corpus_dir=folder)
    (folder / INDEX_FILE).unlink()
    conn = sqlite3.connect(folder / INDEX_FILE)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.commit()
    conn.close()
    with pytest.raises(SourceError, match="another version"):
        LocalCorpusSource(corpus_dir=folder)


# -- search -------------------------------------------------------------------------


def test_the_match_expression_is_the_or_of_the_distinct_words() -> None:
    assert (
        match_expression("Apollo, the APOLLO cache?") == '"apollo" OR "the" OR "cache"'
    )
    assert match_expression('say "hi" AND NOT (a) -b') == (
        '"say" OR "hi" OR "and" OR "not" OR "a" OR "b"'
    )
    assert match_expression("  ?!  ") is None
    assert match_expression("") is None


async def test_search_ranks_by_bm25_and_returns_local_records(corpus: Path) -> None:
    source = LocalCorpusSource(corpus_dir=corpus)
    found = await source.search("saltlog logging overhead per request", limit=3)
    assert [c.work.local_id for c in found[:2]] == [
        f"local:{SALTLOG_D}",
        "local:1005",
    ]
    assert [c.rank for c in found] == [0, 1, 2]
    top = found[0]
    assert top.sources == ["local_corpus"]
    assert top.queries == ["saltlog logging overhead per request"]
    assert top.work.title == "SaltLog logging overhead per request"
    assert top.work.access == "abstract" and "3 ms" in top.work.abstract
    assert top.work.licence == ""
    assert top.snippet and top.work.abstract.startswith(top.snippet)
    assert not top.work.doi and not top.work.url


async def test_search_reads_the_whole_question_as_or_terms(corpus: Path) -> None:
    """A long question that no record contains in full still ranks the records
    sharing its words: AND would return nothing."""
    source = LocalCorpusSource(corpus_dir=corpus)
    question = (
        "Which paper reports how much the Apollo cache reduced request latency "
        "compared with a service that kept every request uncached and unlogged?"
    )
    found = await source.search(question, limit=20)
    ids = [c.work.local_id for c in found]
    assert ids[0] == f"local:{APOLLO_A}"
    # plain BM25 keeps stop words, so a record sharing only "a" is found, below
    # the records about the Apollo cache
    assert ids.index(f"local:{GARDEN}") > max(
        ids.index(f"local:{n}") for n in (APOLLO_A, APOLLO_B, FOLLOW)
    )
    assert len(ids) == len(set(ids))


async def test_search_stems_and_breaks_ties_by_corpus_order(corpus: Path) -> None:
    source = LocalCorpusSource(corpus_dir=corpus)
    assert [c.work.local_id for c in await source.search("pruned rose", limit=5)] == [
        f"local:{GARDEN}"
    ]
    first = await source.search("latency", limit=20)
    again = await source.search("latency", limit=20)
    assert [c.work.local_id for c in first] == [c.work.local_id for c in again]


async def test_search_handles_odd_queries_and_limits(corpus: Path) -> None:
    source = LocalCorpusSource(corpus_dir=corpus)
    assert await source.search("?!  ...", limit=5) == []
    assert await source.search("apollo", limit=0) == []
    assert len(await source.search("apollo", limit=1)) == 1
    assert await source.search("zzzz-no-such-word-zzzz") == []
    assert await source.search('apollo" OR "x', limit=2)  # quotes are not syntax
    assert await source.search("apollo", filters=None)


async def test_a_record_with_no_text_is_never_found_by_search(corpus: Path) -> None:
    source = LocalCorpusSource(corpus_dir=corpus)
    ids = {
        c.work.local_id
        for q in ("untitled", "cache sizing", "empty", "title", "no title")
        for c in await source.search(q, limit=20)
    }
    assert f"local:{EMPTY}" not in ids
    assert f"local:{NO_ABSTRACT}" in ids  # it has a title


# -- text and links -----------------------------------------------------------------


async def test_the_text_is_the_title_and_the_abstract(corpus: Path) -> None:
    source = LocalCorpusSource(corpus_dir=corpus)
    fetched = await source.fetch_text(local(APOLLO_B))
    assert fetched is not None
    assert fetched.access == "abstract" and fetched.source == "local_corpus"
    assert fetched.url == f"local:{APOLLO_B}"
    assert fetched.text.startswith("Latency effects of the Apollo caching layer\n\n")
    assert "lowered median latency by about half" in fetched.text
    title_only = await source.fetch_text(local(NO_ABSTRACT))
    assert (
        title_only is not None and title_only.text == "An untitled note on cache sizing"
    )


async def test_there_is_no_text_for_what_the_corpus_does_not_hold(corpus: Path) -> None:
    source = LocalCorpusSource(corpus_dir=corpus)
    assert (
        await source.fetch_text(Work(title="A web page", url="https://x.example/"))
        is None
    )
    assert await source.fetch_text(local("424242")) is None
    assert await source.fetch_text(local(EMPTY)) is None  # neither title nor abstract


async def test_references_and_citations_follow_the_corpus_links(corpus: Path) -> None:
    source = LocalCorpusSource(corpus_dir=corpus)
    refs = await source.references(local("1002"))
    assert [c.work.local_id for c in refs] == [f"local:{APOLLO_A}"]  # 9999 is outside
    assert refs[0].work.title.startswith("Apollo cache halves")
    cites = await source.citations(local(APOLLO_A))
    assert [c.work.local_id for c in cites] == [
        "local:1002",
        f"local:{FOLLOW}",
        f"local:{EMPTY}",
    ]
    assert [c.rank for c in cites] == [0, 1, 2]
    assert len(await source.citations(local(APOLLO_A), limit=1)) == 1
    assert await source.references(local(APOLLO_A)) == []
    assert await source.citations(local(SALTLOG_D), limit=5)  # 1005 cites 1004


async def test_a_work_the_corpus_cannot_place_is_a_config_error(corpus: Path) -> None:
    source = LocalCorpusSource(corpus_dir=corpus)
    with pytest.raises(SourceError, match="no local id") as web:
        await source.citations(Work(doi="10.5555/x"))
    assert web.value.kind == "config"
    with pytest.raises(SourceError, match="cannot place local:424242") as unknown:
        await source.references(local("424242"))
    assert unknown.value.kind == "config"


def test_the_source_is_a_searching_reading_expanding_source(corpus: Path) -> None:
    source = LocalCorpusSource(corpus_dir=corpus)
    assert isinstance(source, Source) and source.name == "local_corpus"
    assert isinstance(source, TextSource) and isinstance(source, ExpandingSource)
    assert source.summary.n_records == len(records())


def test_a_record_becomes_a_work_with_only_its_local_id() -> None:
    work = work_of(
        CorpusRecord(corpus_id="12", title="  A  title ", abstract="Some\ntext.")
    )
    assert work.local_id == "local:12" and work.title == "A title"
    assert work.abstract == "Some text." and work.access == "abstract"
    bare = work_of(CorpusRecord(corpus_id="13"))
    assert bare.title == "" and bare.access == "metadata"
    assert bare.identity_keys() == [("local", "local:13")]


# -- building the sources of a run --------------------------------------------------


def test_build_sources_makes_the_local_source_from_the_config(corpus: Path) -> None:
    with ResearchStore.in_memory() as store:
        built = build_sources(
            SourceConfig(scholarly=("local_corpus",), local_corpus=corpus),
            store=store,
            mode="scholarly",
        )
    assert [s.name for s in built.search] == ["local_corpus"]
    assert [s.name for s in built.text] == ["local_corpus"]
    assert [s.name for s in built.expand] == ["local_corpus"]


def test_the_local_source_is_read_before_the_network_sources(corpus: Path) -> None:
    with ResearchStore.in_memory() as store:
        built = build_sources(
            SourceConfig(scholarly=("europepmc", "local_corpus"), local_corpus=corpus),
            store=store,
            mode="scholarly",
        )
    assert [s.name for s in built.text] == ["local_corpus", "europepmc"]


def test_naming_the_local_source_without_a_folder_is_a_config_error() -> None:
    with ResearchStore.in_memory() as store, pytest.raises(SourceError) as error:
        build_sources(
            SourceConfig(scholarly=("local_corpus",)), store=store, mode="scholarly"
        )
    assert error.value.kind == "config" and "local_corpus" in error.value.reason
