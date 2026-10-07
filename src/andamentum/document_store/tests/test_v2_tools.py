"""Tests for the v2 store's agent tools: list, glob, grep, search, similar, outline, read, note, decisions.

Synthetic multi-document store, fake vectors, no model.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from andamentum.document_store.v2 import DocEntry, OutlineEntry, SectionStore
from andamentum.document_store.v2.store import longest_literal

DIM = 4

DOCS = {
    "notes/apollo.md": (
        "Apollo notes",
        {"kind": "note", "year": 2024},
        "# Apollo\n\nThe Apollo widget is assembled in Brightwater.\n\n"
        "## Release v2.1\n\nRelease v2.1 fixed the SaltLog import of NM_000256.3 records.\n"
        "Also NM_000256.3 again.\n",
    ),
    "notes/saltlog.md": (
        "SaltLog notes",
        {"kind": "note", "year": 2023},
        "# SaltLog\n\nSaltLog stores logs of salt.\n\n## Format\n\nJane Doe wrote the format.\n",
    ),
    "papers/jane_doe.md": (
        "Jane Doe paper",
        {"kind": "paper", "year": 2024},
        "# Widgets\n\nWidget theory by Jane Doe.\n\n## Results\n\nResults of widget tests: 42 passed.\n",
    ),
    "papers/100%_sure.md": (
        "Odd name",
        None,
        "# Odd\n\nOdd file with a percent sign.\n",
    ),
}


def _vec(text: str) -> list[float]:
    t = text.lower()
    return [float("widget" in t), float("salt" in t), float("release" in t), 0.1]


async def _embed(texts: list[str]) -> list[list[float]]:
    return [_vec(t) for t in texts]


@pytest.fixture
async def store(tmp_path: Path) -> SectionStore:
    s = SectionStore(tmp_path / "s.db", embedding_dim=DIM)
    s.create()
    for path, (title, meta, text) in DOCS.items():
        await s.add_document(
            path=path,
            title=title,
            text=text,
            meta=json.dumps(meta) if meta is not None else None,
            embed_fn=_embed,
        )
    return s


# ── listing ───────────────────────────────────────────────────────────────


async def test_list_documents_prefix_and_limit(store: SectionStore):
    notes = store.list_documents(prefix="notes/")
    assert [d.path for d in notes] == ["notes/apollo.md", "notes/saltlog.md"]
    assert notes[0] == DocEntry("notes/apollo.md", "Apollo notes", 2)
    assert len(store.list_documents()) == 4
    assert len(store.list_documents(limit=1)) == 1


async def test_list_prefix_treats_like_wildcards_literally(store: SectionStore):
    assert [d.path for d in store.list_documents(prefix="papers/100%")] == [
        "papers/100%_sure.md"
    ]
    assert store.list_documents(prefix="papers/1%0") == []
    assert store.list_documents(prefix="papers/_") == []


async def test_glob(store: SectionStore):
    assert [d.path for d in store.glob("papers/*")] == [
        "papers/100%_sure.md",
        "papers/jane_doe.md",
    ]
    assert [d.path for d in store.glob("*/saltlog.md")] == ["notes/saltlog.md"]
    assert store.glob("nothing*") == []
    assert len(store.glob("*.md", limit=2)) == 2


async def test_collections_upsert(store: SectionStore):
    store.add_collection(path="papers/", description="papers")
    store.add_collection(path="notes/", description="notes")
    store.add_collection(path="papers/", description="all papers")
    assert store.collections() == [("notes/", "notes"), ("papers/", "all papers")]


# ── grep ──────────────────────────────────────────────────────────────────


async def test_grep_literal_counts_matches(store: SectionStore):
    hits = store.grep("NM_000256")
    assert len(hits) == 1
    assert hits[0].path == "notes/apollo.md"
    assert hits[0].score == 2.0
    assert hits[0].heading == "Release v2.1"


async def test_grep_escaped_dot_is_literal(store: SectionStore):
    assert len(store.grep(r"NM_000256\.3")) == 1
    assert store.grep(r"NM_000256\.4") == []


async def test_grep_regex_without_three_char_literal_scans(store: SectionStore):
    assert longest_literal(r"\d\d") is None
    hits = store.grep(r"\b\d\d\b")
    assert [h.path for h in hits] == ["papers/jane_doe.md"]
    assert [h.path for h in store.grep(r"J.ne D.e")] == [
        "notes/saltlog.md",
        "papers/jane_doe.md",
    ]


async def test_grep_path_prefix_and_limit(store: SectionStore):
    assert {h.path for h in store.grep("Jane Doe")} == {
        "notes/saltlog.md",
        "papers/jane_doe.md",
    }
    assert [h.path for h in store.grep("Jane Doe", path_prefix="papers/")] == [
        "papers/jane_doe.md"
    ]
    assert len(store.grep("Jane Doe", limit=1)) == 1


async def test_grep_alternation_and_optional_group_are_not_narrowed(
    store: SectionStore,
):
    assert longest_literal("Brightwater|Jane Doe") is None
    assert longest_literal("(Jane Doe)?") is None
    hits = store.grep("Brightwater|Brightwood")
    assert [h.path for h in hits] == ["notes/apollo.md"]
    # a quantifier removes the optional character from the literal run
    assert longest_literal("colou?rful") == "colo"
    assert longest_literal(r"abc[xyz]defg") == "defg"


async def test_grep_is_case_sensitive_like_the_regex(store: SectionStore):
    assert store.grep("jane doe") == []
    assert len(store.grep("(?i)jane doe")) == 2


async def test_grep_invalid_regex_raises_value_error(store: SectionStore):
    with pytest.raises(ValueError, match="invalid regex"):
        store.grep("(unclosed")


# ── search / similar ──────────────────────────────────────────────────────


async def test_search_bm25_or_terms(store: SectionStore):
    hits = store.search("widget Brightwater", k=5)
    assert hits[0].path == "notes/apollo.md"
    assert {h.path for h in hits} >= {"notes/apollo.md", "papers/jane_doe.md"}
    assert all(h.score > 0 for h in hits)
    assert store.search("???") == []


async def test_search_path_prefix_and_meta_filters(store: SectionStore):
    assert {h.path for h in store.search("Jane Doe", path_prefix="notes/")} == {
        "notes/saltlog.md"
    }
    papers = store.search("Jane Doe widget", meta={"kind": "paper"})
    assert {h.path for h in papers} == {"papers/jane_doe.md"}
    both = store.search("Doe", path_prefix="notes/", meta={"year": "2023"})
    assert {h.path for h in both} == {"notes/saltlog.md"}  # numbers compare as text
    assert store.search("Doe", meta={"kind": "nothing"}) == []
    assert store.search("Doe", meta={"missing": "x"}) == []


async def test_similar_ranks_by_vector_and_filters(store: SectionStore):
    q = [1.0, 0.0, 0.0, 0.1]
    hits = store.similar(q, k=2)
    assert len(hits) == 2
    assert all("widget" in h.text.lower() for h in hits)
    notes = store.similar(q, k=3, path_prefix="notes/")
    assert notes and all(h.path.startswith("notes/") for h in notes)
    assert len(notes) <= 3
    kinded = store.similar(q, k=5, meta={"kind": "paper"})
    assert kinded and all(h.path == "papers/jane_doe.md" for h in kinded)
    assert store.similar(q, k=5, meta={"kind": "nothing"}) == []


async def test_similar_needs_vector_index(tmp_path: Path):
    s = SectionStore(tmp_path / "lex.db", embedding_dim=None)
    s.create()
    with pytest.raises(ValueError, match="no vector index"):
        s.similar([0.0] * DIM)


# ── outline / read ────────────────────────────────────────────────────────


async def test_outline(store: SectionStore):
    out = store.outline("notes/apollo.md")
    assert [(e.heading, e.level) for e in out] == [("Apollo", 1), ("Release v2.1", 2)]
    assert isinstance(out[0], OutlineEntry)
    assert out[0].line_from == 1 and out[0].line_to < out[1].line_from
    assert store.section_path(out[0].section_id) == "notes/apollo.md"


async def test_outline_unknown_path(store: SectionStore):
    with pytest.raises(KeyError):
        store.outline("nope.md")


async def test_read_prefixes_line_numbers(store: SectionStore):
    text = store.read("notes/apollo.md", offset=1, limit=3)
    assert text.split("\n") == [
        "1: # Apollo",
        "2: ",
        "3: The Apollo widget is assembled in Brightwater.",
    ]
    assert store.read("notes/apollo.md", offset=3, limit=1).startswith("3: The Apollo")


async def test_read_offset_past_end_and_overlong_limit(store: SectionStore):
    assert store.read("notes/apollo.md", offset=500) == ""
    whole = store.read("notes/apollo.md", limit=1000)
    assert whole.split("\n")[0] == "1: # Apollo"
    assert len(whole.split("\n")) == len(DOCS["notes/apollo.md"][2].split("\n"))


async def test_read_errors(store: SectionStore):
    with pytest.raises(KeyError):
        store.read("nope.md")
    with pytest.raises(ValueError):
        store.read("notes/apollo.md", offset=0)
    with pytest.raises(ValueError):
        store.read("notes/apollo.md", limit=0)


# ── evidence / decisions ──────────────────────────────────────────────────


async def test_note_and_evidence(store: SectionStore):
    sid = store.outline("notes/saltlog.md")[0].section_id
    first = store.note(
        run_id="r1", section_id=sid, line_from=3, line_to=3, claim="stores logs"
    )
    second = store.note(
        run_id="r1", section_id=sid, line_from=1, line_to=2, claim="title"
    )
    store.note(run_id="r2", section_id=sid, line_from=1, line_to=1, claim="other run")
    assert second > first
    assert store.evidence("r1") == [(sid, 3, 3, "stores logs"), (sid, 1, 2, "title")]
    assert store.evidence("r2") == [(sid, 1, 1, "other run")]
    assert store.evidence("none") == []


async def test_note_validates(store: SectionStore):
    sid = store.outline("notes/saltlog.md")[0].section_id
    with pytest.raises(ValueError):
        store.note(run_id="r", section_id=sid, line_from=5, line_to=2, claim="x")
    with pytest.raises(ValueError):
        store.note(run_id="r", section_id=99999, line_from=1, line_to=1, claim="x")


async def test_log_decision_roundtrip(store: SectionStore):
    store.log_decision(
        run_id="r1",
        step=0,
        kind="stop_after_round0",
        features={"n_hits": 10, "top": [0.9, 0.5]},
        choice="loop",
        decided_by="m",
    )
    store.log_decision(
        run_id="r1",
        step=1,
        kind="tool:grep",
        features={},
        choice="NM_1",
        decided_by="m",
        confidence=0.7,
    )
    store.log_decision(
        run_id="r2", step=0, kind="k", features={}, choice="c", decided_by="m"
    )
    rows = store.decisions("r1")
    assert [r["kind"] for r in rows] == ["stop_after_round0", "tool:grep"]
    assert rows[0]["features"] == {"n_hits": 10, "top": [0.9, 0.5]}
    assert rows[0]["confidence"] is None and rows[1]["confidence"] == 0.7
    assert rows[0]["outcome"] is None
    assert store.decisions("none") == []


async def test_section_path_unknown(store: SectionStore):
    with pytest.raises(KeyError):
        store.section_path(99999)


# ── schema ────────────────────────────────────────────────────────────────


async def test_create_is_idempotent_and_adds_tables(store: SectionStore):
    store.create()
    with sqlite3.connect(store.path) as db:
        names = {r[0] for r in db.execute("SELECT name FROM sqlite_master")}
    assert {"collections", "evidence", "decisions"} <= names


def test_hits_returns_sections_in_the_given_order(store):
    ids = [h.section_id for h in store.search("apollo", k=3)]
    assert [h.section_id for h in store.hits(list(reversed(ids)))] == list(
        reversed(ids)
    )
