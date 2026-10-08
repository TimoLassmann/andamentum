"""Enrichers: each writes its own metadata key, keeps the others, writes a revision."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from andamentum.core.decisions import Answer, DecisionError
from andamentum.core.decisions.testing import StubDecisions, choice_answer, noul_answer
from andamentum.document_store import Store
from andamentum.document_store.enrich import (
    EnrichError,
    EnrichResult,
    enrich_all,
    enrich_bibliographic,
    enrich_classify,
    enrich_extract,
    enrich_facts,
    enrich_identifiers,
)
from andamentum.document_store.enrich import extract as extract_module
from andamentum.document_store.extraction import ExtractionUnavailable
from andamentum.document_store.metadata_models import DocumentMetadataFields
from andamentum.document_store.types import DocumentNotFound
from andamentum.references import CrossrefError, ReferenceHttp

DOI = "10.5555/saltlog.001"
TEXT = (
    "# Salt logging\n\nAn invented study. doi:" + DOI + "\n\n## Methods\n\nWe logged salt.\n"
)
PUBMED_XML = """<?xml version="1.0"?>
<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>9000001</PMID>
<Article><Journal><JournalIssue><Volume>12</Volume>
<PubDate><Year>2021</Year></PubDate></JournalIssue><Title>Invented Salt Journal</Title></Journal>
<ArticleTitle>Salt logging</ArticleTitle>
<AuthorList><Author><LastName>Doe</LastName><Initials>J</Initials></Author></AuthorList>
</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>
"""


@pytest.fixture
async def store(tmp_path: Path):
    s = await Store.open(tmp_path / "docs.db")
    yield s
    await s.close()


def _crossref_http(status: int = 200) -> ReferenceHttp:
    def handler(request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(status, json={})
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "message": {
                    "DOI": DOI,
                    "title": ["Salt logging"],
                    "type": "journal-article",
                    "author": [{"family": "Doe", "given": "Jane"}],
                },
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ReferenceHttp(client=client, mailto="jane@example.invalid")


async def test_identifiers_preserves_other_keys_and_writes_revision(store: Store):
    doc_id = await store.put(TEXT, metadata={"owner": "jane", "tags": ["a"]})
    result = await enrich_identifiers(store, doc_id)
    assert isinstance(result, EnrichResult)
    assert result.key == "identifiers" and result.rev == 2 and result.skipped is None
    doc = await store.get(doc_id)
    assert doc is not None and doc.rev == 2
    assert doc.metadata["owner"] == "jane" and doc.metadata["tags"] == ["a"]
    ids = doc.metadata["identifiers"]
    assert ids["doi"] == DOI
    assert ids["source"] == "andamentum.references.extract_identifiers"
    assert "pmid" not in ids and "urls" not in ids
    old = await store.get(doc_id, rev=1)
    assert old is not None and "identifiers" not in old.metadata


async def test_facts(store: Store):
    doc_id = await store.put(TEXT, metadata={"keep": 1})
    result = await enrich_facts(store, doc_id)
    assert result.rev == 2
    doc = await store.get(doc_id)
    assert doc is not None
    assert doc.metadata["keep"] == 1
    assert doc.metadata["facts"] == {
        "chars": len(TEXT),
        "words": len(TEXT.split()),
        "lines": len(TEXT.splitlines()),
        "headings": 2,
        "media_type": "text/markdown",
    }


async def test_missing_document_raises(store: Store):
    with pytest.raises(DocumentNotFound):
        await enrich_facts(store, "nope")


async def test_bibliographic_crossref(store: Store):
    doc_id = await store.put(TEXT, metadata={"keep": 1})
    result = await enrich_bibliographic(store, doc_id, http=_crossref_http())
    assert result.key == "bib" and result.rev == 2
    doc = await store.get(doc_id)
    assert doc is not None
    assert doc.metadata["keep"] == 1
    bib = doc.metadata["bib"]
    assert bib["source"] == "crossref" and bib["id"] == DOI
    assert bib["title"] == "Salt logging"


async def test_bibliographic_reads_stored_identifiers(store: Store):
    doc_id = await store.put("no identifier in the text", metadata={"identifiers": {"doi": DOI}})
    result = await enrich_bibliographic(store, doc_id, http=_crossref_http())
    assert result.rev == 2


async def test_bibliographic_pubmed_when_no_doi(store: Store):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=PUBMED_XML)

    doc_id = await store.put("An invented note. PMID: 9000001")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await enrich_bibliographic(
            store, doc_id, http=_crossref_http(), pubmed_client=client, api_key="k"
        )
    assert result.rev == 2
    assert seen[0].url.params["api_key"] == "k"
    doc = await store.get(doc_id)
    assert doc is not None
    bib = doc.metadata["bib"]
    assert bib["source"] == "pubmed" and bib["id"] == "9000001"
    assert bib["title"] == "Salt logging"
    assert bib["issued"] == {"date-parts": [[2021]]}


async def test_bibliographic_skips_without_identifier(store: Store):
    doc_id = await store.put("Nothing citable here.")
    result = await enrich_bibliographic(store, doc_id, http=_crossref_http())
    assert result.rev is None and result.skipped == "skipped: no identifier"
    doc = await store.get(doc_id)
    assert doc is not None and doc.rev == 1 and "bib" not in doc.metadata


async def test_bibliographic_lookup_error_raises(store: Store):
    doc_id = await store.put(TEXT)
    with pytest.raises(CrossrefError):
        await enrich_bibliographic(store, doc_id, http=_crossref_http(404))
    doc = await store.get(doc_id)
    assert doc is not None and doc.rev == 1


async def test_bibliographic_pubmed_http_error_raises(store: Store):
    doc_id = await store.put("PMID: 9000001")
    transport = httpx.MockTransport(lambda r: httpx.Response(500))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await enrich_bibliographic(
                store, doc_id, http=_crossref_http(), pubmed_client=client
            )


async def test_bibliographic_pubmed_no_record_raises(store: Store):
    doc_id = await store.put("PMID: 9000001")
    xml = "<PubmedArticleSet></PubmedArticleSet>"
    transport = httpx.MockTransport(lambda r: httpx.Response(200, text=xml))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(EnrichError):
            await enrich_bibliographic(
                store, doc_id, http=_crossref_http(), pubmed_client=client
            )


async def test_extract(store: Store, monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, str]] = []

    async def fake(content: str, model: str | None = None, **_: object):
        calls.append((content, model or ""))
        return DocumentMetadataFields(title="Salt", projects=["Apollo"], people=["Jane Doe"])

    monkeypatch.setattr(extract_module, "extract_document_metadata", fake)
    doc_id = await store.put(TEXT, metadata={"keep": 1})
    result = await enrich_extract(store, doc_id, model="ollama:test-model")
    assert result.key == "extracted" and result.rev == 2
    assert calls == [(TEXT, "ollama:test-model")]
    doc = await store.get(doc_id)
    assert doc is not None and doc.metadata["keep"] == 1
    assert doc.metadata["extracted"] == {
        "title": "Salt",
        "projects": ["Apollo"],
        "people": ["Jane Doe"],
        "model": "ollama:test-model",
    }


async def test_extract_requires_model_and_raises_unavailable(
    store: Store, monkeypatch: pytest.MonkeyPatch
):
    doc_id = await store.put(TEXT)
    with pytest.raises(ValueError):
        await enrich_extract(store, doc_id, model="")

    async def down(content: str, model: str | None = None, **_: object):
        raise ExtractionUnavailable("down")

    monkeypatch.setattr(extract_module, "extract_document_metadata", down)
    with pytest.raises(ExtractionUnavailable):
        await enrich_extract(store, doc_id, model="ollama:m")
    doc = await store.get(doc_id)
    assert doc is not None and doc.rev == 1


LABELS = {"methods": "describes methods", "results": "reports results", "other": "anything else"}


async def test_classify_single(store: Store):
    stub = StubDecisions(
        lambda state, qs: {k: choice_answer(list(LABELS), "methods", 0.8) for k in qs}
    )
    doc_id = await store.put("x" * 50, metadata={"keep": 1})
    result = await enrich_classify(
        store, doc_id, decision_model=stub.model(), labels=LABELS, max_chars=20
    )
    assert result.key == "labels" and result.rev == 2
    assert len(stub.calls) == 1
    state, questions = stub.calls[0]
    assert state == {"document": "x" * 20}
    (question,) = questions.values()
    assert question["type"] == "choice" and set(question["criteria"]) == set(LABELS)  # type: ignore[arg-type]
    doc = await store.get(doc_id)
    assert doc is not None and doc.metadata["keep"] == 1
    value = doc.metadata["labels"]
    assert value["labels"] == ["methods"]
    assert value["probabilities"]["methods"] == pytest.approx(0.8)
    assert value["decision_model"] == stub.name and value["digest"] == stub.digest
    assert value["truncated"] is True and value["multi"] is False


async def test_classify_multi(store: Store):
    probs = {"methods": 0.9, "results": 0.5, "other": 0.1}

    def respond(state: object, qs: dict[str, dict[str, object]]) -> dict[str, Answer]:
        return {k: noul_answer(probs[k]) for k in qs}

    stub = StubDecisions(respond)
    doc_id = await store.put("short")
    await enrich_classify(
        store, doc_id, decision_model=stub.model(), labels=LABELS, multi=True, max_chars=100
    )
    doc = await store.get(doc_id)
    assert doc is not None
    value = doc.metadata["labels"]
    assert value["labels"] == ["methods", "results"]
    assert value["probabilities"] == probs
    assert value["truncated"] is False


async def test_classify_validates_arguments(store: Store):
    stub = StubDecisions(lambda s, q: {})
    doc_id = await store.put("t")
    dm = stub.model()
    with pytest.raises(ValueError):
        await enrich_classify(store, doc_id, decision_model=dm, labels={"a": "x"}, max_chars=10)
    with pytest.raises(ValueError):
        await enrich_classify(store, doc_id, decision_model=dm, labels=LABELS, max_chars=0)
    with pytest.raises(ValueError):
        await enrich_classify(store, doc_id, decision_model=dm, labels={}, max_chars=10)


async def test_classify_server_error_raises(store: Store):
    def respond(state: object, qs: dict[str, dict[str, object]]) -> dict[str, Answer]:
        raise RuntimeError("boom")

    stub = StubDecisions(respond)
    doc_id = await store.put("t")
    with pytest.raises((DecisionError, RuntimeError)):
        await enrich_classify(
            store, doc_id, decision_model=stub.model(), labels=LABELS, max_chars=10
        )
    doc = await store.get(doc_id)
    assert doc is not None and doc.rev == 1


async def test_enrich_all_sequential_over_where(store: Store):
    a = await store.put("alpha", metadata={"kind": "x"})
    b = await store.put("beta", metadata={"kind": "x"})
    c = await store.put("gamma", metadata={"kind": "y"})
    active = 0
    peak = 0
    order: list[str] = []

    async def probe(s: Store, doc_id: str) -> EnrichResult:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        order.append(doc_id)
        result = await enrich_facts(s, doc_id)
        active -= 1
        return result

    results = await enrich_all(store, probe, where={"kind": "x"})
    assert [r.doc_id for r in results] == [a, b] == order
    assert peak == 1
    cdoc = await store.get(c)
    assert cdoc is not None and cdoc.rev == 1
    adoc = await store.get(a)
    assert adoc is not None and adoc.metadata["kind"] == "x" and "facts" in adoc.metadata


async def test_enrich_all_passes_arguments_and_propagates_errors(store: Store):
    await store.put("no identifier")
    results = await enrich_all(store, enrich_bibliographic, http=_crossref_http())
    assert len(results) == 1 and results[0].skipped is not None
    await store.put(TEXT)
    with pytest.raises(CrossrefError):
        await enrich_all(store, enrich_bibliographic, http=_crossref_http(404))
