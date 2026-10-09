"""read_work: a work's text from the text sources, and the part relevant to a query."""

from __future__ import annotations

import httpx

from andamentum.deep_research import READ_SOURCES, read_work
from andamentum.references import LICENCE_NO_REDISTRIBUTE, Work

from .source_stub import SourceKit, ok

PMC = "PMC13245737"
SENTENCE = "Sparse sampling recovers the lattice constant."


def routes(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/fullTextXML"):
        return ok("europepmc_fulltext.xml")
    return httpx.Response(404, text=f"unrouted {request.url.path}")


async def test_full_text_comes_from_the_first_text_source_that_holds_it() -> None:
    kit = SourceKit(routes)
    work = Work(pmcid=PMC, title="Sparse sampling of invented lattices")
    found = await read_work(
        work, query=SENTENCE, store=kit.store, source_client=kit.client
    )
    assert found.via == "fetched" and found.access == "full_text"
    assert found.url.endswith(f"/{PMC}/fullTextXML")
    assert "94 of 100 trials" in found.text
    assert "94 of 100 trials" in found.window


async def test_the_window_is_the_part_relevant_to_the_query() -> None:
    kit = SourceKit(routes)
    work = Work(pmcid=PMC, title="Sparse sampling of invented lattices")
    found = await read_work(
        work,
        query=SENTENCE,
        store=kit.store,
        source_client=kit.client,
        window_chars=200,
    )
    assert found.text and len(found.window) < len(found.text)
    assert found.window in found.text or "[...]" in found.window


async def test_without_full_text_the_free_abstract_stands_in() -> None:
    kit = SourceKit(routes)
    work = Work(
        doi="10.1000/invented.2",
        title="An invented SaltLog study",
        abstract="SaltLog sampled forty invented lattices.",
    )
    found = await read_work(
        work, query=SENTENCE, store=kit.store, source_client=kit.client
    )
    assert found.via == "abstract" and found.text == work.abstract


async def test_an_abstract_the_licence_forbids_is_never_read() -> None:
    kit = SourceKit(routes)
    work = Work(
        doi="10.1000/invented.3",
        title="An invented restricted record",
        abstract="A restricted abstract about invented lattices.",
        licence=LICENCE_NO_REDISTRIBUTE,
    )
    found = await read_work(
        work, query=SENTENCE, store=kit.store, source_client=kit.client
    )
    assert found.via == "none" and found.text == "" and found.window == ""


def test_pdfs_are_read_from_their_text_layer() -> None:
    assert READ_SOURCES.pdf_extraction == "fast"
