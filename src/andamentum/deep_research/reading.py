"""``read_work``: one work's text, whole, and the part of it relevant to a query.

The reading half of a citation step: ``find_works`` names candidate works,
``read_work`` gets one of them to read. It asks the text sources in the
engine's reading order (Europe PMC full text, arXiv, the open-access copy
OpenAlex or Semantic Scholar names) and takes the first that holds the text,
with the engine's rules: a source's failure is a note, and an abstract the
licence forbids (Semantic Scholar's) is never used. Without full text the
work's own abstract stands in when it may be used. No model is called.

``window`` is the part of the text most relevant to *query* (BM25 passages in
document order, at most ``window_chars``; the whole text when it fits), the
part a model reads to decide whether the work supports a sentence.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from andamentum.references import Work, WorkAccess

from .read import build_window, provider_text
from .sources import SourceClient, SourceConfig, build_sources
from .store import ResearchStore
from .types import DEFAULT_WINDOW_CHARS, PageVia

# The text sources read_work asks, in the engine's reading order. PDFs are read
# from their text layer: the layout model is not loaded for one citation.
READ_SOURCES = SourceConfig(
    scholarly=("europepmc", "arxiv", "openalex", "semantic_scholar"),
    pdf_extraction="fast",
)
WINDOW_GAP = "\n\n[...]\n\n"


class WorkText(BaseModel):
    """What ``read_work`` found for one work.

    ``text`` is the whole text ("" when none could be read), ``window`` the part
    relevant to the query, ``via`` how it was got (``fetched`` full text,
    ``abstract``, or ``none``), ``url`` where the full text came from.
    """

    work: Work
    text: str = ""
    window: str = ""
    access: WorkAccess = "metadata"
    via: PageVia = "none"
    url: str = ""
    notes: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)


async def read_work(
    work: Work,
    *,
    query: str,
    sources: SourceConfig = READ_SOURCES,
    store: ResearchStore | None = None,
    http_cache: ResearchStore | None = None,
    source_client: SourceClient | None = None,
    window_chars: int = DEFAULT_WINDOW_CHARS,
) -> WorkText:
    """The text of *work* and its part most relevant to *query* (see the module
    docstring). ``sources`` names the text sources (``scholarly``), the keys and
    the PDF extraction. ``store`` is an open research store whose HTTP cache
    answers repeated requests (``None``: an in-memory one for this call, reading
    through ``http_cache`` when given); ``source_client`` must belong to it."""
    if source_client is not None and source_client.store is not store:
        raise ValueError("source_client must belong to the open store passed as store")
    if http_cache is not None and store is not None:
        raise ValueError(
            "http_cache goes to a store this call opens: open the store with "
            "shared_cache= instead"
        )
    opened = store or ResearchStore.in_memory(shared_cache=http_cache)
    client = source_client or SourceClient(
        opened, pdf_extraction=sources.pdf_extraction
    )
    try:
        source_set = build_sources(
            sources, store=opened, mode="scholarly", client=client
        )
        found, _ = await provider_text(work, source_set.text)
    finally:
        if source_client is None:
            await client.aclose()
        if store is None:
            opened.close()
    result = WorkText(
        work=work,
        text=found.text,
        access=found.access,
        via=found.via,
        url=found.url,
        notes=found.notes,
        failed=found.failed,
    )
    if not result.text and work.abstract.strip() and work.redistributable:
        result.text, result.via, result.access = work.abstract, "abstract", "abstract"
    if result.text:
        passages = build_window(
            result.text,
            work_id=work.work_id or "work",
            query=query,
            max_chars=window_chars,
        )
        result.window = WINDOW_GAP.join(p.text for p in passages)
    return result
