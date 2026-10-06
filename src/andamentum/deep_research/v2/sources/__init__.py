"""The sources a v2 run searches and reads: one protocol, six adapters.

``build_sources`` turns a ``SourceConfig`` into the adapters for a mode. The
adapters share one ``SourceClient`` (cache, spacing, retry), so the spacing
between requests to one source holds across every caller.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..store import ResearchStore
from ..types import Mode
from .arxiv import ArxivSource
from .base import (
    ExpandingSource,
    FetchedText,
    HttpResult,
    RateInfo,
    SearchFilters,
    SharedPacing,
    Source,
    SourceClient,
    SourceConfig,
    SourceError,
    TextSource,
    process_pacing,
)
from .crossref import CrossrefSource
from .europepmc import EuropePmcSource
from .local_corpus import LocalCorpusSource
from .openalex import OpenAlexSource
from .semantic_scholar import SemanticScholarSource
from .web import WebSource

__all__ = [
    "ArxivSource",
    "CrossrefSource",
    "EuropePmcSource",
    "ExpandingSource",
    "FetchedText",
    "HttpResult",
    "LocalCorpusSource",
    "OpenAlexSource",
    "RateInfo",
    "SearchFilters",
    "SemanticScholarSource",
    "SharedPacing",
    "process_pacing",
    "Source",
    "SourceClient",
    "SourceConfig",
    "SourceError",
    "SourceSet",
    "TextSource",
    "WebSource",
    "build_sources",
]


@dataclass(frozen=True)
class SourceSet:
    """The adapters of one run.

    ``search`` holds those a run's mode searches; ``text`` those that can give a
    work's text, in the order the reader should try them; ``expand`` those that
    follow citation links; ``crossref`` is for verification.
    """

    client: SourceClient
    search: tuple[Source, ...]
    text: tuple[TextSource, ...]
    expand: tuple[ExpandingSource, ...]
    crossref: CrossrefSource


def build_sources(
    config: SourceConfig,
    *,
    store: ResearchStore,
    mode: Mode,
    client: SourceClient | None = None,
) -> SourceSet:
    """The adapters *config* sets up for *mode* (``web``, ``scholarly`` or ``both``).

    Web mode needs ``searxng_url`` and raises a ``config`` error without it.
    The scholarly sources that read text come first (Europe PMC's open-access
    XML, then arXiv, then OpenAlex's open-access links), the web source last.
    ``config.expand_with`` limits which sources follow citation links (``None``:
    all that can); it does not limit which are searched or read.
    """
    shared = client or SourceClient.for_store(store)
    scholarly: list[Source] = []
    for name in config.scholarly:
        if name == "openalex":
            scholarly.append(
                OpenAlexSource(
                    client=shared,
                    api_key=config.openalex_api_key,
                    mailto=config.mailto,
                    mode=config.openalex_search,
                    tdm_allowed_hosts=config.tdm_allowed_hosts,
                )
            )
        elif name == "europepmc":
            scholarly.append(EuropePmcSource(client=shared))
        elif name == "local_corpus":
            if config.local_corpus is None:
                raise SourceError(
                    "local_corpus",
                    "the local_corpus source needs SourceConfig.local_corpus (the "
                    "corpus folder)",
                    kind="config",
                )
            scholarly.append(LocalCorpusSource(corpus_dir=config.local_corpus))
        elif name == "arxiv":
            scholarly.append(
                ArxivSource(
                    client=shared,
                    pdf_interval_s=config.arxiv_pdf_interval_s,
                    query_form=config.arxiv_query,
                )
            )
        else:
            scholarly.append(
                SemanticScholarSource(
                    client=shared,
                    api_key=config.semantic_scholar_api_key,
                    tdm_allowed_hosts=config.tdm_allowed_hosts,
                )
            )
    web: list[Source] = (
        [
            WebSource(
                searxng_url=config.searxng_url,
                tdm_allowed_hosts=config.tdm_allowed_hosts,
                client=shared,
            )
        ]
        if mode in ("web", "both")
        else []
    )
    searched = (scholarly if mode in ("scholarly", "both") else []) + web
    pool = scholarly + web
    reading = {
        "local_corpus": -1,
        "europepmc": 0,
        "arxiv": 1,
        "openalex": 2,
        "semantic_scholar": 3,
        "web": 4,
    }
    return SourceSet(
        client=shared,
        search=tuple(searched),
        text=tuple(
            sorted(
                (s for s in pool if isinstance(s, TextSource)),
                key=lambda s: reading.get(s.name, 9),
            )
        ),
        expand=tuple(
            s
            for s in pool
            if isinstance(s, ExpandingSource)
            and (config.expand_with is None or s.name in config.expand_with)
        ),
        crossref=CrossrefSource(client=shared, mailto=config.mailto),
    )
