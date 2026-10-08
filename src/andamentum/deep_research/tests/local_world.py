"""A tiny invented local corpus for the local-corpus tests: the engine world's
papers (Apollo cache, SaltLog logging) as records of a corpus folder, plus a few
records that match nothing, one with no abstract and one with neither title nor
abstract. Synthetic; no real paper."""

from __future__ import annotations

from pathlib import Path

from ..sources.local_corpus import CORPUS_FILE, CorpusRecord, build_index
from .engine_world import FOLLOW_UP, PAPERS

# the corpus id of each engine-world paper: its OpenAlex number
APOLLO_A = "1001"
APOLLO_B = "1002"
APOLLO_X = "1003"
SALTLOG_D = "1004"
SALTLOG_E = "1005"
FOLLOW = "1006"
GARDEN = "2001"
NO_ABSTRACT = "2002"
EMPTY = "2003"


def records() -> list[CorpusRecord]:
    papers = [*PAPERS, FOLLOW_UP]
    cites = {
        "W1006": ["1001"],
        "W1002": ["1001", "9999"],  # 9999 is not in the corpus
        "W1005": ["1004"],
    }
    out = [
        CorpusRecord(
            corpus_id=p.openalex_id.removeprefix("W"),
            title=p.title,
            abstract=p.abstract,
            citations=cites.get(p.openalex_id, []),
        )
        for p in papers
    ]
    out += [
        CorpusRecord(
            corpus_id=GARDEN,
            title="Pruning roses in cold climates",
            abstract="A survey of pruning dates for roses grown in cold climates.",
        ),
        CorpusRecord(corpus_id=NO_ABSTRACT, title="An untitled note on cache sizing"),
        CorpusRecord(corpus_id=EMPTY, citations=["1001"]),
    ]
    return out


def write_corpus(folder: Path, *, index: bool = True) -> Path:
    """Write ``corpus.jsonl`` (and, by default, its index) into *folder*."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / CORPUS_FILE).write_text(
        "".join(r.model_dump_json() + "\n" for r in records()), encoding="utf-8"
    )
    if index:
        build_index(folder)
    return folder
