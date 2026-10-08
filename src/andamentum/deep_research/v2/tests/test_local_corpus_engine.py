"""The v2 engine end to end over a local corpus: no network at all.

The corpus is the engine world's papers as records of a folder (``local_world``);
the language model is the world's stand-in. What is checked is that the whole
run (plan, search, choose, citation expansion, read, reflect, write, verify)
works when the local corpus is the only source, that the folder is not recorded
in the run, and that nothing reaches a network.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from .. import research
from ..sources import SourceConfig
from ..sources.base import SourceClient
from ..store import ResearchStore
from ..types import Budget
from .engine_world import QUESTION, world_llm
from .local_world import FOLLOW, write_corpus


async def test_a_run_over_the_local_corpus_alone(tmp_path: Path) -> None:
    corpus = write_corpus(tmp_path / "corpus")
    requests: list[httpx.Request] = []

    def no_network(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500, text="no network in this test")

    async def no_sleep(_seconds: float) -> None:
        return None

    store = ResearchStore.in_memory()
    client = SourceClient(
        store,
        client=httpx.AsyncClient(transport=httpx.MockTransport(no_network)),
        sleep=no_sleep,
    )
    result = await research(
        QUESTION,
        mode="scholarly",
        model=world_llm().model,
        store=store,
        budget=Budget(),
        sources=SourceConfig(
            scholarly=("local_corpus",),
            expand_with=("local_corpus",),
            local_corpus=corpus,
        ),
        source_client=client,
    )

    assert requests == []
    assert result.degraded == []
    assert result.stop_reason == "answered"
    assert result.works and all(w.local_id for w in result.works)
    assert all(not w.doi and not w.url for w in result.works)
    assert result.findings
    passages = {p.passage_id: p for p in result.passages}
    for finding in result.findings:
        assert finding.quote in passages[finding.passage_id].text
    # the abstracts were read as text: every finding rests on a record read as an abstract
    stored = {w.work_id: w for w in store.iter_works()}
    assert all(stored[f.work_id].local_id for f in result.findings)
    assert all(stored[f.work_id].access == "abstract" for f in result.findings)
    # citation expansion found the follow-up study through the corpus' own links
    follow_up = [w for w in result.works if w.local_id == f"local:{FOLLOW}"]
    assert follow_up and follow_up[0].work_id in {f.work_id for f in result.findings}
    assert "local_corpus" not in store.get_run(result.run_id).config["sources"]  # type: ignore[union-attr]
