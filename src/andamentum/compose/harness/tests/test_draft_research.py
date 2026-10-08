"""``draft_document`` with research: verified works with findings are cited, the rest is reported.

The research run is a stub with the ``ResearchRun`` shape; ``research_literature``
(the real one, over a mock Crossref) has its own tests below.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from andamentum.compose.evidence import RawFacts
from andamentum.compose.genres import Genre
from andamentum.compose.harness.docx.reader import open_docx
from andamentum.compose.harness.draft import CitableWork, draft_document
from andamentum.compose.harness.draft.search import research_literature
from andamentum.compose.harness.draft.works import ResearchFinding
from andamentum.compose.research import ResearchSetup
from andamentum.compose.write import SectionBody
from andamentum.deep_research import SourceConfig
from andamentum.references import ReferenceHttp

from ..draft.run import DraftResult
from ..draft.verb import DraftParams, run_draft
from ..tests._draft_helpers import (
    ROLES,
    WORK_CSL,
    install_script,
    make_project,
)
from ...tests._research_helpers import FakeResearch, make_result, study

TASK = "Write the SaltLog pilot article"


def _finding(*, with_blog: bool = True, unverified: list[str] | None = None):
    result = make_result(with_blog=with_blog)
    verified = {"w1": CitableWork(csl=WORK_CSL, abstract="", uri=None)}
    notes = (
        [
            "work not used: Apollo cache latency guide (no DOI, so it cannot be verified or cited in Word)"
        ]
        if with_blog
        else []
    )
    return ResearchFinding(
        result=result,
        works=verified,
        unverified=unverified or [],
        notes=notes,
    )


def _run_with(finding: ResearchFinding, calls: list[tuple[str, Genre, Path]]):
    async def run(task: str, *, genre: Genre, database: Path) -> ResearchFinding:
        calls.append((task, genre, database))
        return finding

    return run


async def _write(tmp_path: Path, research) -> DraftResult:  # type: ignore[no-untyped-def]
    notes, results = make_project(tmp_path)
    return await draft_document(
        TASK,
        out=tmp_path / "paper.docx",
        roles=ROLES,
        sources=[str(notes)],
        results=results,
        research=research,
        offline=True,
    )


async def test_a_verified_research_work_is_cited_from_its_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = install_script(monkeypatch)
    calls: list[tuple[str, Genre, Path]] = []
    result = await _write(tmp_path, _run_with(_finding(), calls))

    (task, genre, database) = calls[0]
    assert task == TASK and genre.name == "article"
    assert database == tmp_path / "state" / "research.db"
    # [[f3]] is the research fact (after the notes and the scores file)
    text = open_docx(tmp_path / "paper.docx").text()
    assert "(Okafor 2020)" in text
    assert result.citations == 1
    assert result.works == ["doi:10.5555/saltlog.2020.1"]
    # the findings were not sent through the extraction model
    assert script.count(RawFacts) == 2
    drafts = [m for t, m in script.calls if t is SectionBody]
    assert any("Another published work (Structured session logs" in m for m in drafts)


async def test_the_work_without_a_doi_is_reported_not_cited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_script(monkeypatch)
    result = await _write(tmp_path, _run_with(_finding(), []))
    assert any("no DOI" in note for note in result.notes)
    assert "Apollo" not in open_docx(tmp_path / "paper.docx").text()


async def test_an_unverified_research_work_is_an_author_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_script(monkeypatch)
    line = "A study that does not exist (2019): no such work in Crossref"
    result = await _write(tmp_path, _run_with(_finding(unverified=[line]), []))
    assert any(line in q for q in result.author_questions)
    assert result.exit_code == 1


async def test_the_statement_says_what_searched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_script(monkeypatch)
    result = await _write(tmp_path, _run_with(_finding(), []))
    assert (
        "Further evidence was found by searching scholarly databases"
        in result.statement
    )
    assert "stub:drafter planned the search" in result.statement
    assert "screened" not in result.statement


async def test_without_research_the_statement_is_silent_about_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_script(monkeypatch)
    result = await _write(tmp_path, None)
    assert "Further evidence" not in result.statement


# -- the real research_literature over a mock Crossref -------------------------------


def _crossref(doi_known: bool) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host != "api.crossref.org":
            return httpx.Response(404)
        if not doi_known:
            return httpx.Response(404, json={"status": "failed"})
        message = {
            "DOI": "10.5555/saltlog.2020.1",
            "title": [study().title],
            "author": [{"family": "Okafor", "given": "Ada"}],
            "issued": {"date-parts": [[2020]]},
            "container-title": ["Journal of Invented Studies"],
            "type": "journal-article",
        }
        return httpx.Response(200, json={"message": message})

    return httpx.MockTransport(handler)


def _http(doi_known: bool) -> ReferenceHttp:
    return ReferenceHttp(
        client=httpx.AsyncClient(transport=_crossref(doi_known)),
        url_check=lambda url: (True, ""),
    )


async def _research_literature(doi_known: bool, monkeypatch: pytest.MonkeyPatch):
    fake = FakeResearch()
    monkeypatch.setattr("andamentum.deep_research.research", fake)
    run = research_literature(
        setup=ResearchSetup(sources=SourceConfig(), mode="scholarly"),
        model="stub:drafter",
        http=_http(doi_known),
        progress=lambda _line: None,
    )
    from andamentum.compose.genres import GENRES

    return await run(TASK, genre=GENRES["article"], database=Path("research.db")), fake


async def test_research_literature_verifies_each_work_with_a_finding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finding, fake = await _research_literature(True, monkeypatch)
    assert fake.calls[0]["mode"] == "scholarly"
    assert "decision_model" not in fake.calls[0]
    assert list(finding.works) == ["w1"]  # the blog post has no DOI
    assert finding.works["w1"].csl["DOI"] == "10.5555/saltlog.2020.1"
    assert finding.unverified == []
    assert any(
        "Apollo cache latency guide" in n and "no DOI" in n for n in finding.notes
    )


async def test_research_literature_never_cites_a_work_crossref_does_not_know(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finding, _ = await _research_literature(False, monkeypatch)
    assert finding.works == {}
    assert (
        len(finding.unverified) == 1
        and "Structured session logs" in finding.unverified[0]
    )


# -- the verb ---------------------------------------------------------------------------


async def test_the_word_verb_replaces_crossref_discovery_with_research(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_script(monkeypatch)
    fake = FakeResearch()
    monkeypatch.setattr("andamentum.deep_research.research", fake)
    notes, results = make_project(tmp_path)
    crossref_search_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("query.bibliographic") or request.url.params.get(
            "query"
        ):
            crossref_search_calls.append(str(request.url))
        return _crossref(True).handler(request)  # type: ignore[attr-defined,return-value]

    result = await run_draft(
        DraftParams(
            task=TASK,
            out=tmp_path / "paper.docx",
            model="stub:model",
            source=[str(notes)],
            results=results,
            research=True,
            research_mode="scholarly",
        ),
        transport=httpx.MockTransport(handler),
        url_check=lambda url: (True, ""),
    )
    assert fake.calls[0]["mode"] == "scholarly"
    assert crossref_search_calls == []  # Crossref only verifies; it no longer discovers
    assert result.citations == 1
