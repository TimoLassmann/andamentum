"""The cite loop: sentences that cite nothing get a paper that was read and supports them.

Offline: the model is the SaltLog ``Script``, the finder, reader and
verification are stand-ins, and the document store is a real one in tmp_path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.compose.harness.draft.cite import (
    MAX_SENTENCES,
    READS_PER_SENTENCE,
    CiteTools,
    CiteVerdict,
    paper_key,
)
from andamentum.compose.harness.draft.run import draft_document
from andamentum.compose.harness.draft.works import CitableWork
from andamentum.deep_research import WorkText
from andamentum.document_store import Store
from andamentum.references import Work

from ._draft_helpers import (
    ROLES,
    default_script,
    install_script,
    make_project,
    saltlog_literature,
)

TASK = "Write the SaltLog pilot article"
DISCUSSION = (
    "The gain suggests that logging helps teachers deliver sessions consistently."
)
QUOTE = "structured logging helped teachers deliver sessions consistently"
PAPER = (
    "# An invented trial of session logs\n\nIn an invented trial in twelve schools, "
    f"{QUOTE} across the school year. Reading outcomes were not measured.\n"
)
SUPPORTING_DOI = "10.5555/logs.2021.7"
OTHER_DOI = "10.5555/other.2018.2"
SUPPORTING = Work(
    doi=SUPPORTING_DOI,
    title="An invented trial of session logs",
    authors=["Jane Doe"],
    year=2021,
)
OTHER = Work(doi=OTHER_DOI, title="An invented unrelated study", year=2018)


def csl_of(work: Work) -> dict[str, object]:
    return {
        "id": work.doi,
        "type": "article-journal",
        "title": work.title,
        "author": [{"family": "Doe", "given": "Jane"}],
        "issued": {"date-parts": [[work.year]]},
        "DOI": work.doi,
    }


class Tools:
    """Stand-in tools that record what the loop asked for."""

    def __init__(
        self,
        papers: Store,
        store_path: Path,
        *,
        found: list[Work],
        texts: dict[str, str],
        library: list[Work] | None = None,
        refuse: set[str] = frozenset(),  # type: ignore[assignment]
    ) -> None:
        self.found, self.texts, self.refuse = found, texts, refuse
        self.library_works = library
        self.read_calls: list[str] = []
        self.verified: list[str] = []
        self.tools = CiteTools(
            library=self.library if library is not None else None,
            finder=self.finder,
            read=self.read,
            verify=self.verify,
            papers=papers,
            store_path=store_path,
        )

    async def library(self, sentence: str) -> list[Work]:
        return list(self.library_works or [])

    async def finder(self, sentence: str, *, paragraph: str) -> list[Work]:
        return list(self.found)

    async def read(self, work: Work, *, query: str) -> WorkText:
        self.read_calls.append(work.doi or work.title)
        text = self.texts.get(work.doi or "", "")
        return WorkText(
            work=work,
            text=text,
            window=text,
            via="fetched" if text else "none",
            url=f"https://example.org/{work.doi}" if text else "",
        )

    async def verify(self, work: Work) -> CitableWork | str:
        self.verified.append(work.doi or "")
        if work.doi in self.refuse:
            return f"{work.title}: retracted"
        return CitableWork(csl=csl_of(work), abstract="")


def verdicts(**by_title: CiteVerdict):  # noqa: ANN201
    """A CiteVerdict handler: the verdict for the paper the message names."""

    def answer(message: str) -> CiteVerdict:
        for title, verdict in by_title.items():
            if title in message and f"SENTENCE: {DISCUSSION}" in message:
                return verdict
        return CiteVerdict(supports="no")

    return answer


SUPPORTS = CiteVerdict(supports="yes", quote=QUOTE)


async def draft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tools: Tools,
    *,
    verdict: CiteVerdict | None = None,
):  # noqa: ANN201
    handlers = default_script()
    handlers[CiteVerdict] = verdicts(**{SUPPORTING.title: verdict or SUPPORTS})
    script = install_script(monkeypatch, handlers)
    notes, results = make_project(tmp_path)
    result = await draft_document(
        TASK,
        out=tmp_path / "paper.docx",
        roles=ROLES,
        sources=[str(notes)],
        results=results,
        literature=saltlog_literature(),
        cite=tools.tools,
        offline=True,
    )
    return result, script


async def test_a_supporting_paper_is_read_stored_and_cited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "papers.db"
    async with await Store.open(path) as papers:
        tools = Tools(papers, path, found=[SUPPORTING], texts={SUPPORTING_DOI: PAPER})
        result, script = await draft(tmp_path, monkeypatch, tools)
        assert script.count(CiteVerdict) >= 1
        stored = await papers.list(limit=10)
        assert len(stored) == 1
        doc = await papers.get(stored[0].doc_id)
        assert doc is not None and doc.key == f"doi:{SUPPORTING_DOI}"
        assert doc.text == PAPER and doc.metadata["title"] == SUPPORTING.title
    report = result.report.read_text(encoding="utf-8")
    assert "cite loop:" in report and "1 added to" in report
    assert SUPPORTING_DOI in tools.verified
    assert any(SUPPORTING_DOI in w for w in result.works)


async def test_only_papers_read_go_into_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed_doi = "10.5555/closed.2019.1"
    unreadable = Work(doi=closed_doi, title="An invented closed paper")
    path = tmp_path / "papers.db"
    async with await Store.open(path) as papers:
        tools = Tools(
            papers,
            path,
            found=[unreadable, SUPPORTING, OTHER],
            texts={SUPPORTING_DOI: PAPER},
        )
        await draft(tmp_path, monkeypatch, tools)
        keys = {d.key for d in await papers.list(limit=10)}
    assert keys == {f"doi:{SUPPORTING_DOI}"}
    assert closed_doi in tools.read_calls  # tried, but nothing to read: not stored


async def test_the_library_is_searched_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "papers.db"
    async with await Store.open(path) as papers:
        tools = Tools(
            papers,
            path,
            found=[OTHER],
            texts={SUPPORTING_DOI: PAPER, OTHER_DOI: "An unrelated invented text."},
            library=[SUPPORTING],
        )
        await draft(tmp_path, monkeypatch, tools)
    assert tools.read_calls[0] == SUPPORTING_DOI


async def test_a_paper_already_in_the_store_is_not_fetched_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "existing.db"
    async with await Store.open(path) as papers:
        await papers.put(PAPER, key=paper_key(SUPPORTING), metadata={"title": "x"})
    async with await Store.open(path) as papers:  # an existing store is reused
        tools = Tools(papers, path, found=[SUPPORTING], texts={})
        result, _ = await draft(tmp_path, monkeypatch, tools)
        assert await papers.count() == 1
    assert tools.read_calls == []
    # both candidate sentences took the stored paper; nothing was fetched
    assert "0 added to" in result.report.read_text(encoding="utf-8")
    assert "2 taken from it" in result.report.read_text(encoding="utf-8")


async def test_a_new_store_path_makes_a_new_store(tmp_path: Path) -> None:
    path = tmp_path / "new" / "folder" / "papers.db"
    async with await Store.open(path) as papers:
        assert await papers.count() == 0
    assert path.exists()


async def test_a_work_that_fails_verification_is_never_cited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "papers.db"
    async with await Store.open(path) as papers:
        tools = Tools(
            papers,
            path,
            found=[SUPPORTING],
            texts={SUPPORTING_DOI: PAPER},
            refuse={SUPPORTING_DOI},
        )
        result, _ = await draft(tmp_path, monkeypatch, tools)
    report = result.report.read_text(encoding="utf-8")
    assert "could not be verified" in report and "retracted" in report
    assert not any(SUPPORTING_DOI in w for w in result.works)


async def test_a_quote_that_is_not_in_the_paper_is_not_cited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "papers.db"
    async with await Store.open(path) as papers:
        tools = Tools(papers, path, found=[SUPPORTING], texts={SUPPORTING_DOI: PAPER})
        result, _ = await draft(
            tmp_path,
            monkeypatch,
            tools,
            verdict=CiteVerdict(supports="yes", quote="words the paper never says"),
        )
    assert "the supporting quote is not in the paper" in result.report.read_text(
        encoding="utf-8"
    )
    assert tools.verified == []


async def test_a_sentence_is_changed_to_match_the_paper_and_keeps_its_citation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changed = (
        "In one invented trial, logging helped teachers deliver sessions consistently."
    )
    path = tmp_path / "papers.db"
    async with await Store.open(path) as papers:
        tools = Tools(papers, path, found=[SUPPORTING], texts={SUPPORTING_DOI: PAPER})
        result, _ = await draft(
            tmp_path,
            monkeypatch,
            tools,
            verdict=CiteVerdict(supports="after_change", quote=QUOTE, sentence=changed),
        )
    report = result.report.read_text(encoding="utf-8")
    assert f"changed to match the paper: {DISCUSSION} -> {changed}" in report
    assert any(SUPPORTING_DOI in w for w in result.works)


def test_the_loop_is_bounded() -> None:
    assert MAX_SENTENCES == 8 and READS_PER_SENTENCE == 2


async def test_a_paper_that_supports_two_sentences_is_one_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handlers = default_script()
    handlers[CiteVerdict] = lambda _m: SUPPORTS
    install_script(monkeypatch, handlers)
    notes, results = make_project(tmp_path)
    path = tmp_path / "papers.db"
    async with await Store.open(path) as papers:
        tools = Tools(papers, path, found=[SUPPORTING], texts={SUPPORTING_DOI: PAPER})
        result = await draft_document(
            TASK,
            out=tmp_path / "paper.docx",
            roles=ROLES,
            sources=[str(notes)],
            results=results,
            literature=saltlog_literature(),
            cite=tools.tools,
            offline=True,
        )
        assert await papers.count() == 1
    report = result.report.read_text(encoding="utf-8")
    assert (
        "2 cited" in report and "1 added to" in report and "1 taken from it" in report
    )
    assert sum(1 for w in result.works if SUPPORTING_DOI in w) == 1
