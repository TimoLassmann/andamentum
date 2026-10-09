"""The cite loop: sentences that cite nothing get a paper that was read and supports them.

Offline: the model is the SaltLog ``Script``, the finder, reader and
verification are stand-ins, and the document store is a real one in tmp_path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.compose.harness.draft.cite import (
    MIN_QUOTE_WORDS,
    READS_PER_SENTENCE,
    CiteTools,
    CiteVerdict,
    paper_key,
)
from andamentum.compose.harness.literature import LiteratureError
from andamentum.compose import store as bundle_store
from andamentum.compose.evidence import stored_source_path
from andamentum.compose.write import SectionBody
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
from ._draft_helpers import BODIES, WORK_CSL, _section_of

TASK = "Write the SaltLog pilot article"
DISCUSSION = (
    "The gain suggests that logging helps teachers deliver sessions consistently."
)
QUOTE = "structured session logging helped teachers deliver sessions consistently"
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
    discussion: str | None = None,
    answer: object = None,
):  # noqa: ANN201
    handlers = default_script()
    if discussion is not None:
        handlers[SectionBody] = lambda m: SectionBody(
            body=discussion
            if _section_of(m) == "Discussion"
            else BODIES[_section_of(m)]
        )
    handlers[CiteVerdict] = answer or verdicts(  # type: ignore[assignment]
        **{SUPPORTING.title: verdict or SUPPORTS}
    )
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
            verdict=CiteVerdict(
                supports="yes", quote="words that the paper never says anywhere at all"
            ),
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


def docx_paragraphs(path: Path) -> list[str]:
    """Paragraph texts including tracked insertions (python-docx skips those)."""
    from docx import Document
    from docx.oxml.ns import qn

    return [
        "".join(t.text or "" for t in p._p.iter(qn("w:t"), qn("w:delText")))
        for p in Document(str(path)).paragraphs
    ]


async def test_a_work_made_a_source_by_the_literature_step_gets_its_own_full_text_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doi = str(WORK_CSL["DOI"])
    work = Work(doi=doi, title=str(WORK_CSL["title"]), year=2020)
    path = tmp_path / "papers.db"
    async with await Store.open(path) as papers:
        tools = Tools(papers, path, found=[work], texts={doi: PAPER})
        tools.verify = _verify_as(WORK_CSL)  # type: ignore[method-assign]
        tools.tools.verify = tools.verify
        result, _ = await draft(
            tmp_path,
            monkeypatch,
            tools,
            answer=lambda m: (
                SUPPORTS
                if f"SENTENCE: {DISCUSSION}" in m
                else CiteVerdict(supports="no")
            ),
        )
    project = result.project / "compose"
    sources = {s.id: s for s in bundle_store.load_sources(project)}
    facts = [f for f in bundle_store.load_facts(project) if f.quote == QUOTE]
    assert facts
    for fact in facts:
        source = sources[fact.source_id]
        text = stored_source_path(project, source).read_text(encoding="utf-8")
        assert text[fact.start : fact.end] == fact.quote


def _verify_as(csl: dict[str, object]):  # noqa: ANN202
    async def verify(work: Work) -> CitableWork | str:
        return CitableWork(csl=csl, abstract="")

    return verify


async def test_a_library_outage_is_a_note_and_the_finder_is_still_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "papers.db"
    async with await Store.open(path) as papers:
        tools = Tools(
            papers,
            path,
            found=[SUPPORTING],
            texts={SUPPORTING_DOI: PAPER},
            library=[],
        )

        async def broken(sentence: str) -> list[Work]:
            raise LiteratureError("library offline")

        tools.tools.library = broken
        result, _ = await draft(tmp_path, monkeypatch, tools)
    report = result.report.read_text(encoding="utf-8")
    assert report.count("library search failed:") == 1  # once, not once per sentence
    assert SUPPORTING_DOI in tools.read_calls
    assert any(SUPPORTING_DOI in w for w in result.works)


async def test_source_notes_from_the_tools_reach_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "papers.db"
    async with await Store.open(path) as papers:
        tools = Tools(papers, path, found=[], texts={})
        tools.tools.source_notes.append("scholarly search is down")
        result, _ = await draft(tmp_path, monkeypatch, tools)
    assert "scholarly search is down" in result.report.read_text(encoding="utf-8")
    assert tools.tools.source_notes == []


async def test_a_sentence_with_a_placeholder_is_not_a_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = "The gain suggests that [TO FILL: the size of the gain] helps teachers."
    path = tmp_path / "papers.db"
    async with await Store.open(path) as papers:
        tools = Tools(papers, path, found=[SUPPORTING], texts={SUPPORTING_DOI: PAPER})
        result, script = await draft(
            tmp_path,
            monkeypatch,
            tools,
            discussion=body,
            answer=lambda m: SUPPORTS,
        )
    assert all("[TO FILL" not in w for w in result.works)
    assert any(
        "[TO FILL: the size of the gain]" in p for p in docx_paragraphs(result.output)
    )
    assert not any(
        "TO FILL" in message for kind, message in script.calls if kind is CiteVerdict
    )


async def test_a_sentence_that_is_the_tail_of_a_longer_one_is_cited_in_its_own_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tail = (
        "the gain suggests that logging helps teachers deliver sessions consistently."
    )
    body = f"Overall, {tail}\n\n{tail}"
    path = tmp_path / "papers.db"
    async with await Store.open(path) as papers:
        tools = Tools(papers, path, found=[SUPPORTING], texts={SUPPORTING_DOI: PAPER})
        result, _ = await draft(
            tmp_path,
            monkeypatch,
            tools,
            discussion=body,
            answer=lambda m: (
                SUPPORTS if f"SENTENCE: {tail}" in m else CiteVerdict(supports="no")
            ),
        )
    paragraphs = docx_paragraphs(result.output)
    first = next(p for p in paragraphs if p.startswith("Overall,"))
    second = next(p for p in paragraphs if p.startswith("the gain suggests"))
    assert "Doe" not in first and "Doe" in second


async def test_a_sentence_wrapped_over_two_lines_is_cited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = (
        "The gain suggests that logging helps\nteachers deliver sessions consistently."
    )
    path = tmp_path / "papers.db"
    async with await Store.open(path) as papers:
        tools = Tools(papers, path, found=[SUPPORTING], texts={SUPPORTING_DOI: PAPER})
        result, _ = await draft(tmp_path, monkeypatch, tools, discussion=body)
    assert any(SUPPORTING_DOI in w for w in result.works)


async def test_a_short_quote_is_not_enough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    short = "helped teachers deliver"
    assert len(short.split()) < MIN_QUOTE_WORDS
    path = tmp_path / "papers.db"
    async with await Store.open(path) as papers:
        tools = Tools(papers, path, found=[SUPPORTING], texts={SUPPORTING_DOI: PAPER})
        result, _ = await draft(
            tmp_path,
            monkeypatch,
            tools,
            verdict=CiteVerdict(supports="yes", quote=short),
        )
    assert "quote too short to show support" in result.report.read_text(
        encoding="utf-8"
    )
    assert tools.verified == []


async def test_a_changed_sentence_is_recorded_as_a_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from andamentum.compose.harness.draft.cite import cite_sentences

    from andamentum.compose.harness.draft import cite as cite_module

    seen: list[object] = []
    real = cite_module.cite_sentences

    async def spy(*args: object, **kwargs: object):  # noqa: ANN202
        outcome = await real(*args, **kwargs)  # type: ignore[arg-type]
        seen.append(outcome)
        return outcome

    monkeypatch.setattr("andamentum.compose.harness.draft.run.cite_sentences", spy)
    assert cite_sentences is real
    changed = (
        "In one invented trial, logging helped teachers deliver sessions consistently."
    )
    path = tmp_path / "papers.db"
    async with await Store.open(path) as papers:
        tools = Tools(papers, path, found=[SUPPORTING], texts={SUPPORTING_DOI: PAPER})
        await draft(
            tmp_path,
            monkeypatch,
            tools,
            verdict=CiteVerdict(supports="after_change", quote=QUOTE, sentence=changed),
        )
    (outcome,) = seen
    rewrites = outcome.rewrites  # type: ignore[attr-defined]
    assert [(r.section_key, r.old, r.new) for r in rewrites] == [
        (rewrites[0].section_key, DISCUSSION, changed)
    ]
    assert rewrites[0].section_key


async def test_a_paper_the_judge_rejects_leaves_the_sentence_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    third = Work(doi="10.5555/third.2017.3", title="An invented third study", year=2017)
    path = tmp_path / "papers.db"
    async with await Store.open(path) as papers:
        tools = Tools(
            papers,
            path,
            found=[SUPPORTING, OTHER, third],
            texts={
                SUPPORTING_DOI: PAPER,
                OTHER_DOI: "An unrelated invented text.",
                "10.5555/third.2017.3": "Another unrelated invented text.",
            },
        )
        result, _ = await draft(
            tmp_path, monkeypatch, tools, answer=lambda m: CiteVerdict(supports="no")
        )
    report = result.report.read_text(encoding="utf-8")
    assert f"no citation found for: {DISCUSSION}" in report
    assert len(tools.read_calls) <= 2 * READS_PER_SENTENCE  # two candidate sentences
    assert tools.verified == []
    assert any(DISCUSSION in p for p in docx_paragraphs(result.output))


async def test_half_of_a_placeholder_split_at_its_full_stop_is_not_a_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = "Logging helps [TO FILL: gain size. It suggests a gain] teachers deliver sessions."
    path = tmp_path / "papers.db"
    async with await Store.open(path) as papers:
        tools = Tools(papers, path, found=[SUPPORTING], texts={SUPPORTING_DOI: PAPER})
        result, script = await draft(
            tmp_path, monkeypatch, tools, discussion=body, answer=lambda m: SUPPORTS
        )
    assert not any(
        "It suggests a gain" in message
        for kind, message in script.calls
        if kind is CiteVerdict
    )
    assert any(
        "[TO FILL: gain size. It suggests a gain]" in p
        for p in docx_paragraphs(result.output)
    )


async def test_a_changed_sentence_that_is_not_one_plain_sentence_is_not_cited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "papers.db"
    async with await Store.open(path) as papers:
        tools = Tools(papers, path, found=[SUPPORTING], texts={SUPPORTING_DOI: PAPER})
        result, _ = await draft(
            tmp_path,
            monkeypatch,
            tools,
            verdict=CiteVerdict(
                supports="after_change",
                quote=QUOTE,
                sentence="Logging helped teachers [[f1]] deliver sessions.",
            ),
        )
    report = result.report.read_text(encoding="utf-8")
    assert "the changed sentence is not one plain sentence" in report
    assert not any(SUPPORTING_DOI in w for w in result.works)
