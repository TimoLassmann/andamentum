"""The cite loop in the Word document: a rewritten sentence carries a comment, and
the statement on the use of AI says what the loop did, from the counts."""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.compose.harness.docx.reader import open_docx
from andamentum.document_store import Store

from .test_draft_cite import (
    DISCUSSION,
    PAPER,
    QUOTE,
    SUPPORTING,
    SUPPORTING_DOI,
    Tools,
    draft,
)
from andamentum.compose.harness.draft.cite import CiteVerdict

CHANGED = (
    "In one invented trial, logging helped teachers deliver sessions consistently."
)


def _statement(path: Path) -> str:
    view = open_docx(path)
    for paragraph in view.paragraphs:
        if "drafted with the help of an AI system" in paragraph.text:
            return paragraph.text
    raise AssertionError("no statement on the use of AI in the document")


async def test_a_rewritten_sentence_carries_a_comment_with_the_drafted_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "papers.db"
    async with await Store.open(path) as papers:
        tools = Tools(papers, path, found=[SUPPORTING], texts={SUPPORTING_DOI: PAPER})
        result, _ = await draft(
            tmp_path,
            monkeypatch,
            tools,
            verdict=CiteVerdict(supports="after_change", quote=QUOTE, sentence=CHANGED),
        )
    view = open_docx(result.output)
    notes = [c for c in view.comments if DISCUSSION in c.text]
    assert len(notes) == 1
    assert notes[0].text.startswith("Changed to match the cited paper.")
    paragraph = next(p for p in view.paragraphs if p.id == notes[0].paragraph_id)
    assert CHANGED.removesuffix(".") in paragraph.text
    assert paragraph.text[notes[0].start : notes[0].end].startswith("In one invented")


async def test_the_statement_gives_the_counts_of_the_cite_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "papers.db"
    async with await Store.open(path) as papers:
        tools = Tools(papers, path, found=[SUPPORTING], texts={SUPPORTING_DOI: PAPER})
        result, _ = await draft(
            tmp_path,
            monkeypatch,
            tools,
            verdict=CiteVerdict(supports="after_change", quote=QUOTE, sentence=CHANGED),
        )
    text = _statement(result.output)
    assert "checked 3 sentences that cited nothing" in text
    assert "1 sentence was cited" in text and "1 of them changed" in text


async def test_the_statement_claims_no_paper_when_none_was_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "papers.db"
    async with await Store.open(path) as papers:
        tools = Tools(papers, path, found=[], texts={})
        result, _ = await draft(tmp_path, monkeypatch, tools)
    text = _statement(result.output)
    assert "found no supporting paper" in text
    assert "read the papers" not in text
