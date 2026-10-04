from __future__ import annotations

from typing import Any

import pytest

from andamentum.compose.brief import BriefDraft, compile_brief
from andamentum.compose.genres import GENRES
from andamentum.compose.log import ActionLog
from andamentum.compose.schemas import ReaderQuestion, Source

from ._stub import install


def _draft(**overrides) -> BriefDraft:
    base: dict[str, Any] = dict(
        genre="memo",
        title="Apollo pilot",
        audience="Jane Doe",
        purpose="Confirm the pilot date",
        key_messages=["Pilot starts 1 March"],
        must_include=["1 March"],
        constraints=["polite tone"],
        reader_questions=[
            ReaderQuestion(question="When?", expected_answer="1 March"),
            ReaderQuestion(question="Who?", expected_answer="Apollo Labs"),
        ],
    )
    base.update(overrides)
    return BriefDraft(**base)


SRC = Source(id="src1", path="a.md", title="A", sha256="0" * 64, chars=5, kind="file")


async def test_brief_with_fixed_genre_uses_genre_word_range(monkeypatch):
    stub = install(monkeypatch, [_draft()])
    brief = await compile_brief(
        "Write a memo",
        genre=GENRES["memo"],
        sources=[SRC],
        source_text={"src1": "hello"},
        model="m",
    )
    assert brief.genre == "memo"
    assert brief.word_range == GENRES["memo"].word_range
    assert brief.task == "Write a memo"
    assert "src1" in stub.calls[0]["user_message"]
    assert stub.calls[0]["model"] == "m"


async def test_brief_picks_genre_and_honours_length_override(monkeypatch):
    install(
        monkeypatch,
        [
            _draft(
                genre="Email", word_min=80, word_max=120, email_subject=" Pilot date "
            )
        ],
    )
    brief = await compile_brief(
        "Email Jane", genre=None, sources=[], source_text={}, model="m"
    )
    assert brief.genre == "email"
    assert brief.word_range == (80, 120)
    assert brief.email_subject == "Pilot date"


async def test_unknown_genre_raises(monkeypatch):
    install(monkeypatch, [_draft(genre="sonnet")])
    with pytest.raises(Exception, match="sonnet"):
        await compile_brief("x", genre=None, sources=[], source_text={}, model="m")


async def test_too_few_questions_raises(monkeypatch):
    install(
        monkeypatch,
        [
            _draft(
                reader_questions=[
                    ReaderQuestion(question="When?", expected_answer="1 March")
                ]
            )
        ],
    )
    with pytest.raises(ValueError, match="reader"):
        await compile_brief(
            "x", genre=GENRES["memo"], sources=[], source_text={}, model="m"
        )


async def test_non_email_drops_subject_and_logs(monkeypatch, tmp_path):
    install(monkeypatch, [_draft(email_subject="ignored")])
    log = ActionLog(tmp_path)
    brief = await compile_brief(
        "x", genre=GENRES["memo"], sources=[], source_text={}, model="m", log=log
    )
    assert brief.email_subject == ""
    rec = log.records()
    assert len(rec) == 1 and rec[0]["node"] == "brief" and rec[0]["input_tokens"] == 11
