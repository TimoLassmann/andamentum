from __future__ import annotations

from andamentum.compose.evals.baseline import (
    BaselineDocument,
    BaselineSection,
    single_call,
)
from andamentum.compose.genres import GENRES
from andamentum.compose.schemas import Fact

from ._stub import install


async def test_single_call_returns_document_and_uses_one_call(monkeypatch):
    doc = BaselineDocument(
        title="T", sections=[BaselineSection(title="Body", body="Hi.")]
    )
    stub = install(monkeypatch, [doc])
    out = await single_call(
        "Email Jane Doe",
        {"src1": "Apollo Labs pilot"},
        GENRES["email"],
        model="drafter-model",
        facts=[
            Fact(
                id="f1", source_id="src1", statement="Pilot", quote="q", start=0, end=1
            )
        ],
    )
    assert out == doc
    assert len(stub.calls) == 1
    assert stub.calls[0]["model"] == "drafter-model"
    assert "Apollo Labs pilot" in stub.calls[0]["user_message"]
    assert "Body" in stub.calls[0]["user_message"]
    # the baseline cites with the same marker rules and fact list as compose
    assert "[[f12]]" in stub.calls[0]["instructions"]
    assert "f1: Pilot" in stub.calls[0]["user_message"]
