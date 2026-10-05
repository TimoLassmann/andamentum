"""``python -m andamentum.core.decisions``: check (stubbed server)."""

from __future__ import annotations

from andamentum.core.decisions import Question
from andamentum.core.decisions.cli import main
from andamentum.core.decisions.testing import StubDecisions, noul_answer

QUESTION = Question(type="noul", instructions="Is it a claim?")


def server() -> StubDecisions:
    return StubDecisions(
        lambda state, questions: {
            k: noul_answer(0.95 if "yes" in str(state) else 0.05) for k in questions
        }
    )


def test_check_prints_server_capability_digest_and_a_smoke_answer(capsys) -> None:
    stub = server()
    assert (
        main(
            ["check", "--model", "nimble:9b", "--base-url", "http://stub.invalid"],
            transport=stub.transport(),
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "nimble:9b" in out and "decision" in out and stub.digest in out
    assert "p(true)" in out and len(stub.calls) == 1


def test_check_reports_a_missing_model(capsys) -> None:
    stub = server()
    code = main(["check", "--model", "absent:1b"], transport=stub.transport())
    assert code == 1
    assert "ollama pull absent:1b" in capsys.readouterr().err
