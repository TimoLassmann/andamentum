"""Stage 3 with a decision model replacing the LLM judge (stubbed, offline)."""

from __future__ import annotations

from andamentum.chunker import cli
from andamentum.chunker.extractor import extract_units
from andamentum.chunker.judge import MERGE_QUESTION, JudgeVerdict, judge_cut_decided
from andamentum.chunker.types import ChunkingResult
from andamentum.core.decisions import Answer

SRC = "## Section\n\n" + "Filler text " * 200 + "\n\n" + "More filler " * 200


class StubDecisionModel:
    def __init__(self, p_true: float | None = None, *, fail: bool = False) -> None:
        self.p_true = p_true
        self.fail = fail
        self.calls: list[tuple[object, dict]] = []

    async def decide(self, state, questions):
        self.calls.append((state, questions))
        if self.fail:
            raise RuntimeError("server down")
        p = self.p_true
        return {
            k: Answer(
                choice=None if p is None else ("yes" if p >= 0.5 else "no"),
                probabilities={} if p is None else {"yes": p, "no": 1 - p},
                confidence=0.9,
                p_true=p,
            )
            for k in questions
        }


async def _embed(texts):
    return [[1.0, 0.0] if i == 0 else [0.0, 1.0] for i, _ in enumerate(texts)]


async def _run(**kwargs):
    return await extract_units(
        SRC,
        target_min_chars=500,
        target_max_chars=2_000,
        embedding_fn=_embed,
        judge_low_pct=0.0,
        judge_high_pct=1.0,
        **kwargs,
    )


async def test_without_decision_model_the_llm_judge_is_used():
    asked = []

    async def executor(*, instructions, user_message, output_type, validators):
        asked.append(user_message)
        return output_type(decision="keep")

    result = await _run(judge_executor=executor)

    assert asked and result.model_calls == len(asked)
    assert result.decision_calls == 0


async def test_decision_model_replaces_the_llm():
    asked = []

    async def executor(*, instructions, user_message, output_type, validators):
        asked.append(user_message)
        return output_type(decision="keep")

    baseline = await _run(judge_executor=executor)
    asked.clear()
    stub = StubDecisionModel(0.9)
    result = await _run(judge_executor=executor, decision_model=stub)  # type: ignore[arg-type]

    assert asked == []
    assert result.model_calls == 0
    assert result.decision_calls == len(stub.calls) > 0
    assert len(result.units) < len(baseline.units)


async def test_decision_model_alone_runs_stage_three():
    stub = StubDecisionModel(0.1)
    result = await _run(decision_model=stub)  # type: ignore[arg-type]

    assert stub.calls and result.model_calls == 0


async def test_threshold_is_half():
    for p, expected in ((0.5, "merge"), (0.49, "keep")):
        verdict = await judge_cut_decided(
            decision_model=StubDecisionModel(p),  # type: ignore[arg-type]
            source=SRC,
            cut_offset=len(SRC) // 2,
        )
        assert verdict.decision == expected


async def test_failure_keeps_the_cut():
    verdict = await judge_cut_decided(
        decision_model=StubDecisionModel(fail=True),  # type: ignore[arg-type]
        source=SRC,
        cut_offset=len(SRC) // 2,
    )
    assert isinstance(verdict, JudgeVerdict) and verdict.decision == "keep"


async def test_question_is_noul_and_state_has_both_windows():
    stub = StubDecisionModel(0.9)
    await judge_cut_decided(
        decision_model=stub,  # type: ignore[arg-type]
        source=SRC,
        cut_offset=len(SRC) // 2,
    )
    state, questions = stub.calls[0]
    assert questions == {"merge": MERGE_QUESTION}
    assert MERGE_QUESTION.type == "noul"
    assert set(state) == {"left", "right"}  # type: ignore[arg-type]


def test_cli_passes_the_opened_decision_model(tmp_path, monkeypatch):
    seen: dict = {}

    async def fake_extract_units(source, **kwargs):
        seen.update(kwargs)
        return ChunkingResult(
            units=[],
            gaps=[],
            total_chars=0,
            model_calls=0,
            retries_used=0,
            windows_processed=0,
        )

    stub = StubDecisionModel(0.9)

    class FakeOpen:
        async def __aenter__(self):
            return stub

        async def __aexit__(self, *exc):
            return None

    async def fake_open(name):
        assert name == "nimble:9b"
        return FakeOpen()

    monkeypatch.setattr(cli, "extract_units", fake_extract_units)
    monkeypatch.setattr(cli.DecisionModel, "open", fake_open)
    src = tmp_path / "doc.md"
    src.write_text("## A\n\nbody")

    rc = cli.main(
        [str(src), "--decision-model", "nimble:9b", "-o", str(tmp_path / "o.json")]
    )

    assert rc == 0 and seen["decision_model"] is stub
