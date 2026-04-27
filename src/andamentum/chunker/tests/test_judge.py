"""Tests for stage 3 — LLM judge for grey-zone boundaries."""

from andamentum.chunker.judge import JudgeVerdict, judge_cut


async def test_judge_returns_verdict_from_executor():
    async def fake_executor(*, instructions, user_message, output_type, validators):
        assert "LEFT" in user_message and "RIGHT" in user_message
        return JudgeVerdict(decision="merge", reason="same paragraph")

    src = "left side text. " * 50 + "right side text. " * 50
    cut = src.index("right side")
    verdict = await judge_cut(executor=fake_executor, source=src, cut_offset=cut)
    assert verdict.decision == "merge"
    assert "same paragraph" in verdict.reason


async def test_judge_defaults_to_keep_on_executor_error():
    async def failing_executor(*, instructions, user_message, output_type, validators):
        raise RuntimeError("model unavailable")

    verdict = await judge_cut(executor=failing_executor, source="abc", cut_offset=1)
    # Conservative default: don't merge unrelated content silently
    assert verdict.decision == "keep"
    assert "judge unavailable" in verdict.reason


async def test_judge_truncates_context_marker_on_long_text():
    captured = {}

    async def capture_executor(*, instructions, user_message, output_type, validators):
        captured["msg"] = user_message
        return JudgeVerdict(decision="keep")

    long_left = "x" * 5_000
    long_right = "y" * 5_000
    src = long_left + long_right
    await judge_cut(
        executor=capture_executor, source=src, cut_offset=len(long_left), ctx_chars=200
    )
    # The prompt should show truncation markers since both sides exceed ctx_chars
    assert "…" in captured["msg"]
