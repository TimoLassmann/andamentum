"""Tests for _extract_one — the per-window LLM call wrapper.

Uses a fake executor callable that returns canned NextUnitResult objects.
"""

from andamentum.chunker.extractor import ExtractionAttempt, _extract_one
from andamentum.chunker.types import NextUnitResult
from andamentum.chunker.windowing import Window


def _window(text: str) -> Window:
    return Window(
        text=text,
        cursor=0,
        window_end_offset=len(text),
        full_end_offset=len(text),
    )


def _make_executor(programmed: list):
    """Builds a fake executor that returns/raises items in order."""
    items = list(programmed)

    async def executor(*, instructions, user_message, output_type, validators):
        item = items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return executor


async def test_extract_one_returns_attempt_with_unit():
    text = "Hello world. This is a test. End of test."
    executor = _make_executor(
        [
            NextUnitResult(
                found=True,
                title="Greeting",
                start_anchor="Hello world",
                end_anchor="End of test.",
                kind="prose",
            )
        ]
    )
    attempt = await _extract_one(
        executor=executor,
        window=_window(text),
        domain="general",
        prior_unit_titles=[],
    )
    assert isinstance(attempt, ExtractionAttempt)
    assert attempt.result is not None
    assert attempt.result.found is True
    assert attempt.calls == 1
    assert attempt.error is None


async def test_extract_one_returns_attempt_with_not_found():
    executor = _make_executor([NextUnitResult(found=False, skip_to="end of nav")])
    attempt = await _extract_one(
        executor=executor,
        window=_window("junk junk junk end of nav"),
        domain="general",
        prior_unit_titles=[],
    )
    assert attempt.result is not None
    assert attempt.result.found is False
    assert attempt.result.skip_to == "end of nav"


async def test_extract_one_propagates_runtime_errors():
    """A non-validation error from the executor becomes attempt.error."""
    executor = _make_executor([RuntimeError("model timeout")])
    attempt = await _extract_one(
        executor=executor,
        window=_window("text"),
        domain="general",
        prior_unit_titles=[],
    )
    assert attempt.error is not None
    assert "timeout" in str(attempt.error)
