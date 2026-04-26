"""End-to-end tests for the main extract_units loop with a fake executor."""

from andamentum.chunker.extractor import extract_units
from andamentum.chunker.types import ChunkingResult, NextUnitResult


def _make_executor(programmed):
    items = list(programmed)

    async def executor(*, instructions, user_message, output_type, validators):
        return items.pop(0)

    return executor


async def test_extract_units_simple_two_unit_doc():
    text = (
        "Hello world. This is the first unit.\n\n"
        "Goodbye world. This is the second unit."
    )
    executor = _make_executor(
        [
            NextUnitResult(
                found=True,
                title="First",
                start_anchor="Hello world",
                end_anchor="first unit.",
                kind="prose",
            ),
            NextUnitResult(
                found=True,
                title="Second",
                start_anchor="Goodbye world",
                end_anchor="second unit.",
                kind="prose",
            ),
        ]
    )

    result = await extract_units(
        text,
        primary_executor=executor,
        window_size=200,
        lookahead=50,
        domain="general",
    )
    assert isinstance(result, ChunkingResult)
    assert len(result.units) == 2
    assert result.units[0].title == "First"
    assert result.units[1].title == "Second"
    # text is byte-identical to source spans
    for u in result.units:
        assert text[u.source_start : u.source_end] == u.text


async def test_extract_units_handles_skip():
    text = "junk junk junk SKIP_HERE Real content. End of content."
    executor = _make_executor(
        [
            NextUnitResult(found=False, skip_to="SKIP_HERE"),
            NextUnitResult(
                found=True,
                title="Content",
                start_anchor="Real content.",
                end_anchor="End of content.",
                kind="prose",
            ),
        ]
    )

    result = await extract_units(
        text,
        primary_executor=executor,
        window_size=200,
        lookahead=50,
        domain="general",
    )
    assert len(result.units) == 1
    assert len(result.gaps) == 1
    assert result.gap_chars > 0
    assert result.gaps[0].text.startswith("junk")


async def test_extract_units_records_coverage_metric():
    text = "Hello world. This is content."
    executor = _make_executor(
        [
            NextUnitResult(
                found=True,
                title="t",
                start_anchor="Hello world",
                end_anchor="content.",
                kind="prose",
            )
        ]
    )

    result = await extract_units(
        text,
        primary_executor=executor,
        window_size=200,
        lookahead=50,
        domain="general",
    )
    assert result.total_chars == len(text)
    assert result.coverage > 0.9
    assert result.windows_processed >= 1
