"""The CLI resolves its input once, and the .docx it annotates is that input.

Everything else in this change is only true if the wiring holds: the review has
to be handed the document, and the renderer has to be handed the same one. The
review itself is stubbed — what is under test is which text reached it and
which file came back annotated.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import pytest
from docx import Document
from rich.console import Console

from andamentum.whetstone import cli as cli_mod
from andamentum.whetstone.docx.atoms import document_atoms, projection
from andamentum.whetstone.docx.reading import read_docx
from andamentum.whetstone.reviewed_document import ReviewedDocument
from andamentum.whetstone.schemas import Edit, Finding, Quote, ReviewResult


def _draft(path: Path) -> None:
    document = Document()
    document.add_heading("Methods", level=1)
    paragraph = document.add_paragraph("We met the ")
    paragraph.add_run("goal")
    paragraph.add_run("3,4").font.superscript = True
    paragraph.add_run(". Then we stopped.")
    document.save(str(path))


def _seen_and_result() -> tuple[dict[str, Any], Any]:
    """A stub review that annotates whatever text it was given."""
    seen: dict[str, Any] = {}

    async def fake_review(source: Any, **kwargs: Any) -> ReviewResult:
        seen["source"] = source
        text = source.text
        start = text.index("aim3,4.")
        return ReviewResult(
            summary="Stubbed.",
            edits=[
                Edit(
                    title="Space the citation",
                    rationale="Readability.",
                    section_id="s1",
                    char_start=start,
                    char_end=start + len("aim3,4."),
                    original_text="aim3,4.",
                    new_text="aim 3 and 4.",
                )
            ],
            findings=[
                Finding(
                    title="Vague",
                    severity="minor",
                    confidence="medium",
                    rationale="Which goal?",
                    quotes=[
                        Quote(
                            section_id="s1",
                            char_start=text.index("Then we stopped."),
                            char_end=text.index("Then we stopped.")
                            + len("Then we stopped."),
                            text="Then we stopped.",
                        )
                    ],
                )
            ],
        )

    return seen, fake_review


def _args(argv: list[str]):
    parser = cli_mod._build_parser()
    return parser.parse_args(argv)


async def test_a_docx_input_is_the_file_that_gets_annotated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "draft.docx"
    _draft(source)
    out = tmp_path / "reviewed.docx"

    seen, fake_review = _seen_and_result()
    monkeypatch.setattr(
        "andamentum.whetstone.v3.review_document_v3", fake_review, raising=True
    )

    await cli_mod._run(
        _args([str(source), "--model", "stub:none", "--out", str(out)]),
        Console(stderr=True),
    )

    # The review was handed the opened document, not a path string.
    document = seen["source"]
    assert isinstance(document, ReviewedDocument)
    assert document.docx_path == source
    assert document.text == projection(document_atoms(read_docx(source).paragraphs))

    # And its offsets landed: the tracked change is in the output.
    assert out.exists()
    with zipfile.ZipFile(out) as archive:
        body = archive.read("word/document.xml").decode()
    assert "<w:ins " in body and "<w:del " in body
    assert '<w:commentRangeStart w:id="1"/>' in body
    assert source.exists(), "the user's input must survive the run"


async def test_a_markdown_input_leaves_no_baseline_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generated baseline is a temp file the CLI owns and removes."""
    source = tmp_path / "draft.md"
    source.write_text("# Methods\n\nWe met the aim3,4. Then we stopped.\n")
    out = tmp_path / "reviewed.docx"

    seen, fake_review = _seen_and_result()
    monkeypatch.setattr(
        "andamentum.whetstone.v3.review_document_v3", fake_review, raising=True
    )

    await cli_mod._run(
        _args([str(source), "--model", "stub:none", "--out", str(out)]),
        Console(stderr=True),
    )

    document = seen["source"]
    assert document.docx_is_temp is True
    assert not document.docx_path.exists()
    assert out.exists()


async def test_the_reviewed_text_does_not_depend_on_which_outputs_were_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One resolution, before any mode and before any renderer.

    The baseline used to be built only when a ``.docx`` output was requested,
    by a SECOND harvest of the same input — so what was reviewed and what was
    annotated were two independent renderings of the source.
    """
    source = tmp_path / "draft.md"
    source.write_text("# Methods\n\nWe met the aim3,4. Then we stopped.\n")

    texts: list[str] = []
    for outputs in (["review.md"], ["review.md", "reviewed.docx"]):
        seen, fake_review = _seen_and_result()
        monkeypatch.setattr(
            "andamentum.whetstone.v3.review_document_v3", fake_review, raising=True
        )
        argv = [str(source), "--model", "stub:none"]
        for name in outputs:
            argv += ["--out", str(tmp_path / name)]
        await cli_mod._run(_args(argv), Console(stderr=True))
        texts.append(seen["source"].text)

    assert texts[0] == texts[1]


async def test_a_crashed_review_still_removes_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run that leaves a stray temp file is the one nobody is watching."""
    source = tmp_path / "draft.md"
    source.write_text("# Methods\n\nWe met the aim3,4. Then we stopped.\n")

    opened: list[Any] = []

    from andamentum.whetstone import reviewed_document as rd

    original = rd.open_source

    async def spy(arg: str):
        document = await original(arg)
        opened.append(document)
        return document

    monkeypatch.setattr(rd, "open_source", spy)

    async def boom(*_a: Any, **_k: Any) -> ReviewResult:
        raise RuntimeError("the model fell over")

    monkeypatch.setattr("andamentum.whetstone.v3.review_document_v3", boom)

    with pytest.raises(SystemExit):
        await cli_mod._run(
            _args(
                [str(source), "--model", "stub:none", "--out", str(tmp_path / "r.md")]
            ),
            Console(stderr=True),
        )

    assert opened and not opened[0].docx_path.exists()
