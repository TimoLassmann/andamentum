"""Reviewer B reviews what reviewer A left behind.

The whole of sequential review is one parameter — ``author`` — and one
property: **round two reads the document, so it reads round one's changes as
Word will show them once accepted.** Nothing tells the pipeline it is round
two; the input file already says so.

Three things had to be true for that to hold, and each is pinned below:

* Round one's review report is not part of round two's manuscript. It is
  bracketed by a bookmark, and the paragraph reader skips it.
* A revision id is allocated above the ones already in the file. Word groups
  revisions by ``w:id``, so a restart at a fixed base makes accepting A's
  insertion also accept B's deletion of it. Measured before the fix:
  ``ins id=1000 (A)`` and ``del id=1000 (B)`` in one document.
* A comment id likewise. Before the fix, two comments both claimed ``id=1``
  and two ``commentRangeStart`` markers both pointed at it.

The .docx assertions are made on the XML rather than by eye in Word, so the
one failure mode they cannot see is Word's repair prompt. The id-uniqueness
tests are the proxy for it: an id collision is the defect Word repairs.
"""

from __future__ import annotations

import types
import zipfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from docx import Document
from lxml import etree  # type: ignore

from andamentum.whetstone.docx.atoms import document_atoms, projection
from andamentum.whetstone.docx.reading import read_docx
from andamentum.whetstone.renderers.docx import render_docx
from andamentum.whetstone.reviewed_document import ReviewedDocument, open_source
from andamentum.whetstone.pipeline.consolidate import _Consolidation
from andamentum.whetstone.pipeline.editor import EditorOutput, EditProposal
from andamentum.whetstone.pipeline.extract import _ClaimSpans, _Requote
from andamentum.whetstone.pipeline.gaps import _DemandList
from andamentum.whetstone.pipeline.graph import review_document
from andamentum.whetstone.pipeline.review import _CriterionFindings, _RawFinding
from andamentum.whetstone.pipeline.synth import StructuredReview

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

ORIGINAL_SENTENCE = "The evidence layer earns its place in the analysis."
SECOND_SENTENCE = "We sampled twelve participants across three sites."


# ── the document, and the two rounds over it ────────────────────────────


def _source_docx(path: Path) -> Path:
    document = Document()
    document.add_heading("Methods", level=1)
    document.add_paragraph(ORIGINAL_SENTENCE)
    document.add_paragraph(SECOND_SENTENCE)
    document.save(str(path))
    return path


def _stub_agents(*, rewrite: tuple[str, str], quote: str) -> ExitStack:
    """Every LLM call in the graph, answered without a network.

    *rewrite* is the one edit the editor proposes (``(original, replacement)``)
    and *quote* the passage the criterion review flags.
    """

    def _router(defn, _model):
        name = defn.name

        class _Agent:
            def output_validator(self, fn):  # absorbs the lock-and-refine validator
                return fn

            async def run(self, _prompt, **_kwargs):
                if name == "extract_claims":
                    out = _ClaimSpans(claims=[quote])
                elif name == "requote":
                    out = _Requote(quote="")
                elif name == "gap_analysis":
                    out = _DemandList(demands=[])
                elif name == "consolidate":
                    out = _Consolidation(groups=[])
                elif name in ("synthesise", "critique_revise"):
                    out = StructuredReview(
                        synopsis="A short methods note.",
                        strengths=["clear"],
                        weaknesses=["thin"],
                    )
                elif name == "editor":
                    original, replacement = rewrite
                    out = EditorOutput(
                        edits=[
                            EditProposal(
                                title="Sharpen the idiom",
                                rationale="clarity",
                                original_text=original,
                                new_text=replacement,
                            )
                        ]
                    )
                else:
                    out = _CriterionFindings(findings=[])
                return types.SimpleNamespace(output=out)

        return _Agent()

    def _review_agent(_criterion, _agent_model):
        class _Agent:
            def output_validator(self, fn):
                return fn

            async def run(self, _prompt, **_kwargs):
                return types.SimpleNamespace(
                    output=_CriterionFindings(
                        findings=[
                            _RawFinding(
                                issue="sample size is not justified",
                                quote=quote,
                                severity="major",
                            )
                        ]
                    )
                )

        return _Agent()

    stack = ExitStack()
    for module in ("extract", "gaps", "consolidate", "synth", "editor"):
        stack.enter_context(
            patch(
                f"andamentum.whetstone.pipeline.{module}.build_pydantic_ai_agent", new=_router
            )
        )
        stack.enter_context(
            patch(f"andamentum.whetstone.pipeline.{module}.resolve_model", new=lambda x: None)
        )
    stack.enter_context(
        patch("andamentum.whetstone.pipeline.review._build_agent", new=_review_agent)
    )
    return stack


async def _round(
    document: ReviewedDocument,
    *,
    out: Path,
    author: str,
    rewrite: tuple[str, str],
    quote: str,
) -> Path:
    """One complete review round: pipeline, then annotated .docx."""
    with _stub_agents(rewrite=rewrite, quote=quote):
        result = await review_document(
            document, model="stub", cap=1, document_type="academic", editor=True
        )
    assert result.edits, "the stubbed editor proposed an edit and it must survive"
    render_docx(result, document=document, output_path=out, author=author)
    return out


# ── XML readers ─────────────────────────────────────────────────────────


def _document_xml(path: Path):
    with zipfile.ZipFile(path) as archive:
        return etree.fromstring(archive.read("word/document.xml"))


def _comment_ids(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        if "word/comments.xml" not in archive.namelist():
            return []
        root = etree.fromstring(archive.read("word/comments.xml"))
    return [c.get(f"{W}id") for c in root.iter(f"{W}comment")]


def _revisions(root) -> list[tuple[str, str, str]]:
    """``(kind, author, id)`` for every tracked change in the document."""
    return [
        (element.tag[len(W) :], element.get(f"{W}author"), element.get(f"{W}id"))
        for element in root.iter()
        if element.tag in (f"{W}ins", f"{W}del")
    ]


def _accepted_text(path: Path) -> str:
    """The document as a reader sees it with the changes accepted."""
    parts = read_docx(path)
    return projection(document_atoms(parts.paragraphs))


# ── the two-round run ───────────────────────────────────────────────────


async def test_two_rounds_stack_without_losing_the_first(tmp_path: Path) -> None:
    """A reviews the draft, B reviews A's output and edits A's own insertion."""
    source = _source_docx(tmp_path / "draft.docx")

    first = await open_source(str(source))
    assert first.text.startswith("Methods\n")
    round_one = await _round(
        first,
        out=tmp_path / "round1.docx",
        author="Reviewer A",
        rewrite=("earns its place", "earns its keep"),
        quote=SECOND_SENTENCE,
    )

    # ── what round two reads ────────────────────────────────────────
    second = await open_source(str(round_one))
    # A's insertion, as Word will show it once accepted — and nothing of A's
    # report, which is this pipeline's output rather than the author's draft.
    assert "earns its keep" in second.text
    assert "earns its place" not in second.text
    assert "Whetstone Review" not in second.text
    assert second.text == first.text.replace("earns its place", "earns its keep")

    round_two = await _round(
        second,
        out=tmp_path / "round2.docx",
        author="Reviewer B",
        # B edits the words A inserted.
        rewrite=("earns its keep", "earns its salt"),
        quote=SECOND_SENTENCE,
    )

    root = _document_xml(round_two)
    revisions = _revisions(root)
    authors = {author for _, author, _ in revisions}
    assert authors == {"Reviewer A", "Reviewer B"}

    # A's change is intact, not flattened: both halves still carry A's name.
    assert ("ins", "Reviewer A", None) not in revisions  # every one is attributed
    assert any(
        kind == "ins" and author == "Reviewer A" for kind, author, _ in revisions
    )
    assert any(
        kind == "del" and author == "Reviewer A" for kind, author, _ in revisions
    )

    # Accepting everything leaves B's wording, on the author's own sentence.
    assert "earns its salt" in _accepted_text(round_two)
    assert "earns its keep" not in _accepted_text(round_two)


async def test_b_deleting_as_inside_as_insertion_nests_the_way_word_writes_it(
    tmp_path: Path,
) -> None:
    """``<w:ins A><w:del B>`` — Word's markup for "B revised what A wrote".

    It falls out of the surgery rule rather than being coded for: an edit
    rewrites the runs its span covers, in place, and the run holding the text A
    inserted lives inside A's ``<w:ins>``. Nothing here special-cases a second
    round, which is the point.
    """
    source = _source_docx(tmp_path / "draft.docx")

    first = await open_source(str(source))
    round_one = await _round(
        first,
        out=tmp_path / "round1.docx",
        author="Reviewer A",
        rewrite=("earns its place", "earns its keep"),
        quote=SECOND_SENTENCE,
    )
    second = await open_source(str(round_one))
    round_two = await _round(
        second,
        out=tmp_path / "round2.docx",
        author="Reviewer B",
        rewrite=("earns its keep", "earns its salt"),
        quote=SECOND_SENTENCE,
    )

    root = _document_xml(round_two)
    nested = [
        element
        for element in root.iter(f"{W}del")
        if element.get(f"{W}author") == "Reviewer B"
        and element.getparent().tag == f"{W}ins"
        and element.getparent().get(f"{W}author") == "Reviewer A"
    ]
    assert nested, "B's deletion of A's inserted text must sit inside A's <w:ins>"
    assert "keep" in "".join(t.text or "" for t in nested[0].iter(f"{W}delText"))


async def test_every_revision_and_comment_id_is_unique_after_two_rounds(
    tmp_path: Path,
) -> None:
    """Word groups by id: a reused one makes accepting A's change accept B's."""
    source = _source_docx(tmp_path / "draft.docx")

    first = await open_source(str(source))
    round_one = await _round(
        first,
        out=tmp_path / "round1.docx",
        author="Reviewer A",
        rewrite=("earns its place", "earns its keep"),
        quote=SECOND_SENTENCE,
    )
    second = await open_source(str(round_one))
    round_two = await _round(
        second,
        out=tmp_path / "round2.docx",
        author="Reviewer B",
        rewrite=("earns its keep", "earns its salt"),
        quote=SECOND_SENTENCE,
    )

    root = _document_xml(round_two)
    revision_ids = [rid for _, _, rid in _revisions(root)]
    assert len(revision_ids) == len(set(revision_ids)), revision_ids

    comment_ids = _comment_ids(round_two)
    assert len(comment_ids) == 2, comment_ids  # one finding anchored per round
    assert len(comment_ids) == len(set(comment_ids)), comment_ids
    marker_ids = [e.get(f"{W}id") for e in root.iter(f"{W}commentRangeStart")]
    assert sorted(marker_ids) == sorted(comment_ids)


async def test_the_same_name_twice_is_a_valid_document_word_cannot_split(
    tmp_path: Path,
) -> None:
    """What happens when one reviewer runs the tool twice.

    Nothing refuses it, and nothing should: the second run is a review of the
    revised text, which is a reasonable thing to want. But ``w:author`` is the
    only thing Word groups changes by, so the two rounds merge into one name in
    the reviewing pane and the author cannot accept round two's work while
    holding round one's back. The ids stay distinct, so the file is valid and
    each change is still accepted or rejected on its own — the loss is
    attribution, not integrity. Pass a different --patch-author to keep them
    apart.
    """
    source = _source_docx(tmp_path / "draft.docx")
    name = "andamentum-whetstone (AI)"

    first = await open_source(str(source))
    round_one = await _round(
        first,
        out=tmp_path / "round1.docx",
        author=name,
        rewrite=("earns its place", "earns its keep"),
        quote=SECOND_SENTENCE,
    )
    second = await open_source(str(round_one))
    round_two = await _round(
        second,
        out=tmp_path / "round2.docx",
        author=name,
        rewrite=("earns its keep", "earns its salt"),
        quote=SECOND_SENTENCE,
    )

    root = _document_xml(round_two)
    revisions = _revisions(root)
    assert {author for _, author, _ in revisions} == {name}
    ids = [rid for _, _, rid in revisions]
    assert len(ids) == len(set(ids)), ids
    assert "earns its salt" in _accepted_text(round_two)


async def test_a_third_round_reads_neither_report(tmp_path: Path) -> None:
    """Reports stack; the manuscript round three reads is still the manuscript.

    Each round prepends its own report above the last rather than replacing it
    — an earlier report is the earlier reviewer's record, and the author may
    not have worked through it yet. That is only tolerable because every one of
    them is bracketed, so what stacks is the file, not the text under review.
    """
    source = _source_docx(tmp_path / "draft.docx")

    document = await open_source(str(source))
    for index, (before, after) in enumerate(
        [("earns its place", "earns its keep"), ("earns its keep", "earns its salt")],
        start=1,
    ):
        out = await _round(
            document,
            out=tmp_path / f"round{index}.docx",
            author=f"Reviewer {index}",
            rewrite=(before, after),
            quote=SECOND_SENTENCE,
        )
        document = await open_source(str(out))

    assert document.text == (
        f"Methods\nThe evidence layer earns its salt in the analysis.\n"
        f"{SECOND_SENTENCE}"
    )
    # Both reports are in the file, and neither is in the reading.
    body = _document_xml(tmp_path / "round2.docx")
    titles = [
        "".join(t.text or "" for t in p.iter(f"{W}t")) for p in body.iter(f"{W}p")
    ]
    assert titles.count("Whetstone Review") == 2
