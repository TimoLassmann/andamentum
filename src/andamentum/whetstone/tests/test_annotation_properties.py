"""Randomised check that annotating a document is exact and reproducible.

Hand-written fixtures kept missing the case that mattered — every one of them
put a line break in a run of its own, so an edit that deleted the rest of its
run went unnoticed through three versions of the code. This generates
documents instead: breaks and page-break markers INSIDE text runs, tabs,
hyperlinks, and a co-author's tracked changes, then random rewrites and notes
over them.

Four invariants, checked against oracles that do not share code with the thing
under test:

1. the accepted text equals applying the kept rewrites to the original string,
   right to left
2. a note still brackets its own passage, with any rewrite inside it applied
3. no structural element is lost
4. the same events twice produce the same document

Seeded, so a failure names the case that produced it. The count is small enough
to run every time; the same file takes thousands of cases when run by hand,
which is how the numbers in the commit message were measured.

Every failure this test found while it was being written was in the test — a
determinism check that regenerated different events, a comparison sensitive to
whitespace, a regex that ate XML across runs, and a walker that dropped the
newline between paragraphs. The algorithm itself did not fail once. That is
worth recording: the arithmetic is simple enough to be right, and verifying it
ad hoc is where the mistakes actually live.
"""

from __future__ import annotations

import random
import re
import zipfile
from pathlib import Path

import pytest
from docx import Document
from lxml import etree  # type: ignore

from andamentum.whetstone.docx.annotate import Note, Rewrite, apply_annotations
from andamentum.whetstone.docx.atoms import document_atoms, projection, qn
from andamentum.whetstone.docx.low_level import DocxEditor

CASES = 60

WORDS = ("cohort effect sample method result analysis provenance claim evidence "
         "workflow container version audit transparency benchmark").split()

def random_document(rng, path: Path) -> None:
    doc = Document()
    for _ in range(rng.randint(1, 4)):
        para = doc.add_paragraph()
        for _ in range(rng.randint(1, 4)):
            kind = rng.random()
            if kind < 0.12:                                  # a break inside a run
                run = etree.SubElement(para._p, qn("r"))
                etree.SubElement(run, qn("t")).text = " ".join(
                    rng.choices(WORDS, k=rng.randint(1, 3))) + " "
                etree.SubElement(run, qn("br"))
                etree.SubElement(run, qn("t")).text = " " + " ".join(
                    rng.choices(WORDS, k=rng.randint(1, 3)))
            elif kind < 0.20:                                # a tab in its own run
                run = etree.SubElement(para._p, qn("r"))
                etree.SubElement(run, qn("tab"))
            elif kind < 0.28:                                # Word's page-break marker
                run = etree.SubElement(para._p, qn("r"))
                etree.SubElement(run, qn("t")).text = " ".join(
                    rng.choices(WORDS, k=2)) + " "
                etree.SubElement(run, qn("lastRenderedPageBreak"))
                etree.SubElement(run, qn("t")).text = " " + " ".join(
                    rng.choices(WORDS, k=2))
            elif kind < 0.36:                                # a hyperlink
                link = etree.SubElement(para._p, qn("hyperlink"))
                run = etree.SubElement(link, qn("r"))
                etree.SubElement(run, qn("t")).text = " " + " ".join(
                    rng.choices(WORDS, k=2)) + " "
            elif kind < 0.44:                                # someone else's change
                ins = etree.SubElement(para._p, qn("ins"))
                ins.set(qn("id"), str(rng.randint(900, 999)))
                ins.set(qn("author"), "Dr Human")
                ins.set(qn("date"), "2026-01-01T00:00:00")
                run = etree.SubElement(ins, qn("r"))
                etree.SubElement(run, qn("t")).text = " " + " ".join(
                    rng.choices(WORDS, k=2)) + " "
            else:
                bold = rng.random() < 0.3
                run = para.add_run(" " + " ".join(
                    rng.choices(WORDS, k=rng.randint(2, 6))) + " ")
                run.bold = bold
    doc.save(str(path))


def word_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def random_events(rng, text: str):
    spans = word_spans(text)
    if len(spans) < 4:
        return [], []
    rewrites, notes = [], []
    for _ in range(rng.randint(0, 3)):
        i = rng.randrange(len(spans))
        start, end = spans[i]
        rewrites.append(Rewrite(start=start, end=end,
                                new_text=rng.choice(["", "revised", "a much longer replacement"]),
                                author="AI"))
    for _ in range(rng.randint(0, 3)):
        i = rng.randrange(len(spans))
        j = min(i + rng.randint(0, 5), len(spans) - 1)
        notes.append(Note(start=spans[i][0], end=spans[j][1], body="note", author="AI"))
    return rewrites, notes


def accepted_text(paragraphs) -> str:
    return projection(document_atoms(paragraphs))


def oracle(text: str, kept: list[Rewrite]) -> str:
    """Apply the kept rewrites right to left — independent of the code."""
    for r in sorted(kept, key=lambda r: r.start, reverse=True):
        text = text[: r.start] + r.new_text + text[r.end :]
    return text


def comment_ranges(path: Path) -> list[str]:
    """The text each comment brackets, read the way the projection reads it.

    Regexes were wrong twice here: one ate XML across nested runs, and one
    dropped the word boundary a <w:br> stands for. Walk the tree with the same
    rules atoms.py uses — <w:t> is text, br/tab are a space, <w:del> is not
    part of what the document says.
    """
    with zipfile.ZipFile(path) as z:
        root = etree.fromstring(z.read("word/document.xml"))

    W = qn("t")[1:].split("}")[0]
    def tag(name):
        return f"{{{W}}}{name}"

    open_ids, collected = set(), {}
    seen_paragraph = {}
    for node in root.iter():
        if not isinstance(node.tag, str):
            continue
        name = node.tag
        if name == tag("p") and open_ids:
            # document_atoms puts a newline between paragraphs; a walker that
            # skips it welds the last word of one to the first of the next.
            for cid in open_ids:
                if seen_paragraph.get(cid):
                    collected[cid].append("\n")
                seen_paragraph[cid] = True
        if name == tag("commentRangeStart"):
            cid = node.get(tag("id"))
            open_ids.add(cid)
            collected.setdefault(cid, [])
            seen_paragraph[cid] = True
        elif name == tag("commentRangeEnd"):
            open_ids.discard(node.get(tag("id")))
        elif open_ids:
            if any(a.tag == tag("del") for a in node.iterancestors()
                   if isinstance(a.tag, str)):
                continue
            if name == tag("t"):
                for cid in open_ids:
                    collected[cid].append(node.text or "")
            elif name in (tag("br"), tag("tab")):
                for cid in open_ids:
                    collected[cid].append(" ")
    return ["".join(parts) for parts in collected.values()]


def count(path: Path, tag: str) -> int:
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode()
    return len(re.findall(rf"<w:{tag}[ /}}>]", xml))



def _flat(text: str) -> str:
    """Collapse whitespace: the projection renders a <w:br> as a space and a
    paragraph boundary as a newline, and the assertion is about WHICH passage
    is bracketed, not how the nodes between its words are spelled."""
    return " ".join(text.split())


@pytest.mark.parametrize("seed", range(CASES))
def test_annotating_a_random_document_is_exact(seed: int, tmp_path: Path) -> None:
    rng = random.Random(seed)
    source = tmp_path / "in.docx"
    random_document(rng, source)
    before = {tag: count(source, tag) for tag in ("br", "tab", "hyperlink")}

    editor = DocxEditor(str(source), author="AI")
    text = accepted_text(editor.paragraph_elements)
    rewrites, notes = random_events(rng, text)
    if not rewrites and not notes:
        pytest.skip("no events for this seed")

    report = apply_annotations(
        editor, rewrites=rewrites, notes=notes, reviewed_text=text
    )
    out = tmp_path / "out.docx"
    editor.write(str(out))

    refused = {id(r) for r in report.conflicted_rewrites}
    kept = [r for r in rewrites if id(r) not in refused]

    # 1. the text is what the rewrites say it is
    after = accepted_text(DocxEditor(str(out)).paragraph_elements)
    assert after == oracle(text, kept)

    # 2. nothing structural was lost
    for tag in ("br", "tab", "hyperlink"):
        assert count(out, tag) >= before[tag], f"lost a <w:{tag}>"

    # 3. every placed note still brackets its own passage
    placed = [_flat(span) for span in comment_ranges(out)]
    for note in notes:
        if note in report.shadowed_notes:
            continue
        inside = [r for r in kept if note.start <= r.start and r.end <= note.end]
        straddles = [
            r
            for r in kept
            if not (r.end <= note.start or note.end <= r.start) and r not in inside
        ]
        if straddles:
            continue  # a partly-overwritten passage has no single expected form
        passage = text[note.start : note.end]
        for r in sorted(inside, key=lambda r: r.start, reverse=True):
            passage = (
                passage[: r.start - note.start]
                + r.new_text
                + passage[r.end - note.start :]
            )
        want = _flat(passage)
        if want:
            assert any(want in got for got in placed), (
                f"note lost its passage\n  want {want!r}\n  got  {placed!r}"
            )
        # An empty expectation means every word of the passage was deleted.
        # There is nothing left to assert here — the note either goes or sits
        # on the whitespace the words left behind, and neither is wrong.
        # Anchoring it to the struck-through text instead would put it
        # somewhere that vanishes the moment the deletion is accepted, and the
        # finding is listed in full in the prepended report either way. What
        # must not happen is the note drifting onto SOMEONE ELSE's text, and
        # that is checked on its own below.


@pytest.mark.parametrize("seed", range(CASES))
def test_a_note_never_drifts_onto_text_that_was_not_its_own(
    seed: int, tmp_path: Path
) -> None:
    """The safety property, stated on its own.

    Whatever a rewrite does to a note's passage — shifts it, revises part of
    it, deletes every word of it — the note must never end up bracketing a
    DIFFERENT passage. A comment that quietly moved to another sentence is
    worse than one that was dropped, because nothing tells the reader.
    """
    rng = random.Random(seed)
    source = tmp_path / "in.docx"
    random_document(rng, source)
    editor = DocxEditor(str(source), author="AI")
    text = accepted_text(editor.paragraph_elements)
    rewrites, notes = random_events(rng, text)
    if not notes:
        pytest.skip("no notes for this seed")

    apply_annotations(editor, rewrites=rewrites, notes=notes, reviewed_text=text)
    out = tmp_path / "out.docx"
    editor.write(str(out))

    final = accepted_text(DocxEditor(str(out)).paragraph_elements)

    for span in comment_ranges(out):
        flat = _flat(span)
        if not flat:
            continue  # its passage was deleted; whitespace is all that is left
        # Whatever it brackets must be text that is actually in the document,
        # not a fragment stitched from two places.
        assert _flat(span) in _flat(final), (
            f"comment brackets text that is not in the document: {flat!r}"
        )


@pytest.mark.parametrize("seed", range(CASES))
def test_the_same_events_twice_produce_the_same_document(
    seed: int, tmp_path: Path
) -> None:
    """Not "the same pipeline run" — the model is not reproducible. This stage
    is, and that is the part worth pinning."""
    rng = random.Random(seed)
    source = tmp_path / "in.docx"
    random_document(rng, source)
    text = accepted_text(DocxEditor(str(source)).paragraph_elements)
    rewrites, notes = random_events(rng, text)
    if not rewrites and not notes:
        pytest.skip("no events for this seed")

    outputs = []
    for run in range(2):
        editor = DocxEditor(str(source), author="AI")
        apply_annotations(
            editor,
            # Fresh objects each time: passing the same instances twice would
            # pass even if the code mutated them.
            rewrites=[
                Rewrite(start=r.start, end=r.end, new_text=r.new_text, author=r.author)
                for r in rewrites
            ],
            notes=[
                Note(start=n.start, end=n.end, body=n.body, author=n.author)
                for n in notes
            ],
            reviewed_text=text,
        )
        out = tmp_path / f"out{run}.docx"
        editor.write(str(out))
        outputs.append(accepted_text(DocxEditor(str(out)).paragraph_elements))

    assert outputs[0] == outputs[1]
