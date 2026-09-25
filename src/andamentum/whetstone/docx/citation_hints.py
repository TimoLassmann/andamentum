"""What a Word file knows about its citations that its text does not.

Two things, both in the coordinates of the reviewed text (the projection of
the body's paragraphs, one atom per character — see :mod:`.atoms`):

*Superscript runs.* A raised "12" after a word is a citation in the Nature,
Vancouver-superscript and AMA styles, and an exponent, a footnote mark or an
affiliation elsewhere. The text alone shows "disease12"; the run formatting
says which characters were raised. Direct formatting and character styles
that raise their text are both read.

*Reference-manager fields.* Zotero, Mendeley and EndNote insert each citation
as a Word field: a hidden instruction naming what is cited, and a visible
result — "(Smith et al., 2020)" or "12". The projection keeps only the result
(``<w:instrText>`` is not text); this recovers which characters are a field's
result and what its instruction says, so the checker knows exactly which
spans are citations without judging them.

Nothing here decides anything. It reports formatting; the reference checker
weighs it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from andamentum.references import Hint

from .atoms import document_atoms, projection, qn
from .reading import read_docx

_CITATION_FIELD_MARKERS = (
    "ZOTERO_ITEM",
    "CSL_CITATION",
    "EN.CITE",
    "MENDELEY CITATION",
    "ADDIN CITATION",
    "CITAVI",
)


def citation_hints(path: str | Path, *, expected_text: str) -> list[Hint]:
    """Superscript and citation-field spans of *path*'s body.

    *expected_text* is the text the offsets must address. If the file does
    not project to exactly that text, no hints are returned — an offset into a
    different text is not a hint but a misplacement — and the caller's check
    runs on the text alone.
    """
    parts = read_docx(path)
    atoms = document_atoms(parts.paragraphs)
    if projection(atoms) != expected_text:
        return []
    raised_styles = _raised_character_styles(parts.styles_root)
    field_of_run = _citation_field_runs(parts.paragraphs)

    hints: list[Hint] = []
    spans: dict[int, list[int]] = {}
    instructions: dict[int, str] = {}
    run_start: int | None = None
    for index, atom in enumerate(atoms):
        run = atom.run
        raised = run is not None and _is_raised(run, raised_styles)
        if raised and atom.char.strip():
            if run_start is None:
                run_start = index
        elif run_start is not None:
            hints.append(Hint(kind="superscript", start=run_start, end=index))
            run_start = None
        if run is not None and id(run) in field_of_run:
            field_index, instruction = field_of_run[id(run)]
            spans.setdefault(field_index, []).append(index)
            instructions[field_index] = instruction
    if run_start is not None:
        hints.append(Hint(kind="superscript", start=run_start, end=len(atoms)))
    for field_index, indices in spans.items():
        hints.append(
            Hint(
                kind="field_citation",
                start=min(indices),
                end=max(indices) + 1,
                payload=instructions[field_index][:4000],
            )
        )
    hints.sort(key=lambda h: (h.start, h.kind))
    return hints


def _is_raised(run: Any, raised_styles: frozenset[str]) -> bool:
    properties = run.find(qn("rPr"))
    if properties is None:
        return False
    align = properties.find(qn("vertAlign"))
    if align is not None:
        return align.get(qn("val")) == "superscript"
    style = properties.find(qn("rStyle"))
    return style is not None and style.get(qn("val")) in raised_styles


def _raised_character_styles(styles_root: Any) -> frozenset[str]:
    """Character styles whose own formatting raises the text."""
    if styles_root is None:
        return frozenset()
    raised: set[str] = set()
    for style in styles_root.iter(qn("style")):
        properties = style.find(qn("rPr"))
        align = properties.find(qn("vertAlign")) if properties is not None else None
        if align is not None and align.get(qn("val")) == "superscript":
            raised.add(style.get(qn("styleId")) or "")
    return frozenset(raised)


def _citation_field_runs(paragraphs: list[Any]) -> dict[int, tuple[int, str]]:
    """``id(run) → (field index, instruction)`` for runs showing a citation
    field's result.

    Complex fields are a begin / instruction / separate / result / end
    sequence of runs, possibly across paragraphs and possibly nested; simple
    fields (``<w:fldSimple w:instr=...>``) wrap their result runs. Only fields
    whose instruction names a reference manager count.
    """
    owned: dict[int, tuple[int, str]] = {}
    stack: list[dict] = []
    counter = 0
    for paragraph in paragraphs:
        for element in paragraph.iter(qn("r"), qn("fldSimple")):
            if element.tag == qn("fldSimple"):
                instruction = element.get(qn("instr")) or ""
                if _is_citation(instruction):
                    counter += 1
                    for run in element.iter(qn("r")):
                        owned[id(run)] = (counter, instruction)
                continue
            char = element.find(qn("fldChar"))
            kind = char.get(qn("fldCharType")) if char is not None else None
            if kind == "begin":
                counter += 1
                stack.append({"index": counter, "instr": "", "result": False})
                continue
            if kind == "separate" and stack:
                stack[-1]["result"] = True
                continue
            if kind == "end" and stack:
                stack.pop()
                continue
            text = element.find(qn("instrText"))
            if text is not None and stack and not stack[-1]["result"]:
                stack[-1]["instr"] += text.text or ""
                continue
            # A result run belongs to the innermost citation field showing it.
            for field in reversed(stack):
                if field["result"] and _is_citation(field["instr"]):
                    owned.setdefault(id(element), (field["index"], field["instr"]))
                    break
    return owned


def _is_citation(instruction: str) -> bool:
    upper = instruction.upper()
    return any(marker in upper for marker in _CITATION_FIELD_MARKERS)
