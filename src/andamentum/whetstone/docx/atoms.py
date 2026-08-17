"""The document as a stream of characters, each knowing where it came from.

This is the single text representation. Everything that needs to know "what
does this document say, and where is that text" reads an atom stream — the
matcher, the comment anchorer, and the edit surgeon alike.

Before this module there were three answers to that question, and they
disagreed on the same paragraph:

    <w:t> join            'See the protocol at the registered plan before readingfurther.'
    visible_text()        'See the protocol at the registered plan before reading further.'
    direct-child runs     'See the protocol at  before reading further.'

The join drops line breaks and welds the words either side together. The
direct-child walk loses the whole hyperlink, because a hyperlink nests its run
inside ``<w:hyperlink>``. Three representations meant every bug had to be
fixed three times, and a span resolved against one could not be trusted
against another.

**Descend by default, exclude by name.** The old walkers collected by name —
"take the direct ``<w:r>`` children" — so any container nobody had thought of
was silently invisible. This one recurses into everything and names only what
is genuinely not readable text: a ``<w:del>`` subtree (already on its way out
of the document). Content controls, smart tags, hyperlinks and insertions are
handled because nothing special is done about them, which is the property that
makes the next unfamiliar construct work too.

An atom is one character a reader sees. Three kinds:

``TEXT``
    A real character in a ``<w:t>``. Editable: an edit may consume it.
``STRUCTURAL``
    A ``<w:br>`` or ``<w:tab>``, rendered as the single space it looks like.
    Not editable — an edit that consumed one would silently delete layout.
``PARAGRAPH_BREAK``
    The boundary between two paragraphs, rendered as a newline. Deleting one
    merges the paragraphs; inserting one splits them. Modelling the break as a
    character is what lets a cross-paragraph edit be an ordinary span rather
    than a special case.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator, Sequence

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def qn(tag: str) -> str:
    """``"t"`` → the fully-qualified ``<w:t>`` tag name."""
    return f"{{{W_NS}}}{tag}"


RUN_TAG = qn("r")
TEXT_TAG = qn("t")
PARAGRAPH_TAG = qn("p")

# Rendered as the one space a reader sees them as.
SPACE_TAGS: frozenset[str] = frozenset({qn("br"), qn("tab")})

# The only subtree excluded from the reading. Deleted text is not part of what
# the document says; a comment bracketing it would vanish the moment the author
# accepts the deletion it belongs to. Field plumbing (``<w:instrText>``) and
# deleted text (``<w:delText>``) are excluded already by not being ``<w:t>``.
EXCLUDED_SUBTREES: frozenset[str] = frozenset({qn("del")})


class AtomKind(Enum):
    TEXT = "text"
    STRUCTURAL = "structural"
    PARAGRAPH_BREAK = "paragraph_break"


@dataclass(frozen=True)
class Atom:
    """One character of the document, with its provenance.

    ``run`` is the ``<w:r>`` the character lives in and ``offset`` its index
    within that run's ``<w:t>`` — together they are the address an edit needs.
    A ``PARAGRAPH_BREAK`` has no run; its ``paragraph`` is the paragraph whose
    mark it is (the one that ends there).
    """

    char: str
    kind: AtomKind
    run: Any | None
    paragraph: Any
    offset: int
    # The node the character came from: the ``<w:t>`` for text, the
    # ``<w:br>``/``<w:tab>`` for structure. ``offset`` indexes into THIS
    # element, not into the run — a run may hold several ``<w:t>`` children
    # with a break or a page-break marker between them, and an offset against
    # the run would then be ambiguous. Addressing the element instead is what
    # lets an edit cut one span and leave every sibling node in place.
    element: Any = None

    @property
    def editable(self) -> bool:
        """May an edit consume this character?

        A break or tab is layout the reviewer did not ask to remove, so an edit
        span is not allowed to swallow one. A paragraph break IS consumable:
        deleting it is how two paragraphs are merged.
        """
        return self.kind is not AtomKind.STRUCTURAL


def paragraph_atoms(p_elem: Any) -> Iterator[Atom]:
    """Every visible character in one paragraph, in reading order."""
    yield from _walk(p_elem, paragraph=p_elem, run=None)


def _walk(node: Any, *, paragraph: Any, run: Any | None) -> Iterator[Atom]:
    for child in node:
        tag = child.tag
        # XML comments and processing instructions carry a callable tag.
        if not isinstance(tag, str) or tag in EXCLUDED_SUBTREES:
            continue
        if tag == TEXT_TAG:
            for index, char in enumerate(child.text or ""):
                yield Atom(
                    char=char,
                    kind=AtomKind.TEXT,
                    run=run,
                    paragraph=paragraph,
                    offset=index,
                    element=child,
                )
        elif tag in SPACE_TAGS:
            yield Atom(
                char=" ",
                kind=AtomKind.STRUCTURAL,
                run=run,
                paragraph=paragraph,
                offset=-1,
                element=child,
            )
        elif tag == RUN_TAG:
            yield from _walk(child, paragraph=paragraph, run=child)
        else:
            # Descend by default: hyperlinks, content controls, smart tags,
            # insertions and anything else that may hold runs.
            yield from _walk(child, paragraph=paragraph, run=run)


def document_atoms(paragraphs: Sequence[Any]) -> list[Atom]:
    """The whole document as one atom stream, paragraph breaks included.

    The break is emitted BETWEEN paragraphs and attributed to the paragraph it
    ends, which is where Word keeps the mark it represents.
    """
    stream: list[Atom] = []
    for index, p_elem in enumerate(paragraphs):
        if index > 0:
            previous = paragraphs[index - 1]
            stream.append(
                Atom(
                    char="\n",
                    kind=AtomKind.PARAGRAPH_BREAK,
                    run=None,
                    paragraph=previous,
                    offset=-1,
                )
            )
        stream.extend(paragraph_atoms(p_elem))
    return stream


def projection(stream: Sequence[Atom]) -> str:
    """The canonical text of an atom stream."""
    return "".join(atom.char for atom in stream)
