"""WordprocessingML element construction shared by the low-level editor.

Only the comment machinery builds elements through here. Tracked changes are
built in :mod:`.surgery`, which cuts and wraps the runs a span covers instead
of constructing replacement runs from a text string.
"""

import datetime
from lxml import etree  # type: ignore
from docx.oxml.ns import qn

# Word XML namespaces
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def utc_timestamp() -> str:
    """Now, as Word writes a comment date: UTC, ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def comment_initials(author: str) -> str:
    """Up to three capitals from the author's words, as Word shows them on a comment."""
    letters = "".join(word[0] for word in author.split() if word[:1].isalnum())
    return (letters or "AC")[:3].upper()


class XMLElementBuilder:
    """Element factories for the parts of a .docx the editor writes."""

    @staticmethod
    def create_part_root(local_name: str) -> etree.Element:
        """Create the root element of a NEW WordprocessingML XML part.

        Use this for any element that becomes the root of an independent
        part inside the .docx zip (e.g. ``comments.xml``). It binds the
        WordprocessingML namespace to the ``w:`` prefix explicitly via
        ``nsmap``.

        Why this matters: lxml assigns an auto-generated prefix
        (``ns0:``) to an unbound namespace at serialisation time.
        Microsoft Word looks for the ``w:`` prefix specifically on a
        part's elements and silently ignores content under any other
        prefix — so a ``comments.xml`` written with ``ns0:comment``
        elements contains comments that never appear in Word. Routing
        every new-part root through this helper makes that mistake
        impossible to reintroduce.

        Sub-elements appended under a root created here do NOT need their
        own ``nsmap`` — they inherit the ``w:`` binding from the root at
        serialisation time.

        Args:
            local_name: the part root's local tag name, e.g. ``"comments"``.

        Returns:
            An lxml element ``<w:{local_name}>`` with ``w:`` bound.
        """
        return etree.Element(f"{{{NS['w']}}}{local_name}", nsmap=NS)

    @staticmethod
    def set_text_content(element: etree.Element, text: str) -> None:
        """
        Set text content with proper XML space preservation.

        Args:
            element: Text element to modify
            text: Text content to set
        """
        element.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        element.text = text

    @staticmethod
    def create_comment_range_elements(
        cid: int,
    ) -> tuple[etree.Element, etree.Element, etree.Element]:
        """
        Create comment range start, end, and reference elements.

        Args:
            cid: Comment ID

        Returns:
            Tuple of (range_start, range_end, comment_reference_run)
        """
        # Comment range start
        range_start = etree.Element(f"{{{NS['w']}}}commentRangeStart")
        range_start.set(f"{{{NS['w']}}}id", str(cid))

        # Comment range end
        range_end = etree.Element(f"{{{NS['w']}}}commentRangeEnd")
        range_end.set(f"{{{NS['w']}}}id", str(cid))

        # Comment reference run
        ref_run = etree.Element(f"{{{NS['w']}}}r")
        ref_elem = etree.Element(f"{{{NS['w']}}}commentReference")
        ref_elem.set(qn("w:id"), str(cid))
        ref_run.append(ref_elem)

        return range_start, range_end, ref_run

    @staticmethod
    def create_comment_element(
        cid: int, author: str, text: str, *, date: str
    ) -> etree.Element:
        """
        Create a Word-shaped ``w:comment`` for comments.xml.

        ``text`` is split on newlines: the first line goes in the first paragraph,
        each further line in a paragraph of its own. The first paragraph starts
        with the annotation-reference run that Word uses for the comment mark.

        Args:
            cid: Comment ID
            author: Comment author
            text: Comment text; newlines start new paragraphs
            date: The comment's date, as :func:`utc_timestamp` writes it

        Returns:
            Complete comment element
        """
        comment = etree.Element(f"{{{NS['w']}}}comment")
        comment.set(f"{{{NS['w']}}}id", str(cid))
        comment.set(f"{{{NS['w']}}}author", author)
        comment.set(f"{{{NS['w']}}}date", date)
        comment.set(f"{{{NS['w']}}}initials", comment_initials(author))

        lines = text.split("\n")
        first = etree.SubElement(comment, f"{{{NS['w']}}}p")
        etree.SubElement(
            etree.SubElement(first, f"{{{NS['w']}}}r"), f"{{{NS['w']}}}annotationRef"
        )
        for index, line in enumerate(lines):
            paragraph = first if index == 0 else etree.SubElement(
                comment, f"{{{NS['w']}}}p"
            )
            run = etree.SubElement(paragraph, f"{{{NS['w']}}}r")
            text_elem = etree.SubElement(run, f"{{{NS['w']}}}t")
            XMLElementBuilder.set_text_content(text_elem, line)

        return comment
