#!/usr/bin/env python3
"""
XML element builder for consolidating DOCX XML operations.

This module eliminates the massive duplication in XML element creation
across the docxeditor module by providing reusable builders.
"""

import copy
import datetime
from typing import Optional
from lxml import etree  # type: ignore
from docx.oxml.ns import qn

# Word XML namespaces
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


class XMLElementBuilder:
    """
    Consolidated XML element creation for Word documents.

    This class eliminates the duplication of XML creation patterns
    found throughout docxeditor.py by providing reusable methods.
    """

    @staticmethod
    def create_change_container(
        change_type: str, cid: int, author: str, text: str, rPr: Optional[etree.Element] = None
    ) -> etree.Element:
        """
        Create a change container (ins/del) with proper attributes.

        Consolidates the logic from _build_change_container in docxeditor.py.

        Args:
            change_type: 'ins' or 'del'
            cid: Change ID
            author: Author name
            text: Text content
            rPr: Run properties to copy

        Returns:
            Complete change container element
        """
        # Create container element
        container = etree.Element(f"{{{NS['w']}}}{change_type}")
        container.set(qn("w:id"), str(cid))
        container.set(qn("w:author"), author)
        container.set(qn("w:date"), datetime.datetime.now().isoformat())

        # Create run element
        run = XMLElementBuilder.create_run_element(rPr)

        # Create text element (different for deletions)
        if change_type == "del":
            text_elem = etree.Element(f"{{{NS['w']}}}delText")
        else:
            text_elem = etree.Element(f"{{{NS['w']}}}t")

        XMLElementBuilder.set_text_content(text_elem, text)
        run.append(text_elem)
        container.append(run)

        return container

    @staticmethod
    def create_run_element(rPr: Optional[etree.Element] = None) -> etree.Element:
        """
        Create a run element with optional properties.

        Args:
            rPr: Run properties to copy

        Returns:
            Run element with properties
        """
        run = etree.Element(f"{{{NS['w']}}}r")
        if rPr is not None:
            run.append(copy.deepcopy(rPr))
        return run

    @staticmethod
    def create_text_element(text: str, is_deletion: bool = False) -> etree.Element:
        """
        Create a text element with proper attributes.

        Args:
            text: Text content
            is_deletion: Whether this is a deletion (uses delText)

        Returns:
            Text element with content
        """
        if is_deletion:
            text_elem = etree.Element(f"{{{NS['w']}}}delText")
        else:
            text_elem = etree.Element(f"{{{NS['w']}}}t")

        XMLElementBuilder.set_text_content(text_elem, text)
        return text_elem

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
    def create_simple_run(
        text: str,
        rPr: Optional[etree.Element] = None,
        change_type: Optional[str] = None,
        cid: Optional[int] = None,
        author: Optional[str] = None,
    ) -> etree.Element:
        """
        Create a complete run with text, optionally wrapped in change tracking.

        This method consolidates the _add_run logic from docxeditor.py.

        Args:
            text: Text content
            rPr: Run properties to copy
            change_type: 'ins', 'del', or None for no change tracking
            cid: Change ID for tracked changes
            author: Author for tracked changes

        Returns:
            Run element or change container with run
        """
        if change_type in ("ins", "del"):
            # Create change container
            # Ensure cid and author are not None for tracked changes
            if cid is None or author is None:
                raise ValueError("cid and author are required for tracked changes")
            return XMLElementBuilder.create_change_container(change_type, cid, author, text, rPr)
        else:
            # Create simple run
            run = XMLElementBuilder.create_run_element(rPr)
            text_elem = XMLElementBuilder.create_text_element(text)
            run.append(text_elem)
            return run

    @staticmethod
    def create_comment_range_elements(cid: int) -> tuple[etree.Element, etree.Element, etree.Element]:
        """
        Create comment range start, end, and reference elements.

        Consolidates the comment anchor creation logic.

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
    def create_comment_element(cid: int, author: str, text: str) -> etree.Element:
        """
        Create a complete comment element for comments.xml.

        Args:
            cid: Comment ID
            author: Comment author
            text: Comment text

        Returns:
            Complete comment element
        """
        comment = etree.Element(f"{{{NS['w']}}}comment")
        comment.set(f"{{{NS['w']}}}id", str(cid))
        comment.set(f"{{{NS['w']}}}author", author)
        comment.set(f"{{{NS['w']}}}date", datetime.datetime.now().isoformat())

        # Create paragraph inside comment
        para = etree.SubElement(comment, f"{{{NS['w']}}}p")
        run = etree.SubElement(para, f"{{{NS['w']}}}r")
        text_elem = etree.SubElement(run, f"{{{NS['w']}}}t")
        XMLElementBuilder.set_text_content(text_elem, text)

        return comment

    @staticmethod
    def create_page_break_run() -> etree.Element:
        """
        Create a run with a page break.

        Returns:
            Run element containing page break
        """
        run = etree.Element(f"{{{NS['w']}}}r")
        br = etree.SubElement(run, f"{{{NS['w']}}}br")
        br.set(f"{{{NS['w']}}}type", "page")
        return run

    @staticmethod
    def create_paragraph_element() -> etree.Element:
        """
        Create an empty paragraph element.

        Returns:
            Empty paragraph element
        """
        return etree.Element(f"{{{NS['w']}}}p")

    @staticmethod
    def get_change_attributes(element: etree.Element) -> dict:
        """
        Extract change tracking attributes from an element.

        Args:
            element: Element to extract attributes from

        Returns:
            Dictionary with id, author, date if present
        """
        attrs = {}

        cid = element.get(qn("w:id"))
        if cid:
            attrs["id"] = cid

        author = element.get(qn("w:author"))
        if author:
            attrs["author"] = author

        date = element.get(qn("w:date"))
        if date:
            attrs["date"] = date

        return attrs

    @staticmethod
    def copy_run_properties(source_run: etree.Element) -> Optional[etree.Element]:
        """
        Extract and copy run properties from a source run.

        Args:
            source_run: Run element to copy properties from

        Returns:
            Copied run properties element or None
        """
        rPr = source_run.find("w:rPr", namespaces=NS)
        if rPr is not None:
            return copy.deepcopy(rPr)
        return None


class XMLPatternMatcher:
    """
    Helper class for common XML pattern matching operations.

    Consolidates XML traversal and matching logic.
    """

    @staticmethod
    def find_text_runs(paragraph: etree.Element) -> list[etree.Element]:
        """
        Find all text-containing runs in a paragraph.

        Args:
            paragraph: Paragraph element to search

        Returns:
            List of run elements containing text
        """
        return paragraph.xpath(".//w:r[w:t]", namespaces=NS)

    @staticmethod
    def extract_text_from_element(element: etree.Element) -> str:
        """
        Extract all text content from an element and its children.

        Args:
            element: Element to extract text from

        Returns:
            Combined text content
        """
        text_elements = element.xpath(".//w:t | .//w:delText", namespaces=NS)
        return "".join(elem.text or "" for elem in text_elements)

    @staticmethod
    def is_change_element(element: etree.Element) -> bool:
        """
        Check if element is a change tracking element.

        Args:
            element: Element to check

        Returns:
            True if element is ins or del
        """
        local_name = etree.QName(element.tag).localname
        return local_name in ("ins", "del")

    @staticmethod
    def get_change_type(element: etree.Element) -> Optional[str]:
        """
        Get the change type of an element.

        Args:
            element: Element to check

        Returns:
            'ins', 'del', or None
        """
        if XMLPatternMatcher.is_change_element(element):
            return etree.QName(element.tag).localname
        return None
