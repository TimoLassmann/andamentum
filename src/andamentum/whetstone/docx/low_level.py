"""Word document editor with track changes support.

This module is the core Word XML engine for the document-review
package. A ``.docx`` file is a zip archive containing XML — this
module unzips it, edits the XML, and zips it back up.

The primary class :class:`DocxEditor` exposes a document as a list of
:class:`ParagraphData` objects. Callers mutate each paragraph's
``modified`` text freely; on :meth:`DocxEditor.write`, the editor
diffs original vs modified tokens and emits proper Word track
changes (``<w:ins>`` / ``<w:del>``) with per-author attribution.

:class:`PatchDocxEditor` (in ``patch_editor.py``) extends this engine
with ``DocumentPatch`` application. :class:`DocumentReview` in this
module handles the conversion of markdown review reports into Word
content for prepending to the final document.

Most consumers should use the higher-level
``document_review.renderers.render_docx`` function rather than
constructing ``DocxEditor`` directly.
"""

import zipfile
import tempfile
import shutil
import os
import re
import logging
from dataclasses import dataclass

from pydantic import BaseModel
from typing import Optional, Any

from lxml import etree  # type: ignore
from docx.oxml.ns import qn

from .atoms import paragraph_atoms, projection
from .reading import body_paragraphs, main_document_part
from .xml_builder import XMLElementBuilder

# Logger for this module
logger = logging.getLogger(__name__)

# Namespaces
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

# Elements that a reader sees as whitespace but that Word stores as
# structure rather than as text. Concatenating only <w:t> silently deletes
# them, so "Accuracy was<tab>94%" reads back as "Accuracy was94%" and any
# quote spanning the tab can never be matched — the converted markdown kept
# a space there, and no amount of normalisation can invent one that isn't
# in the other string. Anything derived from run text FOR MATCHING must go
# through :func:`visible_text`; the byte-exact ``<w:t>`` join stays the
# basis for formatting and change tracking, whose character offsets must
# not move.
_WHITESPACE_ELEMENTS = frozenset(f"{{{NS['w']}}}{tag}" for tag in ("tab", "br", "cr"))
_TEXT_ELEMENT = f"{{{NS['w']}}}t"


def visible_text(element) -> str:
    """The text a reader sees under *element*, in document order.

    A thin name over the atom projection rather than a second walker, so the
    two cannot drift apart. Tabs and line breaks contribute the single space
    they visually are; tracked deletions and field plumbing are excluded.
    """
    return projection(list(paragraph_atoms(element)))


REL_NS = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
CT_NS = {"ct": "http://schemas.openxmlformats.org/package/2006/content-types"}


class ParagraphContext(BaseModel):
    """Neighborhood view of a paragraph for agent consumption.

    Provides the paragraph text along with a configurable window of
    preceding and following paragraphs as read-only context. Used by
    agents that need surrounding context to reason about a specific
    paragraph without having to process the entire document.
    """

    index: int
    original: str
    modified: str
    comment: Optional[str]
    before: list[str]
    after: list[str]


@dataclass
class PlacedComment:
    """A comment resolved to the runs it brackets.

    Resolution happens once, in the patch editor, and the two run elements are
    kept. Nothing searches for the text again at write time — that second
    search was how a comment came to be validated, written into
    ``comments.xml``, and then never anchored, while the run reported success.
    """

    author: str
    text: str
    start_run: Any
    end_run: Any


class ParagraphData:
    """A single Word paragraph, plus the annotations attached to it.

    Deliberately holds no text of its own. It used to carry two — a ``<w:t>``
    join and a whitespace-faithful rendering — which disagreed wherever the
    paragraph contained a line break, and a third appeared once anything
    indexed runs directly. Text now comes from
    :mod:`~andamentum.whetstone.docx.atoms`, which is the only place that
    answers "what does this say".
    """

    def __init__(self, xml_path, p_elem):
        """Wrap one ``<w:p>`` element.

        Args:
            xml_path: Filesystem path to the XML file this paragraph
                belongs to (typically ``word/document.xml``).
            p_elem: ``lxml`` element for the ``<w:p>`` node.
        """
        self.xml_path = xml_path
        self.p_elem = p_elem
        self.change_author = None
        # True for paragraphs synthesised by prepend_review_section (the
        # prepended review report), so a finding's quote can't anchor onto the
        # report's own restatement of it.
        self.is_review_report = False

    @property
    def text(self) -> str:
        """What this paragraph says, from the one text representation."""
        from .atoms import paragraph_atoms, projection

        return projection(list(paragraph_atoms(self.p_elem)))


class DocxEditor:
    """Low-level Word (.docx) editor with track changes support.

    A Word document is a zip archive of XML files. This class unzips
    the archive into a temporary directory, loads every XML file as an
    ``lxml`` tree, and exposes each paragraph as a :class:`ParagraphData`
    object whose ``modified`` string can be freely mutated by callers.
    On :meth:`write`, diffs are computed between ``original`` and
    ``modified`` tokens and serialized as ``<w:ins>`` / ``<w:del>``
    elements with proper author attribution, producing a valid Word
    track-changes document.

    This is the core engine used by :class:`PatchDocxEditor` and the
    ``render_docx`` renderer. Most callers should use those higher-level
    interfaces rather than constructing ``DocxEditor`` directly.
    """

    def __init__(self, input_path, author="Python", context_size=1):
        """Open a Word document and prepare it for editing.

        Unzips the ``.docx`` into a temporary directory, loads all XML
        trees, normalizes runs so each ``<w:r>`` contains exactly one
        token, and extracts the paragraph list.

        Args:
            input_path: Path to an existing ``.docx`` file.
            author: Default author name used for track changes that
                are not explicitly attributed to another agent.
            context_size: Number of surrounding paragraphs returned by
                :meth:`get_paragraph_context`.

        Raises:
            FileNotFoundError: If ``input_path`` does not exist.
        """
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        self.author = author
        self.context_size = context_size
        self.tmpdir = tempfile.mkdtemp()
        self._unzip(input_path)
        self._doc_filename = self._detect_document_filename()
        self.trees = self._load_trees()
        self.paragraphs = self._extract_paragraphs()
        # Comments resolved to the runs they bracket. Written out by `write`;
        # nothing re-searches for them, so what was validated is what is placed.
        self.placed_comments: list[PlacedComment] = []

    @property
    def paragraph_elements(self) -> list:
        """The body's outermost ``<w:p>`` elements, in reading order.

        Read from the tree each time rather than cached: an edit may split a
        paragraph in two, and a stale list would hide the new one.

        The selection rule itself lives in :func:`.reading.body_paragraphs`,
        shared with the read-only sectioniser, so the paragraph list an edit is
        addressed against and the one the reviewer's offsets were computed
        against are the same list.
        """
        return body_paragraphs(self.trees[self._doc_path].getroot())

    def _unzip(self, docx_path):
        """Extract the .docx archive into the temporary working directory.

        A Word document is a zip file — unzipping gives us individual
        XML files we can manipulate with lxml. The temporary directory
        is cleaned up in :meth:`_zip` after the final output is written.
        """
        with zipfile.ZipFile(docx_path, "r") as z:
            z.extractall(self.tmpdir)

    def _detect_document_filename(self) -> str:
        """Detect the main document XML filename from the package rels.

        Standard DOCX files use ``word/document.xml``, but recent
        versions of Word sometimes save as ``word/document2.xml`` when
        extended features (threaded comments, @mentions) are used. The
        canonical source for the filename is ``_rels/.rels``, which
        declares the ``officeDocument`` relationship target.

        Falls back to ``document.xml`` if the rels file can't be parsed.
        """
        rels_path = os.path.join(self.tmpdir, "_rels", ".rels")
        if not os.path.isfile(rels_path):
            return "document.xml"
        try:
            with open(rels_path, "rb") as handle:
                # Target is like "word/document2.xml" — we want just the name,
                # because everything here is addressed inside the unzipped tree.
                filename = os.path.basename(main_document_part(handle.read()))
            logger.debug("Detected main document: %s", filename)
            return filename
        except Exception as e:
            logger.warning(
                "Could not parse _rels/.rels: %s — falling back to document.xml", e
            )
        return "document.xml"

    @property
    def _doc_path(self) -> str:
        """Full path to the main document XML in the temp directory."""
        return os.path.join(self.tmpdir, "word", self._doc_filename)

    @property
    def _rels_path(self) -> str:
        """Full path to the main document's .rels file."""
        rels_name = self._doc_filename + ".rels"
        return os.path.join(self.tmpdir, "word", "_rels", rels_name)

    def _load_trees(self):
        """Load every XML file under ``word/`` into an lxml tree.

        Returns a dict mapping absolute file path to the parsed tree.
        The ``word/`` subdirectory contains ``document.xml``,
        ``comments.xml``, ``styles.xml``, relationship files, and
        other parts of the Word package that may need updating.
        """
        trees = {}
        for root, _, files in os.walk(os.path.join(self.tmpdir, "word")):
            for fn in files:
                if fn.endswith(".xml"):
                    path = os.path.join(root, fn)
                    trees[path] = etree.parse(path)
        return trees

    def _extract_paragraphs(self):
        """Build the ordered list of ParagraphData objects from all trees.

        Walks every ``<w:p>`` element across all loaded XML trees and
        keeps only paragraphs that contain at least one ``<w:t>`` text
        element (skipping empty structural paragraphs). Each kept
        paragraph becomes a :class:`ParagraphData` instance.
        """
        paras = []
        for path, tree in self.trees.items():
            root = tree.getroot()
            for p in root.xpath(".//w:p", namespaces=NS):
                if any(paragraph_atoms(p)):
                    paras.append(ParagraphData(path, p))
        return paras

    def get_selection(self, start: int, end: int) -> str:
        """
        Return the concatenated (modified) text of paragraphs in [start, end],
        each prefixed with its paragraph index.
        """
        selection = []
        # clamp end to last paragraph
        last = min(end, len(self.paragraphs) - 1)
        for i in range(start, last + 1):
            text = self.paragraphs[i].modified
            # put the index right before the paragraph
            selection.append(f"[{i}]\n{text}")
        # separate paragraphs with a blank line
        return "\n\n".join(selection)

    def get_paragraph_context(self, index: int) -> ParagraphContext:
        """
        Return the ParagraphContext for the paragraph at `index`.
        """
        pd = self.paragraphs[index]
        N = self.context_size
        before = [p.original for p in self.paragraphs[max(0, index - N) : index]]
        after = [p.original for p in self.paragraphs[index + 1 : index + N + 1]]
        return ParagraphContext(
            index=index,
            original=pd.original,
            modified=pd.modified,
            comment=pd.comment or None,
            before=before,
            after=after,
        )

    def get_paragraphs(self):
        """Return the list of :class:`ParagraphData` instances.

        The list is in document order and can be mutated freely —
        each paragraph's ``modified`` attribute is what gets written
        back on :meth:`write`.
        """
        return self.paragraphs

    def _clear_runs(self, p_elem):
        """Remove all child elements from a paragraph except ``<w:pPr>``.

        Paragraph properties (``<w:pPr>``) are preserved because they
        hold indentation, alignment, and style settings that must
        survive the rewrite. Everything else (runs, hyperlinks,
        comment anchors) is cleared so :meth:`write` can rebuild the
        paragraph content from the current token state.
        """
        for child in list(p_elem):
            if child.tag != f"{{{NS['w']}}}pPr":
                p_elem.remove(child)

    def _add_run(
        self, parent, text, rPr, change_type=None, cid=None, change_author=None
    ):
        """Append a single ``<w:r>`` run element to a parent paragraph.

        Delegates XML construction to :class:`XMLElementBuilder`. When
        ``change_type`` is ``"ins"`` or ``"del"``, the run is wrapped
        in the appropriate track changes element with author attribution.
        Plain runs (``change_type=None``) are appended without track
        changes markup.

        Args:
            parent: ``<w:p>`` element to append to.
            text: Run text content.
            rPr: Run properties element (formatting) to preserve, or None.
            change_type: ``"ins"``, ``"del"``, or None for a plain run.
            cid: Change ID for track changes (ignored when ``change_type`` is None).
            change_author: Author name override; falls back to ``self.author``.
        """
        # Use XMLElementBuilder to create run elements
        run_author = change_author or self.author
        # Only warn for actual track changes (not simple runs)
        if (
            change_type in ("ins", "del")
            and run_author == self.author
            and change_author is None
        ):
            logger.warning(
                f"Track change using fallback author '{self.author}' for text '{text[:20]}...'. change_author: {change_author}"
            )
        # Monitor for Sequential Agent attributions (should be rare now)
        if change_type in ("ins", "del") and run_author == "Sequential Agent":
            logger.warning(
                f"Sequential Agent track change: {change_type} '{text[:15]}...' - check attribution logic"
            )

        run_element = XMLElementBuilder.create_simple_run(
            text=text, rPr=rPr, change_type=change_type, cid=cid, author=run_author
        )
        parent.append(run_element)

    def _ensure_comments_relationship(self):
        """Ensure the document's ``.rels`` file declares a comments relationship.

        Word requires an explicit relationship entry pointing to
        ``comments.xml`` before any ``<w:commentRangeStart>`` markers
        in the main document will be recognized. If the relationship
        is already present this is a no-op.
        """
        rels_path = self._rels_path
        tree = etree.parse(rels_path)
        root = tree.getroot()
        existing = root.xpath(
            "./r:Relationship[@Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments']",
            namespaces=REL_NS,
        )
        if not existing:
            root.append(
                etree.Element(
                    "{http://schemas.openxmlformats.org/package/2006/relationships}Relationship",
                    Id="comments",
                    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments",
                    Target="comments.xml",
                )
            )
            tree.write(
                rels_path, xml_declaration=True, encoding="UTF-8", standalone="yes"
            )

    def _ensure_comments_content_type(self):
        """Ensure ``[Content_Types].xml`` declares a content type for comments.xml.

        Word requires an ``<Override>`` entry declaring the MIME type
        for ``word/comments.xml`` before the file is considered valid.
        Without this, Word silently drops all comments when the
        document is opened. Idempotent — skips if the entry exists.
        """
        ct_path = os.path.join(self.tmpdir, "[Content_Types].xml")
        tree = etree.parse(ct_path)
        root = tree.getroot()
        exists = root.xpath(
            "./ct:Override[@PartName='/word/comments.xml']", namespaces=CT_NS
        )
        if not exists:
            etree.SubElement(
                root,
                f"{{{CT_NS['ct']}}}Override",
                PartName="/word/comments.xml",
                ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml",
            )
            tree.write(
                ct_path, xml_declaration=True, encoding="UTF-8", standalone="yes"
            )

    def _insert_comment_anchor(self, p_elem, cid):
        """Insert comment range markers around the last run in a paragraph.

        Word comments are anchored via a triple of elements:
        ``<w:commentRangeStart>``, ``<w:commentRangeEnd>``, and
        ``<w:commentReference>``. This helper places them around the
        final ``<w:r>`` of ``p_elem`` so the comment attaches to the
        end of the paragraph text. No-op if the paragraph has no runs.

        Args:
            p_elem: ``<w:p>`` element to anchor the comment in.
            cid: Comment ID matching the entry in ``comments.xml``.
        """
        # find the last run in this paragraph
        runs = p_elem.findall(".//w:r", namespaces=NS)
        if not runs:
            return
        last = runs[-1]
        parent = last.getparent()
        idx = parent.index(last)

        # Use XMLElementBuilder to create comment range elements
        crs, cre, rref = XMLElementBuilder.create_comment_range_elements(cid)

        # splice them in right after the last run
        parent.insert(idx, crs)
        parent.insert(idx + 2, cre)
        parent.insert(idx + 3, rref)

    def _load_or_create_comments_tree(self):
        """Return the ``comments.xml`` tree, creating an empty one if needed.

        Looks up ``word/comments.xml`` in three places, in order:
        (1) the already-loaded ``self.trees`` cache, (2) the filesystem
        under the temp directory, or (3) creates a fresh empty
        ``<w:comments>`` element. In all three cases the tree is
        registered in ``self.trees`` so subsequent writes persist it.

        Returns:
            Tuple of (filesystem path, ``lxml`` ElementTree).
        """
        comments_path = os.path.join(self.tmpdir, "word", "comments.xml")

        if comments_path in self.trees:
            # already loaded at init
            tree = self.trees[comments_path]

        elif os.path.exists(comments_path):
            # parse it and register it
            tree = etree.parse(comments_path)
            self.trees[comments_path] = tree

        else:
            # create a brand-new comments.xml and register it.
            # Must bind the `w:` prefix via XMLElementBuilder.create_part_root:
            # a bare etree.Element() here serialises the root (and every
            # comment under it) with lxml's auto `ns0:` prefix, which Word
            # silently ignores — the comments end up in the file but invisible.
            root = XMLElementBuilder.create_part_root("comments")
            tree = etree.ElementTree(root)
            self.trees[comments_path] = tree

        return comments_path, tree

    def _zip(self, output_path):
        """Repack the temp directory into a .docx archive and clean up.

        Walks the entire temp directory (which holds every file from
        the original archive, possibly with modifications) and writes
        each file into a new zip at ``output_path`` using deflate
        compression. After writing, removes the temp directory.

        This is the terminal step of :meth:`write` — once zipped, the
        editor instance should not be used further because its
        working directory no longer exists.
        """
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as z:
            for folder, _, files in os.walk(self.tmpdir):
                for fn in files:
                    full = os.path.join(folder, fn)
                    arc = os.path.relpath(full, self.tmpdir)
                    z.write(full, arc)
        shutil.rmtree(self.tmpdir)

    def _build_change_container(self, change_type, cid, rPr, text, change_author=None):
        """
        Build one <w:del> or <w:ins> element containing a single <w:r> whose
        text is the given 'text' and whose rPr is copied from the first run.
        """
        container_author = change_author or self.author
        if container_author == self.author and change_author is None:
            logger.warning(
                f"Change container using fallback author '{self.author}' for text '{text[:20]}...', change_author: {change_author}"
            )
        # Monitor for Sequential Agent attributions (should be rare now)
        if container_author == "Sequential Agent":
            logger.warning(
                f"Sequential Agent container: {change_type} '{text[:15]}...' - check attribution logic"
            )

        return XMLElementBuilder.create_change_container(
            change_type=change_type,
            cid=cid,
            author=container_author,
            text=text,
            rPr=rPr,
        )

    def _get_author_from_element(self, element) -> str:
        """Extract author from an existing track change element."""
        if element is not None:
            author = element.get(qn("w:author"))
            if author:
                return author
            else:
                # Log when element has no author attribute
                tag_name = (
                    etree.QName(element.tag).localname if element.tag else "unknown"
                )
                logger.debug(
                    f"Element {tag_name} has no w:author attribute, falling back to '{self.author}'"
                )
        return self.author

    def _parse_inline_markdown(self, text: str, parent_elem):
        """
        Parse inline markdown formatting and create Word XML run elements.

        Handles **bold** and *italic* within text, creating properly formatted runs.

        Args:
            text: Text with inline markdown formatting
            parent_elem: Parent XML element to append runs to
        """
        if not text:
            return

        # Pattern to match **bold** or *italic* (non-greedy)
        pattern = r"(\*\*.*?\*\*|\*.*?\*)"

        parts = re.split(pattern, text)

        for part in parts:
            if not part:
                continue

            r_elem = etree.SubElement(parent_elem, f"{{{NS['w']}}}r")

            if part.startswith("**") and part.endswith("**") and len(part) > 4:
                # Bold text
                rPr = etree.SubElement(r_elem, f"{{{NS['w']}}}rPr")
                etree.SubElement(rPr, f"{{{NS['w']}}}b")
                t_elem = etree.SubElement(r_elem, f"{{{NS['w']}}}t")
                t_elem.text = part[2:-2]
            elif (
                part.startswith("*")
                and part.endswith("*")
                and len(part) > 2
                and not part.startswith("**")
            ):
                # Italic text
                rPr = etree.SubElement(r_elem, f"{{{NS['w']}}}rPr")
                etree.SubElement(rPr, f"{{{NS['w']}}}i")
                t_elem = etree.SubElement(r_elem, f"{{{NS['w']}}}t")
                t_elem.text = part[1:-1]
            else:
                # Regular text
                t_elem = etree.SubElement(r_elem, f"{{{NS['w']}}}t")
                t_elem.text = part
                # Preserve space if text has leading/trailing spaces
                if part and (part[0].isspace() or part[-1].isspace()):
                    t_elem.set(
                        "{http://www.w3.org/XML/1998/namespace}space", "preserve"
                    )

    def prepend_review_section(self, review_markdown: str) -> None:
        """
        Prepend review report as formatted content at document beginning.

        Parses the markdown into Word paragraphs with appropriate styles:

          • ``# / ## / ### / #### / ##### / ######`` → Heading 1–6
          • ``- item`` or ``* item``                  → ``List Bullet`` style
          • ``1. item`` (any leading digits + dot)    → ``List Number`` style
          • ``> text``                                → ``Quote`` style
          • ``---``                                   → horizontal rule
          • ``| a | b |`` (+ a ``| --- |`` row)       → a real Word table
          • ``**whole line**`` / ``*whole line*``     → bold / italic paragraph
          • everything else                           → regular paragraph

        Inline ``**bold**`` and ``*italic*`` are applied inside any
        block (including headings, list items, and table cells) via
        ``_parse_inline_markdown``.

        Tables are the one multi-line construct, so the markdown is
        converted block-at-a-time by
        ``_convert_markdown_to_block_elements`` rather than line-at-a-time.
        The resulting ``<w:tbl>`` elements are inserted into the body but
        are deliberately NOT registered in ``self.paragraphs``: that list
        drives the run-level diff/track-changes machinery in
        :meth:`write` and the patch matching in ``PatchDocxEditor``, and
        a table is neither an editable manuscript paragraph nor a valid
        input to either (it has no direct-child ``<w:r>`` runs).

        After the parsed content, the method appends one final paragraph
        containing a hard page break so the original manuscript body
        starts on its own page — visually separating the review report
        from the document being reviewed.

        Args:
            review_markdown: Markdown-formatted review report
        """
        doc_path = self._doc_path
        tree = self.trees[doc_path]
        root = tree.getroot()
        body = root.find(".//w:body", namespaces=NS)

        if body is None:
            return  # No body to prepend to

        elements_to_prepend = self._convert_markdown_to_block_elements(review_markdown)
        # Trailing page break separates the review report from the
        # manuscript that follows. Always added — review reports want
        # a clean visual boundary.
        elements_to_prepend.append(self._build_page_break_paragraph())

        first_child = body[0] if len(body) > 0 else None

        # Create ParagraphData objects WITHOUT re-extracting from XML so
        # any changes made by apply_patches() are preserved.
        prepended_paragraphs = []
        for elem in reversed(elements_to_prepend):
            if first_child is not None:
                body.insert(0, elem)
            else:
                body.append(elem)

            if elem.tag != f"{{{NS['w']}}}p":
                # A <w:tbl> is in the body but out of the paragraph model
                # — see this method's docstring for why.
                continue

            para_data = ParagraphData(doc_path, elem)
            para_data.is_review_report = True
            prepended_paragraphs.insert(0, para_data)

        self.paragraphs = prepended_paragraphs + self.paragraphs

    # ── markdown → Word paragraph helpers ─────────────────────────────

    def _convert_markdown_to_block_elements(self, review_markdown: str) -> list:
        """Convert review markdown into a flat list of Word body elements.

        Almost every line maps 1:1 onto a ``<w:p>`` via
        :meth:`_build_review_paragraph`. Tables are the exception: they
        are the only construct in the report vocabulary that spans
        several lines, so consecutive pipe-delimited lines are buffered
        and handed to :meth:`_build_markdown_table_elements` as a block.
        A line-at-a-time cascade structurally cannot build a table, which
        is why this wrapper exists rather than another branch in
        ``_build_review_paragraph``.
        """
        elements: list = []
        table_buffer: list[str] = []

        for line in review_markdown.split("\n"):
            if self._is_markdown_table_row(line):
                table_buffer.append(line)
                continue
            if table_buffer:
                elements.extend(self._build_markdown_table_elements(table_buffer))
                table_buffer = []
            elements.append(self._build_review_paragraph(line))

        if table_buffer:
            elements.extend(self._build_markdown_table_elements(table_buffer))

        return elements

    def _build_review_paragraph(self, line: str):
        """Convert one markdown line into a Word `<w:p>` element.

        Returns the constructed XML element. The line classification
        cascade is order-sensitive: more specific patterns first.
        """
        # Match heading prefixes (# through ######)
        for level in range(6, 0, -1):
            prefix = "#" * level + " "
            if line.startswith(prefix):
                return self._make_styled_paragraph(
                    style_name=f"Heading{level}",
                    body=line[len(prefix) :],
                )

        stripped = line.strip()

        # Horizontal rule
        if stripped == "---":
            return self._make_horizontal_rule_paragraph()

        # Bullet list:  "- item" or "* item"  (single space; not "**")
        if (
            len(line) >= 2
            and line[0] in ("-", "*")
            and line[1] == " "
            and not line.startswith("**")
        ):
            return self._make_styled_paragraph(
                style_name="ListBullet",
                body=line[2:],
            )

        # Numbered list:  "1. item" / "12. item"
        if stripped and stripped[0].isdigit():
            digits_end = 0
            while digits_end < len(stripped) and stripped[digits_end].isdigit():
                digits_end += 1
            if (
                digits_end > 0
                and digits_end + 1 < len(stripped)
                and stripped[digits_end] == "."
                and stripped[digits_end + 1] == " "
            ):
                return self._make_styled_paragraph(
                    style_name="ListNumber",
                    body=stripped[digits_end + 2 :],
                )

        # Blockquote
        if line.startswith("> "):
            return self._make_styled_paragraph(
                style_name="Quote",
                body=line[2:],
            )
        if stripped == ">":
            return self._make_styled_paragraph(style_name="Quote", body="")

        # Whole-line bold / italic (no surrounding text)
        if line.startswith("**") and line.endswith("**") and len(line) > 4:
            return self._make_inline_marked_paragraph(line[2:-2], bold=True)
        if (
            line.startswith("*")
            and line.endswith("*")
            and not line.startswith("**")
            and len(line) > 2
        ):
            return self._make_inline_marked_paragraph(line[1:-1], italic=True)

        # Regular paragraph — empty lines preserve spacing
        return self._make_styled_paragraph(style_name=None, body=line)

    def _make_styled_paragraph(self, *, style_name, body: str):
        """Build a paragraph with the given style; body uses inline markdown."""
        p_elem = etree.Element(f"{{{NS['w']}}}p")
        if style_name:
            pPr = etree.SubElement(p_elem, f"{{{NS['w']}}}pPr")
            pStyle = etree.SubElement(pPr, f"{{{NS['w']}}}pStyle")
            pStyle.set(f"{{{NS['w']}}}val", style_name)

        if body.strip():
            self._parse_inline_markdown(body, p_elem)
        else:
            # Empty paragraph — preserve vertical spacing in the docx
            r_elem = etree.SubElement(p_elem, f"{{{NS['w']}}}r")
            t_elem = etree.SubElement(r_elem, f"{{{NS['w']}}}t")
            t_elem.text = ""
            t_elem.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        return p_elem

    def _make_horizontal_rule_paragraph(self):
        """Empty paragraph with a bottom border = a markdown horizontal rule."""
        p_elem = etree.Element(f"{{{NS['w']}}}p")
        pPr = etree.SubElement(p_elem, f"{{{NS['w']}}}pPr")
        pBdr = etree.SubElement(pPr, f"{{{NS['w']}}}pBdr")
        bottom = etree.SubElement(pBdr, f"{{{NS['w']}}}bottom")
        bottom.set(f"{{{NS['w']}}}val", "single")
        bottom.set(f"{{{NS['w']}}}sz", "6")
        bottom.set(f"{{{NS['w']}}}space", "1")
        bottom.set(f"{{{NS['w']}}}color", "auto")
        return p_elem

    def _make_inline_marked_paragraph(
        self, text: str, *, bold: bool = False, italic: bool = False
    ):
        """Single-run paragraph with bold/italic applied to its only run."""
        p_elem = etree.Element(f"{{{NS['w']}}}p")
        r_elem = etree.SubElement(p_elem, f"{{{NS['w']}}}r")
        rPr = etree.SubElement(r_elem, f"{{{NS['w']}}}rPr")
        if bold:
            etree.SubElement(rPr, f"{{{NS['w']}}}b")
        if italic:
            etree.SubElement(rPr, f"{{{NS['w']}}}i")
        t_elem = etree.SubElement(r_elem, f"{{{NS['w']}}}t")
        t_elem.text = text
        return p_elem

    # ── markdown table → Word table ───────────────────────────────────

    @staticmethod
    def _is_markdown_table_row(line: str) -> bool:
        """True for a line shaped like a table row: ``| a | b |``.

        Shape only. Whether a buffered run of such lines is *actually* a
        table is decided by :meth:`_build_markdown_table_elements`, which
        additionally requires the ``| --- |`` separator row — a lone
        pipe-wrapped line is ordinary prose.
        """
        stripped = line.strip()
        return (
            len(stripped) >= 2 and stripped.startswith("|") and stripped.endswith("|")
        )

    @staticmethod
    def _split_markdown_table_row(line: str) -> list[str]:
        r"""Split ``| a | b |`` into ``["a", "b"]``.

        Splits on unescaped pipes only and unescapes ``\|`` afterwards,
        so a finding whose text contains a literal pipe does not silently
        gain a column and make the whole table look ragged.
        """
        inner = line.strip()[1:-1]
        return [
            cell.replace("\\|", "|").strip() for cell in re.split(r"(?<!\\)\|", inner)
        ]

    @classmethod
    def _is_markdown_table_separator_row(cls, line: str) -> bool:
        """True for the ``| --- | :---: |`` row that follows a table header."""
        cells = cls._split_markdown_table_row(line)
        return bool(cells) and all(
            re.fullmatch(r":?-+:?", cell) is not None for cell in cells
        )

    def _build_markdown_table_elements(self, rows: list[str]) -> list:
        """Convert a buffered block of pipe-delimited lines into Word XML.

        Returns a single-element list holding a ``<w:tbl>`` when the block
        is a well-formed markdown table, and otherwise one literal
        paragraph per line — the pre-table behaviour.

        Falling back instead of raising is deliberate: this markdown is
        prepended to the author's manuscript, and a malformed table in
        generated report prose must not cost them the entire render. The
        ragged case is logged at warning level so the malformation is
        still visible rather than silently absorbed.
        """
        if len(rows) < 2 or not self._is_markdown_table_separator_row(rows[1]):
            # Pipe-wrapped lines with no separator row are not a table.
            return [self._build_review_paragraph(line) for line in rows]

        header_cells = self._split_markdown_table_row(rows[0])
        body_rows = [self._split_markdown_table_row(line) for line in rows[2:]]

        ragged_count = sum(1 for cells in body_rows if len(cells) != len(header_cells))
        if ragged_count:
            logger.warning(
                "[markdown-table] ragged table: header has %d column(s) but "
                "%d of %d body row(s) differ — rendering the %d line(s) as "
                "literal paragraphs instead of a Word table",
                len(header_cells),
                ragged_count,
                len(body_rows),
                len(rows),
            )
            return [self._build_review_paragraph(line) for line in rows]

        return [
            self._make_table_element(header_cells=header_cells, body_rows=body_rows)
        ]

    def _make_table_element(
        self, *, header_cells: list[str], body_rows: list[list[str]]
    ):
        """Build a ``<w:tbl>`` with a bolded, page-repeating header row."""
        tbl = etree.Element(f"{{{NS['w']}}}tbl")

        tblPr = etree.SubElement(tbl, f"{{{NS['w']}}}tblPr")
        tblW = etree.SubElement(tblPr, f"{{{NS['w']}}}tblW")
        tblW.set(f"{{{NS['w']}}}w", "5000")  # 100% in fiftieths of a percent
        tblW.set(f"{{{NS['w']}}}type", "pct")
        # Borders are written out explicitly rather than referenced via
        # <w:tblStyle w:val="TableGrid"/>: the manuscript being reviewed
        # owns styles.xml and need not define any table style, and a
        # dangling style reference renders as an invisible grid.
        tblBorders = etree.SubElement(tblPr, f"{{{NS['w']}}}tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            border = etree.SubElement(tblBorders, f"{{{NS['w']}}}{edge}")
            border.set(f"{{{NS['w']}}}val", "single")
            border.set(f"{{{NS['w']}}}sz", "4")
            border.set(f"{{{NS['w']}}}space", "0")
            border.set(f"{{{NS['w']}}}color", "auto")

        # Equal columns across a 6.5in text block (9360 twips). Word
        # re-flows these to fit content; they only need to be present and
        # sane, since tblW above asks for the full text width.
        column_width = 9360 // max(len(header_cells), 1)
        tblGrid = etree.SubElement(tbl, f"{{{NS['w']}}}tblGrid")
        for _ in header_cells:
            gridCol = etree.SubElement(tblGrid, f"{{{NS['w']}}}gridCol")
            gridCol.set(f"{{{NS['w']}}}w", str(column_width))

        tbl.append(self._make_table_row(cells=header_cells, is_header=True))
        for cells in body_rows:
            tbl.append(self._make_table_row(cells=cells, is_header=False))

        return tbl

    def _make_table_row(self, *, cells: list[str], is_header: bool):
        """Build one ``<w:tr>``; each cell gets the inline-markdown treatment."""
        tr = etree.Element(f"{{{NS['w']}}}tr")
        if is_header:
            trPr = etree.SubElement(tr, f"{{{NS['w']}}}trPr")
            # Repeat the header when the table spans a page break.
            etree.SubElement(trPr, f"{{{NS['w']}}}tblHeader")

        for cell in cells:
            tc = etree.SubElement(tr, f"{{{NS['w']}}}tc")
            tcPr = etree.SubElement(tc, f"{{{NS['w']}}}tcPr")
            tcW = etree.SubElement(tcPr, f"{{{NS['w']}}}tcW")
            tcW.set(f"{{{NS['w']}}}w", "0")
            tcW.set(f"{{{NS['w']}}}type", "auto")
            # Same builder as a normal paragraph, so **bold** / *italic*
            # inside a cell renders as formatting, not as literal asterisks.
            cell_paragraph = self._make_styled_paragraph(style_name=None, body=cell)
            if is_header:
                self._apply_bold_to_runs(cell_paragraph)
            tc.append(cell_paragraph)

        return tr

    @staticmethod
    def _apply_bold_to_runs(p_elem) -> None:
        """Mark every run in a paragraph bold.

        Applied *after* ``_parse_inline_markdown`` rather than by wrapping
        the cell text in ``**``: pre-wrapping would collide with any
        markers the header cell already contains. ``<w:b/>`` is inserted
        at the front of ``<w:rPr>`` because ECMA-376 orders ``b`` before
        the ``i`` that inline parsing may already have emitted.
        """
        for r_elem in p_elem.findall(f"{{{NS['w']}}}r"):
            rPr = r_elem.find(f"{{{NS['w']}}}rPr")
            if rPr is None:
                rPr = etree.Element(f"{{{NS['w']}}}rPr")
                r_elem.insert(0, rPr)
            if rPr.find(f"{{{NS['w']}}}b") is None:
                rPr.insert(0, etree.Element(f"{{{NS['w']}}}b"))

    def _build_page_break_paragraph(self):
        """A paragraph containing a hard page break — separates review from body."""
        p_elem = etree.Element(f"{{{NS['w']}}}p")
        r_elem = etree.SubElement(p_elem, f"{{{NS['w']}}}r")
        br_elem = etree.SubElement(r_elem, f"{{{NS['w']}}}br")
        br_elem.set(f"{{{NS['w']}}}type", "page")
        return p_elem

    def write(self, output_path):
        """Serialize the document to a new ``.docx``.

        The document is already in the state the caller asked for: edits were
        applied as surgery on the runs they covered, so there is nothing to
        rebuild here. All that remains is to emit the comment bodies and
        bracket the runs each one was resolved to.

        This method used to clear every touched paragraph's runs and
        regenerate them from a string, which destroyed line breaks, tabs and
        any tracked change already in the paragraph — including a co-author's.

        Args:
            output_path: Path for the output ``.docx`` file.
        """
        comments_path, comments_tree = self._load_or_create_comments_tree()
        comments_root = comments_tree.getroot()

        for cid, placed in enumerate(self.placed_comments, start=1):
            comments_root.append(
                XMLElementBuilder.create_comment_element(
                    cid=cid, author=placed.author, text=placed.text
                )
            )
            self._place_comment_range(placed.start_run, placed.end_run, cid)

        if self.placed_comments:
            self._ensure_comments_relationship()
            self._ensure_comments_content_type()

        for path, tree in self.trees.items():
            tree.write(path, xml_declaration=True, encoding="UTF-8", standalone="yes")
        self._zip(output_path)

    def _place_comment_range(self, start_run, end_run, cid) -> None:
        """Bracket the runs ``start_run``..``end_run`` with comment markers.

        ``start_run`` and ``end_run`` are ``<w:r>`` elements (possibly in
        different paragraphs). Inserts ``commentRangeStart`` immediately
        before ``start_run`` and ``commentRangeEnd`` + the reference run
        immediately after ``end_run``. End markers are inserted first so
        inserting the start marker can't shift the end run's index.
        """
        crs, cre, rref = XMLElementBuilder.create_comment_range_elements(cid)

        end_parent = end_run.getparent()
        end_idx = end_parent.index(end_run)
        end_parent.insert(end_idx + 1, cre)
        end_parent.insert(end_idx + 2, rref)

        start_parent = start_run.getparent()
        start_parent.insert(start_parent.index(start_run), crs)
