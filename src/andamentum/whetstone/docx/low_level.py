"""Word document editor with track changes support.

This module is the core Word XML engine for the document-review
package. A ``.docx`` file is a zip archive containing XML — this
module unzips it, edits the XML, and zips it back up.

The primary class :class:`DocxEditor` owns the archive, the parsed XML
trees, and the comment parts. It does NOT rewrite paragraph text: the
predecessor design let callers mutate a paragraph's text and rebuilt the
runs from a token diff on :meth:`write`, which destroyed everything the
text projection dropped (line breaks, hyperlinks, a co-author's existing
tracked changes). Tracked changes are now made in :mod:`.surgery`, which
cuts only the runs a resolved span covers, and :meth:`write` emits
comments and re-zips — it rebuilds nothing.

Most consumers should use the higher-level
``whetstone.renderers.render_docx`` function rather than constructing
``DocxEditor`` directly.
"""

import zipfile
import tempfile
import shutil
import os
import logging
from dataclasses import dataclass

from typing import Any

from lxml import etree  # type: ignore
from docx.oxml.ns import qn

from andamentum.docx_edit.atoms import paragraph_atoms, projection
from andamentum.docx_edit.body import body_paragraphs, main_document_part
from andamentum.docx_edit.xml_builder import XMLElementBuilder

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

    @property
    def text(self) -> str:
        """What this paragraph says, from the one text representation."""
        from andamentum.docx_edit.atoms import paragraph_atoms, projection

        return projection(list(paragraph_atoms(self.p_elem)))


class DocxEditor:
    """Low-level Word (.docx) editor with track changes support.

    A Word document is a zip archive of XML files. This class unzips
    the archive into a temporary directory, loads every XML file as an
    ``lxml`` tree, and exposes the body's paragraphs via
    :attr:`paragraph_elements` and :class:`ParagraphData`. Tracked changes
    are applied to those elements in place by :mod:`.surgery`;
    :meth:`write` writes the comment parts and re-zips.

    This is the core engine used by the
    ``render_docx`` renderer. Most callers should use that higher-level
    interface rather than constructing ``DocxEditor`` directly.
    """

    def __init__(self, input_path, author="Python"):
        """Open a Word document and prepare it for editing.

        Unzips the ``.docx`` into a temporary directory, loads all XML
        trees, normalizes runs so each ``<w:r>`` contains exactly one
        token, and extracts the paragraph list.

        Args:
            input_path: Path to an existing ``.docx`` file.
            author: Default author name used for track changes that
                are not explicitly attributed to another agent.

        Raises:
            FileNotFoundError: If ``input_path`` does not exist.
        """
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        self.author = author
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

    # ── markdown → Word paragraph helpers ─────────────────────────────

    # ── markdown table → Word table ───────────────────────────────────

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

        first_cid = self._next_comment_id(comments_root)
        for cid, placed in enumerate(self.placed_comments, start=first_cid):
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

    def _next_comment_id(self, comments_root) -> int:
        """One above every comment id the document already uses.

        A second review round opens a file that already holds comments, and
        numbering from 1 again gave two comments the same ``w:id`` — measured:
        round two produced ``id=1`` twice in ``comments.xml`` and two
        ``commentRangeStart`` markers both claiming id 1. Word resolves that by
        showing one comment in two places, or by offering to repair the file.

        The document's own range markers are counted too, not just
        ``comments.xml``: a marker whose comment body was deleted still owns
        its id as far as Word's anchoring is concerned.
        """
        used = [
            element.get(qn("w:id"))
            for root in (comments_root, self.trees[self._doc_path].getroot())
            for element in root.iter()
            if element.tag
            in (
                qn("w:comment"),
                qn("w:commentRangeStart"),
                qn("w:commentRangeEnd"),
            )
        ]
        numeric = [
            int(value) for value in used if value is not None and value.isdigit()
        ]
        return max(numeric) + 1 if numeric else 1

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
