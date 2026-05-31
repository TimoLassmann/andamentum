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
import difflib
import re
import copy
import logging

from pydantic import BaseModel
from typing import Optional, Any, List

from lxml import etree  # type: ignore
from docx.oxml.ns import qn
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT

from .attribution import ChangeAttributionTracker
from .xml_builder import XMLElementBuilder
from .token_processor import TokenProcessor

# Logger for this module
logger = logging.getLogger(__name__)

# Namespaces
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
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


class ParagraphData:
    """Live representation of a single Word paragraph with edit tracking.

    Holds the original text, the current modified text, per-token
    formatting (``rPr``) mappings, and the attribution tracker that
    records which author changed which tokens. Instances are created
    during ``DocxEditor._extract_paragraphs()`` by walking the
    ``<w:p>`` element of the Word XML and preserving run-level
    formatting so track changes can be serialized back to ``.docx``
    with the original styling intact.
    """

    def __init__(self, xml_path, p_elem):
        """Parse a Word paragraph element into a ParagraphData instance.

        Extracts the full paragraph text, builds a character-to-``rPr``
        mapping so formatting can be restored token-by-token, tokenizes
        the text, and initializes a :class:`ChangeAttributionTracker`
        seeded with the original tokens attributed to ``"Original"``.

        Args:
            xml_path: Filesystem path to the XML file this paragraph
                belongs to (typically ``word/document.xml``).
            p_elem: ``lxml`` element for the ``<w:p>`` node.
        """
        self.xml_path = xml_path
        self.p_elem = p_elem

        # 1) grab the full paragraph text
        texts = p_elem.xpath(".//w:t", namespaces=NS)
        full = "".join(t.text or "" for t in texts)
        self.original = full
        self.modified = full
        self.comment = ""  # Legacy single comment - kept for compatibility
        self.comments = []  # New: List of (author, comment_text) tuples for separate comments
        self.change_author = None  # Legacy single author - kept for compatibility
        # True for paragraphs synthesised by prepend_review_section (the
        # prepended review report). Comment anchoring excludes these so a
        # finding's quote can't match the report's restatement of itself.
        self.is_review_report = False

        # Multi-author attribution tracking
        self.attribution_tracker = ChangeAttributionTracker(full, "Original")

        # 2) build a list mapping each character back to its run's rPr
        char_rprs = []
        for run in p_elem.findall(".//w:r", namespaces=NS):
            rpr = run.find("w:rPr", namespaces=NS)
            run_rpr = copy.deepcopy(rpr) if rpr is not None else None
            run_text = "".join(
                t.text or "" for t in run.findall(".//w:t", namespaces=NS)
            )
            # each character in run_text gets this run_rpr
            char_rprs += [run_rpr] * len(run_text)

        # 3) now split into tokens *once*, attaching the rPr of the first char
        self.tokens = []
        for m in TokenProcessor.TOKEN_REGEX.finditer(full):
            tok = m.group(0)
            start = m.start()
            # fall back to the first non-None rPr if out of bounds
            rpr = (
                char_rprs[start]
                if start < len(char_rprs)
                else next((x for x in char_rprs if x is not None), None)
            )
            self.tokens.append({"text": tok, "rPr": rpr})

        # keep the very first rPr as a default fallback
        self.default_rpr = next(
            (t["rPr"] for t in self.tokens if t["rPr"] is not None), None
        )


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
        self._normalize_runs()

        self.paragraphs = self._extract_paragraphs()

    def _normalize_runs(self):
        """
        Ensure each <w:r> contains exactly one token (whitespace or word).
        """
        doc_path = self._doc_path
        tree = self.trees[doc_path]
        root = tree.getroot()
        for p in root.xpath(".//w:p", namespaces=NS):
            # collect runs to process
            for run in list(p.findall("w:r", namespaces=NS)):
                texts = [t.text or "" for t in run.findall("w:t", namespaces=NS)]
                full = "".join(texts)
                tokens = TokenProcessor.tokenize(full)
                if len(tokens) <= 1:
                    continue
                # get original rPr
                rpr = run.find("w:rPr", namespaces=NS)
                # insert new runs before original
                for tok in tokens:
                    new_run = etree.Element(f"{{{NS['w']}}}r")
                    if rpr is not None:
                        new_run.append(copy.deepcopy(rpr))
                    t_el = etree.Element(f"{{{NS['w']}}}t")
                    t_el.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                    t_el.text = tok
                    new_run.append(t_el)
                    p.insert(p.index(run), new_run)
                # remove original run
                p.remove(run)
        # write back document.xml
        tree.write(doc_path, xml_declaration=True, encoding="UTF-8", standalone="yes")

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
            tree = etree.parse(rels_path)
            ns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
            for rel in tree.xpath(".//r:Relationship", namespaces=ns):
                rel_type = rel.get("Type", "")
                if rel_type.endswith("/officeDocument"):
                    target = rel.get("Target", "")
                    # Target is like "word/document2.xml" — we want just "document2.xml"
                    filename = os.path.basename(target)
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
                if p.xpath(".//w:t", namespaces=NS):
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

    def _get_agent_authors_for_paragraph(self, attribution_tracker) -> List[str]:
        """Get list of agent authors that made changes to this paragraph."""
        if not attribution_tracker or not hasattr(
            attribution_tracker, "change_history"
        ):
            return []

        agent_authors = []
        for change in attribution_tracker.change_history:
            author = change.get("author", "")
            if author and author != "Original" and "Specialist" in author:
                if author not in agent_authors:
                    agent_authors.append(author)

        return agent_authors

    def _merge_adjacent_changes(self, p_elem, attribution_tracker=None):
        """
        Scan p_elem’s children; whenever you see consecutive <w:del> or <w:ins>
        containers, merge each run of them into one by concatenating their text.
        """
        new_children = []
        merge_type = None
        merge_cid = None
        merge_rPr = None
        merge_author = None  # Track author of elements being merged
        buffer = []

        for child in list(p_elem):
            local = etree.QName(child.tag).localname
            if local in ("del", "ins"):
                # pull out its single <w:r> inside
                r_el = child.find("./w:r", namespaces=NS)
                rPr_el = r_el.find("./w:rPr", namespaces=NS)
                # pick the right text element
                if local == "del":
                    t_el = r_el.find("./w:delText", namespaces=NS)
                else:
                    t_el = r_el.find(".//w:t", namespaces=NS)

                txt = t_el.text or ""
                if local == merge_type:
                    # same run type: append text (keep existing merge_author)
                    buffer.append(txt)
                else:
                    # flush previous merge block
                    if merge_type and merge_author:
                        new_children.append(
                            self._build_change_container(
                                merge_type,
                                merge_cid,
                                merge_rPr,
                                "".join(buffer),
                                merge_author,
                            )
                        )
                    # start new merge block - get author from current element
                    merge_type = local
                    merge_cid = child.get(qn("w:id"))
                    merge_rPr = copy.deepcopy(rPr_el) if rPr_el is not None else None
                    merge_author = self._get_author_from_element(
                        child
                    )  # Author from first element of this merge
                    buffer = [txt]
            else:
                # flush any open merge
                if merge_type and merge_author:
                    new_children.append(
                        self._build_change_container(
                            merge_type,
                            merge_cid,
                            merge_rPr,
                            "".join(buffer),
                            merge_author,
                        )
                    )
                    merge_type = None
                    merge_author = None
                    buffer = []
                # keep this non-change node as-is
                new_children.append(child)

        # flush trailing
        if merge_type and merge_author:
            new_children.append(
                self._build_change_container(
                    merge_type, merge_cid, merge_rPr, "".join(buffer), merge_author
                )
            )

        # replace old children
        for c in list(p_elem):
            p_elem.remove(c)
        for c in new_children:
            p_elem.append(c)

    def _normalize_whitespace_changes(self, p_elem, attribution_tracker=None):
        """
        Collapse any <w:del> + <w:ins> pair whose non-whitespace chars are identical.
        Rebuild it as:
            <w:r>…shared-chars…</w:r>
            <w:ins>…only-the-new-whitespace…</w:ins>
        """
        children = list(p_elem)
        out = []
        i = 0

        while i < len(children):
            node = children[i]
            tag = etree.QName(node.tag).localname

            if tag == "del" and i + 1 < len(children):
                nxt = children[i + 1]
                if etree.QName(nxt.tag).localname == "ins":
                    old_t = node.find(".//w:delText", namespaces=NS)
                    new_t = nxt.find(".//w:t", namespaces=NS)
                    old_txt = old_t.text or ""
                    new_txt = new_t.text or ""

                    # if non-space content matches
                    if re.sub(r"\s+", "", old_txt) == re.sub(r"\s+", "", new_txt):
                        # 1) build an unchanged run with the shared text
                        shared = re.sub(r"\s+", "", new_txt)
                        rPr_el = node.find(".//w:rPr", namespaces=NS)
                        rPr = copy.deepcopy(rPr_el) if rPr_el is not None else None

                        keep = etree.Element(f"{{{NS['w']}}}r")
                        if rPr is not None:
                            keep.append(copy.deepcopy(rPr))

                        t0 = etree.Element(f"{{{NS['w']}}}t")
                        t0.set(
                            "{http://www.w3.org/XML/1998/namespace}space", "preserve"
                        )
                        t0.text = shared
                        keep.append(t0)  # ← make sure to append the text!

                        out.append(keep)

                        # 2) build an <ins> of just the new whitespace
                        ws_only = "".join(ch for ch in new_txt if ch.isspace())
                        cid = nxt.get(qn("w:id"))
                        # Get author from the insertion element
                        ins_author = self._get_author_from_element(nxt)
                        ins_cont = self._build_change_container(
                            "ins", cid, rPr, ws_only, ins_author
                        )
                        out.append(ins_cont)

                        i += 2
                        continue

            # otherwise, pass through
            out.append(node)
            i += 1

        # replace paragraph runs
        for c in list(p_elem):
            p_elem.remove(c)
        for c in out:
            p_elem.append(c)

    def _lcp(self, a, b):
        """Return the longest common prefix of two sequences.

        Used by :meth:`_split_affix_changes` to find shared leading
        characters between adjacent deletion and insertion runs so
        the track change can be narrowed to just the differing middle.
        """
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return a[:i]

    def _lcs(self, a, b):
        """Return the longest common suffix of two sequences.

        Implemented by reversing both inputs, taking the longest
        common prefix, and reversing the result. Paired with
        :meth:`_lcp` in :meth:`_split_affix_changes`.
        """
        # longest common suffix
        ra, rb = a[::-1], b[::-1]
        rc = self._lcp(ra, rb)
        return rc[::-1]

    def _split_affix_changes(self, p_elem, attribution_tracker=None):
        """
        For any adjacent <w:del> + <w:ins> where old and new text share
        a long common prefix or suffix (>=3 chars), split them into:
            [prefix run] [del(mid)] [ins(mid)] [suffix run]
        so only the differing middle is tracked.
        """

        children = list(p_elem)
        out = []
        i = 0

        while i < len(children):
            node = children[i]
            tag = etree.QName(node.tag).localname

            # look for del → ins
            if tag == "del" and i + 1 < len(children):
                ins_node = children[i + 1]
                if etree.QName(ins_node.tag).localname == "ins":
                    old_txt = node.find(".//w:delText", namespaces=NS).text or ""
                    new_txt = ins_node.find(".//w:t", namespaces=NS).text or ""

                    # compute shared affixes
                    prefix = self._lcp(old_txt, new_txt)
                    suffix = self._lcs(old_txt, new_txt)

                    # ensure prefix + suffix don't overlap the whole string
                    if len(prefix) >= 3 or len(suffix) >= 3:
                        # drop the overlap if both prefix & suffix cover everything
                        if len(prefix) + len(suffix) > min(len(old_txt), len(new_txt)):
                            suffix = ""

                        # capture formatting
                        rPr_el = node.find(".//w:rPr", namespaces=NS)
                        rPr = copy.deepcopy(rPr_el) if rPr_el is not None else None

                        # 1) unchanged prefix run
                        if prefix:
                            run0 = etree.Element(f"{{{NS['w']}}}r")
                            if rPr is not None:
                                run0.append(copy.deepcopy(rPr))
                            t0 = etree.Element(f"{{{NS['w']}}}t")
                            t0.set(
                                "{http://www.w3.org/XML/1998/namespace}space",
                                "preserve",
                            )
                            t0.text = prefix
                            run0.append(t0)
                            out.append(run0)

                        # 2) delete the old middle
                        mid_old = old_txt[
                            len(prefix) : len(old_txt) - len(suffix) if suffix else None
                        ]
                        if mid_old:
                            cid = node.get(qn("w:id"))
                            # Get author from deletion element
                            del_author = self._get_author_from_element(node)
                            out.append(
                                self._build_change_container(
                                    "del", cid, rPr, mid_old, del_author
                                )
                            )

                        # 3) insert the new middle
                        mid_new = new_txt[
                            len(prefix) : len(new_txt) - len(suffix) if suffix else None
                        ]
                        if mid_new:
                            cid = ins_node.get(qn("w:id"))
                            # Get author from insertion element
                            ins_author = self._get_author_from_element(ins_node)
                            out.append(
                                self._build_change_container(
                                    "ins", cid, rPr, mid_new, ins_author
                                )
                            )

                        # 4) unchanged suffix run
                        if suffix:
                            run1 = etree.Element(f"{{{NS['w']}}}r")
                            if rPr is not None:
                                run1.append(copy.deepcopy(rPr))
                            t1 = etree.Element(f"{{{NS['w']}}}t")
                            t1.set(
                                "{http://www.w3.org/XML/1998/namespace}space",
                                "preserve",
                            )
                            t1.text = suffix
                            run1.append(t1)
                            out.append(run1)

                        i += 2
                        continue

            # otherwise, keep the node
            out.append(node)
            i += 1

        # replace all children
        for c in list(p_elem):
            p_elem.remove(c)
        for c in out:
            p_elem.append(c)

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
          • ``**whole line**`` / ``*whole line*``     → bold / italic paragraph
          • everything else                           → regular paragraph

        Inline ``**bold**`` and ``*italic*`` are applied inside any
        block (including headings and list items) via
        ``_parse_inline_markdown``.

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

        elements_to_prepend = [
            self._build_review_paragraph(line) for line in review_markdown.split("\n")
        ]
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

            para_data = ParagraphData(doc_path, elem)
            text = "".join(t.text or "" for t in elem.xpath(".//w:t", namespaces=NS))
            para_data.modified = text
            para_data.original = text
            para_data.comment = ""
            para_data.is_review_report = True
            prepended_paragraphs.insert(0, para_data)

        self.paragraphs = prepended_paragraphs + self.paragraphs

    # ── markdown → Word paragraph helpers ─────────────────────────────

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

    def _build_page_break_paragraph(self):
        """A paragraph containing a hard page break — separates review from body."""
        p_elem = etree.Element(f"{{{NS['w']}}}p")
        r_elem = etree.SubElement(p_elem, f"{{{NS['w']}}}r")
        br_elem = etree.SubElement(r_elem, f"{{{NS['w']}}}br")
        br_elem.set(f"{{{NS['w']}}}type", "page")
        return p_elem

    def write(self, output_path):
        """Serialize the edited document to a new .docx file.

        This is the most important method on ``DocxEditor``. For each
        paragraph in the document, it:

        1. Skips unmodified paragraphs that have no comments (echo mode).
        2. Clears the existing runs and tokenizes the current ``modified`` text.
        3. Runs ``difflib.ndiff`` between original and new tokens.
        4. Emits plain runs for unchanged tokens, ``<w:del>`` wrappers
           for deletions, and ``<w:ins>`` wrappers for insertions — each
           attributed to the appropriate author via the paragraph's
           :class:`ChangeAttributionTracker`.
        5. Attaches any comments as ``<w:commentRangeStart>`` /
           ``<w:commentRangeEnd>`` / ``<w:commentReference>`` markers
           and writes corresponding ``<w:comment>`` entries to
           ``comments.xml``.
        6. Normalizes adjacent track changes and splits affix overlaps
           so only the truly differing text is marked as changed.

        After all paragraphs are processed the XML trees are written
        back to the temp directory and the whole thing is re-zipped
        via :meth:`_zip`.

        Args:
            output_path: Path for the output ``.docx`` file.
        """
        change_id = 0
        comments_path, comments_tree = self._load_or_create_comments_tree()
        comments_root = comments_tree.getroot()
        # Comments are placed in a single document-level pass AFTER all
        # paragraphs are rebuilt, so each anchors precisely to its own target
        # span (no last-run pile-up). Collected here as
        # (cid, author, comment_text, target_text).
        pending_comments: list[tuple[int, str, str, str]] = []

        for para_idx, pd in enumerate(self.paragraphs):
            # Special handling for page break or review paragraphs
            if pd.comment == "PAGE_BREAK":
                # For dedicated page break paragraphs, just ensure the br element is present
                self._clear_runs(pd.p_elem)
                r_elem = etree.SubElement(pd.p_elem, f"{{{NS['w']}}}r")
                br_elem = etree.SubElement(r_elem, f"{{{NS['w']}}}br")
                br_elem.set(f"{{{NS['w']}}}type", "page")
                continue

            # echo mode: skip unmodified paragraphs unless they're review paragraphs with comments
            if pd.modified == pd.original and not pd.comment and not pd.comments:
                continue

            # rebuild paragraph runs
            self._clear_runs(pd.p_elem)
            orig_tokens = [t["text"] for t in pd.tokens]
            new_tokens = TokenProcessor.tokenize(pd.modified)

            # Get attribution mapping for current paragraph
            author_map = pd.attribution_tracker.get_attribution_for_diff_operation(
                orig_tokens, new_tokens
            )

            # For paragraphs edited by agents, get the primary agent(s) involved
            agent_authors = self._get_agent_authors_for_paragraph(
                pd.attribution_tracker
            )
            primary_agent = agent_authors[0] if agent_authors else None

            i = 0
            j = 0  # Track position in new_tokens for attribution lookup
            for diff in difflib.ndiff(orig_tokens, new_tokens):
                tag = diff[0]
                tok = diff[2:]
                if tag == " ":
                    rPr = pd.tokens[i]["rPr"]
                    self._add_run(pd.p_elem, tok, rPr)
                    i += 1
                    j += 1
                elif tag == "-":
                    rPr = pd.tokens[i]["rPr"]
                    change_id += 1
                    # Use unified attribution: all deletions attributed to primary agent
                    deletion_author = (
                        primary_agent or pd.change_author or "ATTRIBUTION_ERROR"
                    )
                    self._add_run(
                        pd.p_elem,
                        tok,
                        rPr,
                        change_type="del",
                        cid=change_id,
                        change_author=deletion_author,
                    )
                    i += 1
                elif tag == "+":
                    rPr = pd.tokens[i - 1]["rPr"] if i > 0 else pd.default_rpr
                    change_id += 1
                    # Use unified attribution: get author from attribution tracker (should be consistent now)
                    raw_author = author_map.get(j)
                    if raw_author and raw_author != "Original":
                        insertion_author = raw_author
                    else:
                        # Fallback to primary agent for unified attribution
                        insertion_author = (
                            primary_agent or pd.change_author or "ATTRIBUTION_ERROR"
                        )

                    self._add_run(
                        pd.p_elem,
                        tok,
                        rPr,
                        change_type="ins",
                        cid=change_id,
                        change_author=insertion_author,
                    )
                    j += 1

            # IMPORTANT: For comment-only paragraphs, ensure we have runs for comment anchors
            # If no text changes occurred but we have comments, we need runs to attach comments to
            if pd.modified == pd.original and (pd.comments or pd.comment):
                # No text changes but has comments - make sure we have runs
                runs = pd.p_elem.findall(".//w:r", namespaces=NS)
                if not runs:
                    # Recreate original runs from tokens since diff generated no runs
                    for token_data in pd.tokens:
                        rPr = token_data["rPr"]
                        self._add_run(pd.p_elem, token_data["text"], rPr)

            # 2) **collapse** any long runs of adjacent <w:del>/<w:ins>
            self._merge_adjacent_changes(pd.p_elem, pd.attribution_tracker)
            # 3) **new**: normalize any pure-whitespace change pairs
            self._normalize_whitespace_changes(pd.p_elem, pd.attribution_tracker)
            # 4) split off large common-prefix/suffix changes
            self._split_affix_changes(pd.p_elem, pd.attribution_tracker)
            # Create separate comments for each agent. The comment BODY goes
            # into comments.xml now; the in-document RANGE markers are placed
            # in a single document-level pass after this loop (see
            # _place_pending_comments) so each anchors to its own target span.
            if pd.comments:
                for comment_author, comment_text, target_text in pd.comments:
                    cid = change_id + 1
                    change_id = cid
                    comment_el = XMLElementBuilder.create_comment_element(
                        cid=cid, author=comment_author, text=comment_text
                    )
                    comments_root.append(comment_el)
                    pending_comments.append(
                        (cid, comment_author, comment_text, target_text)
                    )

            # Legacy single comment support (fallback)
            elif pd.comment:
                cid = change_id + 1
                change_id = cid
                # 1) insert anchor tags into the paragraph XML
                self._insert_comment_anchor(pd.p_elem, cid)
                # 2) then add the <comment> element to comments.xml as before
                # Use primary agent for unified attribution consistency
                comment_author = primary_agent or pd.change_author or self.author
                if comment_author == self.author:
                    logger.warning(
                        f"Comment using fallback author '{self.author}' instead of agent. primary_agent: {primary_agent}, pd.change_author: {pd.change_author}"
                    )
                comment_el = XMLElementBuilder.create_comment_element(
                    cid=cid, author=comment_author, text=pd.comment
                )
                comments_root.append(comment_el)

        # Place all comment ranges precisely, document-level, in one pass.
        self._place_pending_comments(pending_comments)

        # finalize and write unchanged parts
        self._ensure_comments_relationship()
        self._ensure_comments_content_type()
        for path, tree in self.trees.items():
            tree.write(path, xml_declaration=True, encoding="UTF-8", standalone="yes")
        self._zip(output_path)

    def _place_pending_comments(self, pending) -> None:
        """Place each comment's range markers at its resolved text span.

        Builds a document-level normalised index over the FINAL run
        structure (after all paragraph rebuilds) and, for each pending
        comment, locates its target text and brackets the matching runs
        with ``commentRangeStart`` / ``commentRangeEnd`` /
        ``commentReference``. The span may cross runs and paragraphs.

        Targets were validated to resolve at apply time, so a miss here is
        unexpected (e.g. a paragraph was rebuilt with altered text by an
        edit). Such misses are logged loudly and skipped — never anchored
        to a wrong location.
        """
        if not pending:
            return
        from .anchor import DocIndex

        # Build paragraphs-of-runs over the final document, keyed by run
        # element. Only direct-child runs (the structure produced by the
        # rebuild) are indexed. The prepended review report is excluded so a
        # finding's quote can't anchor onto the report's own restatement of it.
        paragraphs_runs: list[list[tuple[Any, str]]] = []
        for pd in self.paragraphs:
            if getattr(pd, "is_review_report", False):
                continue
            runs: list[tuple[Any, str]] = []
            for r in pd.p_elem.findall(f"{{{NS['w']}}}r"):
                text = "".join(t.text or "" for t in r.findall(f"{{{NS['w']}}}t"))
                if text:
                    runs.append((r, text))
            paragraphs_runs.append(runs)

        index = DocIndex(paragraphs_runs)
        for cid, _author, _text, target in pending:
            span = index.find(target)
            if span is None:
                logger.warning(
                    "[anchor] write-time placement miss for comment %s "
                    "(validated at apply but not found in rebuilt runs): %r",
                    cid,
                    (target or "")[:80],
                )
                continue
            self._place_comment_range(span.start_key, span.end_key, cid)

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


class DocumentReview:
    """Handles conversion of markdown reviews to docx formatted content and prepends them to a document."""

    # Breaking the _convert_markdown_to_paragraphs method into smaller helper methods
    # to reduce cyclomatic complexity
    def __init__(self):
        """Initialize heading style table for markdown -> Word conversion.

        Builds the lookup used when converting markdown headings
        (``#``, ``##``, etc.) to Word paragraph formatting. Each
        heading level maps to font size, bold flag, and paragraph
        alignment. Levels 1-6 mirror HTML ``<h1>``-``<h6>``.
        """
        self.heading_styles = {
            1: {"size": 16, "bold": True, "alignment": WD_PARAGRAPH_ALIGNMENT.CENTER},
            2: {"size": 14, "bold": True, "alignment": WD_PARAGRAPH_ALIGNMENT.LEFT},
            3: {"size": 12, "bold": True, "alignment": WD_PARAGRAPH_ALIGNMENT.LEFT},
            4: {"size": 11, "bold": True, "alignment": WD_PARAGRAPH_ALIGNMENT.LEFT},
            5: {"size": 10, "bold": True, "alignment": WD_PARAGRAPH_ALIGNMENT.LEFT},
            6: {"size": 10, "bold": False, "alignment": WD_PARAGRAPH_ALIGNMENT.LEFT},
        }

    def prepend_review(self, docx_editor: DocxEditor, markdown_content: str) -> None:
        """
        Converts markdown content to docx format and prepends it to the document.

        Args:
            docx_editor: The DocxEditor instance being used
            markdown_content: The markdown formatted review content
        """
        # First get the current paragraphs for reference
        if not docx_editor.paragraphs:
            return

        document_path = self._find_document_path(docx_editor)
        if not document_path:
            logger.error("Could not find document.xml path")
            return

        # Get the main document tree and body element
        document_tree, body = self._get_document_body(docx_editor, document_path)

        # Convert markdown to paragraph data structures
        review_dicts = self._convert_markdown_to_paragraphs(markdown_content)

        # Store current paragraphs to preserve their edits
        existing_paragraphs = docx_editor.paragraphs.copy()

        # Create and insert paragraph elements
        review_paragraph_objects = self._create_review_paragraphs(
            review_dicts, document_path, body
        )

        # Add a dedicated page break paragraph
        page_break_data = self._add_page_break(document_path, body, len(review_dicts))

        # Combine the review paragraphs with the page break and existing paragraphs
        docx_editor.paragraphs = (
            review_paragraph_objects + [page_break_data] + existing_paragraphs
        )

    def _find_document_path(self, docx_editor: DocxEditor) -> Optional[str]:
        """Find the main document XML file path in the docx file."""
        return docx_editor._doc_path

    def _get_document_body(self, docx_editor: DocxEditor, document_path: str):
        """Get the document tree and body element."""
        document_tree = docx_editor.trees[document_path]
        body = document_tree.xpath(".//w:body", namespaces=NS)[0]
        return document_tree, body

    def _create_review_paragraphs(self, review_dicts, document_path, body):
        """Create paragraph elements and insert them at the beginning of the body."""
        review_paragraph_objects = []

        # We need to insert them in reverse order since we're always inserting at the start
        for para_dict in reversed(review_dicts):
            p_elem = self._create_paragraph_element(para_dict)

            # Insert at the beginning of the body
            body.insert(0, p_elem)

            # Create a paragraph data object for this paragraph
            para_data = self._create_paragraph_data(para_dict, document_path, p_elem)

            # Add to our review paragraphs list (insert at beginning to maintain order)
            review_paragraph_objects.insert(0, para_data)

        return review_paragraph_objects

    def _create_paragraph_element(self, para_dict):
        """Create a paragraph XML element with properties."""
        p_elem = etree.Element(f"{{{NS['w']}}}p")

        # Apply paragraph properties (like alignment, indentation, borders, and spacing)
        if any(
            prop in para_dict["props"]
            for prop in ["alignment", "left_indent", "bottom_border", "spacing_after"]
        ):
            self._apply_paragraph_properties(p_elem, para_dict["props"])

        # Add the runs with their styling
        for run_dict in para_dict["runs"]:
            self._add_run_to_paragraph(p_elem, run_dict)

        return p_elem

    def _apply_paragraph_properties(self, p_elem, props):
        """Apply properties like alignment and indentation to a paragraph element."""
        pPr = etree.SubElement(p_elem, f"{{{NS['w']}}}pPr")

        # Apply alignment if present
        if "alignment" in props:
            jc = etree.SubElement(pPr, f"{{{NS['w']}}}jc")
            alignment = props["alignment"]
            if alignment == WD_PARAGRAPH_ALIGNMENT.CENTER:
                jc.set(f"{{{NS['w']}}}val", "center")
            elif alignment == WD_PARAGRAPH_ALIGNMENT.LEFT:
                jc.set(f"{{{NS['w']}}}val", "left")
            elif alignment == WD_PARAGRAPH_ALIGNMENT.RIGHT:
                jc.set(f"{{{NS['w']}}}val", "right")

        # Apply left indentation if present
        if "left_indent" in props:
            ind = etree.SubElement(pPr, f"{{{NS['w']}}}ind")
            # Convert inches to twips (1 inch = 1440 twips)
            left_twips = int(props["left_indent"] * 1440)
            ind.set(f"{{{NS['w']}}}left", str(left_twips))

        # Apply bottom border for horizontal rules
        if "bottom_border" in props and props["bottom_border"]:
            pBdr = etree.SubElement(pPr, f"{{{NS['w']}}}pBdr")
            bottom = etree.SubElement(pBdr, f"{{{NS['w']}}}bottom")
            bottom.set(f"{{{NS['w']}}}val", "single")
            bottom.set(f"{{{NS['w']}}}sz", "6")  # Line thickness
            bottom.set(f"{{{NS['w']}}}space", "1")
            bottom.set(f"{{{NS['w']}}}color", "auto")

        # Apply spacing after paragraph
        if "spacing_after" in props:
            spacing = etree.SubElement(pPr, f"{{{NS['w']}}}spacing")
            # Convert points to twips (1 point = 20 twips)
            spacing_twips = int(props["spacing_after"] * 20)
            spacing.set(f"{{{NS['w']}}}after", str(spacing_twips))

    def _add_run_to_paragraph(self, p_elem, run_dict):
        """Add a run with styling to a paragraph element."""
        r_elem = etree.SubElement(p_elem, f"{{{NS['w']}}}r")
        rPr = etree.SubElement(r_elem, f"{{{NS['w']}}}rPr")

        # Apply run properties
        self._apply_run_properties(rPr, run_dict["props"])

        # Add page break if needed
        if "page_break" in run_dict["props"] and run_dict["props"]["page_break"]:
            br = etree.SubElement(r_elem, f"{{{NS['w']}}}br")
            br.set(f"{{{NS['w']}}}type", "page")
        else:
            # Add the text
            t_elem = etree.SubElement(r_elem, f"{{{NS['w']}}}t")
            t_elem.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            t_elem.text = run_dict["text"]

    def _apply_run_properties(self, rPr, props):
        """Apply properties to a run element."""
        if "bold" in props and props["bold"]:
            etree.SubElement(rPr, f"{{{NS['w']}}}b")

        if "italic" in props and props["italic"]:
            etree.SubElement(rPr, f"{{{NS['w']}}}i")

        if "color" in props:
            color = etree.SubElement(rPr, f"{{{NS['w']}}}color")
            color.set(f"{{{NS['w']}}}val", props["color"])

        if "size" in props:
            sz = etree.SubElement(rPr, f"{{{NS['w']}}}sz")
            sz.set(f"{{{NS['w']}}}val", str(props["size"] * 2))  # Word uses half-points

        if "font_name" in props:
            rFonts = etree.SubElement(rPr, f"{{{NS['w']}}}rFonts")
            rFonts.set(f"{{{NS['w']}}}ascii", props["font_name"])
            rFonts.set(f"{{{NS['w']}}}hAnsi", props["font_name"])

    def _create_paragraph_data(self, para_dict, document_path, p_elem):
        """Create a ParagraphData object from a paragraph dictionary."""
        # Use text from the paragraph dictionary for content
        text = "".join(run["text"] for run in para_dict["runs"])
        para_data = ParagraphData(document_path, p_elem)

        # For review paragraphs, set modified and original to the same value
        # to avoid showing tracked changes in the review section
        para_data.modified = text
        para_data.original = text

        # Override the skip condition in write() by making this a special case
        # We'll add a comment to force the paragraph to be written
        para_data.comment = ""

        return para_data

    def _add_page_break(self, document_path, body, position):
        """Add a page break paragraph after the review content."""
        page_break_p = etree.Element(f"{{{NS['w']}}}p")
        page_break_r = etree.SubElement(page_break_p, f"{{{NS['w']}}}r")
        page_break_br = etree.SubElement(page_break_r, f"{{{NS['w']}}}br")
        page_break_br.set(f"{{{NS['w']}}}type", "page")

        # Insert the page break into the document body
        body.insert(position, page_break_p)

        # Create a ParagraphData object for the page break
        page_break_data = ParagraphData(document_path, page_break_p)
        page_break_data.modified = " "  # Need some content to ensure it's written
        page_break_data.original = " "
        page_break_data.comment = "PAGE_BREAK"  # Special comment to force processing

        return page_break_data

    def _process_heading(self, line: str) -> dict[str, Any]:
        """Process a markdown heading line and return paragraph data"""
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2)

            style = self.heading_styles.get(level, self.heading_styles[6])

            return {
                "runs": [
                    {
                        "text": text,
                        "props": {
                            "bold": style["bold"],
                            "size": style["size"],
                        },
                    }
                ],
                "props": {"alignment": style["alignment"]},
            }

        # Default return if no match (shouldn't happen with proper validation)
        return {
            "runs": [{"text": line, "props": {}}],
            "props": {"alignment": WD_PARAGRAPH_ALIGNMENT.LEFT},
        }

    def _process_unordered_list(self, line: str) -> dict[str, Any]:
        """Process an unordered list item and return paragraph data"""
        list_match = re.match(r"^([-*])\s+(.+)$", line)
        if list_match:
            content_text = list_match.group(2)

            # Process for inline formatting
            formatted_para = self._process_inline_formatting(content_text)

            # Add bullet at the beginning
            runs = [{"text": "• ", "props": {"bold": True}}]
            runs.extend(formatted_para["runs"])

            return {
                "runs": runs,
                "props": {"alignment": WD_PARAGRAPH_ALIGNMENT.LEFT, "left_indent": 0.5},
            }

        # Default return if no match (shouldn't happen with proper validation)
        return {
            "runs": [{"text": line, "props": {}}],
            "props": {"alignment": WD_PARAGRAPH_ALIGNMENT.LEFT},
        }

    def _process_ordered_list(self, line: str, list_item_number: int) -> dict[str, Any]:
        """Process an ordered list item and return paragraph data"""
        list_match = re.match(r"^(\d+)\.?\s+(.+)$", line)
        if list_match:
            content_text = list_match.group(2)

            # Process for inline formatting
            formatted_para = self._process_inline_formatting(content_text)

            # Add number at the beginning
            runs = [{"text": f"{list_item_number}. ", "props": {"bold": True}}]
            runs.extend(formatted_para["runs"])

            return {
                "runs": runs,
                "props": {"alignment": WD_PARAGRAPH_ALIGNMENT.LEFT, "left_indent": 0.5},
            }

        # Default return if no match (shouldn't happen with proper validation)
        return {
            "runs": [{"text": line, "props": {}}],
            "props": {"alignment": WD_PARAGRAPH_ALIGNMENT.LEFT},
        }

    def _process_indented_paragraph(
        self, content: str, indent_level: float
    ) -> dict[str, Any]:
        """Process an indented paragraph and return paragraph data with proper indentation"""
        # Process the content for inline formatting
        formatted_para = self._process_inline_formatting(content)

        # Add left indentation based on indent level
        # indent_level is already in inches
        left_indent = indent_level

        formatted_para["props"]["left_indent"] = left_indent
        return formatted_para

    def _process_horizontal_rule(self) -> dict[str, Any]:
        """Process a horizontal rule and return paragraph data with a bottom border"""
        return {
            "runs": [{"text": " ", "props": {}}],  # Empty paragraph with border
            "props": {"alignment": WD_PARAGRAPH_ALIGNMENT.LEFT, "bottom_border": True},
        }

    def _process_grey_right_aligned(self, content: str) -> dict[str, Any]:
        """Process grey right-aligned text for metadata display"""
        return {
            "runs": [{"text": content, "props": {"color": "808080"}}],  # Grey color
            "props": {"alignment": WD_PARAGRAPH_ALIGNMENT.RIGHT},
        }

    def _process_indented_grey_metadata(self, content: str) -> dict[str, Any]:
        """Process indented grey metadata text with proper indentation"""
        return {
            "runs": [{"text": content, "props": {"color": "808080"}}],  # Grey color
            "props": {
                "alignment": WD_PARAGRAPH_ALIGNMENT.LEFT,
                "left_indent": 0.2,  # 0.2 inches indentation
            },
        }

    def _process_indented_heading(self, content: str) -> dict[str, Any]:
        """Process indented headings (e.g., '  #### Title')"""
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", content)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2)

            # Use the existing heading styles but add indentation
            style = self.heading_styles.get(level, self.heading_styles[6])

            return {
                "runs": [
                    {
                        "text": text,
                        "props": {
                            "bold": style["bold"],
                            "size": style["size"],
                        },
                    }
                ],
                "props": {
                    "alignment": WD_PARAGRAPH_ALIGNMENT.LEFT,  # Left-aligned instead of center
                    "left_indent": 0.2,  # 0.2 inches indentation
                },
            }

        # Fallback if regex doesn't match (shouldn't happen)
        return {
            "runs": [{"text": content, "props": {}}],
            "props": {"alignment": WD_PARAGRAPH_ALIGNMENT.LEFT, "left_indent": 0.2},
        }

    def _process_spacing_paragraph(self) -> dict[str, Any]:
        """Process a spacing paragraph that adds vertical space"""
        return {
            "runs": [{"text": " ", "props": {}}],  # Single space
            "props": {
                "alignment": WD_PARAGRAPH_ALIGNMENT.LEFT,
                "spacing_after": 12,  # Extra spacing after paragraph (in points)
            },
        }

    def _process_inline_formatting(self, line: str) -> dict[str, Any]:
        """Process a line with inline markdown formatting and return paragraph data"""
        paragraph = {"runs": [], "props": {"alignment": WD_PARAGRAPH_ALIGNMENT.LEFT}}

        j = 0
        while j < len(line):
            # Bold text (**bold**)
            if j + 3 < len(line) and line[j : j + 2] == "**" and "**" in line[j + 2 :]:
                end = line.find("**", j + 2)
                paragraph["runs"].append(
                    {"text": line[j + 2 : end], "props": {"bold": True}}
                )
                j = end + 2

            # Italic text (*italic*)
            elif j + 2 < len(line) and line[j] == "*" and "*" in line[j + 1 :]:
                end = line.find("*", j + 1)
                paragraph["runs"].append(
                    {"text": line[j + 1 : end], "props": {"italic": True}}
                )
                j = end + 1

            # Code (`code`)
            elif j + 2 < len(line) and line[j] == "`" and "`" in line[j + 1 :]:
                end = line.find("`", j + 1)
                paragraph["runs"].append(
                    {"text": line[j + 1 : end], "props": {"font_name": "Courier New"}}
                )
                j = end + 1

            # Regular text
            else:
                # Find the next special character
                next_special = len(
                    line
                )  # Default to end of line instead of float('inf')
                for special in ["**", "*", "`"]:
                    pos = line.find(special, j)
                    if pos != -1 and pos < next_special:
                        next_special = pos

                if next_special == len(line):
                    paragraph["runs"].append({"text": line[j:], "props": {}})
                    j = len(line)
                else:
                    paragraph["runs"].append(
                        {"text": line[j:next_special], "props": {}}
                    )
                    j = next_special

        return paragraph

    def _convert_markdown_to_paragraphs(
        self, markdown_content: str
    ) -> list[dict[str, Any]]:
        """
        Converts markdown content to a list of paragraph data dictionaries.

        Args:
            markdown_content: The markdown formatted content

        Returns:
            A list of paragraph data dictionaries ready to be added to the document
        """
        lines = markdown_content.split("\n")
        paragraphs = []
        current_list_type = None
        list_item_number = 0

        i = 0
        while i < len(lines):
            line = lines[i]  # Don't strip here - we need to preserve indentation

            # Skip empty lines
            if not line.strip():
                i += 1
                continue

            # Process indented paragraphs (2+ spaces at start) - CHECK BEFORE STRIPPING!
            indent_match = re.match(r"^( {2,})(.+)$", line)
            if indent_match:
                content = indent_match.group(2)

                # Check if this is an indented ~~~ pattern that should be processed as metadata
                grey_match = re.match(r"^~~~(.+)~~~$", content)
                if grey_match:
                    metadata_content = grey_match.group(1)
                    if metadata_content == "SPACING":
                        # Create spacing paragraph
                        paragraphs.append(self._process_spacing_paragraph())
                    else:
                        # Process as indented grey metadata
                        paragraphs.append(
                            self._process_indented_grey_metadata(metadata_content)
                        )
                    i += 1
                    continue

                # Check if this is an indented heading (e.g., "  #### Title")
                heading_match = re.match(r"^(#{1,6})\s+(.+)$", content)
                if heading_match:
                    # Process as indented heading
                    paragraphs.append(self._process_indented_heading(content))
                    i += 1
                    continue

                # Regular indented paragraph
                # Convert to 0.5cm indentation (approximately 0.2 inches)
                indent_level = 0.2  # Fixed 0.2 inches regardless of spaces
                paragraphs.append(
                    self._process_indented_paragraph(content, indent_level)
                )
                i += 1
                continue

            # Now strip for all other patterns (headings, lists, etc.)
            line = line.strip()

            # Process headings (# Heading) - support levels 1-6
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if heading_match:
                paragraphs.append(self._process_heading(line))
                i += 1
                continue

            # Process unordered lists (- Item or * Item)
            list_match = re.match(r"^([-*])\s+(.+)$", line)
            if list_match:
                if current_list_type != "unordered":
                    current_list_type = "unordered"
                    list_item_number = 0

                paragraphs.append(self._process_unordered_list(line))
                i += 1
                continue

            # Process ordered lists (1. Item)
            list_match = re.match(r"^(\d+)\.?\s+(.+)$", line)
            if list_match:
                if current_list_type != "ordered":
                    current_list_type = "ordered"
                    list_item_number = int(list_match.group(1))
                else:
                    list_item_number += 1

                paragraphs.append(self._process_ordered_list(line, list_item_number))
                i += 1
                continue

            # Process horizontal rules (*** or ---)
            if line in ["***", "---"]:
                paragraphs.append(self._process_horizontal_rule())
                i += 1
                continue

            # Process grey right-aligned text (~~~text~~~)
            grey_match = re.match(r"^~~~(.+)~~~$", line)
            if grey_match:
                content = grey_match.group(1)
                if content == "SPACING":
                    # Create spacing paragraph
                    paragraphs.append(self._process_spacing_paragraph())
                else:
                    paragraphs.append(self._process_grey_right_aligned(content))
                i += 1
                continue

            # Process normal paragraphs with inline formatting
            paragraphs.append(self._process_inline_formatting(line))
            i += 1

        return paragraphs
