"""Tiered provenance watermarking for whetstone output artifacts.

Three layers:

1. **Invisible metadata** — always on. Embedded into the artifact's
   metadata fields (docx core properties, HTML <meta>, markdown
   HTML-comment header). Survives "Save As" and copy. Recoverable by
   any docx/HTML/markdown-aware tool.

2. **customXml provenance part** (docx only) — always on. A separate
   ``customXml/itemN.xml`` part inside the .docx zip, in our own
   namespace (``urn:andamentum:provenance:v1``). Layer-1 core-properties
   fields (``author``, ``keywords``, ``comments``) are user-editable
   from Word's File→Info pane and from one-click metadata-clear
   workflows; a customXml part requires manual zip surgery to remove,
   so the provenance survives the most common "scrub before sharing"
   actions.

3. **Visible banner** — default ON for standalone review reports
   (markdown / HTML / docx that summarise the review), default OFF
   for ``--apply-patches`` output (the user's manuscript with edits
   applied). Override via ``--visible-watermark`` /
   ``--no-visible-watermark``.

All three layers stamp the same ``provenance_line()``. The
``verify-provenance`` subcommand reads any docx and reports which
layers are present.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

DISCLAIMER_SHORT = (
    "This report was generated for your own drafts. Whetstone is not a "
    "peer-review tool — do not use it on manuscripts other authors have "
    "sent you confidentially."
)

BANNER_TITLE = "AI-generated review content"

DISCLOSURE_REMINDER = (
    "Remember to disclose AI assistance in your methods / acknowledgements "
    "section per your target journal's policy. See "
    "whetstone/RESPONSIBLE_USE.md for suggested wording."
)


def provenance_line(*, model: str | None, version: str | None = None) -> str:
    """Single human-readable provenance line for banners and metadata."""
    import andamentum

    v = version or getattr(andamentum, "__version__", "unknown")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    model_tag = f"model={model}" if model else "no-llm"
    return f"andamentum-whetstone v{v} ({model_tag}) on {timestamp}"


def banner_markdown(*, model: str | None) -> str:
    """Top-of-file banner for the markdown renderer."""
    prov = provenance_line(model=model)
    return f"> **⚠ {BANNER_TITLE}.** {DISCLAIMER_SHORT}\n> \n> *Produced by {prov}.*"


def banner_html_callout() -> dict[str, str]:
    """Visible banner atom for the HTML renderer (a typeset callout)."""
    return {
        "kind": "callout",
        "tone": "warning",
        "content": f"{BANNER_TITLE}. {DISCLAIMER_SHORT}",
    }


def metadata_html_meta_tags(*, model: str | None) -> str:
    """HTML <meta> tags embedded in the typeset output's <head>."""
    prov = provenance_line(model=model)
    return (
        f'<meta name="generator" content="andamentum-whetstone">\n'
        f'<meta name="andamentum:produced-by" content="{prov}">\n'
        f'<meta name="andamentum:ai-generated" content="true">'
    )


def metadata_markdown_comment(*, model: str | None) -> str:
    """HTML-comment header for markdown — recoverable, invisible in renders."""
    prov = provenance_line(model=model)
    return (
        f"<!-- andamentum-whetstone\n"
        f"     produced-by: {prov}\n"
        f"     ai-generated: true\n"
        f"-->"
    )


def stamp_docx_core_properties(docx_path: str | Path, *, model: str | None) -> None:
    """Write AI-provenance markers to docx ``core.xml`` properties.

    Visible in Word's File→Info pane and recoverable by any docx-reading
    tool. Stamps three Dublin Core fields:
      - ``author`` (creator) — appends an AI contributor entry
      - ``keywords`` — appends ``andamentum:ai-generated`` for searchable detection
      - ``comments`` — short AI-provenance line (kept under 255 chars per
        python-docx's enforced limit for that field)

    Idempotent: re-running on a stamped file produces the same result.
    Failure to open / save the docx is best-effort and never raises.
    """
    try:
        from docx import Document as DocxDocument
    except ImportError:
        return

    try:
        doc = DocxDocument(str(docx_path))
    except Exception:
        return
    prov = provenance_line(model=model)

    # Author: append, don't clobber.
    try:
        existing_contrib = doc.core_properties.author or ""
        if "andamentum-whetstone" not in existing_contrib:
            ai_contrib = f"andamentum-whetstone (AI; {prov})"
            doc.core_properties.author = (
                f"{existing_contrib}; {ai_contrib}" if existing_contrib else ai_contrib
            )
    except Exception:
        pass

    # Keywords: append, don't clobber. This is where most detection
    # tooling will look.
    try:
        existing_kw = (doc.core_properties.keywords or "").strip()
        ai_keyword = "andamentum:ai-generated"
        if ai_keyword not in existing_kw:
            doc.core_properties.keywords = (
                f"{existing_kw}, {ai_keyword}" if existing_kw else ai_keyword
            )
    except Exception:
        pass

    # Comments: SHORT line — python-docx enforces a 255-char limit.
    # The full disclaimer lives in the visible banner / RESPONSIBLE_USE.md.
    try:
        ai_desc = f"AI-generated review content. Produced by {prov}."
        existing_desc = (doc.core_properties.comments or "").strip()
        if "andamentum-whetstone" not in existing_desc and len(ai_desc) <= 255:
            doc.core_properties.comments = ai_desc
    except Exception:
        pass

    try:
        doc.save(str(docx_path))
    except Exception:
        return

    # Layer 2: customXml part. Stamped AFTER python-docx saves so we can
    # operate directly on the resulting zip. Best-effort; failure here
    # leaves layer 1 untouched.
    try:
        write_provenance_customxml(docx_path, model=model)
    except Exception:
        return


# ── Layer 2: customXml provenance part ──────────────────────────────────

PROVENANCE_NS = "urn:andamentum:provenance:v1"
"""XML namespace for the provenance customXml part. Versioned so a future
schema can co-exist with v1 readers."""

_PROVENANCE_PART_NAME = "/customXml/andamentum-provenance.xml"
"""Fixed path inside the docx zip. We always use this name (not
``item1.xml`` / ``item2.xml``) so verify-provenance can find it without
walking the Open Packaging Conventions relationship graph."""

_PROVENANCE_REL_TYPE = (
    "http://schemas.andamentum.org/provenance/2026/relationships/provenance"
)
"""Custom relationship type added to ``_rels/.rels`` (the package-root
relationship file). The relationship anchors our customXml part into the
OPC tree; without it, python-docx's package model treats the part as an
orphan and drops it on the next save."""

_PROVENANCE_REL_ID = "rIdAndamentumProvenance"
"""Stable relationship id so re-stamping is idempotent."""


def _build_provenance_xml(*, model: str | None) -> bytes:
    """Build the customXml part body — a single ``<provenance>`` element
    in the andamentum provenance namespace."""
    import andamentum

    v = getattr(andamentum, "__version__", "unknown")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    model_attr = model or "no-llm"
    # Hand-crafted (not lxml) so this layer has no extra runtime
    # dependency. The schema is intentionally tiny — five attributes,
    # one root element — so reader code stays simple.
    body = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<provenance xmlns="{PROVENANCE_NS}"'
        f' generator="andamentum-whetstone"'
        f' version="{v}"'
        f' model="{_xml_escape(model_attr)}"'
        f' produced-at="{timestamp}"'
        ' ai-generated="true"/>\n'
    )
    return body.encode("utf-8")


def _xml_escape(value: str) -> str:
    """Escape the small set of characters that can appear in a model id."""
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_provenance_customxml(docx_path: str | Path, *, model: str | None) -> bool:
    """Embed the provenance customXml part into the docx zip.

    Returns ``True`` if the part was written (or already present and
    refreshed), ``False`` if the file is unreadable / not a zip. Never
    raises — failure is best-effort by design (same contract as
    ``stamp_docx_core_properties``).

    Implementation note: we rewrite the .docx zip rather than using
    python-docx, because python-docx does not expose customXml part
    creation as a public API. The rewrite preserves all other parts
    byte-identically.
    """
    import zipfile
    from io import BytesIO

    path = Path(docx_path)
    try:
        original_bytes = path.read_bytes()
    except OSError:
        return False
    if not _looks_like_zip(original_bytes):
        return False

    new_body = _build_provenance_xml(model=model)
    part_name_no_slash = _PROVENANCE_PART_NAME.lstrip("/")
    content_type = (
        '<Override PartName="' + _PROVENANCE_PART_NAME + '"'
        ' ContentType="application/xml"/>'
    )

    rel_xml = (
        f'<Relationship Id="{_PROVENANCE_REL_ID}"'
        f' Type="{_PROVENANCE_REL_TYPE}"'
        f' Target="{_PROVENANCE_PART_NAME[1:]}"/>'
    )

    buf = BytesIO()
    try:
        with (
            zipfile.ZipFile(BytesIO(original_bytes), "r") as src,
            zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as dst,
        ):
            for item in src.infolist():
                if item.filename == part_name_no_slash:
                    # Skip — we'll re-write below to refresh the timestamp.
                    continue
                data = src.read(item.filename)
                if item.filename == "[Content_Types].xml":
                    data = _ensure_content_type_override(data, content_type)
                elif item.filename == "_rels/.rels":
                    data = _ensure_package_relationship(data, rel_xml)
                dst.writestr(item, data)
            dst.writestr(part_name_no_slash, new_body)
    except (zipfile.BadZipFile, OSError):
        return False

    try:
        path.write_bytes(buf.getvalue())
    except OSError:
        return False
    return True


def _looks_like_zip(data: bytes) -> bool:
    """Cheap PK\x03\x04 signature check — avoids opening a zip we can't read."""
    return len(data) >= 4 and data[:4] == b"PK\x03\x04"


def _ensure_content_type_override(content_types_xml: bytes, override_xml: str) -> bytes:
    """Add the provenance content-type override to ``[Content_Types].xml``
    if it isn't already present. Idempotent — a repeated call is a no-op.

    We do a literal substring check + insert before ``</Types>`` rather
    than parsing the XML. The Content_Types.xml format is stable and the
    override fragment is small + uniquely keyed by ``PartName``, so this
    stays correct without pulling in lxml here.
    """
    text = content_types_xml.decode("utf-8")
    if _PROVENANCE_PART_NAME in text:
        return content_types_xml
    closing = "</Types>"
    idx = text.rfind(closing)
    if idx == -1:
        # Malformed — leave untouched. The provenance part will still be
        # written; only the content-type declaration will be missing,
        # which Word tolerates for customXml parts.
        return content_types_xml
    new_text = text[:idx] + override_xml + text[idx:]
    return new_text.encode("utf-8")


def _ensure_package_relationship(rels_xml: bytes, rel_xml: str) -> bytes:
    """Add the provenance relationship to ``_rels/.rels`` if it isn't
    already present. Idempotent — re-stamping is a no-op.

    Without this relationship, python-docx (and any other OPC-strict
    consumer) treats our customXml part as an unreferenced orphan and
    drops it on the next save. The relationship is anchored at the
    package root so it survives any in-document mutation.
    """
    text = rels_xml.decode("utf-8")
    if _PROVENANCE_REL_ID in text:
        return rels_xml
    closing = "</Relationships>"
    idx = text.rfind(closing)
    if idx == -1:
        return rels_xml
    new_text = text[:idx] + rel_xml + text[idx:]
    return new_text.encode("utf-8")


# ── Verification (read-only) ────────────────────────────────────────────


def read_provenance_markers(docx_path: str | Path) -> dict[str, object]:
    """Inspect a .docx and report which provenance markers are present.

    Returns a dict with these keys:
      - ``core_properties_marker``: bool — keyword ``andamentum:ai-generated``
        in core.xml keywords, or comments referencing the package.
      - ``core_properties_author_marker``: bool — ``andamentum-whetstone``
        in core.xml author/creator field.
      - ``customxml_provenance``: dict | None — parsed attributes from the
        customXml provenance part if present (generator, version, model,
        produced-at, ai-generated), ``None`` otherwise.
      - ``readable``: bool — True if the file opened successfully.

    Read-only — never modifies the input file. Used by the
    ``verify-provenance`` subcommand and by integrity-checking workflows.
    """
    result: dict[str, object] = {
        "readable": False,
        "core_properties_marker": False,
        "core_properties_author_marker": False,
        "customxml_provenance": None,
    }
    path = Path(docx_path)
    try:
        data = path.read_bytes()
    except OSError:
        return result
    if not _looks_like_zip(data):
        return result
    result["readable"] = True

    import zipfile
    from io import BytesIO

    try:
        with zipfile.ZipFile(BytesIO(data), "r") as zf:
            # core.xml — DC keywords + creator.
            try:
                core_xml = zf.read("docProps/core.xml").decode("utf-8", "ignore")
            except KeyError:
                core_xml = ""
            if "andamentum:ai-generated" in core_xml:
                result["core_properties_marker"] = True
            if "andamentum-whetstone" in core_xml:
                result["core_properties_author_marker"] = True

            # customXml provenance part.
            part_name = _PROVENANCE_PART_NAME.lstrip("/")
            try:
                prov_xml = zf.read(part_name).decode("utf-8", "ignore")
            except KeyError:
                prov_xml = ""
            if prov_xml:
                result["customxml_provenance"] = _parse_provenance_attrs(prov_xml)
    except zipfile.BadZipFile:
        return result
    return result


def _parse_provenance_attrs(xml_text: str) -> dict[str, str]:
    """Pull the five attributes off the single ``<provenance>`` element.

    Hand-rolled (not stdlib ``xml.etree``) so an attacker-controlled
    document can't trigger an XML parser CVE in a release-prep pathway.
    The schema is fixed and one-line; a regex over the relevant span is
    sufficient and safe.
    """
    import re as _re

    out: dict[str, str] = {}
    # Only inspect text within a ``<provenance ...>`` element to scope
    # the attribute scan.
    m = _re.search(r"<provenance\b([^>]*)/?>", xml_text)
    if not m:
        return out
    attrs_blob = m.group(1)
    for attr_match in _re.finditer(r'(\w[\w-]*)\s*=\s*"([^"]*)"', attrs_blob):
        out[attr_match.group(1)] = attr_match.group(2)
    return out
