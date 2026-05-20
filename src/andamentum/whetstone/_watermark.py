"""Tiered provenance watermarking for whetstone output artifacts.

Two layers:

1. **Invisible metadata** — always on. Embedded into the artifact's
   metadata fields (docx core properties, HTML <meta>, markdown
   HTML-comment header). Survives "Save As" and copy. Recoverable by
   any docx/HTML/markdown-aware tool. The point of this layer is that
   an editor or integrity workflow can discover AI provenance without
   parsing the body text.

2. **Visible banner** — default ON for standalone review reports
   (markdown / HTML / docx that summarise the review), default OFF
   for `--apply-patches` output (the user's manuscript with edits
   applied). Reason: review reports are internal scratch, so an
   explicit banner is honest; the modified manuscript is the user's
   submission artifact, and a visible banner would pollute it. The
   `--visible-watermark` and `--no-visible-watermark` CLI flags
   override the default.
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
    return (
        f"> **⚠ {BANNER_TITLE}.** {DISCLAIMER_SHORT}\n"
        f"> \n"
        f"> *Produced by {prov}.*"
    )


def banner_html_callout() -> dict[str, str]:
    """Visible banner atom for the HTML renderer (a typeset callout)."""
    return {"kind": "callout", "tone": "warning", "content": f"{BANNER_TITLE}. {DISCLAIMER_SHORT}"}


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
        existing_contrib = (doc.core_properties.author or "")
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
