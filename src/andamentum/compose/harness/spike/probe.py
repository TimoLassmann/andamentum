"""What survived in a file: read it with the harness reader and report."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from ..docx.ooxml import w
from ..docx.reader import parse_docx
from . import make


class ProbeAnchor(BaseModel):
    id: str
    kind: str
    paragraph_id: str
    text: str


class ProbeField(BaseModel):
    kind: str
    manager: str
    display: str
    parse_error: str | None
    data_keys: list[str]
    dois: list[str]


class ProbeComment(BaseModel):
    id: str
    author: str
    text: str
    parent_id: str | None
    resolved: bool
    anchored_text: str


class TrackedChange(BaseModel):
    author: str
    text: str


class ProbeReport(BaseModel):
    path: str
    project_id: str | None
    paragraphs: int
    anchors: list[ProbeAnchor]
    fields: list[ProbeField]
    comments: list[ProbeComment]
    insertions: list[TrackedChange]
    deletions: list[TrackedChange]


def probe(path: Path) -> ProbeReport:
    """Anchors, fields, comment threads, project id and tracked changes in *path*."""
    parsed = parse_docx(path)
    view = parsed.view
    text_of = {p.id: p.text for p in view.paragraphs}

    def tracked(tag: str, text_tag: str) -> list[TrackedChange]:
        return [
            TrackedChange(
                author=element.get(w("author")) or "",
                text="".join(t.text or "" for t in element.iter(w(text_tag))),
            )
            for element in parsed.root.iter(w(tag))
        ]

    error_of = [f.data.get("parse_error") for f in view.fields]
    return ProbeReport(
        path=str(path),
        project_id=view.project_id,
        paragraphs=len(view.paragraphs),
        anchors=[
            ProbeAnchor(
                id=a.id,
                kind=a.kind,
                paragraph_id=a.paragraph_id,
                text=text_of[a.paragraph_id][a.start : a.end],
            )
            for a in view.anchors
        ],
        fields=[
            ProbeField(
                kind=f.kind,
                manager=f.manager,
                display=f.display,
                parse_error=str(error) if error is not None else None,
                data_keys=sorted(f.data),
                dois=_dois(f.data),
            )
            for f, error in zip(view.fields, error_of, strict=True)
            if f.manager != "none"
        ],
        comments=[
            ProbeComment(
                id=c.id,
                author=c.author,
                text=c.text,
                parent_id=c.parent_id,
                resolved=c.resolved,
                anchored_text=text_of.get(c.paragraph_id, "")[c.start : c.end],
            )
            for c in view.comments
        ],
        insertions=tracked("ins", "t"),
        deletions=tracked("del", "delText"),
    )


def _dois(data: dict[str, object]) -> list[str]:
    """DOIs in a Zotero/Mendeley CSL payload or an EndNote record."""
    found: list[str] = []
    items = data.get("citationItems")
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                item_data = item.get("itemData")
                if isinstance(item_data, dict) and isinstance(
                    item_data.get("DOI"), str
                ):
                    found.append(item_data["DOI"])
    cites = data.get("cites")
    if isinstance(cites, list):
        for cite in cites:
            if (
                isinstance(cite, dict)
                and isinstance(cite.get("doi"), str)
                and cite["doi"]
            ):
                found.append(cite["doi"])
    return found


@dataclass(frozen=True)
class Finding:
    name: str
    present: bool
    detail: str


def evaluate(report: ProbeReport) -> list[Finding]:
    """Compare a probe report with what ``make_spike_docx`` wrote.

    ``present`` is about structure (is the element still there), ``detail``
    says what text it holds now; a changed text is reported, not failed,
    because the user edits the returned file on purpose.
    """
    anchors = {a.id: a for a in report.anchors}
    managers = {(f.kind, f.manager): f for f in report.fields}
    findings: list[Finding] = []

    findings.append(
        Finding(
            "project id",
            report.project_id == make.SPIKE_PROJECT_ID,
            repr(report.project_id),
        )
    )
    for anchor_id, expected in (
        (make.SPIKE_NUMBER_ANCHOR, make.SPIKE_NUMBER_TEXT),
        (make.SPIKE_CLAIM_ANCHOR, make.SPIKE_CLAIM_TEXT),
    ):
        anchor = anchors.get(anchor_id)
        findings.append(
            Finding(
                f"anchor {anchor_id}",
                anchor is not None,
                "missing"
                if anchor is None
                else f"{anchor.kind}, text {anchor.text!r}"
                + ("" if anchor.text == expected else f" (was {expected!r})"),
            )
        )
    zotero = managers.get(("citation", "zotero"))
    findings.append(
        Finding(
            "zotero citation field",
            zotero is not None
            and zotero.parse_error is None
            and "10.9999/spike.001" in zotero.dois,
            "missing"
            if zotero is None
            else f"display {zotero.display!r}, parse_error {zotero.parse_error}, dois {zotero.dois}",
        )
    )
    biblio = managers.get(("bibliography", "zotero"))
    findings.append(
        Finding(
            "zotero bibliography field",
            biblio is not None,
            "missing" if biblio is None else repr(biblio.display[:60]),
        )
    )
    endnote = managers.get(("citation", "endnote"))
    findings.append(
        Finding(
            "endnote citation field",
            endnote is not None
            and endnote.parse_error is None
            and "10.9999/spike.002" in endnote.dois,
            "missing"
            if endnote is None
            else f"display {endnote.display!r}, dois {endnote.dois}",
        )
    )
    crossrefs = [f for f in report.fields if f.kind == "crossref"]
    findings.append(
        Finding(
            "word REF and SEQ fields",
            len(crossrefs) >= 2,
            f"{len(crossrefs)} cross-reference fields",
        )
    )
    comments = report.comments
    replies = [c for c in comments if c.parent_id is not None]
    findings.append(
        Finding(
            "comment",
            any(c.text == make.SPIKE_COMMENT_TEXT for c in comments),
            f"{len(comments)} comments",
        )
    )
    findings.append(
        Finding(
            "reply thread",
            any(c.text == make.SPIKE_REPLY_TEXT for c in replies),
            f"{len(replies)} replies; "
            + ", ".join(f"{c.author} -> {c.parent_id}" for c in replies),
        )
    )
    insertion = [i for i in report.insertions if i.text == make.SPIKE_INSERTION_TEXT]
    findings.append(
        Finding(
            "tracked insertion",
            bool(insertion),
            f"{len(report.insertions)} insertions, {len(report.deletions)} deletions",
        )
    )
    return findings


def format_findings(report: ProbeReport, findings: list[Finding]) -> str:
    lines = [f"File: {report.path}"]
    for finding in findings:
        mark = "SURVIVED" if finding.present else "LOST    "
        lines.append(f"  {mark}  {finding.name}: {finding.detail}")
    return "\n".join(lines)
