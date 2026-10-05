"""Cross-reference and citation-field integrity (spec section 4, Tier 0).

What it checks, on a ``DocumentView`` only:

- every ``crossref`` field names a target (``data["target"]``), and, when the
  reader recorded whether that bookmark exists (``data["target_exists"]`` is a
  bool), the target exists;
- every citation field of a reference manager (zotero, endnote, mendeley)
  carries parsed data (``data`` is not empty and has no ``parse_error``, the
  key the reader sets when the payload does not parse);
- a bibliography field is present when any manager citation field is;
- "Figure N" / "Table N" mentions in running text match a caption. A caption is
  a paragraph whose style contains "caption", or one that starts with the label,
  the number and a ``.``, ``:`` or dash ("Figure 3. ...").

Coverage limits: the view carries no bookmark list, so a REF/PAGEREF/NOTEREF
target is verified only through ``target_exists``, which the reader sets from
the document's ``w:bookmarkStart`` names (SEQ fields name a sequence, not a
bookmark, and get no flag). Mentions are found
for the first number only ("Figures 1 and 2" checks Figure 1; "Figure 1-3" is
skipped). A mention is an error when captions of that kind exist and none has
the number; a warning when the document has no caption of that kind at all
(the figures may be unlabelled images). Plain-text citations (no field) are not
looked at here.
"""

from __future__ import annotations

import re

from ..kernel import (
    CheckContext,
    CheckSpec,
    Diagnostic,
    DocumentView,
    FieldToken,
    Paragraph,
    register,
)

_MANAGERS = {"zotero", "endnote", "mendeley"}
_MENTION = re.compile(
    r"\b(?P<label>Figure|Fig\.|Table)s?\s+(?P<num>S?\d+)(?![\d–—-]\d)",
)
_CAPTION_TEXT = re.compile(
    r"^\s*(?P<label>Figure|Fig\.|Table)\s+(?P<num>S?\d+)\s*[.:–—-]",
)


def _kind(label: str) -> str:
    return "table" if label == "Table" else "figure"


def _is_caption_style(paragraph: Paragraph) -> bool:
    return "caption" in paragraph.style.lower()


class CrossrefsCheck:
    spec = CheckSpec(id="crossrefs", tier=0, scope="document", needs=[])
    version = "1"

    def run(self, view: DocumentView, ctx: CheckContext) -> list[Diagnostic]:
        out: list[Diagnostic] = []
        out.extend(self._fields(view))
        out.extend(self._mentions(view))
        return out

    def _diag(
        self,
        severity: str,
        message: str,
        *,
        paragraph_id: str | None,
        start: int | None,
        end: int | None,
        evidence: list[str],
        kind: str,
    ) -> Diagnostic:
        return Diagnostic(
            check=self.spec.id,
            tier=self.spec.tier,
            severity=severity,  # type: ignore[arg-type]
            paragraph_id=paragraph_id,
            start=start,
            end=end,
            message=message,
            evidence=evidence,
            data={"kind": kind},
        )

    def _field_diag(
        self, severity: str, message: str, token: FieldToken, *, kind: str
    ) -> Diagnostic:
        return self._diag(
            severity,
            message,
            paragraph_id=token.paragraph_id,
            start=token.start,
            end=token.end,
            evidence=[token.display],
            kind=kind,
        )

    def _fields(self, view: DocumentView) -> list[Diagnostic]:
        out: list[Diagnostic] = []
        managed = 0
        has_bibliography = False
        for token in view.fields:
            if token.kind == "crossref":
                target = token.data.get("target")
                if not isinstance(target, str) or not target.strip():
                    out.append(
                        self._field_diag(
                            "error",
                            "Cross-reference field has no target.",
                            token,
                            kind="no_target",
                        )
                    )
                elif token.data.get("target_exists") is False:
                    out.append(
                        self._field_diag(
                            "error",
                            f"Cross-reference target {target!r} does not exist.",
                            token,
                            kind="dangling_target",
                        )
                    )
            elif token.kind == "citation" and token.manager in _MANAGERS:
                managed += 1
                if not token.data or "parse_error" in token.data:
                    reason = token.data.get("parse_error", "no data")
                    raw = token.data.get("raw")
                    out.append(
                        self._diag(
                            "error",
                            f"{token.manager} citation field has no parseable data "
                            f"({reason}).",
                            paragraph_id=token.paragraph_id,
                            start=token.start,
                            end=token.end,
                            evidence=[token.display]
                            + ([str(raw)[:300]] if raw else []),
                            kind="unparseable_field",
                        )
                    )
            elif token.kind == "bibliography":
                has_bibliography = True
        if managed and not has_bibliography:
            out.append(
                self._diag(
                    "warning",
                    f"{managed} citation field(s) but no bibliography field.",
                    paragraph_id=None,
                    start=None,
                    end=None,
                    evidence=[],
                    kind="no_bibliography",
                )
            )
        return out

    def _mentions(self, view: DocumentView) -> list[Diagnostic]:
        captions: dict[str, set[str]] = {"figure": set(), "table": set()}
        caption_ids: set[str] = set()
        for paragraph in view.paragraphs:
            match = _CAPTION_TEXT.match(paragraph.text)
            if match:
                captions[_kind(match["label"])].add(match["num"])
                caption_ids.add(paragraph.id)
            elif _is_caption_style(paragraph):
                caption_ids.add(paragraph.id)
                found = re.search(r"(Figure|Fig\.|Table)\s+(S?\d+)", paragraph.text)
                if found:
                    captions[_kind(found[1])].add(found[2])
        out: list[Diagnostic] = []
        for paragraph in view.paragraphs:
            if paragraph.id in caption_ids:
                continue
            for match in _MENTION.finditer(paragraph.text):
                kind = _kind(match["label"])
                if match["num"] in captions[kind]:
                    continue
                severity = "error" if captions[kind] else "warning"
                reason = (
                    f"no {kind} caption numbered {match['num']}"
                    if captions[kind]
                    else f"the document has no {kind} captions"
                )
                out.append(
                    self._diag(
                        severity,
                        f"{match.group(0)!r} is mentioned but {reason}.",
                        paragraph_id=paragraph.id,
                        start=match.start(),
                        end=match.end(),
                        evidence=[match.group(0)],
                        kind="missing_caption",
                    )
                )
        return out


register(CrossrefsCheck())
