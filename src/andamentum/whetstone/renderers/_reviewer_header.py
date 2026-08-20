"""The reviewer block, in the two shapes the three renderers need.

``andamentum.whetstone.v3.reviewers.resolve`` builds one
:class:`~andamentum.whetstone.schemas.ReviewerHeader` per run; this module
turns it into text. Markdown and Word share the markdown form (the Word
renderer prepends markdown to the review report it already builds), and HTML
gets typeset atoms — so there are two formatters over one object rather than
three descriptions of the same run that can drift apart.

What it says, and why each part is in it:

* **identity and description** — the reader of a review has to know which
  reviewer wrote it before the findings mean anything. A methods reviewer
  finding nothing wrong with the argument is not a clean bill of health for
  the argument.
* **the brief** — the stance and the bar every finding was held to. It is what
  the model was told, printed for the reader who was not.
* **the fit warning**, when there is one — the run happened anyway, and this
  is the reader's cue to weigh it accordingly.
* **the effective configuration**, each value beside where it came from, so a
  value the profile supplied can be told from one the caller typed.
* **the package version** — with no per-profile version, this is the marker
  that says whether two reports on the same draft were written by the same
  questions. It is printed unconditionally rather than riding the visible
  watermark, which is a flag the caller can turn off.

``header is None`` — the default reviewer — never reaches here: each renderer
checks first and emits nothing, which is what keeps a default run's output
byte-identical to the output it produced before profiles existed.
"""

from __future__ import annotations

from typing import Any

from ..schemas import ReviewerHeader

_HEADING = "Reviewer"


def header_markdown(header: ReviewerHeader) -> str:
    """The reviewer block as markdown, for the ``.md`` and ``.docx`` reports.

    Plain paragraphs and a bulleted list rather than a table: the Word report
    builder renders this text as body paragraphs, and a markdown table would
    arrive there as a row of pipe characters.
    """
    lines: list[str] = [f"## {_HEADING}: {header.identity}", ""]
    if header.description:
        lines += [header.description, ""]
    if header.brief:
        lines += [f"**Brief.** {header.brief}", ""]
    if header.fit_warning:
        lines += [f"**Fit.** {header.fit_warning}", ""]
    return "\n".join(lines).rstrip() + "\n"


def configuration_markdown(header: ReviewerHeader) -> str:
    """The configuration as ONE line, for the foot of the report.

    Separate from the header, and deliberately quiet. What a reader needs
    before the findings mean anything is WHO reviewed and under what brief; the
    exact criteria, probe ids and author string are provenance — worth having
    when you want to check what ran or compare two reports, and not worth the
    weight of a heading and a bulleted list at the top of the page, which is
    what it had. Same information, one italic line, at the end.
    """
    parts = [f"{line.label}: {line.value} ({line.source})" for line in header.configuration]
    version = header.version or "unknown"
    return (
        f"*{header.identity} · "
        + " · ".join(parts)
        + f" · andamentum-whetstone {version}*"
    )


def header_atoms(header: ReviewerHeader) -> list[dict[str, Any]]:
    """The reviewer block as typeset atoms, for the HTML report."""
    atoms: list[dict[str, Any]] = [
        {"kind": "heading", "content": f"{_HEADING}: {header.identity}", "level": 2}
    ]
    if header.description:
        atoms.append({"kind": "prose", "content": header.description})
    if header.brief:
        atoms.append({"kind": "prose", "content": f"**Brief.** {header.brief}"})
    if header.fit_warning:
        atoms.append(
            {"kind": "callout", "tone": "warning", "content": header.fit_warning}
        )
    return atoms


def configuration_atoms(header: ReviewerHeader) -> list[dict[str, Any]]:
    """The configuration as one quiet line, for the foot of the HTML report."""
    return [{"kind": "aside", "content": configuration_markdown(header)}]



