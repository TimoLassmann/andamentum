"""The synthetic SaltLog example project: notes, results, references and a collaborator's draft.

Everything here is invented (the study, the people, the DOIs under the
``10.5555`` test prefix). ``make_example(dest)`` copies the committed files
(``saltlog/notes.md``, ``results/``, ``refs.json``, ``reviews.txt``) into *dest*
and builds ``collaborator.docx`` there: a draft with a threaded comment, a
resolved comment, two open questions, Zotero citations, a bibliography field,
and a hand-edited number (48 participants changed to 52 as a tracked change,
while ``results/summary.csv`` says 48).

The CSL data in the document is read from ``refs.json`` so the two cannot
drift. ``saltlog/fixtures/`` holds Crossref answers in the format of
``bench/fixtures``; ``fixture_transport_for_example`` serves them so tests and
the smoke run need no network.

    uv run python -m andamentum.compose.harness.examples DEST
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import httpx

from ..bench.fixture_http import fixture_transport
from ..bench.planted import load_fixtures
from ..spike.make import (
    build_docx,
    comment_part,
    comments_extended_part,
    complex_field,
    paragraph,
    run,
)

EXAMPLE_DIR = Path(__file__).parent / "saltlog"
FIXTURES_DIR = EXAMPLE_DIR / "fixtures"

_COLLABORATOR = "Dana Reyes"
_WHEN = "2026-10-01T09:00:00Z"
_SCHEMA = (
    "https://github.com/citation-style-language/schema/raw/master/csl-citation.json"
)

Csl = dict[str, object]


@dataclass(frozen=True)
class ExampleProject:
    root: Path
    notes: Path
    results: Path
    library: Path
    collaborator: Path
    reviews: Path


def fixture_transport_for_example() -> httpx.MockTransport:
    """Crossref answers for the example's three works, as an ``httpx`` transport."""
    return fixture_transport(load_fixtures(FIXTURES_DIR))


def _library() -> list[Csl]:
    return json.loads((EXAMPLE_DIR / "refs.json").read_text(encoding="utf-8"))


def _year(item: Csl) -> int:
    issued = item["issued"]
    assert isinstance(issued, dict)
    return issued["date-parts"][0][0]


def _authors(item: Csl) -> list[dict[str, str]]:
    authors = item["author"]
    assert isinstance(authors, list)
    return authors


def _short(item: Csl) -> str:
    authors = _authors(item)
    first = authors[0]["family"]
    if len(authors) == 1:
        return f"({first}, {_year(item)})"
    return f"({first} and {authors[1]['family']}, {_year(item)})"


def _entry(item: Csl) -> str:
    names = ", ".join(f"{a['family']}, {a['given'][0]}." for a in _authors(item))
    return (
        f"{names} ({_year(item)}). {item['title']}. {item['container-title']}, "
        f"{item['volume']}({item['issue']}), {item['page']}."
    )


def _zotero(item: Csl, *, number: int) -> str:
    uri = f"http://zotero.org/users/local/saltlog/items/{str(item['id']).upper()}"
    payload = {
        "citationID": f"SALTLOG-{number}",
        "properties": {
            "formattedCitation": _short(item),
            "plainCitation": _short(item),
            "noteIndex": 0,
        },
        "citationItems": [{"id": uri, "uris": [uri], "itemData": item}],
        "schema": _SCHEMA,
    }
    return complex_field(
        f"ADDIN ZOTERO_ITEM CSL_CITATION {json.dumps(payload)}", _short(item)
    )


def _bibliography(items: list[Csl]) -> str:
    """One field whose result spans one paragraph per entry, as Zotero writes it."""
    begin = (
        '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
        '<w:r><w:instrText xml:space="preserve">ADDIN ZOTERO_BIBL '
        '{"uncited":[],"omitted":[],"custom":[]} CSL_BIBLIOGRAPHY</w:instrText></w:r>'
        '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
    )
    end = '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
    paragraphs = []
    for index, item in enumerate(items):
        content = (begin if index == 0 else "") + run(_entry(item))
        content += end if index == len(items) - 1 else ""
        paragraphs.append(paragraph(content, para_id=f"1000010{index}"))
    return "".join(paragraphs)


def _commented(content: str, *ids: int) -> str:
    starts = "".join(f'<w:commentRangeStart w:id="{i}"/>' for i in ids)
    ends = "".join(f'<w:commentRangeEnd w:id="{i}"/>' for i in ids)
    refs = "".join(f'<w:r><w:commentReference w:id="{i}"/></w:r>' for i in ids)
    return f"{starts}{content}{ends}{refs}"


def _tracked(*, deleted: str, inserted: str) -> str:
    return (
        f'<w:del w:id="201" w:author="{_COLLABORATOR}" w:date="{_WHEN}"><w:r>'
        f'<w:delText xml:space="preserve">{deleted}</w:delText></w:r></w:del>'
        f'<w:ins w:id="202" w:author="{_COLLABORATOR}" w:date="{_WHEN}">'
        f"{run(inserted)}</w:ins>"
    )


def make_collaborator_docx(path: Path) -> Path:
    """Write the collaborator's draft of the SaltLog paper to *path*."""
    okafor, marlowe, lindqvist = _library()
    body = "".join(
        [
            paragraph(
                run("SaltLog: a pilot of meal-by-meal salt logging"),
                style="Title",
                para_id="10000001",
            ),
            paragraph(run("Abstract"), style="Heading1", para_id="10000002"),
            paragraph(
                run("In this single-arm pilot, ")
                + _tracked(deleted="48", inserted="52")
                + run(" adults logged every meal for four weeks. ")
                + _commented(
                    run("Daily logging lowered reported sodium intake by 412 mg."),
                    0,
                    1,
                ),
                para_id="10000003",
            ),
            paragraph(run("Introduction"), style="Heading1", para_id="10000004"),
            paragraph(
                run("Sodium intake exceeds guidelines in most adults ")
                + _zotero(okafor, number=1)
                + run(", and phone-based meal logs are a feasible way to measure it ")
                + _zotero(marlowe, number=2)
                + run(". Self-reported intake is known to be inaccurate ")
                + _zotero(lindqvist, number=3)
                + run("."),
                para_id="10000005",
            ),
            paragraph(run("Methods"), style="Heading1", para_id="10000006"),
            paragraph(
                _commented(run("Sodium was estimated from the logged meals."), 2)
                + run(" Weekly means are shown in Table 1."),
                para_id="10000007",
            ),
            paragraph(
                '<w:bookmarkStart w:id="100" w:name="_Ref100"/>'
                + run("Table ")
                + complex_field("SEQ Table \\* ARABIC", "1")
                + '<w:bookmarkEnd w:id="100"/>'
                + run(": mean daily sodium intake by week."),
                style="Caption",
                para_id="1000000C",
            ),
            paragraph(run("Results"), style="Heading1", para_id="10000008"),
            paragraph(
                run(
                    "Mean daily sodium fell by 412 mg, from 3,310 mg in week 0 to "
                    "2,898 mg in week 4 (t(47) = 3.12, p = .003)."
                ),
                para_id="10000009",
            ),
            paragraph(
                _commented(
                    run("The effect is probably larger in frequent diners out."), 3
                ),
                para_id="1000000A",
            ),
            paragraph(run("References"), style="Heading1", para_id="1000000B"),
            _bibliography([okafor, marlowe, lindqvist]),
        ]
    )
    return build_docx(
        path,
        body,
        comments_xml=comment_part(
            [
                (
                    0,
                    _COLLABORATOR,
                    _WHEN,
                    "Is this causal? There was no control group.",
                    "20000001",
                ),
                (1, "Author", _WHEN, "Agreed, I will soften the wording.", "20000002"),
                (
                    2,
                    _COLLABORATOR,
                    _WHEN,
                    "Which food-composition table did you use?",
                    "20000003",
                ),
                (
                    3,
                    _COLLABORATOR,
                    _WHEN,
                    "Do we have data for this? Nothing in results/.",
                    "20000004",
                ),
            ]
        ),
        comments_extended_xml=comments_extended_part(
            [
                ("20000001", None, False),
                ("20000002", "20000001", False),
                ("20000003", None, False),
                ("20000004", None, True),
            ]
        ),
    )


def make_example(dest: Path) -> ExampleProject:
    """Copy the example project into *dest* and build ``collaborator.docx`` there.

    *dest* is created; an existing file in it is overwritten.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("notes.md", "refs.json", "reviews.txt"):
        shutil.copy(EXAMPLE_DIR / name, dest / name)
    shutil.copytree(EXAMPLE_DIR / "results", dest / "results", dirs_exist_ok=True)
    return ExampleProject(
        root=dest,
        notes=dest / "notes.md",
        results=dest / "results",
        library=dest / "refs.json",
        collaborator=make_collaborator_docx(dest / "collaborator.docx"),
        reviews=dest / "reviews.txt",
    )
