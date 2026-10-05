"""The literature adapter: verified works with abstracts, one line per unverified work."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from andamentum.compose.harness.draft import literature_search
from andamentum.compose.harness.draft.search import abstract_of
from andamentum.compose.harness.literature import LiteratureError, Sources

from .test_literature import RETRACTED, SALT, Crossref

ABSTRACT = (
    "<jats:p>Daily logging lowered <jats:italic>reported</jats:italic> intake.</jats:p>"
)
QUERY = "daily salt logging sodium intake sleep invented adults"


def library(tmp_path: Path, *works: dict) -> Path:
    path = tmp_path / "refs.json"
    path.write_text(json.dumps(list(works)))
    return path


def test_abstract_tags_are_removed_and_missing_is_empty() -> None:
    assert (
        abstract_of({"abstract": ABSTRACT}) == "Daily logging lowered reported intake."
    )
    assert abstract_of({}) == ""


async def test_verified_works_carry_their_abstract(tmp_path: Path) -> None:
    fake = Crossref()
    search = literature_search(
        sources=Sources(library=library(tmp_path, {**SALT, "abstract": ABSTRACT})),
        http=fake.http(),
    )
    finding = await search(QUERY, limit=5)
    (work,) = finding.works
    assert work.csl["DOI"] == "10.5555/saltlog.001"
    assert work.abstract == "Daily logging lowered reported intake."
    assert work.uri is None
    assert finding.unverified == []


async def test_unverified_works_become_lines_never_works(tmp_path: Path) -> None:
    fake = Crossref()
    invented = {
        **SALT,
        "id": "inv",
        "DOI": "10.5555/saltlog.nowhere",
        "abstract": ABSTRACT,
    }
    search = literature_search(
        sources=Sources(
            library=library(tmp_path, invented, {**RETRACTED, "abstract": ABSTRACT})
        ),
        http=fake.http(),
    )
    finding = await search(QUERY, limit=5)
    assert finding.works == []
    assert len(finding.unverified) == 2
    assert any("10.5555/saltlog.nowhere" in line for line in finding.unverified)
    assert any("retracted" in line and "(2018)" in line for line in finding.unverified)


async def test_a_source_that_cannot_answer_raises(tmp_path: Path) -> None:
    search = literature_search(
        sources=Sources(library=tmp_path / "missing.json"), http=Crossref().http()
    )
    with pytest.raises(LiteratureError):
        await search(QUERY, limit=5)
