from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from andamentum.compose.genres import GENRES, Genre, genre_for

NAMES = ["email", "memo", "report", "article", "howto", "proposal", "grant"]


def test_seven_packs():
    assert list(GENRES) == NAMES


@pytest.mark.parametrize("name", NAMES)
def test_pack_valid_and_roundtrips(name):
    g = genre_for(name)
    assert g.name == name
    assert Genre.model_validate(g.model_dump()) == g
    lo, hi = g.word_range
    assert sum(s.words[0] for s in g.sections if s.required) <= hi
    assert sum(s.words[1] for s in g.sections) >= lo


def test_grant_is_generic_and_provenance_required():
    g = genre_for("grant")
    assert g.provenance_required and g.flexible_outline and g.cites
    assert "[TO FILL" in g.style_notes
    assert not any(k in g.style_notes for k in ("NIH", "NHMRC", "ARC"))


def test_email_shape():
    g = genre_for("email")
    assert [s.title for s in g.sections] == ["Body"]
    assert g.word_range == (50, 300) and g.outputs[0] == "txt"
    assert g.max_grade_level == 10


def test_unknown_name_lists_valid():
    with pytest.raises(ValueError) as e:
        genre_for("novel")
    for n in NAMES:
        assert n in str(e.value)


def test_yaml_path_loading(tmp_path):
    data = genre_for("memo").model_dump(mode="json")
    data["name"] = "custom"
    p = tmp_path / "custom.yaml"
    p.write_text(yaml.safe_dump(data))
    assert genre_for(str(p)).name == "custom"


def test_yaml_missing_file_and_invalid(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        genre_for(str(tmp_path / "nope.yaml"))
    bad = genre_for("memo").model_dump(mode="json")
    bad["whetstone_document_type"] = "poetry"
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValidationError):
        genre_for(str(p))


def test_duplicate_titles_rejected():
    data = genre_for("memo").model_dump()
    data["sections"] = data["sections"] + data["sections"][:1]
    with pytest.raises(ValidationError):
        Genre.model_validate(data)
