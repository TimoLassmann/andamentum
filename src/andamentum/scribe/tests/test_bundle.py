"""Bundle lifecycle, sections, revisions, references, figures."""

from pathlib import Path

import pytest
import yaml

from andamentum.scribe import Bundle, ScribeError, StaleRevisionError

from ._png import PNG_BYTES


def _png(path: Path) -> Path:
    path.write_bytes(PNG_BYTES)
    return path


def test_create_layout_and_scaffold(tmp_path):
    b = Bundle.create(
        tmp_path / "doc.compose", title="Apollo Report", scaffold="article"
    )
    root = tmp_path / "doc.compose"
    assert (root / "bundle.yaml").is_file()
    assert (root / "figures").is_dir() and (root / "out").is_dir()
    assert [s.title for s in b.sections()] == [
        "Abstract",
        "Introduction",
        "Methods",
        "Results",
        "Discussion",
    ]
    assert [s.order for s in b.sections()] == [10, 20, 30, 40, 50]
    manifest = yaml.safe_load((root / "bundle.yaml").read_text())
    assert manifest["format_version"] == 1 and manifest["title"] == "Apollo Report"


def test_create_refuses_non_empty_dir(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "f").write_text("a")
    with pytest.raises(ScribeError):
        Bundle.create(tmp_path / "x", title="T")
    Bundle.create(tmp_path / "empty_ok", title="T")


def test_create_unknown_scaffold(tmp_path):
    with pytest.raises(ScribeError, match="unknown scaffold"):
        Bundle.create(tmp_path / "d", title="T", scaffold="grant")


def test_open_roundtrip_and_metadata(tmp_path):
    b = Bundle.create(tmp_path / "d", title="T", metadata={"genre": "memo"})
    b.set_metadata(stage="draft")
    again = Bundle.open(tmp_path / "d")
    assert again.title == "T"
    assert again.metadata == {"genre": "memo", "stage": "draft"}


def test_open_rejects_non_bundle_and_bad_version(tmp_path):
    with pytest.raises(ScribeError, match="not a scribe bundle"):
        Bundle.open(tmp_path)
    Bundle.create(tmp_path / "d", title="T")
    manifest = tmp_path / "d" / "bundle.yaml"
    manifest.write_text(
        manifest.read_text().replace("format_version: 1", "format_version: 9")
    )
    with pytest.raises(ScribeError, match="format_version"):
        Bundle.open(tmp_path / "d")


def test_section_file_format(tmp_path):
    b = Bundle.create(tmp_path / "d", title="T")
    s = b.add_section(title="Summary", body="Hello.\n", status="drafted")
    text = (tmp_path / "d" / "sections" / "s01.md").read_text()
    assert text.startswith("---\nid: s01\ntitle: Summary\norder: 10\nrevision: 1\n")
    assert text.endswith("---\nHello.\n")
    assert "content_hash" not in text
    assert s.content_hash == b.section("s01").content_hash


def test_body_with_horizontal_rule_survives(tmp_path):
    b = Bundle.create(tmp_path / "d", title="T")
    s = b.add_section(title="A", body="one\n\n---\n\ntwo")
    assert b.section(s.id).body == "one\n\n---\n\ntwo"


def test_add_section_ids_and_after_with_gap_and_renumber(tmp_path):
    b = Bundle.create(tmp_path / "d", title="T")
    a = b.add_section(title="A")
    c = b.add_section(title="C")
    mid = b.add_section(title="B", after=a.id)  # gap 10..20 -> 15
    assert [s.title for s in b.sections()] == ["A", "B", "C"]
    assert mid.order == 15 and mid.id == "s03"
    b.add_section(title="B2", after=a.id)  # 10..15 -> 12
    b.add_section(title="B3", after=a.id)  # 10..12 -> 11
    last = b.add_section(title="B4", after=a.id)  # 10..11 no gap -> renumber
    titles = [s.title for s in b.sections()]
    assert titles == ["A", "B4", "B3", "B2", "B", "C"]
    assert [s.order for s in b.sections()] == [10, 20, 30, 40, 50, 60]
    assert last.order == 20
    assert c.id == "s02"


def test_add_section_unknown_after(tmp_path):
    b = Bundle.create(tmp_path / "d", title="T")
    with pytest.raises(ScribeError):
        b.add_section(title="X", after="s99")


def test_write_section_revision_and_stale(tmp_path):
    b = Bundle.create(tmp_path / "d", title="T")
    s = b.add_section(title="A")
    s2 = b.write_section(s.id, body="v2", expected_revision=1, status="drafted")
    assert (s2.revision, s2.status, s2.body) == (2, "drafted", "v2")
    with pytest.raises(StaleRevisionError) as err:
        b.write_section(s.id, body="v3", expected_revision=1)
    assert err.value.actual == 2
    assert b.section(s.id).body == "v2"


def test_write_is_idempotent_bytes(tmp_path):
    b = Bundle.create(tmp_path / "d", title="T")
    s = b.add_section(title="A", body="x")
    path = tmp_path / "d" / "sections" / "s01.md"
    first = path.read_bytes()
    b._save(b.section(s.id))
    assert path.read_bytes() == first
    assert not list((tmp_path / "d" / "sections").glob(".*tmp"))


def test_missing_section_errors(tmp_path):
    b = Bundle.create(tmp_path / "d", title="T")
    with pytest.raises(ScribeError):
        b.section("s09")
    with pytest.raises(ScribeError):
        b.section("../bundle")
    with pytest.raises(ScribeError):
        b.section_by_title("Nope")
    with pytest.raises(ScribeError):
        b.remove_section("s09")


def test_rename_remove_reorder(tmp_path):
    b = Bundle.create(tmp_path / "d", title="T")
    a, c = b.add_section(title="A"), b.add_section(title="C")
    renamed = b.rename_section(a.id, title="Alpha", expected_revision=1)
    assert renamed.revision == 2 and b.section_by_title("Alpha").id == a.id
    with pytest.raises(StaleRevisionError):
        b.rename_section(a.id, title="X", expected_revision=1)
    b.reorder([c.id, a.id])
    assert [s.title for s in b.sections()] == ["C", "Alpha"]
    with pytest.raises(ScribeError):
        b.reorder([c.id])
    with pytest.raises(ScribeError):
        b.reorder([c.id, c.id])
    b.remove_section(c.id)
    assert [s.title for s in b.sections()] == ["Alpha"]
    assert b.add_section(title="N").id == "s02"  # next id is max + 1


def test_references(tmp_path):
    b = Bundle.create(tmp_path / "d", title="T")
    assert b.references() == []
    b.add_reference(
        key="smith2020", text="Smith. A study.", url="https://example.org/a"
    )
    b.add_reference(key="doe2021", text="Doe. Another.")
    b.add_reference(key="smith2020", text="Smith. A study, revised.")
    refs = Bundle.open(tmp_path / "d").references()
    assert [r.key for r in refs] == ["smith2020", "doe2021"]
    assert refs[0].text.endswith("revised.") and refs[0].url is None


def test_add_figure_copies_and_appends(tmp_path):
    b = Bundle.create(tmp_path / "d", title="T")
    s = b.add_section(title="Results", body="Text.")
    img = _png(tmp_path / "plot.PNG")
    updated = b.add_figure(
        s.id,
        image=img,
        caption="Scores [by] method",
        label="fig:scores",
        expected_revision=1,
    )
    assert (tmp_path / "d" / "figures" / "scores.png").is_file()
    assert updated.revision == 2
    assert updated.body.endswith(
        "![Scores (by) method](figures/scores.png){#fig:scores}"
    )
    assert updated.body.startswith("Text.\n\n")
    with pytest.raises(StaleRevisionError):
        b.add_figure(s.id, image=img, caption="c", label="x", expected_revision=1)
    with pytest.raises(ScribeError, match="not found"):
        b.add_figure(
            s.id,
            image=tmp_path / "nope.png",
            caption="c",
            label="x",
            expected_revision=2,
        )
