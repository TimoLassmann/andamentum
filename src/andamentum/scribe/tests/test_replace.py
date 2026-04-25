"""Tests for Document.replace and revision audit trail."""

import pytest

from andamentum.scribe.api import Document, Paragraph
from andamentum.scribe.database import open_db
from andamentum.scribe.models import StaleRevisionError


def test_replace_bumps_revision_and_updates_content(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    bid = doc.append(Paragraph("v1"))

    doc.replace(bid, "v2", expected_revision=1, reason="pass-2")

    blk = doc.query()[0]
    assert blk.content == "v2"
    assert blk.revision == 2


def test_replace_writes_revision_row(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    bid = doc.append(Paragraph("v1"))
    doc.replace(bid, "v2", expected_revision=1, reason="pass-2")

    with open_db("t") as conn:
        row = conn.execute(
            "SELECT previous_content, new_content, reason, revision "
            "FROM scribe_revisions WHERE block_id = ?",
            (bid,),
        ).fetchone()
    assert row["previous_content"] == "v1"
    assert row["new_content"] == "v2"
    assert row["reason"] == "pass-2"
    assert row["revision"] == 2


def test_replace_stale_revision_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    bid = doc.append(Paragraph("v1"))
    doc.replace(bid, "v2", expected_revision=1)

    with pytest.raises(StaleRevisionError) as excinfo:
        doc.replace(bid, "v3", expected_revision=1)
    assert excinfo.value.expected == 1
    assert excinfo.value.actual == 2


def test_replace_unknown_block_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    with pytest.raises(KeyError):
        doc.replace("nope", "x", expected_revision=1)
