"""Document.validate() tests."""

from andamentum.scribe.api import Document, Figure, Paragraph


def test_validate_clean_document_returns_no_issues(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    fig = tmp_path / "f.png"
    fig.write_bytes(b"")
    doc = Document.create(title="P", database="t")
    doc.add_reference(cite_key="smith2023")
    doc.append(Paragraph("As shown [@smith2023]."))
    doc.append(Figure(path=str(fig), caption="C", label="fig:c"))

    issues = doc.validate()
    assert issues == []


def test_validate_flags_missing_citation_key(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.append(Paragraph("Cited [@unknown2024]."))
    issues = doc.validate()
    assert any(i.severity == "error" and "unknown2024" in i.message for i in issues)


def test_validate_flags_missing_figure_file(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.append(Figure(path="/no/such/file.png", caption="C", label="fig:x"))
    issues = doc.validate()
    assert any(i.severity == "error" and "fig:x" in i.location for i in issues)


def test_validate_warns_on_unused_reference(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.add_reference(cite_key="smith2023")
    doc.append(Paragraph("No citations here."))
    issues = doc.validate()
    assert any(i.severity == "warning" and "smith2023" in i.message for i in issues)


def test_validate_reports_unresolved_markers(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.append(Paragraph("Foundational work [verify] established it."))
    doc.append(Paragraph("Some claim [citation needed]."))

    issues = doc.validate()
    msgs = [i.message for i in issues if i.severity == "info"]
    assert any("verify" in m for m in msgs)
    assert any("citation needed" in m for m in msgs)


def test_validate_warns_on_ai_drafted_marker(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.append(Paragraph("This paragraph [ai-drafted] needs author review."))

    issues = doc.validate()
    ai_warnings = [
        i for i in issues if i.severity == "warning" and "ai-drafted" in i.message
    ]
    assert len(ai_warnings) == 1
    assert "disclose AI assistance" in ai_warnings[0].message


def test_validate_warns_on_ai_edited_marker(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.append(Paragraph("Polished [ai-edited] from earlier draft."))

    issues = doc.validate()
    assert any(i.severity == "warning" and "ai-edited" in i.message for i in issues)


def test_validate_clean_when_no_ai_markers(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.append(Paragraph("Plain paragraph with no markers."))

    issues = doc.validate()
    assert not any(
        "ai-drafted" in i.message or "ai-edited" in i.message for i in issues
    )
