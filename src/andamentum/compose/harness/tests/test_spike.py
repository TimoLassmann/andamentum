"""M0 spike: survival of the spike document through other tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.compose.harness.spike import make_spike_docx, probe
from andamentum.compose.harness.spike.__main__ import main
from andamentum.compose.harness.spike.probe import evaluate
from andamentum.compose.harness.spike.survival import (
    SOFFICE,
    ProjectionResult,
    SurvivalResults,
    ToolResult,
    libreoffice_roundtrip,
    python_docx_roundtrip,
    render_report,
    whetstone_projection,
)


def _missing(path: Path) -> list[str]:
    return [f.name for f in evaluate(probe(path)) if not f.present]


def test_spike_document_has_everything(tmp_path: Path):
    assert _missing(make_spike_docx(tmp_path / "spike.docx")) == []


def test_python_docx_roundtrip_keeps_everything(tmp_path: Path):
    original = make_spike_docx(tmp_path / "spike.docx")
    out = python_docx_roundtrip(original, tmp_path / "out.docx")
    assert _missing(out) == []


@pytest.mark.skipif(not SOFFICE, reason="LibreOffice not installed")
def test_libreoffice_roundtrip_keeps_everything(tmp_path: Path):
    original = make_spike_docx(tmp_path / "spike.docx")
    out = libreoffice_roundtrip(original, workdir=tmp_path / "work")
    assert _missing(out) == []


def test_whetstone_projection_includes_content_control_text(tmp_path: Path):
    original = make_spike_docx(tmp_path / "spike.docx")
    projection = whetstone_projection(original)
    assert projection.number_in_text and projection.claim_in_text


def test_probe_command_prints_the_survival_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    original = make_spike_docx(tmp_path / "spike.docx")
    assert main(["probe", str(original)]) == 0
    output = capsys.readouterr().out
    assert "SURVIVED  anchor spike-number" in output
    assert "LOST" not in output


def test_probe_reports_what_an_edit_destroyed(tmp_path: Path):
    """A returned file with the claim control stripped must read as LOST."""
    import zipfile

    original = make_spike_docx(tmp_path / "spike.docx")
    stripped = tmp_path / "stripped.docx"
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(stripped, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "word/document.xml":
                data = data.replace(b'w:val="am:spike-claim"', b'w:val="other"')
            target.writestr(item, data)
    assert _missing(stripped) == ["anchor spike-claim"]


def test_report_renders_a_table_per_tool(tmp_path: Path):
    original = make_spike_docx(tmp_path / "spike.docx")
    done = ToolResult(findings=evaluate(probe(original)))
    text = render_report(
        SurvivalResults(
            original=done,
            python_docx=done,
            libreoffice=ToolResult(skipped="not installed"),
            whetstone=ProjectionResult(
                number_in_text=True, claim_in_text=True, characters=1
            ),
        )
    )
    assert "## python-docx" in text and "Skipped: not installed" in text
