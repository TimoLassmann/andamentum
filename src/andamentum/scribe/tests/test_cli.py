"""Smoke tests for andamentum-scribe CLI."""

import os
import subprocess
import sys


def _run(args: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "andamentum.scribe.cli", *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _envwith(tmp_path) -> dict:
    env = os.environ.copy()
    env["SCRIBE_DIR"] = str(tmp_path)
    return env


def test_help_lists_subcommands():
    result = _run(["--help"])
    assert result.returncode == 0
    for sub in (
        "init",
        "list-sections",
        "read-section",
        "write-section",
        "insert-figure",
        "insert-table",
        "render",
    ):
        assert sub in result.stdout


def test_init_with_scaffold(tmp_path):
    env = _envwith(tmp_path)
    result = _run(
        ["init", "--database", "t", "--title", "P", "--scaffold", "article"],
        env=env,
    )
    assert result.returncode == 0, result.stderr
    doc_id = result.stdout.strip()
    assert len(doc_id) >= 8


def test_list_sections_after_scaffold(tmp_path):
    env = _envwith(tmp_path)
    create = _run(
        ["init", "--database", "t", "--title", "P", "--scaffold", "article"],
        env=env,
    )
    doc_id = create.stdout.strip()

    result = _run(["list-sections", "--database", "t", "--id", doc_id], env=env)
    assert result.returncode == 0
    assert "Introduction" in result.stdout
    assert "Methods" in result.stdout


def test_write_section_replaces_content(tmp_path):
    env = _envwith(tmp_path)
    create = _run(
        ["init", "--database", "t", "--title", "P", "--scaffold", "article"],
        env=env,
    )
    doc_id = create.stdout.strip()
    content_file = tmp_path / "intro.md"
    content_file.write_text("Brand new intro paragraph.")

    result = _run(
        [
            "write-section",
            "--database",
            "t",
            "--id",
            doc_id,
            "--section",
            "Introduction",
            "--content-file",
            str(content_file),
        ],
        env=env,
    )
    assert result.returncode == 0, result.stderr

    read = _run(
        [
            "read-section",
            "--database",
            "t",
            "--id",
            doc_id,
            "--section",
            "Introduction",
        ],
        env=env,
    )
    assert "Brand new intro paragraph." in read.stdout


def test_insert_table_from_csv(tmp_path):
    env = _envwith(tmp_path)
    create = _run(
        ["init", "--database", "t", "--title", "P", "--scaffold", "article"], env=env
    )
    doc_id = create.stdout.strip()
    csv = tmp_path / "data.csv"
    csv.write_text("Col A,Col B\n1,2\n3,4\n")

    result = _run(
        [
            "insert-table",
            "--database",
            "t",
            "--id",
            doc_id,
            "--section",
            "Results",
            "--csv",
            str(csv),
            "--caption",
            "demo",
            "--label",
            "tab:demo",
        ],
        env=env,
    )
    assert result.returncode == 0, result.stderr


def test_render_unknown_doc_exits_nonzero(tmp_path):
    env = _envwith(tmp_path)
    _run(["init", "--database", "t", "--title", "x"], env=env)
    result = _run(
        [
            "render",
            "--database",
            "t",
            "--id",
            "nope",
            "--output",
            str(tmp_path / "x.docx"),
        ],
        env=env,
    )
    assert result.returncode != 0
