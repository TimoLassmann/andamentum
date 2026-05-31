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
        "add-reference",
        "list-references",
        "list-citations",
        "validate",
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


# ---------------------------------------------------------------------------
# Reference + validation subcommands
# ---------------------------------------------------------------------------


def _init_doc(env):
    """Create a scaffolded article doc and return its id."""
    create = _run(
        ["init", "--database", "t", "--title", "P", "--scaffold", "article"],
        env=env,
    )
    assert create.returncode == 0, create.stderr
    return create.stdout.strip()


def test_add_reference_and_list_references(tmp_path):
    env = _envwith(tmp_path)
    doc_id = _init_doc(env)

    add = _run(
        [
            "add-reference",
            "--database",
            "t",
            "--id",
            doc_id,
            "--cite-key",
            "Smith2023",
            "--bibtex",
            "@article{Smith2023, title={A paper}, year={2023}}",
        ],
        env=env,
    )
    assert add.returncode == 0, add.stderr
    assert add.stdout.strip()  # ref id is non-empty

    listed = _run(["list-references", "--database", "t", "--id", doc_id], env=env)
    assert listed.returncode == 0
    assert "Smith2023" in listed.stdout


def test_add_reference_reads_bibtex_from_file(tmp_path):
    env = _envwith(tmp_path)
    doc_id = _init_doc(env)

    bib = tmp_path / "ref.bib"
    bib.write_text("@article{Doe2024, title={Another paper}, year={2024}}")

    add = _run(
        [
            "add-reference",
            "--database",
            "t",
            "--id",
            doc_id,
            "--cite-key",
            "Doe2024",
            "--bibtex-file",
            str(bib),
        ],
        env=env,
    )
    assert add.returncode == 0, add.stderr

    listed = _run(["list-references", "--database", "t", "--id", doc_id], env=env)
    assert "Doe2024" in listed.stdout
    assert "Another paper" in listed.stdout


def test_list_citations_returns_dedup_keys_from_paragraph(tmp_path):
    env = _envwith(tmp_path)
    doc_id = _init_doc(env)

    intro = tmp_path / "intro.md"
    intro.write_text("This builds on [@Smith2023] and [@Doe2024]; also [@Smith2023].")
    _run(
        [
            "write-section",
            "--database",
            "t",
            "--id",
            doc_id,
            "--section",
            "Introduction",
            "--content-file",
            str(intro),
        ],
        env=env,
    )

    listed = _run(["list-citations", "--database", "t", "--id", doc_id], env=env)
    assert listed.returncode == 0
    keys = listed.stdout.strip().splitlines()
    assert "Smith2023" in keys
    assert "Doe2024" in keys
    # deduped
    assert keys.count("Smith2023") == 1


def test_validate_flags_missing_reference_as_error(tmp_path):
    env = _envwith(tmp_path)
    doc_id = _init_doc(env)

    intro = tmp_path / "intro.md"
    intro.write_text("Citing [@Ghost2099] without defining it.")
    _run(
        [
            "write-section",
            "--database",
            "t",
            "--id",
            doc_id,
            "--section",
            "Introduction",
            "--content-file",
            str(intro),
        ],
        env=env,
    )

    val = _run(["validate", "--database", "t", "--id", doc_id], env=env)
    # Missing-citation issues are error-severity, so exit code is non-zero.
    assert val.returncode != 0
    assert "Ghost2099" in val.stdout


def test_validate_clean_doc_returns_ok(tmp_path):
    env = _envwith(tmp_path)
    doc_id = _init_doc(env)

    val = _run(["validate", "--database", "t", "--id", doc_id], env=env)
    assert val.returncode == 0
    assert "ok" in val.stdout.lower()
