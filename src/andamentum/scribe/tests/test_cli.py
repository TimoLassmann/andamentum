"""andamentum-scribe CLI against a bundle directory."""

import io

import pytest

from andamentum.scribe.cli import main


def test_full_flow(tmp_path, capsys, monkeypatch):
    path = str(tmp_path / "b")
    assert main(["init", path, "--title", "Apollo", "--scaffold", "article"]) == 0
    capsys.readouterr()

    assert main(["sections", path]) == 0
    out = capsys.readouterr().out
    assert "s01" in out and "Abstract" in out and "planned" in out

    assert main(["add", path, "--title", "Appendix", "--after", "s01"]) == 0
    assert capsys.readouterr().out.strip() == "s06"

    body = tmp_path / "body.md"
    body.write_text("Hello world.\n", encoding="utf-8")
    assert main(["write", path, "s01", "--from", str(body), "--revision", "1"]) == 0
    assert "revision 2" in capsys.readouterr().out

    monkeypatch.setattr("sys.stdin", io.StringIO("From stdin."))
    assert main(["write", path, "s02", "--from", "-", "--revision", "1"]) == 0
    capsys.readouterr()

    assert main(["show", path, "s01"]) == 0
    shown = capsys.readouterr().out
    assert "revision: 2" in shown and "Hello world." in shown

    assert main(["validate", path]) == 0
    capsys.readouterr()

    out = tmp_path / "o.docx"
    assert main(["render", path, "--format", "docx", "--out", str(out)]) == 0
    assert out.is_file()


def test_stale_revision_is_reported_exit_2(tmp_path, capsys):
    path = str(tmp_path / "b")
    main(["init", path, "--title", "T", "--scaffold", "article"])
    body = tmp_path / "b.md"
    body.write_text("x", encoding="utf-8")
    capsys.readouterr()
    assert main(["write", path, "s01", "--from", str(body), "--revision", "5"]) == 2
    assert "Stale revision" in capsys.readouterr().err


def test_validate_exit_1_on_error(tmp_path, capsys):
    path = str(tmp_path / "b")
    main(["init", path, "--title", "T"])
    main(["add", path, "--title", "A"])
    main(["add", path, "--title", "A"])
    capsys.readouterr()
    assert main(["validate", path]) == 1
    assert "error" in capsys.readouterr().out


def test_open_missing_bundle_exit_2(tmp_path, capsys):
    assert main(["sections", str(tmp_path / "nope")]) == 2
    assert "not a scribe bundle" in capsys.readouterr().err


def test_init_refuses_existing_non_empty(tmp_path, capsys):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "f").write_text("a")
    assert main(["init", str(tmp_path / "x"), "--title", "T"]) == 2


def test_requires_subcommand():
    with pytest.raises(SystemExit):
        main([])
