"""``--reviewer`` and the ``reviewers`` listing, and the reviewer block in reports.

Two halves, and they share one property: **naming no reviewer changes
nothing**. The command line gains a flag whose default is the run that
already existed, and the three renderers gain a block that only appears when
a reviewer other than the default one produced the result.

The rest is refusals. A flag that reaches a subcommand it cannot configure is
rejected rather than ignored — panel writes its own reviewers, apply-patches
runs no review at all, and the line editor selects no checks, so a checklist
run under it would print a coverage table with no rows and look like a
document that passed everything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from docx import Document
from rich.console import Console

from andamentum.whetstone import cli as cli_mod
from andamentum.whetstone.renderers import render_docx, render_html, render_markdown
from andamentum.whetstone.renderers.docx import DEFAULT_AI_AUTHOR
from andamentum.whetstone.reviewed_document import open_source
from andamentum.whetstone.schemas import (
    ConfigurationLine,
    ReviewerHeader,
    ReviewResult,
)
from andamentum.whetstone.pipeline.reviewers import PROFILES, resolve


def _args(argv: list[str]):
    return cli_mod.build_parser().parse_args(
        argv if argv[0] in cli_mod._COMMANDS else ["review", *argv]
    )


BASE = ["draft.md", "--model", "stub:none", "--out", "review.md"]


# ── the flag ────────────────────────────────────────────────────────────────


def test_the_default_reviewer_is_the_general_one() -> None:
    assert _args(BASE).reviewer == "default"


def test_every_shipped_profile_is_an_accepted_value() -> None:
    for name in PROFILES:
        assert _args([*BASE, "--reviewer", name]).reviewer == name


def test_an_unknown_reviewer_is_refused_before_anything_is_loaded() -> None:
    """Exit 1 (configuration), not 2 (argparse), and from _validate.

    --reviewer takes a comma-separated list, which argparse `choices` cannot
    express, so the check moved into _validate. It stays EARLY on purpose: by
    run time the document has been loaded and, for a URL, fetched, and a typo
    should cost neither.
    """
    from andamentum.whetstone import cli as cli_mod

    with pytest.raises(SystemExit) as exc:
        cli_mod._parse_reviewers("statistician")
    assert exc.value.code == 2


# ── the refusals ────────────────────────────────────────────────────────────


def _validate(argv: list[str]) -> None:
    cli_mod._validate_args(_args(argv))


def test_panel_refuses_a_reviewer(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        _validate(
            ["panel", *BASE, "--i-am-the-author", "--reviewer", "claims"]
        )
    assert exc.value.code == 2
    assert "unrecognized arguments: --reviewer" in capsys.readouterr().err


def test_panel_needs_the_flag_not_an_environment_variable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The affirmation is a flag on the command line; nothing in the
    environment stands in for it."""
    monkeypatch.setenv("ANDAMENTUM_PANEL_OWN_AUTHOR", "1")
    with pytest.raises(SystemExit) as exc:
        _validate(["panel", *BASE])
    assert exc.value.code == 2
    assert "--i-am-the-author" in capsys.readouterr().err
    _validate(["panel", *BASE, "--i-am-the-author"])


def test_apply_patches_refuses_a_reviewer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        _args(
            [
                "apply-patches",
                str(tmp_path / "draft.docx"),
                "--patches",
                str(tmp_path / "patches.json"),
                "-o",
                str(tmp_path / "out.docx"),
                "--reviewer",
                "claims",
            ]
        )
    assert exc.value.code == 2
    assert "unrecognized arguments: --reviewer" in capsys.readouterr().err


def test_checklist_refuses_the_line_editor(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        _validate(["checklist", *BASE, "--reviewer", "editor"])
    assert exc.value.code == 2
    assert "runs no checks" in capsys.readouterr().err


def test_checklist_accepts_a_reviewer_that_selects_probes() -> None:
    _validate(["checklist", *BASE, "--reviewer", "methods"])


# ── the listing ─────────────────────────────────────────────────────────────


def _listing(argv: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    assert cli_mod.main(argv) == 0
    return capsys.readouterr().out


def test_listing_needs_no_input_file_and_no_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = _listing(["reviewers"], capsys)
    for name, profile in PROFILES.items():
        assert out.count(profile.description) == 1
        assert name in out


def test_listing_prints_every_profile_once_in_display_order(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One row per profile, no repeats, no omissions, registry order."""
    out = _listing(["reviewers"], capsys)
    rows = [
        line
        for line in out.splitlines()
        if any(line.strip().startswith(name) for name in PROFILES)
    ]
    assert len(rows) == len(PROFILES)
    assert [row.split()[0] for row in rows] == list(PROFILES)


def test_listing_marks_the_default(capsys: pytest.CaptureFixture[str]) -> None:
    out = _listing(["reviewers"], capsys)
    assert "default (default)" in out


def test_listing_carries_the_recommended_sequence(
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = _listing(["reviewers"], capsys)
    assert "Recommended sequence" in out
    assert "Editor last" in out


def test_listing_shows_fit_only_when_the_type_is_stated(
    capsys: pytest.CaptureFixture[str],
) -> None:
    without = _listing(["reviewers"], capsys)
    assert "suits" not in without.splitlines()[2]

    with_type = _listing(["reviewers", "--document-type", "creative"], capsys)
    assert "suits creative?" in with_type
    claims_row = next(
        line for line in with_type.splitlines() if line.strip().startswith("claims")
    )
    assert " no " in claims_row


# ── the author ──────────────────────────────────────────────────────────────


def test_the_default_reviewer_keeps_the_package_author_verbatim() -> None:
    assert cli_mod._resolve_author(_args(BASE)) == DEFAULT_AI_AUTHOR


def test_a_named_reviewer_signs_its_own_changes_and_keeps_the_marker() -> None:
    author = cli_mod._resolve_author(_args([*BASE, "--reviewer", "methods"]))
    assert author == "Methods reviewer (andamentum-whetstone, AI)"


def test_an_explicit_author_still_wins_and_still_warns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "draft.md",
        "--model",
        "stub:none",
        "--out",
        str(tmp_path / "out.docx"),
        "--reviewer",
        "methods",
        "--patch-author",
        "Second pass (AI)",
        "--allow-author-override",
    ]
    assert cli_mod._resolve_author(_args(argv)) == "Second pass (AI)"
    assert "research misconduct" in capsys.readouterr().err


# ── the wiring into the pipelines ───────────────────────────────────────────


def _draft(path: Path) -> None:
    document = Document()
    document.add_heading("Methods", level=1)
    document.add_paragraph("We met the goal. Then we stopped.")
    document.save(str(path))


async def test_the_cli_hands_the_review_the_named_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "draft.docx"
    _draft(source)
    seen: dict[str, Any] = {}

    async def fake_review(_source: Any, **kwargs: Any) -> ReviewResult:
        seen.update(kwargs)
        return ReviewResult(summary="Stubbed.")

    monkeypatch.setattr(
        "andamentum.whetstone.pipeline.review_document", fake_review, raising=True
    )
    await cli_mod._run(
        _args(
            [
                str(source),
                "--model",
                "stub:none",
                "--out",
                str(tmp_path / "review.md"),
                "--reviewer",
                "adversary",
            ]
        ),
        Console(stderr=True),
    )
    assert seen["reviewer"] is PROFILES["adversary"]
    assert seen["patch_author"] is None


async def test_the_cli_hands_the_checklist_the_named_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "draft.docx"
    _draft(source)
    seen: dict[str, Any] = {}

    async def fake_checklist(_source: Any, **kwargs: Any) -> ReviewResult:
        seen.update(kwargs)
        return ReviewResult(summary="Stubbed.")

    monkeypatch.setattr(
        "andamentum.whetstone.pipeline.checklist.checklist_document",
        fake_checklist,
        raising=True,
    )
    await cli_mod._run(
        _args(
            [
                "checklist",
                str(source),
                "--model",
                "stub:none",
                "--out",
                str(tmp_path / "review.md"),
                "--reviewer",
                "reader",
            ]
        ),
        Console(stderr=True),
    )
    assert seen["reviewer"] is PROFILES["reader"]


# ── the report header ───────────────────────────────────────────────────────


def _header() -> ReviewerHeader:
    return resolve(profile=PROFILES["methods"], document_type="creative").header or (
        pytest.fail("the methods reviewer must produce a header")
    )


def _fitting_header() -> ReviewerHeader:
    """The same reviewer on a document type it suits — no warning to carry."""
    return resolve(profile=PROFILES["methods"], document_type="academic").header or (
        pytest.fail("the methods reviewer must produce a header")
    )


def _result(header: ReviewerHeader | None) -> ReviewResult:
    return ReviewResult(summary="A short review.", reviewer_header=header)


def test_markdown_prints_the_reviewer_block() -> None:
    text = render_markdown(_result(_header()), model="stub:none")
    assert "Reviewer: Methods reviewer" in text
    assert PROFILES["methods"].brief in text
    assert "methods reviewer on a document classified as creative" in text
    assert "criteria: Reproducibility" in text
    assert "andamentum-whetstone 0.3.0" in text


def test_markdown_prints_nothing_for_the_default_reviewer() -> None:
    assert "Reviewer:" not in render_markdown(_result(None), model="stub:none")


def test_html_prints_the_reviewer_block() -> None:
    html = render_html(_result(_header()), model="stub:none")
    assert "Reviewer: Methods reviewer" in html
    assert "classified as creative" in html
    assert "andamentum-whetstone 0.3.0" in html


def test_html_prints_nothing_for_the_default_reviewer() -> None:
    assert "Reviewer:" not in render_html(_result(None), model="stub:none")


def test_the_clean_document_message_survives_the_reviewer_block() -> None:
    """``prelude_atoms`` used to be the literal 2. With the reviewer block
    above it, a hard-coded count silences this message on an empty review."""
    empty = ReviewResult(summary="", reviewer_header=_header())
    assert "document looks clean" in render_html(empty, model="stub:none")


async def test_the_word_report_carries_the_reviewer_block(tmp_path: Path) -> None:
    source = tmp_path / "draft.docx"
    _draft(source)
    document = await open_source(str(source))
    try:
        out = tmp_path / "reviewed.docx"
        render_docx(
            _result(_header()),
            document=document,
            output_path=out,
            author="Methods reviewer (andamentum-whetstone, AI)",
            model="stub:none",
        )
        import zipfile

        with zipfile.ZipFile(out) as archive:
            body = archive.read("word/document.xml").decode()
        assert "Methods reviewer" in body
        assert "classified as creative" in body
    finally:
        document.cleanup()


# ── the fit warning, only on a miss ─────────────────────────────────────────
#
# The warning is the one part of the header that is a *judgement* about this
# run rather than a restatement of the reviewer, so a report that carries it
# when the pair fits would be telling the reader to discount findings for no
# reason. Checked in every renderer, both ways.


def test_the_reviewer_block_still_appears_when_the_pair_fits() -> None:
    header = _fitting_header()
    assert header.fit_warning == ""
    assert "Reviewer: Methods reviewer" in render_markdown(
        _result(header), model="stub:none"
    )
    assert "Reviewer: Methods reviewer" in render_html(
        _result(header), model="stub:none"
    )


def test_markdown_carries_the_fit_warning_only_on_a_miss() -> None:
    missed = render_markdown(_result(_header()), model="stub:none")
    fitted = render_markdown(_result(_fitting_header()), model="stub:none")
    assert "**Fit.**" in missed
    assert "classified as creative" in missed
    assert "**Fit.**" not in fitted
    assert "classified as" not in fitted


def test_html_carries_the_fit_warning_only_on_a_miss() -> None:
    missed = render_html(_result(_header()), model="stub:none")
    fitted = render_html(_result(_fitting_header()), model="stub:none")
    assert "classified as creative" in missed
    assert "classified as" not in fitted


async def test_the_word_report_carries_the_fit_warning_only_on_a_miss(
    tmp_path: Path,
) -> None:
    assert "classified as" not in await _word_report(
        _fitting_header(), tmp_path=tmp_path
    )
    assert "classified as creative" in await _word_report(_header(), tmp_path=tmp_path)


async def _word_report(header: ReviewerHeader, *, tmp_path: Path) -> str:
    """The rendered document's body XML, as text."""
    import zipfile

    source = tmp_path / f"{header.fit_warning and 'miss' or 'fit'}.docx"
    _draft(source)
    document = await open_source(str(source))
    try:
        out = tmp_path / f"{source.stem}.reviewed.docx"
        render_docx(
            _result(header),
            document=document,
            output_path=out,
            author="Methods reviewer (andamentum-whetstone, AI)",
            model="stub:none",
        )
        with zipfile.ZipFile(out) as archive:
            return archive.read("word/document.xml").decode()
    finally:
        document.cleanup()


def test_a_configuration_line_names_its_own_source() -> None:
    header = _header()
    by_label = {line.label: line for line in header.configuration}
    assert isinstance(by_label["criteria"], ConfigurationLine)
    assert by_label["criteria"].source == "reviewer methods"
    assert by_label["editor criteria"].source == "editor default"


# ── a review that would review nothing ──────────────────────────────────────


def test_review_refuses_the_line_editor_without_the_editor(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The mirror of the checklist refusal, for the mode that has no rows.

    With ``--no-editor`` the line editor makes no criterion call, runs no gap
    round and proposes no rewrite, but still pays for claim extraction,
    synthesis and self-critique — a full-price run whose report says nothing.
    """
    with pytest.raises(SystemExit) as exc:
        _validate([*BASE, "--reviewer", "editor", "--no-editor"])
    assert exc.value.code == 2
    assert "runs no review criteria" in capsys.readouterr().err


def test_review_accepts_the_line_editor_with_the_editor() -> None:
    _validate([*BASE, "--reviewer", "editor", "--editor"])
    # The editor is on by default, so naming the reviewer alone is enough.
    _validate([*BASE, "--reviewer", "editor"])


def test_review_accepts_the_line_editor_given_a_brief_of_its_own() -> None:
    """A brief replaces the profile's empty set of criteria, so the run reviews."""
    _validate([*BASE, "--reviewer", "editor", "--no-editor", "--brief", "Be brief."])


def test_the_refusal_is_stated_as_a_property_not_as_a_name() -> None:
    """Every reviewer that selects criteria is accepted without --editor.

    Written this way so a later prose-only profile is covered by the same
    check rather than needing an edit to it.
    """
    for name, profile in PROFILES.items():
        if profile.criteria == ():
            continue
        _validate([*BASE, "--reviewer", name])


# ── the listing's own parser ────────────────────────────────────────────────


def test_the_listing_refuses_a_document_type_the_run_would_refuse() -> None:
    """`reviewers` has its own parser; it must not be more permissive.

    An unrecognised type matches no profile's ``fits`` set, so the fit column
    would read "no" against every reviewer — including the default one, which
    fits all six by construction.
    """
    assert cli_mod.main(["reviewers", "--document-type", "acadmic"]) == 2


def test_the_listing_accepts_every_type_the_run_accepts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for document_type in cli_mod._DOCUMENT_TYPE_CHOICES:
        out = _listing(["reviewers", "--document-type", document_type], capsys)
        assert "default (default)" in out
