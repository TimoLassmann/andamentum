"""Subcommand front-end for andamentum-whetstone.

The CLI keeps a single flat argparse parser as the canonical surface
(every existing test builds against it). A thin alias layer in front of
``main()`` rewrites subcommand-style invocations into the equivalent
flat argv before parsing. These tests cover that rewriter — the
existing test_apply_patches_cli.py / test_panel_mode.py / etc. cover
the underlying flat parser unchanged.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from andamentum.whetstone.cli import (
    _KNOWN_SUBCOMMANDS,
    _rewrite_subcommand,
    main,
)


def test_known_subcommand_set_matches_documented_verbs() -> None:
    """The recognised subcommands are exactly review / panel / proofread /
    apply-patches. Adding one (or removing one) is a UX change and must
    be deliberate."""
    assert _KNOWN_SUBCOMMANDS == frozenset(
        {"review", "panel", "proofread", "apply-patches"}
    )


def test_review_subcommand_strips_the_verb() -> None:
    """`review draft.md --model X --out r.md` → `draft.md --model X
    --out r.md` — the rewriter just drops the leading verb."""
    assert _rewrite_subcommand(
        ["review", "draft.md", "--model", "openai:gpt-5.4-nano", "--out", "r.md"]
    ) == ["draft.md", "--model", "openai:gpt-5.4-nano", "--out", "r.md"]


def test_panel_subcommand_inserts_mode_panel_after_positional() -> None:
    """`panel draft.md --n-experts 4 ...` → `draft.md --mode panel
    --n-experts 4 ...` — the underlying flat parser sees --mode panel
    set; tests asserting on args.mode continue to work."""
    assert _rewrite_subcommand(
        ["panel", "draft.md", "--n-experts", "4", "--out", "r.md"]
    ) == ["draft.md", "--mode", "panel", "--n-experts", "4", "--out", "r.md"]


def test_panel_subcommand_no_positional_emits_mode_only() -> None:
    """`panel --help` (no positional) still inserts --mode panel so the
    flat parser shows panel-relevant validation errors."""
    assert _rewrite_subcommand(["panel", "--help"]) == [
        "--mode",
        "panel",
        "--help",
    ]


def test_apply_patches_subcommand_translates_to_flat_flag() -> None:
    """`apply-patches draft.docx --patches p.json --out r.docx` →
    `draft.docx --apply-patches p.json --out r.docx` (the existing
    --apply-patches flag form, which all existing tests cover)."""
    assert _rewrite_subcommand(
        [
            "apply-patches",
            "draft.docx",
            "--patches",
            "p.json",
            "--out",
            "r.docx",
        ]
    ) == ["draft.docx", "--apply-patches", "p.json", "--out", "r.docx"]


def test_apply_patches_without_input_exits_with_clear_message() -> None:
    """A bare `apply-patches --patches p.json` (no INPUT positional) is
    a clear user error; exit with code 1 and a usage message."""
    with pytest.raises(SystemExit) as excinfo:
        _rewrite_subcommand(["apply-patches", "--patches", "p.json"])
    assert excinfo.value.code == 1


def test_bare_positional_invocation_unchanged() -> None:
    """No subcommand verb present → argv passes through verbatim. This
    is the back-compat case every existing test relies on."""
    assert _rewrite_subcommand(
        ["draft.md", "--model", "openai:gpt-5.4-nano", "--out", "r.md"]
    ) == ["draft.md", "--model", "openai:gpt-5.4-nano", "--out", "r.md"]


def test_empty_argv_returns_empty() -> None:
    """argparse will then show its top-level help error itself."""
    assert _rewrite_subcommand([]) == []


def test_proofread_subcommand_routes_to_proofread_module() -> None:
    """`proofread draft.md` calls into andamentum.proofread.cli.main
    with the remaining argv (no whetstone review pipeline involved)."""
    with patch("andamentum.proofread.cli.main", return_value=0) as mock_proofread:
        with pytest.raises(SystemExit) as excinfo:
            main(["proofread", "draft.md", "--format", "json"])
    assert excinfo.value.code == 0
    mock_proofread.assert_called_once_with(["draft.md", "--format", "json"])
