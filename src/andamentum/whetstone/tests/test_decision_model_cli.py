"""``--decision-model`` on the whetstone CLI."""

from __future__ import annotations

from unittest import mock

import pytest

from andamentum.core.decisions import DecisionModel, DecisionModelUnavailable
from andamentum.whetstone import cli as cli_mod

BASE = ["draft.md", "--model", "stub:none", "--out", "review.md"]


def _args(argv: list[str]):
    return cli_mod._build_parser().parse_args(argv)


def _validate(argv: list[str]) -> None:
    cli_mod._validate_args(_args(argv))


def test_the_flag_defaults_to_off() -> None:
    assert _args(BASE).decision_model is None


def test_review_and_checklist_accept_a_decision_model() -> None:
    _validate([*BASE, "--decision-model", "nimble:9b"])
    _validate([*BASE, "--mode", "checklist", "--decision-model", "nimble:9b"])


def test_panel_refuses_a_decision_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        _validate(
            [
                *BASE,
                "--mode",
                "panel",
                "--i-am-the-author",
                "--decision-model",
                "nimble:9b",
            ]
        )
    assert exc.value.code == 1
    assert "no effect under `panel`" in capsys.readouterr().err


def test_an_explicit_document_type_leaves_nothing_to_decide(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        _validate([*BASE, "--document-type", "essay", "--decision-model", "nimble:9b"])
    assert exc.value.code == 1
    assert "already settles" in capsys.readouterr().err


async def test_an_unavailable_decision_model_stops_the_run_before_any_review(
    capsys: pytest.CaptureFixture[str],
) -> None:
    document = mock.Mock(warnings=[])
    review = mock.AsyncMock()
    unavailable = DecisionModelUnavailable("run: ollama pull nimble:9b")
    with (
        mock.patch(
            "andamentum.whetstone.reviewed_document.open_source",
            mock.AsyncMock(return_value=document),
        ),
        mock.patch.object(
            DecisionModel, "open", mock.AsyncMock(side_effect=unavailable)
        ),
        mock.patch.object(cli_mod, "_review_and_write", review),
    ):
        with pytest.raises(SystemExit) as exc:
            await cli_mod._run(
                _args([*BASE, "--decision-model", "nimble:9b"]), mock.Mock()
            )
    assert exc.value.code == 1
    assert "ollama pull nimble:9b" in capsys.readouterr().err
    review.assert_not_called()
    document.cleanup.assert_called_once()


async def test_the_opened_handle_reaches_the_review_and_is_closed() -> None:
    document = mock.Mock(warnings=[])
    handle = mock.Mock(aclose=mock.AsyncMock())
    review = mock.AsyncMock()
    with (
        mock.patch(
            "andamentum.whetstone.reviewed_document.open_source",
            mock.AsyncMock(return_value=document),
        ),
        mock.patch.object(DecisionModel, "open", mock.AsyncMock(return_value=handle)),
        mock.patch.object(cli_mod, "_review_and_write", review),
    ):
        await cli_mod._run(_args([*BASE, "--decision-model", "nimble:9b"]), mock.Mock())
    assert review.call_args.kwargs["decision_model"] is handle
    handle.aclose.assert_awaited_once()
    document.cleanup.assert_called_once()


async def test_without_the_flag_no_handle_is_opened() -> None:
    document = mock.Mock(warnings=[])
    review = mock.AsyncMock()
    opener = mock.AsyncMock()
    with (
        mock.patch(
            "andamentum.whetstone.reviewed_document.open_source",
            mock.AsyncMock(return_value=document),
        ),
        mock.patch.object(DecisionModel, "open", opener),
        mock.patch.object(cli_mod, "_review_and_write", review),
    ):
        await cli_mod._run(_args(BASE), mock.Mock())
    opener.assert_not_called()
    assert review.call_args.kwargs["decision_model"] is None
