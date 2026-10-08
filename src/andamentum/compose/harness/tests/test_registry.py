"""Check cache key, promotion policy, model registry, verb registry and CLI generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, Field

from andamentum.compose.harness.kernel import verbs
from andamentum.compose.harness.kernel.checks import (
    PROMOTE_AT,
    CheckContext,
    CheckSpec,
    cache_key,
    effective_tier,
    load_precisions,
)
from andamentum.compose.harness.kernel.diagnostics import Diagnostic
from andamentum.compose.harness.kernel.models import ModelEntry, ModelRegistry
from andamentum.compose.harness.kernel.view import DocumentView


class _Check:
    version = "1"

    def __init__(
        self,
        *,
        id: str = "c",
        tier: Literal[0, 1, 2] = 1,
        needs: list[str] | None = None,
    ):
        self.spec = CheckSpec(id=id, tier=tier, scope="document", needs=needs or [])

    def run(self, view: DocumentView, ctx: CheckContext) -> list[Diagnostic]:
        return []


def test_cache_key_depends_on_id_version_and_inputs() -> None:
    base = cache_key(_Check(), version="1", inputs=["a", "b"])
    assert base == cache_key(_Check(), version="1", inputs=["a", "b"])
    assert base != cache_key(_Check(id="d"), version="1", inputs=["a", "b"])
    assert base != cache_key(_Check(), version="2", inputs=["a", "b"])
    assert base != cache_key(_Check(), version="1", inputs=["a", "c"])
    assert base != cache_key(_Check(), version="1", inputs=["b", "a"])
    # no ambiguity between one joined input and two inputs
    assert cache_key(_Check(), version="1", inputs=["ab"]) != cache_key(
        _Check(), version="1", inputs=["a", "b"]
    )


def test_deterministic_check_keeps_declared_tier() -> None:
    assert effective_tier(_Check(tier=0), precision=None) == 0
    assert effective_tier(_Check(tier=1, needs=["network"]), precision=None) == 1


def test_model_check_promoted_only_at_threshold() -> None:
    check = _Check(tier=1, needs=["model"])
    assert PROMOTE_AT == 0.9
    assert effective_tier(check, precision=None) == 2
    assert effective_tier(check, precision=0.89) == 2
    assert effective_tier(check, precision=0.9) == 1
    assert effective_tier(check, precision=1.0) == 1


def test_load_precisions(tmp_path: Path) -> None:
    assert load_precisions(tmp_path / "none.json") == {}
    path = tmp_path / "p.json"
    path.write_text('{"numbers": {"precision": 0.93, "n": 40}}')
    assert load_precisions(path) == {"numbers": 0.93}
    path.write_text("[]")
    with pytest.raises(ValueError):
        load_precisions(path)


def test_model_registry() -> None:
    registry = ModelRegistry()
    registry.register(ModelEntry(role="numbers", id="anthropic:claude-haiku-4-5"))
    registry.register(
        ModelEntry(
            role="nouls", kind="decision", id="ollama:nimble:9b", digest="sha256:ab"
        )
    )
    assert registry.get("nouls").digest == "sha256:ab"
    assert registry.as_context() == {
        "numbers": "anthropic:claude-haiku-4-5",
        "nouls": "ollama:nimble:9b",
    }
    with pytest.raises(ValueError):
        registry.register(ModelEntry(role="numbers", id="x"))
    with pytest.raises(KeyError):
        registry.get("missing")


def test_model_registry_from_config() -> None:
    registry = ModelRegistry.from_config(
        {
            "models": {
                "a": "m1",
                "b": {"id": "m2", "kind": "decision"},
            }
        }
    )
    assert [e.id for e in registry.entries()] == ["m1", "m2"]
    assert registry.get("b").kind == "decision"
    assert ModelRegistry.from_config({}).entries() == []
    with pytest.raises(ValueError):
        ModelRegistry.from_config({"models": ["x"]})
    with pytest.raises(ValueError):
        ModelRegistry.from_config({"models": {"a": 3}})


# -- verbs ----------------------------------------------------------------


class _Params(BaseModel):
    file: Path = Field(json_schema_extra={"positional": True}, description="the file")
    checks: list[str] = Field(default_factory=list, description="check ids")
    max_items: int = 3
    offline: bool = False
    mode: Literal["a", "b"] = "a"
    out: Path | None = None


@pytest.fixture
def clean_verbs():
    saved = dict(verbs._VERBS)
    verbs._clear_verbs()
    yield
    verbs._clear_verbs()
    verbs._VERBS.update(saved)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="t")
    verbs.build_subparsers(parser.add_subparsers(dest="command", required=True))
    return parser


def test_verbs_become_subcommands_and_dispatch(clean_verbs: None) -> None:
    seen: list[_Params] = []

    def handler(params: _Params) -> int:
        seen.append(params)
        return 1

    verbs.register_verb(verbs.Verb("demo", "demo verb", _Params, handler))
    args = _parser().parse_args(
        [
            "demo",
            "f.docx",
            "--checks",
            "x",
            "--checks",
            "y",
            "--max-items",
            "5",
            "--offline",
            "--mode",
            "b",
            "--out",
            "o.docx",
        ]
    )
    assert verbs.dispatch_verb(args) == 1
    assert seen[0] == _Params(
        file=Path("f.docx"),
        checks=["x", "y"],
        max_items=5,
        offline=True,
        mode="b",
        out=Path("o.docx"),
    )


def test_verb_defaults(clean_verbs: None) -> None:
    verbs.register_verb(verbs.Verb("demo", "d", _Params, lambda p: 0))
    args = _parser().parse_args(["demo", "f.docx"])
    params = _Params.model_validate({n: getattr(args, n) for n in _Params.model_fields})
    assert (
        params.checks == []
        and params.max_items == 3
        and not params.offline
        and params.out is None
    )


def test_verb_bad_choice_is_usage_error(clean_verbs: None) -> None:
    verbs.register_verb(verbs.Verb("demo", "d", _Params, lambda p: 0))
    with pytest.raises(SystemExit) as exc:
        _parser().parse_args(["demo", "f.docx", "--mode", "z"])
    assert exc.value.code == 2


def test_dispatch_returns_2_on_invalid_params(clean_verbs: None) -> None:
    class Strict(BaseModel):
        n: int = Field(ge=1)

    verbs.register_verb(verbs.Verb("strict", "s", Strict, lambda p: 0))
    args = _parser().parse_args(["strict", "--n", "0"])
    assert verbs.dispatch_verb(args) == 2


def test_required_flag_is_required(clean_verbs: None) -> None:
    class Req(BaseModel):
        name: str

    verbs.register_verb(verbs.Verb("req", "r", Req, lambda p: 0))
    with pytest.raises(SystemExit):
        _parser().parse_args(["req"])


def test_unsupported_field_type_fails_loud(clean_verbs: None) -> None:
    class Bad(BaseModel):
        x: dict[str, int] = {}

    verbs.register_verb(verbs.Verb("bad", "b", Bad, lambda p: 0))
    with pytest.raises(ValueError, match="unsupported type"):
        _parser()


def test_registered_verbs_become_subcommands(clean_verbs: None) -> None:
    verbs.register_verb(verbs.Verb("demo-harness", "demo", _Params, lambda p: 7))
    parser = _parser()
    args = parser.parse_args(["demo-harness", "f.docx"])
    assert args.verb == "demo-harness"
    assert verbs.dispatch_verb(args) == 7
