"""The configuration: presets as the spec table sets them, a stable hash, and
the ``criteria`` preset doing exactly what the engine did before presets existed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from andamentum.core import fetch_gate, url_safety

from .. import ResearchStore, SourceConfig, research
from ..ask import Asker
from ..config import (
    DEFAULT_SEED,
    PRESETS,
    ResearchConfig,
    canonical_json,
    config_hash,
    preset,
)
from ..steps import MAX_EXPAND_SEEDS
from ..strategies import GENERIC_DROP_BELOW
from ..types import Budget, RunStats
from . import engine_world
from .preset_trace import SCENARIOS, trace

FIXTURE = Path(__file__).parent / "fixtures" / "criteria_preset_trace.json"


@pytest.fixture(autouse=True)
def _public_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(url_safety, "is_safe_url", lambda url, **_kw: (True, ""))
    monkeypatch.setattr(
        "andamentum.references.http.is_safe_url", lambda url, **_kw: (True, "")
    )
    fetch_gate._robots_cache_clear()


# -- the presets -------------------------------------------------------------------


def test_the_baseline_preset_is_the_spec_table() -> None:
    base = PRESETS["baseline"]
    assert (
        base.screen,
        base.refine,
        base.rank,
        base.passages,
        base.expand,
        base.facets,
    ) == (
        "generic",
        "off",
        "rrf",
        "lexical",
        "citations",
        "off",
    )
    assert base.drop_below == GENERIC_DROP_BELOW == 0.5
    assert base.screen_per_subquestion == 25
    assert base.read_per_subquestion == 8
    assert base.expand_seeds == 3 and base.expand_per_round == 20
    assert base.passages_per_work == 4
    assert not base.needs_criteria  # no LLM-written criteria anywhere


def test_the_criteria_preset_is_the_first_build() -> None:
    crit = PRESETS["criteria"]
    assert (
        crit.screen,
        crit.refine,
        crit.rank,
        crit.passages,
        crit.expand,
        crit.facets,
    ) == (
        "criteria",
        "on",
        "score",
        "criteria",
        "citations",
        "on",
    )
    # the caps are the budget's, as built
    assert crit.screen_per_subquestion is None
    assert crit.read_per_subquestion is None
    assert crit.expand_per_round is None
    assert crit.passages_per_work is None
    assert crit.expand_seeds == MAX_EXPAND_SEEDS
    assert crit.needs_criteria


def test_both_presets_send_the_same_determinism_knobs() -> None:
    for config in PRESETS.values():
        assert config.model_settings() == {"temperature": 0.0, "seed": DEFAULT_SEED}
    loose = PRESETS["baseline"].model_copy(update={"temperature": None, "seed": None})
    assert loose.model_settings() is None


def test_research_defaults_to_hybrid() -> None:
    import inspect

    default = inspect.signature(research).parameters["config"].default
    assert default is PRESETS["hybrid"]


def test_unknown_strategies_and_impossible_combinations_are_refused() -> None:
    fields = PRESETS["baseline"].model_dump()
    with pytest.raises(ValidationError, match="unknown screen strategy"):
        ResearchConfig(**{**fields, "screen": "magic"})
    with pytest.raises(ValidationError, match="needs a criteria screen"):
        ResearchConfig(**{**fields, "refine": "on"})
    with pytest.raises(ValidationError):
        ResearchConfig(**{**fields, "surprise": 1})
    with pytest.raises(KeyError, match="known"):
        preset("nope")


# -- the hash ----------------------------------------------------------------------


def test_the_config_hash_is_stable() -> None:
    base = PRESETS["baseline"]
    # pinned: a change here means every recorded run's hash changes meaning
    assert (
        config_hash(base)
        == "9621c6dae218004804ae66dd41983449346745bdf669406159a8a6ff3c90755c"
    )
    assert config_hash(ResearchConfig(**base.model_dump())) == config_hash(base)
    assert base.config_hash() == config_hash(base)
    # the same content under another name hashes the same; a strategy does not
    renamed = base.model_copy(update={"name": "copy", "description": "x"})
    assert config_hash(renamed) == config_hash(base)
    assert config_hash(PRESETS["criteria"]) != config_hash(base)
    changed = base.model_copy(update={"drop_below": 0.6})
    assert config_hash(changed) != config_hash(base)
    # canonical: sorted keys, no whitespace
    text = canonical_json(base)
    assert text == json.dumps(
        json.loads(text), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    assert "name" not in json.loads(text) and "description" not in json.loads(text)


async def test_the_hash_is_on_the_run_and_the_result() -> None:
    store = ResearchStore.in_memory()
    world = engine_world.World()
    result = await research(
        engine_world.QUESTION,
        mode="web",
        model=engine_world.world_llm().model,
        store=store,
        sources=SourceConfig(searxng_url=engine_world.SEARXNG),
        source_client=world.client(store),
        config=PRESETS["baseline"],
    )
    assert result.config_name == "baseline"
    assert result.config_hash == PRESETS["baseline"].config_hash()
    record = store.get_run(result.run_id)
    assert record is not None
    assert record.config["config_hash"] == result.config_hash
    research_config = record.config["research_config"]
    assert isinstance(research_config, dict) and research_config["screen"] == "generic"
    assert record.config["frozen"] is None


# -- the criteria preset is the engine as first built --------------------------------


async def test_the_criteria_preset_reproduces_the_first_build() -> None:
    """Every model request, in order, and the evidence table of four runs over
    the invented world are those the engine at commit cf61b6e produced (the
    fixture was recorded from that commit's code with ``preset_trace``)."""
    recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert [(t["mode"], t["decision"]) for t in recorded] == list(SCENARIOS)
    for expected, (mode, decision) in zip(recorded, SCENARIOS, strict=True):
        fetch_gate._robots_cache_clear()
        got = await trace(
            mode=mode,
            decision=decision,
            run=research,
            world=engine_world,
            store_type=ResearchStore,
            source_config=SourceConfig,
            extra={"config": PRESETS["criteria"], "decision_cache": False},
        )
        got = json.loads(json.dumps(got, sort_keys=True))
        for key in expected:
            assert got[key] == expected[key], f"{mode}/{decision}: {key} differs"


# -- the determinism knobs reach the model -----------------------------------------


async def test_the_model_settings_reach_every_language_model_call() -> None:
    seen: list[object] = []

    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info.model_settings)
        return ModelResponse(parts=[TextPart(content='{"verdicts": []}')])

    from ..ask import Verdicts

    asker = Asker(
        model=FunctionModel(answer),
        store=ResearchStore.in_memory(),
        run_id="r",
        budget=Budget(),
        stats=RunStats(),
        model_settings=PRESETS["baseline"].model_settings(),
    )
    await asker.llm(instructions="x", output_type=Verdicts, user_message="y")
    assert seen and all(
        isinstance(s, dict)
        and s.get("temperature") == 0.0
        and s.get("seed") == DEFAULT_SEED
        for s in seen
    )
