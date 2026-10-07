"""What a run does, as one named and hashed value: the strategy of every slot,
every cap, and the determinism knobs.

``ResearchConfig`` is frozen. Each slot (screen, refine, rank, passages, expand,
facets) holds a strategy *name*; the registries below map names to the
implementations in ``strategies.py``. An experiment adds a strategy (an object
in ``strategies.py`` and a registry entry) and a preset, and never edits the
engine. ``config_hash`` identifies the configuration; the engine records it on
every run (``runs.config_json``) and every ``ResearchResult``.

Four presets (``PRESETS``):

- ``baseline`` (the default of ``research()``): one generic relevance question
  per candidate, dropped below ``GENERIC_DROP_BELOW``; reciprocal-rank fusion;
  at most 25 candidates screened per sub-question a round and 8 works read per
  sub-question over the run; citations of the 3 best kept works, at most 20
  new candidates a round; BM25 passage selection (4 a work); no refinement and
  no criteria.
- ``fast``: the baseline with the ``generic_prefilter`` screen (a BM25 ranking
  puts only the best ``PREFILTER_TOP`` candidates of a sub-question to the
  decision model) and the ``code_first`` verification (a report sentence whose
  content tokens are in its quotes is supported in code). Everything else is
  the baseline's, so the two differ only in those two strategies.
- ``recall``: the baseline with the ``source_rank`` queue (waiting candidates
  are screened best-by-source-rank first instead of in arrival order) and
  nothing else (round 3).
- ``hybrid`` (round 5): the baseline with three strategies swapped: the
  ``llm_plus_question`` queries (the raw question searched too, once, in round
  1), the ``rerank_dense`` rank (retrieve, then rerank: the searches only
  gather the pool, the order is the fusion of dense similarity against the
  question and the sub-question and the screen's p, over every candidate held;
  the result is the fused order) and the ``rank_only`` screen (the generic
  question about the original question; its p is a ranked list, nothing is
  dropped). It needs an embedding
  model (``research(embedding_model=)``).
- ``criteria``: the behaviour built first (LLM-compiled criteria scored by the
  decision model, refinement, criteria passage scoring, criteria facets, the
  budget's caps), kept as an arm to compare against the baseline.

All presets set the same determinism knobs (``temperature`` 0 and a fixed
``seed``), so the arms differ only in their strategies.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai.settings import ModelSettings

from .steps import MAX_EXPAND_SEEDS
from .strategies import (
    GENERIC_DROP_BELOW,
    ArrivalQueue,
    CitationExpand,
    CodeFirstVerify,
    CriteriaFacets,
    CriteriaPassages,
    CriteriaRefine,
    CriteriaScreen,
    ExpandStrategy,
    FacetStrategy,
    GenericPrefilterScreen,
    GenericScreen,
    LexicalPassages,
    LlmPlusQuestion,
    LlmQueries,
    ModelVerify,
    NoExpand,
    NoFacets,
    NoRefine,
    PassageStrategy,
    QueryStrategy,
    QueueStrategy,
    RankOnlyScreen,
    RankStrategy,
    RefineStrategy,
    RerankDenseRank,
    RrfDenseRank,
    RrfRank,
    ScoreRank,
    ScreenStrategy,
    SourceRankQueue,
    VerifyStrategy,
)

# The registries: a strategy name to its implementation, one per slot.
SCREENS: dict[str, ScreenStrategy] = {
    "generic": GenericScreen(),
    "generic_prefilter": GenericPrefilterScreen(),
    "criteria": CriteriaScreen(),
    "rank_only": RankOnlyScreen(),
}
REFINES: dict[str, RefineStrategy] = {"off": NoRefine(), "on": CriteriaRefine()}
RANKS: dict[str, RankStrategy] = {
    "rrf": RrfRank(),
    "score": ScoreRank(),
    "rrf_dense": RrfDenseRank(),
    "rerank_dense": RerankDenseRank(),
}
PASSAGES: dict[str, PassageStrategy] = {
    "lexical": LexicalPassages(),
    "criteria": CriteriaPassages(),
}
EXPANDS: dict[str, ExpandStrategy] = {"citations": CitationExpand(), "off": NoExpand()}
FACETS: dict[str, FacetStrategy] = {"off": NoFacets(), "on": CriteriaFacets()}
VERIFIES: dict[str, VerifyStrategy] = {
    "model": ModelVerify(),
    "code_first": CodeFirstVerify(),
}

QUEUES: dict[str, QueueStrategy] = {
    "arrival": ArrivalQueue(),
    "source_rank": SourceRankQueue(),
}

QUERIES: dict[str, QueryStrategy] = {
    "llm": LlmQueries(),
    "llm_plus_question": LlmPlusQuestion(),
}

_SLOTS: dict[str, Mapping[str, object]] = {
    "screen": SCREENS,
    "refine": REFINES,
    "rank": RANKS,
    "passages": PASSAGES,
    "expand": EXPANDS,
    "facets": FACETS,
    "verify": VERIFIES,
    "queue": QUEUES,
    "queries": QUERIES,
}

# The fixed seed both presets send with every language-model call.
DEFAULT_SEED = 42
# Left out of the hash: what the configuration is called, not what it does.
_UNHASHED = frozenset({"name", "description"})
# Fields added after the first presets were hashed: left out of the hash while
# they hold the value every earlier configuration had, so a configuration that
# predates the field keeps its hash (the recorded hashes keep their meaning).
_UNHASHED_AT_DEFAULT: dict[str, object] = {
    "verify": "model",
    "queue": "arrival",
    "queries": "llm",
}


class ResearchConfig(BaseModel):
    """Every swappable strategy and cap of a run.

    Caps set to ``None`` leave only the run's ``Budget`` bound (as built); a
    set cap applies inside the budget's, never beyond it:

    - ``screen_per_subquestion``: candidates screened per sub-question per round;
    - ``read_per_subquestion``: works read per sub-question over the whole run;
    - ``expand_seeds``: kept works whose citations a round follows;
    - ``expand_per_round``: new candidates a round's expansion may bring;
    - ``passages_per_work``: passages selected per work and sub-question.

    ``drop_below`` is the generic screen's threshold. ``temperature`` and
    ``seed`` go to every language-model call as pydantic-ai model settings
    (``None``: not sent); see ``CLAUDE.md`` for what Ollama honours.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    version: int = Field(default=1, ge=1)
    screen: str
    refine: str
    rank: str
    passages: str
    expand: str
    facets: str
    verify: str = "model"
    queue: str = "arrival"
    queries: str = "llm"
    drop_below: float = Field(default=GENERIC_DROP_BELOW, ge=0.0, le=1.0)
    screen_per_subquestion: int | None = Field(default=None, ge=1)
    read_per_subquestion: int | None = Field(default=None, ge=1)
    expand_seeds: int = Field(default=MAX_EXPAND_SEEDS, ge=0)
    expand_per_round: int | None = Field(default=None, ge=0)
    passages_per_work: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=0.0, ge=0.0)
    seed: int | None = DEFAULT_SEED

    @model_validator(mode="after")
    def _known_strategies(self) -> ResearchConfig:
        for slot, registry in _SLOTS.items():
            chosen = getattr(self, slot)
            if chosen not in registry:
                raise ValueError(
                    f"unknown {slot} strategy {chosen!r}; known: {sorted(registry)}"
                )
        if (
            self.refine_strategy.needs_criteria
            and not self.screen_strategy.needs_criteria
        ):
            raise ValueError(
                f"refine {self.refine!r} revises criteria; it needs a criteria "
                f"screen, not {self.screen!r}"
            )
        return self

    # -- the strategies --------------------------------------------------------

    @property
    def screen_strategy(self) -> ScreenStrategy:
        return SCREENS[self.screen]

    @property
    def refine_strategy(self) -> RefineStrategy:
        return REFINES[self.refine]

    @property
    def rank_strategy(self) -> RankStrategy:
        return RANKS[self.rank]

    @property
    def passage_strategy(self) -> PassageStrategy:
        return PASSAGES[self.passages]

    @property
    def expand_strategy(self) -> ExpandStrategy:
        return EXPANDS[self.expand]

    @property
    def facet_strategy(self) -> FacetStrategy:
        return FACETS[self.facets]

    @property
    def verify_strategy(self) -> VerifyStrategy:
        return VERIFIES[self.verify]

    @property
    def queue_strategy(self) -> QueueStrategy:
        return QUEUES[self.queue]

    @property
    def query_strategy(self) -> QueryStrategy:
        return QUERIES[self.queries]

    @property
    def needs_embeddings(self) -> bool:
        """Whether the run needs an embedding model (dense similarity)."""
        return self.rank_strategy.needs_embeddings

    @property
    def needs_criteria(self) -> bool:
        """Whether the run compiles criteria at Plan (an LLM call per sub-question)."""
        return any(
            s.needs_criteria
            for s in (
                self.screen_strategy,
                self.refine_strategy,
                self.passage_strategy,
                self.facet_strategy,
            )
        )

    def config_hash(self) -> str:
        """:func:`config_hash` of this configuration."""
        return config_hash(self)

    def model_settings(self) -> ModelSettings | None:
        """The pydantic-ai model settings every language-model call sends."""
        settings = ModelSettings()
        if self.temperature is not None:
            settings["temperature"] = self.temperature
        if self.seed is not None:
            settings["seed"] = self.seed
        return settings or None


def canonical_json(config: ResearchConfig) -> str:
    """The configuration as canonical JSON: sorted keys, no spaces, the name
    and description left out."""
    data = config.model_dump(mode="json", exclude=set(_UNHASHED))
    for field, default in _UNHASHED_AT_DEFAULT.items():
        if data.get(field) == default:
            del data[field]
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(config: ResearchConfig) -> str:
    """sha256 of :func:`canonical_json`: equal for two configurations that do
    the same thing, whatever they are called."""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


PRESETS: dict[str, ResearchConfig] = {
    "baseline": ResearchConfig(
        name="baseline",
        description="one generic relevance question per candidate, RRF ranking, "
        "caps, BM25 passages; no criteria, no refinement",
        screen="generic",
        refine="off",
        rank="rrf",
        passages="lexical",
        expand="citations",
        facets="off",
        drop_below=GENERIC_DROP_BELOW,
        screen_per_subquestion=25,
        read_per_subquestion=8,
        expand_seeds=3,
        expand_per_round=20,
        passages_per_work=4,
    ),
    "fast": ResearchConfig(
        name="fast",
        description="the baseline with a lexical pre-filter before the generic "
        "screen question and code-first claim-support verification",
        screen="generic_prefilter",
        refine="off",
        rank="rrf",
        passages="lexical",
        expand="citations",
        facets="off",
        verify="code_first",
        drop_below=GENERIC_DROP_BELOW,
        screen_per_subquestion=25,
        read_per_subquestion=8,
        expand_seeds=3,
        expand_per_round=20,
        passages_per_work=4,
    ),
    "recall": ResearchConfig(
        name="recall",
        description="the baseline with merit-ordered screening admission: waiting "
        "candidates are screened best by source search rank first",
        screen="generic",
        refine="off",
        rank="rrf",
        passages="lexical",
        expand="citations",
        facets="off",
        queue="source_rank",
        drop_below=GENERIC_DROP_BELOW,
        screen_per_subquestion=25,
        read_per_subquestion=8,
        expand_seeds=3,
        expand_per_round=20,
        passages_per_work=4,
    ),
    "hybrid": ResearchConfig(
        name="hybrid",
        description="the baseline with the raw question searched too, dense "
        "similarity and the screen reranking the pool the searches gathered, the "
        "screen asking about the original question and dropping nothing "
        "(needs an embedding model)",
        screen="rank_only",
        refine="off",
        rank="rerank_dense",
        passages="lexical",
        expand="citations",
        facets="off",
        queries="llm_plus_question",
        drop_below=GENERIC_DROP_BELOW,
        screen_per_subquestion=25,
        read_per_subquestion=8,
        expand_seeds=3,
        expand_per_round=20,
        passages_per_work=4,
    ),
    "criteria": ResearchConfig(
        name="criteria",
        description="LLM-compiled criteria scored per candidate, refinement, "
        "criteria passage scoring and facets; the budget's caps (as first built)",
        screen="criteria",
        refine="on",
        rank="score",
        passages="criteria",
        expand="citations",
        facets="on",
        expand_seeds=MAX_EXPAND_SEEDS,
    ),
}
DEFAULT_PRESET = "hybrid"


def preset(name: str) -> ResearchConfig:
    """The preset called *name*; a ``KeyError`` naming the known ones otherwise."""
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(f"unknown preset {name!r}; known: {sorted(PRESETS)}") from None
