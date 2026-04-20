# Core Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract duplicated model resolution, agent execution, and embedding code into a shared `andamentum.core` module. All three sub-modules (epistemic, deep_research, document_store) import from core instead of maintaining independent implementations.

**Architecture:** Bottom-up extraction — create the core module with the superset of all features, then migrate each sub-module one at a time with full test coverage between each migration. Each task produces a working, tested state. No Big Bang refactor.

**Tech Stack:** Python 3.12, pydantic-ai, httpx, pytest (asyncio_mode=auto), pyright, ruff

---

## Risk Assessment

**High-risk touch points:**
1. `document_store/embeddings.py` has a stateful `EmbeddingService` with concurrent requests and a long-lived httpx client. 6 callsites depend on its lifecycle (`close()` in `finally` blocks). Changing the interface here could break ingestion, search, and repair.
2. The epistemic runner's `system_prompt=` vs deep_research's `instructions=` pydantic-ai Agent kwarg. Both are valid but different parameters. Must standardize without breaking either.
3. `document_store/extraction.py` uses `@output_validator` decorators on Agent instances. A shared runner would need to accept validator callables.

**Mitigation strategy:**
- Create core module first (Task 1-2) — no existing code changes yet
- Migrate model resolution first (Task 3) — smallest blast radius, purely string→string
- Migrate agent runners second (Task 4-5) — preserve all existing interfaces as thin wrappers initially
- Migrate embeddings last (Task 6) — highest risk, most callsites
- Run full test suite after EVERY task

---

## File Structure

**New files:**
```
src/andamentum/core/
  __init__.py           — public API re-exports
  models.py             — resolve_model(), env var handling
  agents.py             — AgentDefinition, AgentRunner (with PromptedOutput fallback)
  embeddings.py          — embed_texts(), EmbeddingService (unified)
```

**Modified files (by task):**
- Task 3: `epistemic/runner.py`, `deep_research/runner.py`, `epistemic/cli.py`, `deep_research/cli.py`
- Task 4: `epistemic/runner.py`, `epistemic/operations/base.py`
- Task 5: `deep_research/runner.py`, `deep_research/nodes.py`, `document_store/extraction.py`, `document_store/query_planner.py`, `document_store/public.py`
- Task 6: `epistemic/embeddings.py`, `document_store/embeddings.py`, + 15 callsites

---

## Task 1: Create `core/models.py` — unified model resolution

The lowest-risk starting point. Extract the superset `resolve_model()` function from epistemic's runner (which has ollama + bedrock + passthrough) into a shared module. Also extract CLI model resolution.

**Files:**
- Create: `src/andamentum/core/__init__.py`
- Create: `src/andamentum/core/models.py`
- Test: `src/andamentum/core/tests/__init__.py`
- Test: `src/andamentum/core/tests/test_models.py`

- [ ] **Step 1: Create `src/andamentum/core/__init__.py`**

```python
"""Andamentum Core — shared infrastructure for all sub-modules.

Provides:
- Model resolution (ollama, bedrock, passthrough)
- Agent execution with PromptedOutput fallback
- Embedding client

Sub-modules (epistemic, deep_research, document_store) import from here
instead of maintaining independent implementations.
"""
```

- [ ] **Step 2: Create `src/andamentum/core/models.py`**

Extract from `epistemic/runner.py` lines 41-110 (the `_resolve_model` function plus the bedrock model map and regional prefix map). Make it public:

```python
"""Model resolution for pydantic-ai.

Handles model string prefixes:
- ollama:model_name → OpenAIChatModel with OllamaProvider
- bedrock:friendly_name → BedrockConverseModel with regional inference profiles
- anything else → passthrough (pydantic-ai resolves via infer_model)

Also provides CLI model resolution from args or env vars.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Bedrock friendly-name → model ID map
BEDROCK_MODEL_MAP: dict[str, str] = {
    "claude-haiku-3-5": "anthropic.claude-3-5-haiku-20241022-v1:0",
    "claude-sonnet-3-5": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-haiku-4-5": "anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-sonnet-4-5": "anthropic.claude-sonnet-4-5-20250514-v1:0",
    "claude-opus-4-5": "anthropic.claude-opus-4-5-20250514-v1:0",
    "qwen3-32b": "qwen.qwen3-32b-v1:0",
    "mistral-7b": "mistral.mistral-7b-instruct-v0:2",
    "ministral-8b": "mistral.ministral-8b-2410-v1:0",
    "gemma-3-12b": "google.gemma-3-12b-it-v1:0",
}

REGION_PREFIX_MAP: dict[str, str] = {
    "ap-southeast-2": "au",
    "eu-west-1": "eu",
    "eu-central-1": "eu",
    "ap-northeast-1": "ap",
}


def resolve_model(model: str) -> Any:
    """Resolve a model string to a pydantic-ai model object.

    Handles:
    - "ollama:llama3" → OpenAIChatModel with OllamaProvider
    - "bedrock:claude-haiku-4-5" → BedrockConverseModel
    - "openai:gpt-4o" → passthrough string (pydantic-ai resolves)
    - "anthropic:claude-haiku-4-5" → passthrough string

    Environment variables:
    - OLLAMA_BASE_URL: Ollama API endpoint (default http://localhost:11434/v1)
    - AWS_PROFILE: boto3 session profile for Bedrock
    - AWS_DEFAULT_REGION / AWS_REGION: Bedrock region
    """
    if model.startswith("ollama:"):
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OllamaProvider

        model_name = model.split(":", 1)[1]
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return OpenAIChatModel(model_name=model_name, provider=OllamaProvider(base_url=base_url))

    if model.startswith("bedrock:"):
        from pydantic_ai.models.bedrock import BedrockConverseModel
        from pydantic_ai.providers.bedrock import BedrockProvider
        import boto3

        friendly = model.split(":", 1)[1]
        model_id = BEDROCK_MODEL_MAP.get(friendly, friendly)

        region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
        if region:
            prefix = REGION_PREFIX_MAP.get(region, "")
            if prefix:
                model_id = f"{prefix}.{model_id}"

        profile = os.environ.get("AWS_PROFILE")
        session = boto3.Session(profile_name=profile, region_name=region) if profile or region else boto3.Session()

        return BedrockConverseModel(
            model_name=model_id,
            provider=BedrockProvider(boto3_session=session),
        )

    # Passthrough — pydantic-ai resolves "openai:gpt-4o", "anthropic:...", etc.
    return model


def resolve_model_from_args(model_arg: str | None) -> str:
    """Resolve model from CLI arg or ANDAMENTUM_MAIN_LLM_MODEL env var.

    Args:
        model_arg: Value from --model CLI flag, or None

    Returns:
        Model string

    Raises:
        SystemExit: If no model is available
    """
    import sys

    model = model_arg or os.environ.get("ANDAMENTUM_MAIN_LLM_MODEL")
    if not model:
        print(
            "Error: --model is required (or set ANDAMENTUM_MAIN_LLM_MODEL).",
            file=sys.stderr,
        )
        sys.exit(1)
    return model
```

- [ ] **Step 3: Create tests in `src/andamentum/core/tests/test_models.py`**

```python
"""Tests for core model resolution."""

from andamentum.core.models import resolve_model, BEDROCK_MODEL_MAP


class TestResolveModel:
    def test_passthrough_openai(self):
        """openai: prefix passes through as string."""
        result = resolve_model("openai:gpt-4o")
        assert result == "openai:gpt-4o"

    def test_passthrough_anthropic(self):
        """anthropic: prefix passes through as string."""
        result = resolve_model("anthropic:claude-haiku-4-5")
        assert result == "anthropic:claude-haiku-4-5"

    def test_ollama_creates_model_object(self):
        """ollama: prefix creates OpenAIChatModel."""
        result = resolve_model("ollama:llama3")
        # Should be an OpenAIChatModel, not a string
        assert not isinstance(result, str)
        assert hasattr(result, "model_name")

    def test_bedrock_model_map_has_entries(self):
        """Bedrock model map should have known models."""
        assert "claude-haiku-4-5" in BEDROCK_MODEL_MAP
        assert "claude-sonnet-4-5" in BEDROCK_MODEL_MAP
```

- [ ] **Step 4: Run tests, verify, commit**

```bash
uv run pytest src/andamentum/core/tests/test_models.py -v
uv run pyright src/andamentum/core/
uv run ruff check src/andamentum/core/
git add -A && git commit -m "feat(core): create core module with unified model resolution"
```

---

## Task 2: Create `core/agents.py` — unified agent runner with fallback

Extract the AgentDefinition dataclass and the agent execution pattern (with PromptedOutput fallback) into core.

**Files:**
- Create: `src/andamentum/core/agents.py`
- Test: `src/andamentum/core/tests/test_agents.py`

- [ ] **Step 1: Create `src/andamentum/core/agents.py`**

```python
"""Agent definition and execution with structured output fallback.

Provides:
- AgentDefinition: frozen dataclass describing an agent's config
- AgentRunner: executes agents with caching and PromptedOutput fallback
- run_agent_with_fallback: one-shot agent execution with fallback

The PromptedOutput fallback catches UnexpectedModelBehavior (model
ignores tool definitions) and retries by injecting the JSON schema
directly into the system prompt. This is essential for small models
(Ollama locals, nano-tier APIs) that don't support tool-based
structured output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentDefinition:
    """Configuration for an epistemic/research agent.

    Each definition maps to a pydantic-ai Agent with a system prompt
    and structured output model.
    """

    name: str
    prompt: str
    output_model: type[BaseModel]
    retries: int = 3
    output_retries: int = 5


class AgentRunner:
    """Executes agents with caching and PromptedOutput fallback.

    Usage:
        runner = AgentRunner(model="openai:gpt-4o")
        result = await runner.run(agent_defn, key="value", ...)
    """

    def __init__(self, *, model: Any):
        from .models import resolve_model

        self.model = resolve_model(model) if isinstance(model, str) else model
        self._cache: dict[str, Any] = {}

    async def run(
        self,
        defn: AgentDefinition,
        *,
        validators: list[Callable[..., Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run an agent with PromptedOutput fallback on structured output failure.

        Args:
            defn: Agent definition with prompt and output model
            validators: Optional output validator callables to register
            **kwargs: Passed as "key: value" lines in the user message

        Returns:
            The agent's structured output (instance of defn.output_model)
        """
        from pydantic_ai import Agent
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        # Build user message from kwargs
        user_message = "\n".join(f"{k}: {v}" for k, v in kwargs.items())

        # Get or create cached agent
        if defn.name not in self._cache:
            agent = Agent(
                self.model,
                instructions=defn.prompt,
                output_type=defn.output_model,
                retries=defn.retries,
                output_retries=defn.output_retries,
            )
            if validators:
                for v in validators:
                    agent.output_validator(v)
            self._cache[defn.name] = agent

        try:
            result = await self._cache[defn.name].run(user_message)
            return result.output
        except UnexpectedModelBehavior:
            # Fallback: inject JSON schema into prompt instead of using tools
            logger.info(
                "Agent %s: tool-based output failed, falling back to PromptedOutput",
                defn.name,
            )
            return await self._run_prompted_fallback(defn, user_message, validators)

    async def _run_prompted_fallback(
        self,
        defn: AgentDefinition,
        user_message: str,
        validators: list[Callable[..., Any]] | None = None,
    ) -> Any:
        """Retry with PromptedOutput (schema in prompt, not tools)."""
        from pydantic_ai import Agent
        from pydantic_ai.output import PromptedOutput

        cache_key = f"{defn.name}__prompted"
        if cache_key not in self._cache:
            agent = Agent(
                self.model,
                instructions=defn.prompt,
                output_type=PromptedOutput(defn.output_model),
                retries=defn.retries,
                output_retries=defn.output_retries,
            )
            if validators:
                for v in validators:
                    agent.output_validator(v)
            self._cache[cache_key] = agent

        result = await self._cache[cache_key].run(user_message)
        return result.output

    def clear_cache(self) -> None:
        """Clear the agent cache."""
        self._cache.clear()


async def run_agent_with_fallback(
    model: Any,
    *,
    instructions: str,
    output_type: type[BaseModel],
    user_message: str,
    retries: int = 3,
    output_retries: int = 5,
    validators: list[Callable[..., Any]] | None = None,
) -> Any:
    """One-shot agent execution with PromptedOutput fallback.

    For callsites that don't need a persistent runner (e.g., document_store
    extraction, query planning). Creates a fresh agent each call.

    Args:
        model: Resolved model (from resolve_model) or model string
        instructions: System prompt for the agent
        output_type: Pydantic BaseModel class for structured output
        user_message: The user prompt
        retries: Max retries for the agent
        output_retries: Max retries for output validation
        validators: Optional output validator callables

    Returns:
        Instance of output_type
    """
    from pydantic_ai import Agent
    from pydantic_ai.exceptions import UnexpectedModelBehavior
    from pydantic_ai.output import PromptedOutput

    from .models import resolve_model

    resolved = resolve_model(model) if isinstance(model, str) else model

    agent = Agent(
        resolved,
        instructions=instructions,
        output_type=output_type,
        retries=retries,
        output_retries=output_retries,
    )
    if validators:
        for v in validators:
            agent.output_validator(v)

    try:
        result = await agent.run(user_message)
        return result.output
    except UnexpectedModelBehavior:
        logger.info("One-shot agent: falling back to PromptedOutput")
        fallback = Agent(
            resolved,
            instructions=instructions,
            output_type=PromptedOutput(output_type),
            retries=retries,
            output_retries=output_retries,
        )
        if validators:
            for v in validators:
                fallback.output_validator(v)
        result = await fallback.run(user_message)
        return result.output
```

- [ ] **Step 2: Write tests for `core/agents.py`**

Test AgentDefinition creation, AgentRunner caching, and the PromptedOutput fallback trigger. Use mock agents that simulate UnexpectedModelBehavior.

- [ ] **Step 3: Update `core/__init__.py` with exports**

```python
from .models import resolve_model, resolve_model_from_args
from .agents import AgentDefinition, AgentRunner, run_agent_with_fallback
```

- [ ] **Step 4: Run tests, verify, commit**

```bash
uv run pytest src/andamentum/core/tests/ -v
uv run pyright src/andamentum/core/
git add -A && git commit -m "feat(core): add AgentRunner with PromptedOutput fallback"
```

---

## Task 3: Migrate model resolution to core

Replace the four `_resolve_model` functions with imports from `core.models`.

**Files:**
- Modify: `src/andamentum/epistemic/runner.py` — replace `_resolve_model` with `from andamentum.core.models import resolve_model`
- Modify: `src/andamentum/deep_research/runner.py` — same
- Modify: `src/andamentum/epistemic/cli.py` — replace CLI model resolution
- Modify: `src/andamentum/deep_research/cli.py` — same

**Critical: keep the function signature identical.** The callers pass a string and get back either a string or a model object. `resolve_model()` in core has the same contract.

- [ ] **Step 1: Replace epistemic/runner.py `_resolve_model`**

Delete the local `_resolve_model` function and its imports (bedrock model map, region prefix map, etc.). Replace with:

```python
from andamentum.core.models import resolve_model as _resolve_model
```

All callers of `_resolve_model(model)` continue to work unchanged.

- [ ] **Step 2: Replace deep_research/runner.py `_resolve_model`**

Same pattern. Delete the local function, import from core. Deep research gains bedrock support for free.

- [ ] **Step 3: Replace CLI model resolution in both CLIs**

In `epistemic/cli.py` and `deep_research/cli.py`, replace the local `_resolve_model(args)` functions with:

```python
from andamentum.core.models import resolve_model_from_args
```

- [ ] **Step 4: Run ALL tests for both modules**

```bash
uv run pytest src/andamentum/epistemic/tests/ -v
uv run pytest src/andamentum/deep_research/tests/ -v
uv run pyright
uv run ruff check
git add -A && git commit -m "refactor: migrate model resolution to core module"
```

---

## Task 4: Migrate epistemic runner to core AgentRunner

Replace `DefaultAgentRunner` in epistemic with a thin wrapper around `core.agents.AgentRunner`.

**CRITICAL INTERFACE PRESERVATION:** The epistemic runner's `run(agent_name, **kwargs)` takes an agent NAME (string) and looks it up in AGENT_REGISTRY. The core runner's `run(defn, **kwargs)` takes an AgentDefinition directly. The epistemic wrapper must preserve the name-based interface.

**Files:**
- Modify: `src/andamentum/epistemic/runner.py`
- Modify: `src/andamentum/epistemic/agents/__init__.py` — AgentDefinition now imported from core

- [ ] **Step 1: Update epistemic AgentDefinition to re-export from core**

In `epistemic/agents/__init__.py`, replace the local AgentDefinition with:

```python
from andamentum.core.agents import AgentDefinition
```

Keep `AGENT_REGISTRY`, `register_agent`, `get_agent` — these are module-specific. Only the dataclass is shared.

Verify all agent registration modules still work (they import AgentDefinition from `__init__`).

- [ ] **Step 2: Rewrite DefaultAgentRunner as a thin wrapper**

```python
class DefaultAgentRunner:
    """Epistemic agent runner — wraps core.AgentRunner with name-based lookup."""

    def __init__(self, *, model: str):
        from andamentum.core.agents import AgentRunner
        self._runner = AgentRunner(model=model)
        self.model = model

    async def run(self, agent_name: str, **kwargs: Any) -> Any:
        from .agents import get_agent
        defn = get_agent(agent_name)
        return await self._runner.run(defn, **kwargs)
```

This preserves the exact `run(agent_name, **kwargs)` interface that all 33 `self.run_agent()` callsites in operations use.

- [ ] **Step 3: Run ALL epistemic tests**

```bash
uv run pytest src/andamentum/epistemic/tests/ -v
uv run pyright
git add -A && git commit -m "refactor(epistemic): use core AgentRunner with PromptedOutput fallback"
```

---

## Task 5: Migrate deep_research runner and document_store agents to core

**Files:**
- Modify: `src/andamentum/deep_research/agents/__init__.py` — re-export AgentDefinition from core
- Modify: `src/andamentum/deep_research/runner.py` — wrap core AgentRunner
- Modify: `src/andamentum/deep_research/nodes.py` — use core for agent building
- Modify: `src/andamentum/document_store/extraction.py` — use `run_agent_with_fallback`
- Modify: `src/andamentum/document_store/query_planner.py` — use `run_agent_with_fallback`

**Deep research gains:** PromptedOutput fallback and bedrock support.

**Document store gains:** Model resolution (ollama/bedrock), PromptedOutput fallback with validators preserved, consistent error handling.

- [ ] **Step 1: Migrate deep_research AgentDefinition**

Same as epistemic: import from core, keep local registry. Remove the `has_tools` field (never used).

- [ ] **Step 2: Migrate deep_research runner and nodes**

`DefaultResearchRunner` becomes a thin wrapper like the epistemic one.

`_build_agent` in `nodes.py` changes to use core's `resolve_model` and Agent construction pattern. The key change: add PromptedOutput fallback to the graph node path.

- [ ] **Step 3: Migrate document_store extraction**

Replace the inline Agent creation + PromptedOutput fallback in `extraction.py` with `run_agent_with_fallback()` from core. The validators need to be passed as callables.

Current pattern (repeated twice):
```python
agent = Agent(model, output_type=DocFields, instructions=prompt, retries=3)
@agent.output_validator
def check(ctx, output): ...
try:
    result = await agent.run(text)
except UnexpectedModelBehavior:
    fallback = Agent(model, output_type=PromptedOutput(DocFields), ...)
    result = await fallback.run(text)
```

New pattern:
```python
from andamentum.core.agents import run_agent_with_fallback

result = await run_agent_with_fallback(
    model,
    instructions=prompt,
    output_type=DocFields,
    user_message=text,
    validators=[check_title_not_empty],
)
```

- [ ] **Step 4: Migrate document_store query_planner**

Same pattern but simpler (no PromptedOutput fallback currently — it gains one for free).

- [ ] **Step 5: Run ALL tests for all three modules**

```bash
uv run pytest src/andamentum/deep_research/tests/ -v
uv run pytest src/andamentum/document_store/tests/ -v
uv run pytest src/andamentum/epistemic/tests/ -v
uv run pyright
git add -A && git commit -m "refactor: migrate deep_research and document_store agents to core"
```

---

## Task 6: Migrate embeddings to core

This is the highest-risk task. Two independent implementations with 15+ callsites.

**Strategy:** Create a unified embedding function in core that has:
- Concurrent requests from EmbeddingService (asyncio.gather + semaphore)
- Better error messages from epistemic's embed_texts (ConnectError → "start Ollama")
- Input truncation from epistemic (2000 char limit)
- OLLAMA_BASE_URL env var support (currently missing from both)
- The chunking helper from epistemic's embed_documents

Then update both modules to import from core.

**Files:**
- Create: `src/andamentum/core/embeddings.py`
- Modify: `src/andamentum/epistemic/embeddings.py` — re-export from core
- Modify: `src/andamentum/document_store/embeddings.py` — re-export from core
- Modify: 15+ callsites (no interface changes needed if we keep function signatures)
- Test: `src/andamentum/core/tests/test_embeddings.py`

- [ ] **Step 1: Create core/embeddings.py**

Unified implementation combining both:

```python
"""Embedding client for Ollama-compatible APIs.

Provides both function-based (embed_texts) and class-based (EmbeddingService)
interfaces for generating text embeddings via Ollama's /api/embeddings endpoint.

Features from both original implementations:
- Concurrent requests with semaphore (from document_store)
- Descriptive error messages (from epistemic)
- Input truncation to 2000 chars (from epistemic)
- OLLAMA_BASE_URL env var support (new — neither had this)
- Chunking for long documents (from epistemic)
"""
```

The key decision: keep BOTH interfaces (function-based `embed_texts` for epistemic, class-based `EmbeddingService` for document_store). The class wraps the function. This minimizes callsite changes.

- [ ] **Step 2: Update epistemic/embeddings.py to re-export from core**

```python
"""Embedding utilities for the epistemic system.

Re-exports from andamentum.core.embeddings. This module exists for
backward compatibility — new code should import from core directly.
"""
from andamentum.core.embeddings import embed_texts, embed_documents, cosine_similarity
```

All 10 epistemic callsites continue to work with `from ..embeddings import embed_texts`.

- [ ] **Step 3: Update document_store/embeddings.py to re-export from core**

```python
"""Embedding utilities for the document store.

Re-exports from andamentum.core.embeddings. This module exists for
backward compatibility — new code should import from core directly.
"""
from andamentum.core.embeddings import EmbeddingService, cosine_similarity
```

All 6 document_store callsites continue to work with `from .embeddings import EmbeddingService`.

- [ ] **Step 4: Run ALL tests**

```bash
uv run pytest src/andamentum/core/tests/ -v
uv run pytest src/andamentum/epistemic/tests/ -v
uv run pytest src/andamentum/document_store/tests/ -v
uv run pytest src/andamentum/deep_research/tests/ -v
uv run pyright
git add -A && git commit -m "refactor: migrate embeddings to core module"
```

---

## Task 7: Cleanup — remove dead code, update CLAUDE.md

- [ ] **Step 1: Remove the old implementations**

After verifying all tests pass with core imports:
- Remove the bodies of `epistemic/runner.py::_resolve_model` (now just a re-export)
- Remove the bedrock model map from epistemic/runner.py (now in core)
- Remove the duplicate `_resolve_model` from deep_research/runner.py
- Clean up any remaining direct pydantic-ai imports that are now handled by core

- [ ] **Step 2: Update CLAUDE.md**

Add architectural note about core module:

```
**Core module** (`andamentum.core`) — shared infrastructure for model resolution,
agent execution, and embeddings. All three sub-modules import from core instead
of maintaining independent implementations. When adding LLM-calling code to any
module, use `core.agents.AgentRunner` or `core.agents.run_agent_with_fallback`
instead of creating pydantic-ai Agents directly.
```

- [ ] **Step 3: Final verification**

```bash
uv run pytest
uv run pyright
uv run ruff check
```

---

## Edge Cases and Failure Modes

### Embedding lifecycle mismatch
`EmbeddingService` has a long-lived httpx client that callers close in `finally` blocks. If we change the client lifecycle (e.g., make it ephemeral), document_store ingestion performance could degrade (connection reuse lost). **Mitigation:** Keep the class-based interface with its existing lifecycle. The core `EmbeddingService` preserves `close()` semantics.

### Validator loss on PromptedOutput fallback
Currently document_store's PromptedOutput fallback agents don't carry validators from the primary agent. The core `run_agent_with_fallback` fixes this by accepting validators as a parameter and applying them to both the primary and fallback agents. This is a behavior IMPROVEMENT but could surface validation errors that were previously silently ignored. **Mitigation:** Test extraction with small models that trigger the fallback.

### system_prompt vs instructions
Epistemic runner uses `system_prompt=defn.prompt` (tuple form), deep_research uses `instructions=defn.prompt` (string form). pydantic-ai treats them differently: `instructions` is a simple string prepended to the conversation; `system_prompt` accepts callables. **Mitigation:** Core standardizes on `instructions=` (simpler, sufficient for all current agents). Verify epistemic agents still produce correct output after the switch.

### Import order and circular imports
Core imports pydantic-ai lazily (inside functions) to stay off the critical import path, matching the existing pattern. Sub-modules import core at function call time, not at module level. No circular dependency risk because core never imports from sub-modules.

### Bedrock session caching
The epistemic runner creates a new boto3 Session per `_resolve_model` call. Core preserves this behavior. For high-throughput use, a session cache could be added later.

### OLLAMA_BASE_URL inconsistency
Runners use `http://localhost:11434/v1` (OpenAI compat endpoint for pydantic-ai). Embedding clients use `http://localhost:11434` (native Ollama endpoint for /api/embeddings). Core preserves BOTH defaults — they're correct for their respective use cases. The env var `OLLAMA_BASE_URL` sets the base, and each client appends the appropriate path.
