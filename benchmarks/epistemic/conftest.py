"""Shared configuration for epistemic agent evals.

Model selection:
    EPISTEMIC_EVAL_MODEL env var, e.g.:
        export EPISTEMIC_EVAL_MODEL=bedrock:claude-sonnet-4-5
        export EPISTEMIC_EVAL_MODEL=openai:gpt-4o
        export EPISTEMIC_EVAL_MODEL=ollama:gpt-oss:20b

    Default: bedrock:us.anthropic.claude-sonnet-4-5-v2-0:0
"""

import os
import asyncio
from typing import Any

DEFAULT_MODEL = "bedrock:us.anthropic.claude-sonnet-4-5-v2-0:0"


def get_eval_model() -> str:
    """Get the model to use for evals from env or default."""
    return os.environ.get("EPISTEMIC_EVAL_MODEL", DEFAULT_MODEL)


async def run_agent(agent_name: str, **kwargs: Any) -> Any:
    """Run a single epistemic agent and return its output.

    This is the task function that pydantic-evals evaluates.
    Each call creates a fresh runner to avoid cross-contamination.
    """
    from epistemic.runner import DefaultAgentRunner

    model = get_eval_model()
    runner = DefaultAgentRunner(model=model)
    result = await runner.run(agent_name, **kwargs)
    return result


def run_agent_sync(agent_name: str, **kwargs: Any) -> Any:
    """Sync wrapper for run_agent."""
    return asyncio.run(run_agent(agent_name, **kwargs))
