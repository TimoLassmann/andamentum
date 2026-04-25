"""Andamentum Core — shared infrastructure for all sub-modules.

Provides:
- Model resolution (ollama, bedrock, passthrough)
- Agent execution with PromptedOutput fallback
- Embedding client

Sub-modules (epistemic, deep_research, document_store) import from here
instead of maintaining independent implementations.
"""

# === Functions you can wrap as agent tools ===
# Note: AgentRunner is a class — wrap its `.run()` method as a tool.
from .agents import AgentRunner, run_agent_with_fallback
from .models import resolve_model, resolve_model_from_args

# === Result/data types (config + return values; not tools themselves) ===
from .agents import AgentDefinition

__all__ = [
    # Functions / callables
    "AgentRunner",
    "resolve_model",
    "resolve_model_from_args",
    "run_agent_with_fallback",
    # Data types
    "AgentDefinition",
]
