"""Andamentum Core — shared infrastructure for all sub-modules.

Provides:
- Model resolution (ollama, bedrock, passthrough)
- Agent execution with PromptedOutput fallback
- Embedding client

Sub-modules (epistemic, deep_research, document_store) import from here
instead of maintaining independent implementations.
"""

from .models import resolve_model, resolve_model_from_args

__all__ = [
    "resolve_model",
    "resolve_model_from_args",
]
