"""Andamentum Core — shared infrastructure for all sub-modules.

Provides:
- Model resolution (ollama, bedrock, passthrough)
- Agent execution with PromptedOutput fallback
- Embedding client (Ollama HTTP, cosine similarity, chunked-doc embeds)

Sub-modules (epistemic, deep_research, document_store, chunker, ...)
import from here instead of maintaining independent implementations.
"""

from .agents import (
    AgentDefinition,
    AgentRunner,
    build_pydantic_ai_agent,
    run_agent_with_fallback,
)
from .embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_EMBED_CHARS,
    DEFAULT_OLLAMA_URL,
    DEFAULT_OVERLAP_CHARS,
    EmbeddingFn,
    chunk_text,
    cosine_similarity,
    discover_input_budget_chars,
    embed_documents,
    embed_texts,
    infer_input_budget_chars,
    make_ollama_embedder,
)
from .models import resolve_model, resolve_model_from_args

__all__ = [
    # Agents
    "AgentDefinition",
    "AgentRunner",
    "build_pydantic_ai_agent",
    "run_agent_with_fallback",
    # Models
    "resolve_model",
    "resolve_model_from_args",
    # Embeddings
    "EmbeddingFn",
    "chunk_text",
    "cosine_similarity",
    "discover_input_budget_chars",
    "embed_documents",
    "embed_texts",
    "infer_input_budget_chars",
    "make_ollama_embedder",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_MAX_EMBED_CHARS",
    "DEFAULT_OLLAMA_URL",
    "DEFAULT_OVERLAP_CHARS",
]
