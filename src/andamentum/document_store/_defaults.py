"""Package-level defaults for document-store.

Replaces the mosaic-defaults dependency with inlined constants.
All values are overridable via environment variables.
"""

import os

DEFAULT_EMBEDDING_MODEL = os.environ.get("MOSAIC_EMBEDDING_MODEL", "embeddinggemma:latest")
DEFAULT_LLM_MODEL = os.environ.get("MOSAIC_LLM_MODEL", "ollama:qwen3.5:27b")
