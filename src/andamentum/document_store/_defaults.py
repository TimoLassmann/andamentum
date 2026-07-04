"""Package-level defaults for document-store.

Model names are no longer inlined here. Callers must pass model
as a required keyword argument — no env-var fallbacks exist.
"""

#: Embedding dimensionality for the vec0 virtual tables (doc + chunk).
#: Sized for ``embeddinggemma:latest``. The vec0 DDL declares fixed-width
#: ``FLOAT[EMBEDDING_DIM]`` columns, so this is the single source of truth —
#: changing the embedding model to a different width means changing this value
#: (and rebuilding the vector tables).
EMBEDDING_DIM = 768
