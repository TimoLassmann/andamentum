"""Pin the public surface of andamentum.chunker."""

import andamentum.chunker as chunker


def test_public_all():
    expected = {
        # Functions / callables
        "extract_units",
        "make_ollama_embedder",
        "make_runner_executor",
        # Data types
        "ChunkingResult",
        "EmbeddingFn",
        "ExecutorFn",
        "Gap",
        "JudgeVerdict",
        "Unit",
    }
    assert set(chunker.__all__) == expected


def test_public_imports_are_resolvable():
    for name in chunker.__all__:
        assert hasattr(chunker, name), f"chunker.{name} is missing"
