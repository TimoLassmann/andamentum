"""Epistemic pipeline as a pydantic-graph DAG.

Replaces the pattern-based scheduler with explicit node dependencies.
Every scheduling decision is a typed return value, not a pattern match.
The graph makes the workflow visible, testable, and provably correct.

Usage::

    from andamentum.epistemic.graph import run_epistemic_graph

    result = await run_epistemic_graph(
        question="Does metformin reduce cancer risk?",
        database_name="my_research",
        model="openai:gpt-4o-mini",
    )
"""
