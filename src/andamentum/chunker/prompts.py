"""Size constants for the chunker.

The bulky LLM prompts that lived here have moved to ``judge.py`` (the only
remaining LLM call site). Only the target size band stays here, exposed
as importable constants so downstream consumers can match.
"""

from __future__ import annotations

# Size band that downstream consumers (whetstone critic agents, deep_research
# summariser) are designed around. Tunable per-call via extract_units kwargs.
TARGET_MIN_CHARS = 2_000
TARGET_MAX_CHARS = 10_000
