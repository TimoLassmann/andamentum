"""Data models for novelty checking."""

from dataclasses import dataclass, field
from typing import List
from enum import Enum


class Relevance(str, Enum):
    """How closely related prior work is to the claim."""

    DIRECT = "direct"  # Same claim, established knowledge
    PARTIAL = "partial"  # Related work, partial overlap
    TANGENTIAL = "tangential"  # Loosely related, different context


@dataclass
class SimilarWork:
    """A piece of prior work related to the claim."""

    title: str
    url: str
    relevance: Relevance
    summary: str  # How it relates to the claim


@dataclass
class NoveltyReport:
    """Result of a novelty check."""

    claim: str
    # ``None`` means undetermined — the search could not be completed, so the
    # report makes no novelty claim either way. Consumers must distinguish
    # None from True/False rather than coercing (e.g. ``bool(None)`` would
    # silently read as "not novel").
    is_novel: bool | None
    confidence: float  # 0.0 - 1.0
    assessment: str  # Human-readable explanation
    similar_work: List[SimilarWork] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)  # All URLs consulted
    search_queries_used: List[str] = field(default_factory=list)
