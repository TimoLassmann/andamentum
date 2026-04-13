"""Adversarial query generator for the epistemic system.

Generates adversarial search queries designed to find criticism,
counterarguments, and disconfirming evidence for claims.

Domain detection and key-term extraction should be performed by focused
agents. This module provides structural query templates and simple
template substitution.

Architecture: Layer 1 (framework-agnostic, no model calls)
"""

from typing import List, Optional, Dict


# --- Query Templates ---

# General adversarial query templates (work for any domain)
# Use {topic} for searchable queries
GENERAL_TEMPLATES = [
    '"{topic}" criticism',
    '"{topic}" problems limitations',
    '"{topic}" wrong false',
    '"{topic}" debunked refuted',
    '"{topic}" controversy dispute',
    '"{topic}" failed replication',
    "arguments against {topic}",
    "evidence against {topic}",
    '"{topic}" does not work ineffective',
    '"{topic}" criticism meta-analysis review',
]

# Domain-specific templates (from spec Part 3.2)
DOMAIN_TEMPLATES: Dict[str, List[str]] = {
    "biomedical": [
        "failed clinical trial {topic}",
        "{topic} side effects adverse events",
        "{topic} methodological flaws",
        "{topic} did not replicate",
        "{topic} retracted study",
        "{topic} safety concerns",
    ],
    "statistical": [
        "p-hacking {topic}",
        "{topic} statistical errors",
        "publication bias {topic}",
        "{topic} effect size inflated",
        "{topic} multiple testing problem",
        "{topic} confounding variables",
    ],
    "computational": [
        "{topic} failure cases",
        "{topic} limitations",
        "{topic} benchmark gaming",
        "{topic} did not generalize",
        "{topic} overfitting",
        "{topic} reproducibility issues",
    ],
    "theoretical": [
        "arguments against {topic}",
        "{topic} counterexamples",
        "problems with {topic}",
        "alternatives to {topic}",
        "{topic} criticism philosophy",
        "{topic} refuted",
    ],
    "social_science": [
        "{topic} replication failure",
        "{topic} WEIRD samples",
        "{topic} demand characteristics",
        "{topic} effect size small",
        "{topic} generalization problem",
    ],
}

# Source-specific adversarial queries (from spec Part 3.3)
SOURCE_SPECIFIC_TEMPLATES = [
    "{author} critics",
    "alternative to {approach}",
    "{method} criticism statistical",
    "{field} replication crisis",
    "{topic} controversy investigation",
]


def detect_domain(claim_text: str) -> Optional[str]:
    """Detect the domain of a claim.

    Previously used keyword matching. Now returns None — domain detection
    should be performed by a focused agent and passed as domain_hint to
    generate_adversarial_queries().

    Args:
        claim_text: The claim text (unused).

    Returns:
        Always None. Use agent-based domain detection instead.
    """
    return None


def generate_adversarial_queries(
    claim_text: str,
    claim_domain: Optional[str] = None,
    max_queries: int = 8,
) -> List[str]:
    """Generate adversarial search queries for a claim.

    Uses the claim text directly as the topic for template substitution.
    Domain-specific templates are included when a domain hint is provided.

    Args:
        claim_text: The claim to generate adversarial queries for.
        claim_domain: Optional domain hint ('biomedical', 'computational', etc.)
        max_queries: Maximum number of queries to generate.

    Returns:
        List of adversarial search queries.
    """
    # Use a trimmed version of the claim as the topic
    topic = claim_text.strip()[:80]
    terms = {"topic": topic}

    queries: List[str] = []
    seen: set[str] = set()

    def add_query(q: str) -> bool:
        """Add query if not duplicate and under limit."""
        if len(queries) >= max_queries:
            return False
        q_lower = q.lower()
        if q_lower in seen:
            return False
        queries.append(q)
        seen.add(q_lower)
        return True

    # Get domain-specific templates (or empty list)
    domain_templates = DOMAIN_TEMPLATES.get(claim_domain, []) if claim_domain else []

    # Interleave general and domain-specific queries for variety
    general_count = min(5, max_queries - len(domain_templates) // 2)

    # Add first batch of general queries
    for template in GENERAL_TEMPLATES[:general_count]:
        query = template.format(**terms)
        add_query(query)

    # Add domain-specific queries
    for template in domain_templates:
        query = template.format(**terms)
        add_query(query)

    # Fill remaining slots with more general queries
    for template in GENERAL_TEMPLATES[general_count:]:
        query = template.format(**terms)
        add_query(query)

    return queries[:max_queries]


def get_domain_specific_templates(domain: str) -> List[str]:
    """Get adversarial query templates for a specific domain.

    Args:
        domain: One of 'biomedical', 'statistical', 'computational', 'theoretical', 'social_science'.

    Returns:
        List of template strings with {topic} placeholder.
    """
    return DOMAIN_TEMPLATES.get(domain, GENERAL_TEMPLATES)


def generate_steelmanned_query(criticism_summary: str) -> str:
    """Generate a query to find the strongest version of a criticism.

    From spec Part 7.3: Present the STRONGEST version of criticism.

    Args:
        criticism_summary: Brief summary of the criticism.

    Returns:
        A search query to find the strongest articulation of this criticism.
    """
    return f'"{criticism_summary}" strongest argument evidence study'
