"""arXiv Evidence Provider.

Searches the arXiv API for preprints in physics, mathematics, computer
science, quantitative biology, quantitative finance, statistics, electrical
engineering, and economics.

API docs: https://info.arxiv.org/help/api/user-manual.html
No authentication required. Rate limit: 1 request per 3 seconds.

Architecture: Layer 1 (standalone package)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..preflight import CheckResult

from ..operations import GatheredEvidence

logger = logging.getLogger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"

# Atom and arXiv XML namespaces
_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"


class ArXivProvider:
    """Evidence provider using the arXiv API."""

    description = (
        "Preprint server for physics, mathematics, computer science, "
        "quantitative biology, quantitative finance, statistics, electrical "
        "engineering, and economics. Use for any non-biomedical scientific "
        "claim, especially in physics, AI/ML, mathematics, or computer "
        "science. Also covers quantitative biology preprints not on bioRxiv. "
        "Example queries: 'transformer attention mechanisms', 'quantum error "
        "correction surface codes', 'reinforcement learning from human feedback'."
    )

    query_guidance = (
        "The query is wrapped as `all:{query}` and sent to the arXiv API's "
        "`search_query` parameter, but a query starting with a field prefix "
        "overrides this. Field prefixes: ti: (title), abs: (abstract), au: "
        "(author), cat: (subject category, e.g., cs.LG, stat.ML, q-bio.PE, "
        "math.ST, hep-ph, cond-mat), all: (all fields), jr: (journal-ref), "
        'id: (arXiv ID). Boolean: AND, OR, ANDNOT. Phrase quoting ("...").\n'
        "\n"
        "Query styles that all work:\n"
        "- Plain bag of terms (auto-prefixed `all:`): transformer attention "
        "mechanism\n"
        '- Title-restricted: ti:"reinforcement learning"\n'
        "- Category plus topic: cat:cs.LG AND ti:transformer\n"
        "- Author plus topic: au:Hinton AND backpropagation\n"
        "- Title plus abstract: ti:diffusion AND abs:image\n"
        "- Multi-category: (cat:cs.LG OR cat:stat.ML) AND ti:scaling\n"
        "- arXiv ID lookup: id:2305.12345\n"
        "\n"
        "Coverage: physics, math, CS, quantitative biology, quantitative "
        "finance, statistics, EE, economics. No clinical or wet-lab biomedical "
        "literature here. Use cat: prefixes to scope to the right subdomain."
    )

    query_examples: list[tuple[str, str | None]] = [
        (
            "transformer attention mechanism scaling with sequence length",
            "transformer attention mechanism scaling",
        ),
        (
            "reinforcement learning from human feedback methodology",
            'ti:"reinforcement learning" AND ti:"human feedback"',
        ),
        (
            "machine-learning research on diffusion models for image generation",
            "cat:cs.LG AND ti:diffusion AND abs:image",
        ),
        (
            "computational methods for predicting protein-protein interactions",
            '(cat:q-bio.QM OR cat:q-bio.BM) AND ti:"protein-protein interaction"',
        ),
        (
            "quantum error correction with surface codes",
            '(cat:quant-ph OR cat:cs.IT) AND ti:"quantum error correction" AND ti:"surface code"',
        ),
        (
            "Hinton's published work on deep learning architectures",
            "au:Hinton AND (abs:deep AND abs:learning)",
        ),
        (
            "what does arXiv paper 2305.12345 propose",
            "id:2305.12345",
        ),
        # Out-of-domain — wet-lab biology
        (
            "in vivo CRISPR knockout efficiency in mouse liver",
            None,
        ),
        # Out-of-domain — clinical pharmacology
        (
            "atorvastatin's effect on LDL cholesterol in heart failure patients",
            None,
        ),
        # Out-of-domain — clinical trial
        (
            "phase III trial of pembrolizumab in melanoma",
            None,
        ),
    ]
    output_kind = "assertion_evidence"
    independence_group = "preprint_archive"
    provider_contract_version = 1

    def __init__(self, max_results: int = 10):
        self.max_results = max_results

    async def check_health(self) -> "CheckResult":
        """Test arXiv API reachability."""
        import time
        import xml.etree.ElementTree as ET

        import httpx

        from ..preflight import CheckResult

        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    ARXIV_API,
                    params={"search_query": "all:test", "max_results": 1},
                )
                elapsed = (time.monotonic() - t0) * 1000
                if response.status_code == 200:
                    root = ET.fromstring(response.text)
                    entries = root.findall(f"{{{_ATOM_NS}}}entry")
                    if entries:
                        return CheckResult(
                            name="ArXivProvider",
                            status="pass",
                            message=f"API reachable ({elapsed:.0f}ms)",
                            elapsed_ms=elapsed,
                        )
                    return CheckResult(
                        name="ArXivProvider",
                        status="fail",
                        message="No entries in response",
                        elapsed_ms=elapsed,
                    )
                return CheckResult(
                    name="ArXivProvider",
                    status="fail",
                    message=f"HTTP {response.status_code}",
                    elapsed_ms=elapsed,
                )
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            return CheckResult(
                name="ArXivProvider",
                status="fail",
                message=str(e),
                elapsed_ms=elapsed,
            )

    async def gather(self, query: str) -> list[GatheredEvidence]:
        """Search arXiv for preprints matching the query."""
        import xml.etree.ElementTree as ET

        import httpx

        gathered: list[GatheredEvidence] = []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    ARXIV_API,
                    params={
                        "search_query": f"all:{query}",
                        "start": 0,
                        "max_results": self.max_results,
                        "sortBy": "relevance",
                        "sortOrder": "descending",
                    },
                )
                if response.status_code != 200:
                    return []

                root = ET.fromstring(response.text)
                entries = root.findall(f"{{{_ATOM_NS}}}entry")

                for entry in entries:
                    title = (
                        entry.findtext(f"{{{_ATOM_NS}}}title", "")
                        .strip()
                        .replace("\n", " ")
                    )
                    summary = entry.findtext(f"{{{_ATOM_NS}}}summary", "").strip()
                    published = entry.findtext(f"{{{_ATOM_NS}}}published", "")
                    updated = entry.findtext(f"{{{_ATOM_NS}}}updated", "")

                    # Parse arXiv ID from <id> URL
                    id_url = entry.findtext(f"{{{_ATOM_NS}}}id", "")
                    arxiv_id = (
                        id_url.split("/abs/")[-1] if "/abs/" in id_url else id_url
                    )

                    # Authors
                    authors = [
                        a.findtext(f"{{{_ATOM_NS}}}name", "")
                        for a in entry.findall(f"{{{_ATOM_NS}}}author")
                    ]

                    # Categories
                    categories = [
                        c.get("term", "")
                        for c in entry.findall(f"{{{_ATOM_NS}}}category")
                    ]

                    # arXiv-specific fields (different namespace)
                    primary_cat_el = entry.find(f"{{{_ARXIV_NS}}}primary_category")
                    primary_category = (
                        primary_cat_el.get("term", "")
                        if primary_cat_el is not None
                        else ""
                    )
                    doi = entry.findtext(f"{{{_ARXIV_NS}}}doi", "")
                    journal_ref = entry.findtext(f"{{{_ARXIV_NS}}}journal_ref", "")
                    comment = entry.findtext(f"{{{_ARXIV_NS}}}comment", "")

                    if not title:
                        continue

                    # Build content
                    content_parts = [title]
                    if authors:
                        if len(authors) > 5:
                            content_parts.append(
                                f"Authors: {', '.join(authors[:5])} (et al, {len(authors)} authors total)"
                            )
                        else:
                            content_parts.append(f"Authors: {', '.join(authors)}")
                    if summary:
                        content_parts.append(f"\n{summary}")

                    # Build identifiers
                    identifiers: dict[str, str] = {"arxiv_id": arxiv_id}
                    if doi:
                        identifiers["doi"] = doi

                    # Build structured_data
                    structured_data: dict[str, str | list[str] | None] = {
                        "title": title,
                        "authors": authors,
                        "categories": categories,
                        "primary_category": primary_category,
                        "published": published,
                        "updated": updated,
                        "journal_ref": journal_ref or None,
                        "doi": doi or None,
                        "comment": comment or None,
                    }

                    # Build quality_metadata
                    quality_metadata: dict[str, str | bool] = {
                        "primary_category": primary_category,
                        "has_journal_ref": bool(journal_ref),
                    }

                    # Build limitations
                    limitations: list[str] = (
                        [] if journal_ref else ["Preprint — not peer-reviewed"]
                    )

                    gathered.append(
                        GatheredEvidence(
                            content="\n".join(content_parts),
                            source_ref=f"arXiv:{arxiv_id}",
                            source_type="arxiv",
                            evidence_kind="preprint",
                            identifiers=identifiers,
                            structured_data=structured_data,
                            quality_score=None,
                            quality_metadata=quality_metadata,
                            limitations=limitations,
                        )
                    )

        except Exception as e:
            logger.warning(f"ArXivProvider query failed for '{query}': {e}")

        return gathered
