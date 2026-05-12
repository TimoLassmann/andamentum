"""Dispatch-quality benchmark — Phase 3 of the description-driven-dispatch refactor.

Public API:

- ``harness.run_tier_one_for_provider`` — Tier 1 for one provider.
- ``harness.run_tier_one_all_providers`` — Tier 1 across the catalogue.
- ``harness.run_and_report`` — convenience runner + report writer.
- ``run.main`` — CLI entry point.
- ``fixtures.all_examples`` — re-export of per-provider example pairs.

See ``README.md`` in this directory for invocation details and the
acceptance criteria from the PRD.
"""

from __future__ import annotations
