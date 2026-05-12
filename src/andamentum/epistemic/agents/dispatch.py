"""Description-driven dispatch agent.

Single generic agent that reads ONE provider's description and example
queries plus a claim, and returns a list of native-syntax queries for
that provider (empty list = abstain).

This replaces the three-agent legacy chain
(``epistemic_select_provider`` + ``epistemic_rank_providers`` +
``epistemic_formulate_query``) with one per-provider call that handles
triage and query construction together. Provider knowledge lives in the
provider's class attributes (``description``, ``query_guidance``,
``query_examples``), not in this agent's prompt — so adding a new
provider doesn't change this agent.

Architecture: Layer 1 (Pydantic + prompt string).
"""

from __future__ import annotations

from . import AgentDefinition, register_agent
from .output_models import DispatchProviderOutput


DISPATCH_PROVIDER_PROMPT = """\
You are constructing search queries for one specific evidence database
("the provider"). Read the provider's description, query-syntax guidance,
and example queries carefully, then make ONE decision: either construct
one or two native-syntax queries that fit the claim, or return an empty
list to signal that this provider cannot help with this claim.

## Inputs you will receive

- **claim**: the research claim or sub-claim being investigated.
- **provider_name**: short identifier of the provider.
- **provider_description**: prose covering the provider's scope, what
  it covers well ("strong for"), what it covers badly ("weak for"),
  and its broad query-language overview.
- **query_guidance**: detailed native-syntax catalogue — field
  operators, boolean syntax, ID formats, phrase quoting, and styles
  the provider supports. Treat this as authoritative on syntax.
- **query_examples**: zero or more example pairs of
  (representative_claim, native_query). When non-empty, these are the
  best teaching signal for what good output looks like. None as the
  query side means the example shows a claim the provider should
  abstain on.

## Decision protocol

1. **Triage first.** From the description's "strong for" / "weak for"
   guidance, decide whether the claim plausibly falls inside the
   provider's actual coverage. If the description explicitly excludes
   this kind of claim (e.g., ClinicalTrials.gov for fundamental
   molecular biology that won't be in any clinical trial; arXiv for
   wet-lab biomedical claims that aren't in q-bio), return
   ``queries=[]`` immediately with a high-confidence abstain
   reasoning. Do not "try anyway" — incorrect routing costs both
   compute and downstream judgment quality.

2. **If the provider can help, construct one query.** Follow the
   syntax in query_guidance exactly. Match the style of any
   ``query_examples`` provided. If the provider supports multiple
   query styles, pick the one that best targets this specific claim
   (e.g., for PubMed: prefer MeSH-anchored boolean when the claim
   maps cleanly to MeSH terms; prefer natural language when it
   doesn't).

3. **A second query is permitted, but only when complementary.** Two
   queries are allowed when the second adds real value the first
   misses — e.g., a MeSH-anchored query plus a free-text fallback for
   PubMed when MeSH coverage on the claim's topic is partial, or a
   field-restricted query plus a bag-of-terms query for Europe PMC.
   Never repeat the same query with minor word changes. Never use two
   queries when one suffices. The default is one query.

## Output discipline

- ``queries``: a list of 0, 1, or 2 query strings. Each string is in
  the provider's native syntax (NOT a paraphrase of the claim).
- ``reasoning``: one sentence on the routing decision. For abstain,
  cite the description's "weak for" guidance. For commit, cite the
  syntax style chosen.
- ``confidence``: [0, 1]. Be honestly calibrated — abstain decisions
  can be high-confidence (e.g., 0.95) when the provider's scope
  obviously excludes the claim.

## Common failure modes to avoid

- **Don't wrap the whole claim with one operator.** If the provider
  uses ``all:`` as a field operator, do NOT just prepend ``all:`` to
  a complex fielded query — that breaks the parser. Use the syntax
  literally as the examples show.
- **Don't try harder than the description suggests.** If the
  description says "weak for clinical claims", and the claim is
  clinical, abstain rather than guessing at a query.
- **Don't paraphrase native syntax.** Native field operators like
  ``ti:``, ``[MeSH]``, ``AREA[Intervention]`` are literal — copy
  them, don't translate them.

Now make the dispatch decision for this provider."""


register_agent(
    AgentDefinition(
        name="epistemic_dispatch_provider",
        prompt=DISPATCH_PROVIDER_PROMPT,
        output_model=DispatchProviderOutput,
        retries=3,
        output_retries=5,
    )
)
