"""Single-page fetch summariser agent.

Used by :func:`andamentum.deep_research.run_fetch` to summarise one URL
without a research-question framing. Output is a :class:`FetchSummary`
(no relevance score — the page IS the target, not one of many).
"""

from ..models import FetchSummary
from . import AgentDefinition, register_agent


GENERAL_PAGE_SUMMARIZER_PROMPT = """\
You summarise a web page faithfully and concisely.

Produce a structured summary capturing what the page actually says — its
main argument, methods (if any), findings (if any), and conclusions.

Guidelines:
  • summary — about 200 words of prose, faithful to the source. State
    what the page is about, what it claims, and what evidence it offers
    (if any). Do not add interpretation that is not on the page.
  • key_points — 3 to 5 short bullet-style points. Each captures one
    substantive thing the page says.

Tone: factual and neutral. No promotional language, no editorial
framing. If the page is mostly navigation, ads, or boilerplate, say so
honestly in the summary instead of fabricating substance."""


register_agent(
    AgentDefinition(
        name="general_page_summarizer",
        prompt=GENERAL_PAGE_SUMMARIZER_PROMPT,
        output_model=FetchSummary,
        retries=2,
        output_retries=3,
    )
)
