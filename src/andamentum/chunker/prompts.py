"""Prompt template + domain hints for the chunker."""

from __future__ import annotations

SYSTEM_PROMPT = """You identify the boundaries of self-contained content units in text.

CRITICAL RULES:
- You do NOT rewrite, summarise, or extract any text. You only point at where one unit starts and ends.
- start_anchor and end_anchor must be copied VERBATIM from the visible text — exact wording, exact spelling.
- If you don't see a clear unit (the visible text is navigation, ads, repeated headers, or otherwise not real content), return found=False and use skip_to to indicate where to advance past.
- Prefer larger coherent units over fragmenting into sentences.
- A unit is a span of text that conveys ONE coherent piece of content and could be evaluated independently.
"""

DOMAIN_HINTS: dict[str, str] = {
    "academic": (
        "This text is an academic paper or section thereof. Likely unit boundaries: "
        "section headings (Introduction, Methods, Results, Discussion), figures with "
        "captions, equation blocks, distinct paragraphs of argument."
    ),
    "web": (
        "This text was extracted from a web page. Watch for navigation menus, ads, "
        "cookie banners, and repeated site headers/footers — return found=False for those. "
        "Real content units: article paragraphs, headlines, body text."
    ),
    "code": (
        "This text contains source code or technical documentation. Unit boundaries: "
        "function definitions, class definitions, docstrings, comment blocks, distinct "
        "code blocks separated by blank lines."
    ),
    "transcript": (
        "This text is a conversation transcript. Unit boundaries: speaker turns, "
        "topic shifts, Q&A pairs (treat a question + its answer as a single unit when natural)."
    ),
    "general": (
        "Unit boundaries: paragraphs, sections marked by headings, list blocks, "
        "code blocks, quotation blocks, distinct topical shifts in continuous prose."
    ),
}


def build_user_prompt(
    *,
    window_text: str,
    domain: str,
    window_size: int,
    prior_unit_titles: list[str],
) -> str:
    """Compose the user message sent to the model for one window."""
    domain_hint = DOMAIN_HINTS.get(domain, DOMAIN_HINTS["general"])

    if prior_unit_titles:
        prior_section = (
            "Previous units already extracted (for continuity context):\n"
            + "\n".join(f"  - {t}" for t in prior_unit_titles[-3:])
            + "\n\n"
        )
    else:
        prior_section = ""

    return f"""{domain_hint}

{prior_section}Find the next coherent unit starting at the very beginning of the text below.
The unit's end_anchor must fall within the FIRST {window_size} characters of the text.
You may use any text after that only to verify the boundary is real.

If the unit clearly continues past the visible text (no natural ending), set complete=False.
If the visible text contains no extractable content (junk, navigation), return found=False with skip_to set to a short verbatim phrase from near the end of the visible text.

--- BEGIN TEXT ---
{window_text}
--- END TEXT ---
"""
