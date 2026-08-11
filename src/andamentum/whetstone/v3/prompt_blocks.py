"""Prompt text and substrate blocks shared by review mode and checklist mode.

The paragraphs here — the author-actionability bar, the severity rubric, the
tool menu and the tool-use discipline — were tuned against real drafts, and they
are most of what makes a small model return usable findings instead of generic
commentary. They are also the text most likely to need another round of tuning.

Both modes need all of it. :mod:`..review` runs an open-ended criterion and
:mod:`..checklist.run_probes` runs one narrowly-named check, but a finding is a
finding: the same actionability bar applies, the same severity tiers mean the
same thing, and the same two tools are on offer. The text lived as two
near-verbatim copies until that became the obvious hazard — a fix to the rubric
would have to be made twice and one of the two would be missed. It lives here
instead, and a change made here lands in BOTH modes. That is the point, and the
thing to check before making one.

Deliberately NOT here: anything mode-specific. The cascade's
prior-stage-findings instruction stays in :mod:`..review`, because probes are
independent by design and must never be told what another probe found. The
presence/absence output instructions stay in :mod:`..checklist.run_probes`,
because they select an output schema.

The blocks say "document" throughout, including where review mode's copy said
"paper": review mode already opened with "You are reviewing a document" and
already ran on memos and blog posts, so the narrower noun was a leftover rather
than a claim about the input.

:func:`sections_block` and :func:`claims_block` render the document substrate
both modes put in front of the agent. They live here rather than in
:mod:`..digest` because they are prompt text — the header wording, the size hint
and the section ordering are tuned for a model reading them, not for a caller
querying the model object.

Dependency-light on purpose: ``DocumentModel`` and nothing else, so importing
the shared prompt text never drags a pipeline module in behind it.
"""

from __future__ import annotations

from .model import DocumentModel

ACTIONABILITY_BAR = """Every finding must be author-actionable: the issue \
description should name what the author would change, add, or verify to \
resolve it. If you cannot say what a fix would look like — even abstractly \
("add a power-analysis sentence", "reconcile the n=50 vs n=48 mismatch", \
"drop the 'first-ever' claim or cite a prior-art search") — the finding is \
too vague to keep; either sharpen it or omit it."""

SEVERITY_RUBRIC = """Severity rubric — pick the level by what the author would \
have to do:
  - major: the document's conclusions, validity, or reproducibility are at \
stake. Ignoring this would leave the work wrong, unsupported, or unusable. \
Author must address before the document is sound.
  - moderate: a real weakness that a competent reader will notice and that \
weakens the document, but conclusions survive if it stays. Author should fix \
to strengthen the work.
  - minor: a local improvement — wording, typo, formatting, a single sentence \
that could be sharper. Safe to ignore; nice to fix.

When uncertain between two severity tiers, pick the lower one."""

TOOLS_AVAILABLE = """You have two tools available to investigate the source \
beyond the digest:
  - read_section(section_id) — read a section in full when the gist isn't \
enough.
  - search_paper(query) — find where a term appears in the document; pass \
regex=True for patterns like (limitation|caveat|weakness) or Theorem [0-9]+."""

TOOL_DISCIPLINE = """Tool-use discipline:
  - Don't re-read a section you've already loaded in this conversation. \
Refer back to the earlier tool result; the section text is already in your \
context.
  - When a search returns zero hits, treat the absence as the answer. Do not \
rephrase the same negative-result query with adjacent terms — if the concept \
isn't there, broaden ONCE if needed (e.g. add a synonym) and then accept the \
finding.
  - When a search returns hits but you want more context around them, call \
read_section on the matching section rather than re-searching with a longer \
query that pulls in the surrounding text."""

SECTION_IDS_NOTE = (
    "Section ids are in the SECTIONS block (e.g. 4.2, abstract, sec_004)."
)


def sections_block(model: DocumentModel) -> str:
    """The table of contents: one line per section, id first.

    Both modes emit this unconditionally, because it is also the
    ``read_section`` navigation aid — without it the agent has no valid section
    ids to pass. The size hint tells it where the substance lives; the gist
    gives it enough semantic flavour to choose a section without reading two.
    """
    gist_by_id = {g.section_id: g.gist for g in model.gists}
    if not model.sections:
        return "SECTIONS: (none)"
    lines = [
        f"  - [{s.id}] {s.title} ({len(s.text):,} chars)"
        + (f" — {gist}" if (gist := gist_by_id.get(s.id, "").strip()) else "")
        for s in model.sections
    ]
    return "SECTIONS (id | title | size | gist):\n" + "\n".join(lines)


def claims_block(model: DocumentModel) -> str:
    """The document's verbatim claims, grouped under their origin section.

    Grouped rather than listed flat so the agent sees "what each section
    asserts" without reading every section. Sections are walked in their
    declared order — matching :func:`sections_block` — so the block is stable
    run to run, which checklist mode's comparability depends on.
    """
    claims_by_section: dict[str, list[str]] = {}
    for c in model.claims:
        claims_by_section.setdefault(c.span.section_id, []).append(c.quote)
    if not claims_by_section:
        return "CLAIMS BY SECTION: (none)"
    lines: list[str] = []
    for s in model.sections:
        section_claims = claims_by_section.get(s.id)
        if not section_claims:
            continue
        lines.append(f"  [{s.id}]:")
        lines.extend(f"    - {quote!r}" for quote in section_claims)
    return "CLAIMS BY SECTION:\n" + "\n".join(lines)
