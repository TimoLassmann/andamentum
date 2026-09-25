"""Prompt text and substrate blocks shared by review mode and checklist mode.

The paragraphs here — the author-actionability bar, the writing rules, the
severity rubric, the tool menu and the tool-use discipline — were tuned against real drafts, and they
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

import re

from .model import DocumentModel

ACTIONABILITY_BAR = """Every finding must be author-actionable: its fix names \
what the author would change, add, cut, or check to resolve it. If you cannot \
write a concrete fix — even a short one ("add a power-analysis sentence", \
"reconcile the n=50 vs n=48 mismatch", "drop 'first-ever' or cite a prior-art \
search") — the finding is too vague to keep; either sharpen it or omit it."""

# How a finding is WORDED, as distinct from whether it is worth keeping (the
# bar above). The author reads every finding as a comment in the margin of
# their own draft, so it is written to them, in their terms. Measured on a real
# run before this block existed: comments headed by the criterion's name
# over a complaint, problem and fix run together after a semicolon, a menu of
# five fixes where one was wanted, "may overstate", "validation/quantitative".
# The two examples carry no field names beyond `issue` and `fix`, which both
# modes' schemas have, so the block reads the same in each.
WRITING_RULES = """How to write each finding — the author reads it as a \
comment in the margin of their own draft:
  - Write to the author as "you", in plain words. No review jargon, and never \
the name of the criterion.
  - issue: what is wrong in this passage and what it costs a reader. Do not \
restate the quoted text; the author can see it.
  - fix: ONE action, starting with a verb. One fix, not a list of options. \
Size it to the kind of document — a cover letter does not need confidence \
intervals.
  - Write "may" or "might" only when you are genuinely unsure, and then say \
what would settle it.
  - Pick one word; do not join alternatives with a slash.

Weak:
  issue: "The performance claim may lack supporting details such as \
confidence intervals, absolute counts, evaluation metric or comparator versions."
  fix: "Consider adding more detail/context."
Strong:
  issue: "You give a 12-18% improvement but not where it comes from, so a \
reader cannot check it."
  fix: "Point to the table that reports it, e.g. '(Table 2)'."
"""

# Field descriptions for a finding's schema, shared by the review cascade, the
# gap loop and (``ISSUE_FIELD`` aside, which checklist words per shape) the
# checklist probes. A field description is read more reliably by a small model
# than the same sentence in the prompt, so these carry the wording rules to
# the slot they govern.
QUOTE_FIELD = "A VERBATIM span from the document it concerns."
ISSUE_FIELD = (
    "What is wrong in this passage and what it costs a reader, in one or two "
    "plain sentences written to the author. Do not restate the quote, and do "
    "not put the fix here — it has its own field."
)
FIX_FIELD = (
    "One concrete action, starting with a verb: what the author should "
    "change, add, cut or check. One fix, not a list of options."
)
HEADLINE_FIELD = (
    "The problem in at most eight plain words, as the heading of a margin "
    "comment — e.g. '\"The bulk\" overstates the judge result'. Never the "
    "criterion's name."
)

_FIX_MARKER = re.compile(r"\s*\bFix:\s*", re.IGNORECASE)


def split_fix(issue: str, fix: str) -> tuple[str, str]:
    """Hold a finding to its two fields when the model ran them together.

    ``ISSUE_FIELD`` says the fix does not go in the issue, and a small model
    puts it there anyway — measured on a real manuscript with gpt-5.4-nano:
    issues ending "… Fix: State the precise metric …", followed by the fix
    field saying the same thing, so the comment gave the fix twice. The text
    after a ``Fix:`` marker is cut from the issue. It becomes the fix only if
    the fix field is empty; otherwise the fix field wins, since it is the slot
    the model was asked to fill.

    Returns ``(issue, fix)``, both stripped.
    """
    parts = _FIX_MARKER.split(issue, maxsplit=1)
    if len(parts) == 1:
        return issue.strip(), fix.strip()
    head, tail = parts[0].strip(), parts[1].strip()
    return (head or issue.strip()), (fix.strip() or tail)


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


def brief_block(brief: str) -> str:
    """The reviewer's stance, as the opening of the call's user prompt.

    The one thing a reviewer profile contributes that no argument can express:
    the stance the questions are read from, the bar a finding has to clear, and
    what this reviewer will not report. It goes at the top of the *user* prompt
    rather than into the agent's instructions deliberately — the shared
    scaffold is the general rule and the brief is the specific one, and the
    later, more specific statement is the one a model follows.

    Rendered by all three calls a profile can reach (the criterion call, the
    probe call and the editor call) so a brief means the same thing in each.

    Returns the empty string for an empty brief, and the callers interpolate
    the result directly — so the default reviewer's prompts are byte-identical
    to the prompts the same run produced before profiles existed. That is a
    pinned test, not an aspiration: see ``tests/test_brief_slot.py``.
    """
    stripped = brief.strip()
    if not stripped:
        return ""
    return (
        "REVIEWER BRIEF — the stance this review is written from, and the "
        f"bar a finding has to clear:\n{stripped}\n\n"
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
