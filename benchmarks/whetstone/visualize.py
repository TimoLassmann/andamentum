"""Self-contained side-by-side HTML visualiser for the whetstone evaluation.

One HTML file over the whole benchmark: a sidebar lists every paper; selecting
one shows the two reviews side by side — whetstone (A) on the left, the
whole-document read (B) on the right — plus the judge's diff (what B caught
that A missed) and both verdicts. Styled with the andamentum design system
(``docs/design/components.css``, inlined so the file is portable). App-style
chrome (``.am-*``), serif for the findings (content), sans for labels.
"""

from __future__ import annotations

import html
from pathlib import Path

from .report import summarise_paper
from .types import ArmOutput, PaperResult

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_COMPONENTS_CSS = _REPO_ROOT / "docs" / "design" / "components.css"


def _esc(text: str) -> str:
    return html.escape(text or "")


def _finding_cards(out: ArmOutput) -> str:
    if not out.findings:
        return '<p class="typeset-meta">No findings.</p>'
    cards = []
    for f in out.findings:
        detail = f"<p>{_esc(f.detail)}</p>" if f.detail else ""
        cards.append(
            f'<div class="am-card"><strong>{_esc(f.title)}</strong>{detail}</div>'
        )
    return "\n".join(cards)


def _stat(value: int, label: str, danger: bool = False) -> str:
    cls = "am-stat__value is-danger" if danger else "am-stat__value"
    return (
        f'<div class="am-stat"><div class="{cls}">{value}</div>'
        f'<div class="am-stat__label">{_esc(label)}</div></div>'
    )


# Map each adjudication facet to a design-system badge tone.
_BUCKET_TONE = {"both": "note", "a_only": "info", "b_only": "warn"}
_SEVERITY_TONE = {"critical": "danger", "minor": "note"}
_LOCALITY_TONE = {"cross_section": "info", "local": "note"}

# Display labels per Arm B variant. Arm A is always whetstone v2. The
# short label is for badges/stats; the long label is for verdict prose and
# section titles ("Whole-document top weaknesses" vs "v3 top weaknesses").
_ARM_B_DISPLAY = {
    "whole-doc": {
        "short": "whole-doc",
        "long": "whole-document",
        "verdict_panel_title": "Whole-document top weaknesses",
        "app_title_suffix": "A (whetstone v2) vs B (whole document)",
    },
    "v3": {
        "short": "v3",
        "long": "whetstone v3",
        "verdict_panel_title": "v3 top weaknesses",
        "app_title_suffix": "A (whetstone v2) vs B (whetstone v3)",
    },
}


def _arm_b(result: PaperResult) -> dict[str, str]:
    return _ARM_B_DISPLAY.get(result.arm_b_label, _ARM_B_DISPLAY["whole-doc"])


def _bucket_labels(arm_b_short: str) -> dict[str, str]:
    return {"both": "both", "a_only": "whetstone only", "b_only": f"{arm_b_short} only"}


def _badge(text: str, tone: str) -> str:
    return f'<span class="am-badge am-badge--{tone}">{_esc(text)}</span>'


_VERDICT_TONE = {
    "whetstone": "info",
    "whole-doc": "warn",
    "v3": "warn",
    "comparable": "note",
    "inconsistent": "danger",
}


def _comparison_section(result: PaperResult) -> str:
    """Top-of-paper 'which is better, and why' — the grounded judge verdict
    plus a deterministic scorecard. The verdict prose is a judgement (cites
    issues you can check below); the scorecard is fact (counts of judge tags).
    The judge is blinded and run twice with the order swapped; a flip shows as
    'inconsistent'."""
    s = summarise_paper(result)
    b = _arm_b(result)
    if result.comparison:
        c = result.comparison
        flag = (
            ""
            if c.order_consistent
            else ' <span class="am-badge am-badge--danger">order-sensitive</span>'
        )
        verdict = (
            '<div class="eyebrow">Judge\'s comparative verdict '
            "(blinded, order-checked)</div>"
            f"<p><strong>More useful: {_badge(c.more_useful, _VERDICT_TONE.get(c.more_useful, 'note'))}</strong>{flag}</p>"
            f"<p>{_esc(c.reasoning)}</p>"
        )
    else:
        verdict = '<div class="eyebrow">Judge\'s comparative verdict</div><p>—</p>'

    vmatch = "—" if s.verdict_match is None else ("yes" if s.verdict_match else "no")
    tiles = "".join(
        [
            _stat(s.both_critical, "critical: both caught"),
            _stat(
                s.b_only_critical_crosssection,
                "architecture gaps",
                danger=s.b_only_critical_crosssection > 0,
            ),
            _stat(s.a_only_critical, "critical: whetstone-only"),
            _stat(len(result.arm_a.findings), "whetstone findings"),
            _stat(len(result.arm_b.findings), f"{b['short']} findings"),
            _stat(s.a_only_minor, "whetstone-only minor (noise)"),
        ]
    )
    return (
        '<div class="comparison"><div class="am-card am-card--quiet">'
        f"{verdict}</div>"
        f'<div class="am-stats">{tiles}</div>'
        f'<p class="typeset-meta">Central problem identified by whetstone\'s '
        f"synthesis: {vmatch}</p></div>"
    )


def _adjudication_panel(result: PaperResult) -> str:
    """The judge's per-issue verdict: every aligned issue + its bucket /
    severity / locality, so the LLM judge's classification is visible (and
    auditable against the blinded worksheet), not just the derived counts."""
    adj = result.adjudications
    if not adj:
        return ""
    # Order: architecture gaps first, then other b_only, then a_only, then both.
    order = {"b_only": 0, "a_only": 1, "both": 2}
    rows = sorted(
        adj,
        key=lambda f: (order.get(f.bucket, 3), 0 if f.severity == "critical" else 1),
    )
    bucket_labels = _bucket_labels(_arm_b(result)["short"])
    cards = []
    for f in rows:
        badges = (
            _badge(
                bucket_labels.get(f.bucket, f.bucket),
                _BUCKET_TONE.get(f.bucket, "note"),
            )
            + _badge(f.severity, _SEVERITY_TONE.get(f.severity, "note"))
            + _badge(
                f.locality.replace("_", "-"), _LOCALITY_TONE.get(f.locality, "note")
            )
        )
        note = f"<p>{_esc(f.note)}</p>" if f.note else ""
        cards.append(
            f'<div class="am-card"><div class="am-tags">{badges}</div>'
            f"<p>{_esc(f.text)}</p>{note}</div>"
        )
    return (
        f'<div class="eyebrow">Judge adjudication ({len(adj)} aligned issue(s))</div>'
        f'<div class="adjudication">{"".join(cards)}</div>'
    )


def _paper_section(idx: int, result: PaperResult, *, hidden: bool) -> str:
    p = result.paper
    hide = " hidden" if hidden else ""
    title = _esc(p.title or p.id)
    meta_bits = [p.source, p.id, p.subfield]
    meta = " · ".join(_esc(b) for b in meta_bits if b)
    b = _arm_b(result)

    return f"""<section class="paper" id="paper-{idx}"{hide}>
  <header class="typeset-heading">
    <h1>{title}</h1>
    <p class="typeset-meta">{meta}</p>
  </header>

  {_comparison_section(result)}

  <div class="verdicts">
    <div class="am-card am-card--quiet">
      <div class="eyebrow">Whetstone synthesis</div>
      <p>{_esc(result.arm_a.verdict) or "—"}</p>
    </div>
    <div class="am-card am-card--quiet">
      <div class="eyebrow">{b["verdict_panel_title"]}</div>
      <p>{_esc(result.arm_b.verdict) or "—"}</p>
    </div>
  </div>

  {_adjudication_panel(result)}

  <div class="panes">
    <div class="pane">
      <div class="eyebrow">System A — whetstone v2 ({len(result.arm_a.findings)})</div>
      {_finding_cards(result.arm_a)}
    </div>
    <div class="pane">
      <div class="eyebrow">System B — {b["long"]} ({len(result.arm_b.findings)})</div>
      {_finding_cards(result.arm_b)}
    </div>
  </div>
</section>"""


def _sidebar(results: list[PaperResult]) -> str:
    rows = []
    for i, r in enumerate(results):
        s = summarise_paper(r)
        active = " is-active" if i == 0 else ""
        label = _esc(r.paper.title or r.paper.id)
        gap = (
            f'<span class="am-badge am-badge--danger">{s.b_only_critical_crosssection}</span>'
            if s.b_only_critical_crosssection
            else ""
        )
        rows.append(
            f'<button class="am-sidebar__item{active}" data-idx="{i}" '
            f'onclick="showPaper({i})">{label} {gap}</button>'
        )
    return "\n".join(rows)


_EXTRA_CSS = """
  .layout { display: grid; grid-template-columns: 280px 1fr; gap: var(--am-sp-6);
            max-width: 1400px; margin: 0 auto; padding: var(--am-sp-6); }
  .panes { display: grid; grid-template-columns: 1fr 1fr; gap: var(--am-sp-5);
           margin-top: var(--am-sp-6); }
  .verdicts { display: grid; grid-template-columns: 1fr 1fr; gap: var(--am-sp-5);
              margin-top: var(--am-sp-5); }
  .pane > .am-card { margin-bottom: var(--am-sp-3); }
  .adjudication { margin-top: var(--am-sp-3); }
  .adjudication > .am-card { margin-bottom: var(--am-sp-3); }
  .adjudication .am-tags { margin-bottom: var(--am-sp-2); }
  .eyebrow { font-family: var(--am-font-ui, Inter, sans-serif); font-size: 10px;
             font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px;
             color: var(--am-ink-5); margin-bottom: var(--am-sp-3); }
  .am-sidebar__item { display: block; width: 100%; text-align: left;
                      background: none; border: none; cursor: pointer; }
"""


def build_html(results: list[PaperResult], *, css: str) -> str:
    sections = "\n".join(
        _paper_section(i, r, hidden=(i != 0)) for i, r in enumerate(results)
    )
    sidebar = _sidebar(results)
    total_gaps = sum(summarise_paper(r).b_only_critical_crosssection for r in results)
    # If all papers ran the same Arm B, use that label in the app title;
    # otherwise fall back to a neutral "mixed" header.
    arm_b_labels = {r.arm_b_label for r in results} if results else {"whole-doc"}
    if len(arm_b_labels) == 1:
        app_title_suffix = _ARM_B_DISPLAY[next(iter(arm_b_labels))]["app_title_suffix"]
    else:
        app_title_suffix = "A (whetstone v2) vs B (mixed)"
    return f"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Whetstone evaluation — side by side</title>
<style>
{css}
{_EXTRA_CSS}
</style>
</head>
<body class="am-app">
<header class="am-app-header">
  <div class="am-app-title">Whetstone evaluation — {app_title_suffix}</div>
  <div class="am-app-header__right"><span class="am-badge am-badge--danger">{total_gaps} architecture gaps</span></div>
</header>
<div class="layout">
  <aside class="am-sidebar">
    <div class="am-sidebar__section">
      <div class="am-sidebar__title">Papers ({len(results)})</div>
      {sidebar}
    </div>
  </aside>
  <main>
{sections}
  </main>
</div>
<script>
function showPaper(idx) {{
  document.querySelectorAll('.paper').forEach(function (el) {{ el.hidden = true; }});
  var sel = document.getElementById('paper-' + idx);
  if (sel) sel.hidden = false;
  document.querySelectorAll('.am-sidebar__item').forEach(function (b) {{
    b.classList.toggle('is-active', b.dataset.idx === String(idx));
  }});
}}
</script>
</body>
</html>"""


def write_report(results: list[PaperResult], out_path: Path) -> Path:
    css = (
        _COMPONENTS_CSS.read_text(encoding="utf-8") if _COMPONENTS_CSS.exists() else ""
    )
    out_path.write_text(build_html(results, css=css), encoding="utf-8")
    return out_path
