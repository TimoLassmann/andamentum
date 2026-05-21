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


def _gap_callout(result: PaperResult) -> str:
    """The money items: critical, cross-section issues B caught and A missed."""
    gaps = [
        f
        for f in result.adjudications
        if f.bucket == "b_only"
        and f.severity == "critical"
        and f.locality == "cross_section"
    ]
    if not gaps:
        return (
            '<aside class="typeset-callout tone-success">'
            "<p>No critical cross-section issue was found by the whole-document "
            "read that whetstone missed.</p></aside>"
        )
    items = "".join(f"<li>{_esc(g.text)}</li>" for g in gaps)
    return (
        '<aside class="typeset-callout tone-danger">'
        f"<p><strong>{len(gaps)} architecture gap(s)</strong> — critical, "
        "cross-section issues the whole-document read caught that whetstone "
        f"missed:</p><ul>{items}</ul></aside>"
    )


def _stat(value: int, label: str, danger: bool = False) -> str:
    cls = "am-stat__value is-danger" if danger else "am-stat__value"
    return (
        f'<div class="am-stat"><div class="{cls}">{value}</div>'
        f'<div class="am-stat__label">{_esc(label)}</div></div>'
    )


def _paper_section(idx: int, result: PaperResult, *, hidden: bool) -> str:
    p = result.paper
    s = summarise_paper(result)
    hide = " hidden" if hidden else ""
    title = _esc(p.title or p.id)
    meta_bits = [p.source, p.id, p.subfield]
    meta = " · ".join(_esc(b) for b in meta_bits if b)

    stats = "".join(
        [
            _stat(s.both, "found by both"),
            _stat(s.a_only, "whetstone only"),
            _stat(s.b_only, "whole-doc only"),
            _stat(s.b_only_critical_crosssection, "architecture gaps", danger=True),
        ]
    )

    return f"""<section class="paper" id="paper-{idx}"{hide}>
  <header class="typeset-heading">
    <h1>{title}</h1>
    <p class="typeset-meta">{meta}</p>
  </header>

  <div class="am-stats">{stats}</div>

  {_gap_callout(result)}

  <div class="verdicts">
    <div class="am-card am-card--quiet">
      <div class="eyebrow">Whetstone synthesis</div>
      <p>{_esc(result.arm_a.verdict) or "—"}</p>
    </div>
    <div class="am-card am-card--quiet">
      <div class="eyebrow">Whole-document top weaknesses</div>
      <p>{_esc(result.arm_b.verdict) or "—"}</p>
    </div>
  </div>

  <div class="panes">
    <div class="pane">
      <div class="eyebrow">System A — whetstone ({len(result.arm_a.findings)})</div>
      {_finding_cards(result.arm_a)}
    </div>
    <div class="pane">
      <div class="eyebrow">System B — whole-document ({len(result.arm_b.findings)})</div>
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
  <div class="am-app-title">Whetstone evaluation — A (whetstone) vs B (whole document)</div>
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
