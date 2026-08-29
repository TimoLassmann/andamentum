#!/usr/bin/env python3
"""Build multimodel_report.html — the shareable write-up of the multi-model
confidence revisit (see MULTIMODEL_PLAN.md for the design).

Reads the two result JSONs produced by ``multimodel_experiment.py``
(``results/multimodel.json`` = per-model, ``results/multimodel_combos.json`` =
panels), renders five figures via ``andamentum.figures``, and embeds everything
into one self-contained HTML file styled identically to ``report.html``
(Source Serif 4 body, Inter tables, warm palette, base64 figures).

No model calls. Regenerate the JSONs with ``multimodel_experiment.py`` first,
then:

    uv run python -m experiments.dirichlet_confidence.multimodel_report
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from andamentum.figures import despine, get_palette, savefig, setup_style  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
RESULTS = HERE / "results" / "multimodel.json"
RESULTS_COMBOS = HERE / "results" / "multimodel_combos.json"
OUT = HERE / "multimodel_report.html"

# Ladder order for the x-axis; short labels for plotting.
LADDER = [
    ("ollama:gemma4:e2b-nvfp4", "gemma4\ne2b"),
    ("ollama:gemma4:12b-mxfp8", "gemma4\n12b"),
    ("ollama:gemma4:26b-nvfp4", "gemma4\n26b"),
    ("ollama:gemma4:31b-nvfp4", "gemma4\n31b"),
    ("ollama:gpt-oss:20b", "gpt-oss\n20b"),
    ("ollama:qwen3.6:35b-mlx", "qwen3.6\n35b"),
    ("ollama:qwen3.8:27b-mlx", "qwen3.8\n27b"),
    ("ollama:muse-glimmer:30b-mlx", "muse-glimmer\n30b"),
    ("ollama:ornith-1.5:35b", "ornith-1.5\n35b"),
]

# The full-panel combo key. Size-independent so adding a model to the roster
# doesn't silently turn every panel lookup into a missing key.
FULL_PANEL = "all-models"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data if data.get("results") else None


def img(path: Path) -> str:
    if not path.exists():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def figure_html(path: Path, caption_html: str) -> str:
    uri = img(path)
    if not uri:
        return (
            f'<div class="note warn">Figure <code>{path.name}</code> not found — '
            f"run the experiment, then rebuild this report.</div>"
        )
    return (
        f'<figure class="figure">\n  <img src="{uri}" alt="{path.stem}">\n'
        f"  <figcaption>{caption_html}</figcaption>\n</figure>"
    )


# ---------------------------------------------------------------------------
# Figures — all read from the JSONs; each returns True if it drew something.
# ---------------------------------------------------------------------------


def _auroc_bar(ax, labels, points, los, his, palette, title, ylabel):
    x = np.arange(len(labels))
    yerr = np.vstack([np.array(points) - np.array(los), np.array(his) - np.array(points)])
    ax.bar(x, points, color=palette[0], width=0.62, zorder=3)
    ax.errorbar(x, points, yerr=yerr, fmt="none", ecolor="#555", capsize=4, lw=1.2, zorder=4)
    ax.axhline(0.5, color="#a8604f", ls="--", lw=1.2, zorder=2, label="chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    despine(ax)


def fig_ladder(per_model: dict) -> bool:
    """H1: single-call entropy AUROC up the capability ladder + degeneracy."""
    rows = []
    for key, label in LADDER:
        r = per_model["results"].get(key)
        if r and r.get("single"):
            s = r["single"]
            rows.append((label, s["auroc_entropy"], s["degeneracy_rate"], s["accuracy"]))
    if not rows:
        return False
    setup_style()
    palette = get_palette("npg")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = [r[0] for r in rows]
    _auroc_bar(
        ax1, labels,
        [r[1]["point"] for r in rows], [r[1]["lo"] for r in rows], [r[1]["hi"] for r in rows],
        palette, "Error-detection AUROC of single-call entropy", "AUROC (wrong > right)",
    )
    ax1.legend(frameon=False, fontsize=9, loc="lower right")
    x = np.arange(len(labels))
    ax2.bar(x - 0.2, [r[2] for r in rows], width=0.4, color=palette[1], label="one-hot rate", zorder=3)
    ax2.bar(x + 0.2, [r[3] for r in rows], width=0.4, color=palette[2], label="accuracy", zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("rate")
    ax2.set_title("One-hot degeneracy vs accuracy")
    ax2.legend(frameon=False, fontsize=9)
    despine(ax2)
    fig.tight_layout()
    savefig(fig, FIGS / "mm_fig1_ladder.png")
    plt.close(fig)
    return True


def fig_single_vs_temp(per_model: dict) -> bool:
    """H2: single-call entropy (S1) vs K=5 temperature band (S2) per model."""
    rows = []
    for key, label in LADDER:
        r = per_model["results"].get(key)
        if r and r.get("single") and r.get("temperature"):
            rows.append((label, r["single"]["auroc_entropy"], r["temperature"]["auroc_band"]))
    if not rows:
        return False
    setup_style()
    palette = get_palette("npg")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(rows))
    labels = [r[0] for r in rows]
    s1 = [r[1]["point"] for r in rows]
    s2 = [r[2]["point"] for r in rows]
    s1e = np.vstack([np.array(s1) - [r[1]["lo"] for r in rows], [r[1]["hi"] for r in rows] - np.array(s1)])
    s2e = np.vstack([np.array(s2) - [r[2]["lo"] for r in rows], [r[2]["hi"] for r in rows] - np.array(s2)])
    ax.bar(x - 0.2, s1, width=0.4, color=palette[0], label="S1: 1 call, temp 0", zorder=3)
    ax.bar(x + 0.2, s2, width=0.4, color=palette[3], label="S2: 5 calls, temp 0.7 (band)", zorder=3)
    ax.errorbar(x - 0.2, s1, yerr=s1e, fmt="none", ecolor="#555", capsize=3, lw=1, zorder=4)
    ax.errorbar(x + 0.2, s2, yerr=s2e, fmt="none", ecolor="#555", capsize=3, lw=1, zorder=4)
    ax.axhline(0.5, color="#a8604f", ls="--", lw=1.2, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("error-detection AUROC")
    ax.set_title("Is one deterministic call as good as five sampled ones?")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    despine(ax)
    fig.tight_layout()
    savefig(fig, FIGS / "mm_fig2_single_vs_temp.png")
    plt.close(fig)
    return True


def _compact(name: str) -> str:
    """Short axis label for a model or panel id."""
    return (name.replace("ollama:", "").replace("-nvfp4", "")
            .replace("-mxfp8", "").replace("-mlx", ""))


_NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five",
                 6: "six", 7: "seven", 8: "eight"}


def _nword(n: int) -> str:
    """Spell small counts so the narrative reads as prose, not as a template."""
    return _NUMBER_WORDS.get(n, str(n))


def _widest_panel(combos: dict) -> str | None:
    """Key of the widest panel that actually produced a singles arm.

    Prefers FULL_PANEL, then falls back to the combo with the most members —
    never merely the first one, which on a results file written before the
    roster changed would silently pick a *pair* and describe it as the panel.
    """
    results = combos.get("results", {})
    scored = {k: v for k, v in results.items() if v.get("singles_arm")}
    if not scored:
        return None
    if FULL_PANEL in scored:
        return FULL_PANEL
    members = combos.get("config", {}).get("combos", {})
    return max(scored, key=lambda k: len(members.get(k, [])))


def _panel_members(combos: dict) -> list[str]:
    """Members of the widest panel, read from the run's own config so the prose
    describes the panel that actually ran rather than a hard-coded roster."""
    key = _widest_panel(combos)
    if key is None:
        return []
    return list(combos.get("config", {}).get("combos", {}).get(key, []))


def _panel_singles(combos: dict) -> dict | None:
    """The widest panel's singles arm, or None if no panel produced one."""
    key = _widest_panel(combos)
    if key is None:
        return None
    return combos["results"][key].get("singles_arm")


def _config_entropy_aurocs(per_model: dict | None, combos: dict):
    """Ordered list of (label, auroc_dict, is_panel) — the error-detection AUROC
    of the entropy of each configuration's final answer. For a single model that
    is its own histogram's entropy; for a panel it is the pooled histogram's
    (total-doubt) entropy — the one quantity defined identically for both."""
    configs = []
    if per_model:
        for k, _ in LADDER:
            r = per_model["results"].get(k)
            if r and r.get("single"):
                configs.append((_compact(k), r["single"]["auroc_entropy"], False))
    for name, r in combos["results"].items():
        s = r.get("singles_arm")
        if s:
            configs.append((name, s["auroc_total_entropy"], True))
    return configs


def fig_panels(combos: dict, per_model: dict | None) -> bool:
    """Individual models vs panels on one axis: does combining models improve the
    entropy error-detection signal? Plus the vote gate for the widest panel."""
    key = _widest_panel(combos)
    s = _panel_singles(combos)
    configs = _config_entropy_aurocs(per_model, combos)
    if not s or not configs:
        return False

    setup_style()
    palette = get_palette("npg")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                                   gridspec_kw={"width_ratios": [1.6, 1]})

    # Left: entropy error-detection AUROC per configuration, individual models
    # (one colour) then panels (another), with a divider between the groups.
    x = np.arange(len(configs))
    pts = [c[1]["point"] for c in configs]
    los = [c[1]["lo"] for c in configs]
    his = [c[1]["hi"] for c in configs]
    cols = [palette[4] if not c[2] else palette[0] for c in configs]
    yerr = np.vstack([np.array(pts) - np.array(los), np.array(his) - np.array(pts)])
    ax1.bar(x, pts, color=cols, width=0.66, zorder=3)
    ax1.errorbar(x, pts, yerr=yerr, fmt="none", ecolor="#555", capsize=3, lw=1.1, zorder=4)
    ax1.axhline(0.5, color="#a8604f", ls="--", lw=1.2, zorder=2)
    n_indiv = sum(1 for c in configs if not c[2])
    if 0 < n_indiv < len(configs):
        ax1.axvline(n_indiv - 0.5, color="#bbb", ls="-", lw=1.0, zorder=1)
        ax1.text((n_indiv - 1) / 2, 0.97, "individual models", ha="center",
                 va="top", fontsize=9, color=palette[4], transform=ax1.get_xaxis_transform())
        ax1.text((n_indiv + len(configs) - 1) / 2, 0.97, "panels", ha="center",
                 va="top", fontsize=9, color=palette[0], transform=ax1.get_xaxis_transform())
    ax1.set_xticks(x)
    ax1.set_xticklabels([c[0] for c in configs], fontsize=8.5, rotation=30, ha="right")
    ax1.set_ylim(0.4, 1.0)
    ax1.set_ylabel("error-detection AUROC")
    ax1.set_title("Does combining models sharpen the entropy signal?")
    despine(ax1)

    # Right: the vote gate — accuracy on unanimous vs split, with coverage.
    vg = s["vote_gate"]
    cats, accs, cols, covs = [], [], [], []
    if vg["acc_unanimous"] is not None:
        cats.append("unanimous\n(all agree)"); accs.append(vg["acc_unanimous"])
        cols.append(palette[2]); covs.append(vg["frac_unanimous"])
    if vg["acc_split"] is not None:
        cats.append("split\n(any dissent)"); accs.append(vg["acc_split"])
        cols.append(palette[3]); covs.append(vg["frac_split"])
    x = np.arange(len(cats))
    ax2.bar(x, accs, color=cols, width=0.6, zorder=3)
    ax2.axhline(s["accuracy"], color="#555", ls="--", lw=1.2, zorder=2,
                label=f"overall accuracy ({s['accuracy']:.2f})")
    for xi, acc, cov in zip(x, accs, covs):
        ax2.text(xi, acc + 0.02, f"{acc:.0%}\n({cov:.0%} of claims)",
                 ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(x); ax2.set_xticklabels(cats, fontsize=9)
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("accuracy")
    ax2.set_title(f"The vote gate ({key})")
    ax2.legend(frameon=False, fontsize=8.5, loc="lower left")
    despine(ax2)

    fig.tight_layout()
    savefig(fig, FIGS / "mm_fig3_panels.png")
    plt.close(fig)
    return True


def fig_calibration(combos: dict, per_model: dict | None) -> bool:
    """Calibration of individual models AND panels on the same axes — so we can
    see whether combining models improves the confidence's calibration. Individual
    models drawn dashed, panels solid."""
    curves = []  # (label, calibration_dict, is_panel)
    if per_model:
        for k, _ in LADDER:
            r = per_model["results"].get(k)
            if r and r.get("single") and r["single"].get("calibration"):
                curves.append((_compact(k), r["single"]["calibration"], False))
    for name, r in combos["results"].items():
        s = r.get("singles_arm")
        if s and s.get("calibration"):
            curves.append((name, s["calibration"], True))
    if not curves:
        return False
    setup_style()
    palette = get_palette("npg")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4),
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    ax1.plot([0, 1], [0, 1], color="#999", ls="--", lw=1.2, zorder=2, label="perfect")
    for i, (name, cal, is_panel) in enumerate(curves):
        rc = cal["reliability_curve"]
        conf, acc, count = np.array(rc["confidence"]), np.array(rc["accuracy"]), np.array(rc["count"])
        mask = count > 0
        ax1.plot(conf[mask], acc[mask], "-o" if is_panel else "--o",
                 color=palette[i % len(palette)], ms=3.5,
                 lw=1.8 if is_panel else 1.3, alpha=1.0 if is_panel else 0.75,
                 label=f"{name} ({cal['ece']:.2f})", zorder=3)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.set_xlabel("stated confidence (predicted-class prob)")
    ax1.set_ylabel("empirical accuracy")
    ax1.set_title("Reliability — dashed = single model, solid = panel")
    ax1.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2, title="ECE in ()")
    despine(ax1)
    # ECE bars: individual models then panels, coloured by group.
    names = [c[0] for c in curves]
    x = np.arange(len(names))
    cols = [palette[4] if not c[2] else palette[1] for c in curves]
    ax2.bar(x, [c[1]["ece"] for c in curves], color=cols, width=0.62, zorder=3)
    n_indiv = sum(1 for c in curves if not c[2])
    if 0 < n_indiv < len(curves):
        ax2.axvline(n_indiv - 0.5, color="#bbb", ls="-", lw=1.0, zorder=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=8.5, rotation=30, ha="right")
    ax2.set_ylabel("expected calibration error (↓ better)")
    ax2.set_title("Calibration error — models vs panels")
    despine(ax2)
    fig.tight_layout()
    savefig(fig, FIGS / "mm_fig4_calibration.png")
    plt.close(fig)
    return True


def fig_risk_coverage(combos: dict) -> bool:
    """The training-signal filter: accuracy vs coverage when we keep the
    lowest-disagreement panel judgments as training data."""
    series = []
    for name, r in combos["results"].items():
        s = r.get("singles_arm")
        if s and s.get("risk_coverage"):
            series.append((name, s["risk_coverage"]))
    if not series:
        return False
    setup_style()
    palette = get_palette("npg")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (name, rc) in enumerate(series):
        cov = [p["coverage"] for p in rc]
        acc = [p["accuracy"] for p in rc]
        ax.plot(cov, acc, "-o", color=palette[i % len(palette)], ms=5, lw=1.8, label=name, zorder=3)
    ax.set_xlabel("coverage (fraction of panel judgments kept)")
    ax.set_ylabel("accuracy on kept judgments")
    ax.set_title("Keep the panel's most-agreed judgments → cleaner training signal")
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    despine(ax)
    fig.tight_layout()
    savefig(fig, FIGS / "mm_fig5_risk_coverage.png")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# HTML tables (from JSON)
# ---------------------------------------------------------------------------


def _short(model: str) -> str:
    return model.split(":", 1)[-1]


def _auroc_cell(a: dict | None) -> str:
    if not a or a.get("point") is None or (isinstance(a["point"], float) and np.isnan(a["point"])):
        return '<td class="num">—</td>'
    cls = "win" if a["lo"] > 0.5 else ""
    return f'<td class="num {cls}">{a["point"]:.3f} <span style="color:#999">[{a["lo"]:.2f}, {a["hi"]:.2f}]</span></td>'


def per_model_table(per_model: dict, has_temp: bool) -> str:
    """Full per-model table (header + body). The self-consistency band column is
    only shown when temperature draws were gathered (K≥2)."""
    # Column labels say what they measure. The old "S1"/"S2" arm codes were
    # plan-doc shorthand that never appeared in the report's own prose.
    head = ('<thead><tr><th>model</th><th class="num">accuracy</th>'
            '<th class="num">one-hot rate</th>'
            '<th class="num">entropy AUROC<br><span style="font-weight:400;text-transform:none;'
            'letter-spacing:0">1 call, temp 0</span></th>')
    if has_temp:
        head += ('<th class="num">band AUROC<br><span style="font-weight:400;'
                 'text-transform:none;letter-spacing:0">K draws, temp 0.7</span></th>')
    head += '<th class="num">ECE ↓</th></tr></thead>'
    body = []
    for key, _ in LADDER:
        r = per_model["results"].get(key)
        if not r:
            continue
        s, tp = r.get("single"), r.get("temperature")
        cells = [f"<td>{_short(key)}</td>"]
        cells.append(f'<td class="num">{s["accuracy"]:.2f}</td>' if s else '<td class="num">—</td>')
        cells.append(f'<td class="num">{s["degeneracy_rate"]:.0%}</td>' if s else '<td class="num">—</td>')
        cells.append(_auroc_cell(s["auroc_entropy"]) if s else '<td class="num">—</td>')
        if has_temp:
            cells.append(_auroc_cell(tp["auroc_band"]) if tp else '<td class="num">—</td>')
        cells.append(f'<td class="num">{s["ece"]:.3f}</td>' if s else '<td class="num">—</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table>{head}<tbody>{chr(10).join(body)}</tbody></table>"


def comparison_table(per_model: dict | None, combos: dict) -> str:
    """Individual models AND panels in one table, so combined vs single is a
    direct read. For a single model there is one histogram, so total-doubt =
    solo-doubt = its own entropy and there is no disagreement / vote term (—)."""
    ncols = 7
    head = ('<thead><tr><th>configuration</th><th class="num">accuracy</th>'
            '<th class="num">ECE ↓</th>'
            '<th class="num">total-doubt AUROC</th>'
            '<th class="num">solo-doubt AUROC</th>'
            '<th class="num">disagreement AUROC</th>'
            '<th class="num">unanimous: accuracy (share)</th></tr></thead>')
    body = []

    def subheader(text: str) -> str:
        return (f'<tr><td colspan="{ncols}" style="background:#faf9f6;font-weight:600;'
                f'color:#555;font-size:12px;text-transform:uppercase;letter-spacing:0.5px">'
                f'{text}</td></tr>')

    if per_model:
        body.append(subheader("individual models"))
        for key, _ in LADDER:
            r = per_model["results"].get(key)
            s = r.get("single") if r else None
            if not s:
                continue
            body.append("<tr>" + "".join([
                f"<td>{_compact(key)}</td>",
                f'<td class="num">{s["accuracy"]:.2f}</td>',
                f'<td class="num">{s["calibration"]["ece"]:.3f}</td>',
                _auroc_cell(s["auroc_entropy"]),   # its own entropy = total = solo
                _auroc_cell(s["auroc_entropy"]),
                '<td class="num">—</td>',
                '<td class="num">—</td>',
            ]) + "</tr>")

    body.append(subheader("panels"))
    for name, r in combos["results"].items():
        s = r.get("singles_arm")
        if not s:
            body.append(f'<tr><td>{name}</td>' + '<td class="num">—</td>' * (ncols - 1) + "</tr>")
            continue
        vg = s["vote_gate"]
        unan = (f'{vg["acc_unanimous"]:.0%} <span style="color:#999">({vg["frac_unanimous"]:.0%})</span>'
                if vg["acc_unanimous"] is not None else "—")
        body.append("<tr>" + "".join([
            f"<td>{name}</td>",
            f'<td class="num">{s["accuracy"]:.2f}</td>',
            f'<td class="num">{s["calibration"]["ece"]:.3f}</td>',
            _auroc_cell(s["auroc_total_entropy"]),
            _auroc_cell(s["auroc_solo_entropy"]),
            _auroc_cell(s["auroc_disagreement_mi"]),
            f'<td class="num">{unan}</td>',
        ]) + "</tr>")
    return f"<table>{head}<tbody>{chr(10).join(body)}</tbody></table>"


def _metrics_note_single() -> str:
    """Column definitions for the per-model table, placed directly above it."""
    return """
<div class="note">
<strong>accuracy</strong> — how often the largest of the three numbers matches the gold
label. Claims are drawn balanced, so chance is 33%.<br>
<strong>one-hot rate</strong> — share of answers putting ≥ 95% on a single class. This
caps everything to its right: a model that never hedges has no spread left to read, so
its entropy cannot carry a signal however accurate it is.<br>
<strong>entropy AUROC</strong> — error detection (see above) scored on this model's own
entropy: the spread of its histogram, normalised so 0 = all weight on one class and
1 = a flat 33/33/33 shrug.<br>
<strong>ECE</strong> — expected calibration error. Bucket answers by stated confidence,
then average the gap between that confidence and the accuracy actually observed in each
bucket. Lower is better; 0 means a stated 80% really is right 80% of the time.
</div>
"""


def _metrics_note_panel() -> str:
    """The one column of the comparison table the decomposition text doesn't cover."""
    return """
<div class="note">
The three AUROC columns are the doubt signals defined just above, each scored the same
way. <strong>unanimous: accuracy (share)</strong> reads as the panel's accuracy on the
claims where every member's top pick agreed, with the share of claims that were
unanimous in brackets. Individual models show — for disagreement and the vote gate:
with one histogram there is nothing to disagree with, and total doubt equals solo doubt
equals its own entropy.
</div>
"""


def _metrics_note_calibration() -> str:
    """Definitions for the calibration + risk-coverage figures."""
    return """
<div class="note">
<strong>reliability curve</strong> — stated confidence on one axis against observed
accuracy on the other; the diagonal is perfect. Above the diagonal is under-confident,
below it is over-confident — the direction ECE alone hides.<br>
<strong>ECE</strong> — the average distance from that diagonal, weighted by how many
judgments land in each confidence bin.<br>
<strong>risk–coverage</strong> — drop the highest-disagreement judgments first, then plot
accuracy on what survives. Rising to the right means disagreement is a usable filter for
selecting clean data.
</div>
"""


def _decomposition_explainer(n_models: int) -> str:
    """Plain-language teaching block: what the per-model histograms give us, with
    a worked toy example, before any of our own numbers appear."""
    count = _nword(n_models) if n_models else "several"
    return f"""
<p>When we ask the {count} models the same claim, each returns a small bar chart —
how it splits 100 points across SUPPORT / CONTRADICT / NOINFO. So every claim
gives us <strong>{count} bar charts</strong>. There is more information in those
shapes than in just their average, and there are two genuinely different
reasons a panel might be unsure. Pulling them apart is the point of this
section.</p>

<p><strong>Two kinds of doubt.</strong> A model can be unsure <em>on its own</em>
(it answers something hedged like 40/35/25 instead of 100/0/0), and the models
can be unsure <em>relative to each other</em> (they point at different
answers). These are not the same thing, and a worked example — three judges, for
legibility; the reasoning is the same at any panel size — makes it concrete:</p>

<table>
<thead><tr><th>what three judges answered</th><th>each judge, alone</th>
<th>do they agree?</th><th>so the doubt is…</th></tr></thead>
<tbody>
<tr><td>all three say <code>40 / 35 / 25</code></td><td>all hedging</td>
<td>yes — same hedge</td><td><strong>solo</strong> doubt high, disagreement low</td></tr>
<tr><td><code>100/0/0</code>, <code>0/100/0</code>, <code>0/0/100</code></td>
<td>each certain</td><td>no — all different</td>
<td>solo doubt low, <strong>disagreement</strong> high</td></tr>
<tr><td>all three say <code>100 / 0 / 0</code></td><td>each certain</td>
<td>yes</td><td>both low — confident consensus</td></tr>
</tbody>
</table>

<p>That gives us four numbers to test as "is the panel about to be wrong?"
warning lights — three continuous, one dead-simple:</p>
<ul>
<li><strong>Solo doubt</strong> — take how spread-out each judge's own bar chart is
(its Shannon entropy: flat = unsure, spiky = confident) and <em>average that over
the judges</em>. This is exactly the original one-model signal, just read off
every judge and averaged. High when the judges are individually torn.</li>
<li><strong>Disagreement</strong> — how much the bar charts differ <em>from
each other</em> (the "mutual information", a.k.a. Jensen–Shannon divergence). Zero
if they are all identical; large when they conflict — especially the dangerous
"confidently contradicting" case. Unlike a crude spread it uses the whole shape of
every chart, not just the winning bar.</li>
<li><strong>Total doubt</strong> — mash the charts into one average chart and
measure <em>its</em> spread. Mathematically this is exactly <em>solo + disagreement</em>,
so it's the two above combined into one number.</li>
<li><strong>The vote gate</strong> — forget the shading entirely; just look at each
judge's top pick. Either they all agree (<em>unanimous</em>) or at least one
dissents (<em>split</em>). No threshold to tune.</li>
</ul>

<p>We grade each one the same honest way: <strong>does it get bigger on the
claims the panel actually got wrong?</strong> That is the "error-detection AUROC"
— 0.5 means the signal is useless noise, 1.0 means it perfectly ranks every wrong
answer above every right one, and a green star marks a signal whose 95% confidence
interval clears 0.5 (i.e. we can be reasonably sure it beats noise). None of this
needs the true answer at judging time — only to score the signals here.</p>
"""


def _individual_vs_panel(per_model: dict | None, combos: dict) -> str:
    """Honest read of the comparison the reader most wants: does combining models
    beat the best single model — on accuracy and on error detection?"""
    s = _panel_singles(combos)
    if not s or not per_model:
        return ""
    n_panel = len(_panel_members(combos))
    panel_desc = f"{_nword(n_panel)}-model panel" if n_panel else "full panel"
    singles = [r["single"] for r in per_model["results"].values()
               if r and r.get("single")]
    if not singles:
        return ""
    best_acc = max(x["accuracy"] for x in singles)
    best_auroc = max(x["auroc_entropy"]["point"] for x in singles)
    p_acc = s["accuracy"]
    p_auroc = s["auroc_total_entropy"]["point"]
    acc_word = ("does not beat" if p_acc < best_acc + 1e-9 else "beats")
    return (
        "<h3>Does combining models actually help?</h3>"
        f"<p>Comparing the individual models against the {panel_desc} on the same "
        f"axes (Fig. 3, and the table below): the best single model reaches "
        f"{best_acc:.0%} accuracy and {best_auroc:.2f} error-detection AUROC, while the "
        f"panel reaches {p_acc:.0%} and {p_auroc:.2f}. So combining <strong>{acc_word}</strong> "
        f"the best single model on raw accuracy, and its error-detection AUROC is close "
        f"enough that the confidence intervals overlap — no decisive win there either. "
        f"The panel's advantage is not a better error signal or higher accuracy; it is "
        f"that combining supplies two things a lone model cannot: a calibrated confidence "
        f"(next section) and the discrete vote gate.</p>"
    )


def _calibration_interpretation(per_model: dict | None, combos: dict) -> str:
    """Honest read of the ECE comparison — this is where combining most clearly
    helps (it rescues a poorly-calibrated single model)."""
    if not per_model:
        return ""
    singles = [(_compact(k), r["single"]["calibration"]["ece"])
               for k, _ in LADDER
               if (r := per_model["results"].get(k)) and r.get("single")]
    panels = [(name, r["singles_arm"]["calibration"]["ece"])
              for name, r in combos["results"].items() if r.get("singles_arm")]
    if not singles or not panels:
        return ""
    best_single = min(singles, key=lambda x: x[1])
    worst_single = max(singles, key=lambda x: x[1])
    best_panel = min(panels, key=lambda x: x[1])
    return (
        "<p>This is the one place combining clearly earns its keep. The single models' "
        f"calibration ranges from {best_single[1]:.2f} ({best_single[0]}) to a poor "
        f"{worst_single[1]:.2f} ({worst_single[0]}); every panel lands at or below the best "
        f"single model, with {best_panel[0]} the tightest at {best_panel[1]:.2f}. So pooling "
        "does two useful things at once: it nudges the best case a little better, and — more "
        "importantly — it <em>rescues</em> a badly-calibrated individual model, so you are no "
        "longer at the mercy of whichever single judge you happened to pick. Unlike accuracy "
        "and error-detection, where combining was a wash, calibration is a real win for the "
        "panel.</p>"
    )


def _decomposition_interpretation(combos: dict) -> str:
    """Honest, numbers-in read of the widest panel — states plainly what the data
    does and does NOT support (overlapping CIs at this error count)."""
    s = _panel_singles(combos)
    if not s:
        return ""
    n_panel = len(_panel_members(combos))
    sigs = {
        "total doubt": s["auroc_total_entropy"],
        "solo doubt": s["auroc_solo_entropy"],
        "disagreement": s["auroc_disagreement_mi"],
    }
    best = max(sigs.items(), key=lambda kv: kv[1]["point"])
    worst = min(sigs.items(), key=lambda kv: kv[1]["point"])
    vg = s["vote_gate"]
    n_wrong = s["n_wrong"]
    parts = [f"<strong>{k}</strong> {v['point']:.2f}" for k, v in sigs.items()]
    note = (
        "<h3>What the numbers actually say (honestly)</h3>"
        f"<p>On these {s['n']} claims the panel got {n_wrong} wrong, and that small number "
        f"is the catch: the three continuous signals land at { ', '.join(parts) } "
        f"(error-detection AUROC), but with only {n_wrong} wrong answers their confidence "
        f"intervals overlap heavily — so we <em>cannot</em> honestly claim any one beats "
        f"another. What we can say is the ordering of the best guesses: the strongest is "
        f"<strong>{best[0]}</strong> ({best[1]['point']:.2f}) and the weakest is "
        f"<strong>{worst[0]}</strong> ({worst[1]['point']:.2f}).</p>"
    )
    if best[0] == "solo doubt":
        note += (
            "<p>That ordering is a mild surprise and worth stating plainly: the humble "
            "<em>solo doubt</em> — each model's own hedging, averaged — is at least as good "
            "as the fancier cross-model <em>disagreement</em>. On this task the panel tends "
            "to be wrong because each judge was individually unsure, not because the judges "
            "confidently contradicted each other. So the main value of the panel here is "
            "averaging several independent reads, not harvesting their disagreement.</p>"
        )

    gate = ""
    if vg["acc_unanimous"] is not None and vg["acc_split"] is not None:
        gate = (
            "<h3>The one clearly useful lever: the vote gate</h3>"
            f"<p>When {f'all {_nword(n_panel)} models' if n_panel else 'all models'} agree "
            f"({vg['frac_unanimous']:.0%} of claims) the panel is "
            f"{vg['acc_unanimous']:.0%} accurate; when at least one dissents "
            f"({vg['frac_split']:.0%} of claims) it drops to {vg['acc_split']:.0%}. "
        )
        if vg["share_of_errors_in_split"] is not None:
            gate += (
                f"That small split bucket holds <strong>{vg['share_of_errors_in_split']:.0%} of "
                f"all the panel's errors</strong>. So a rule with no thresholds to tune — "
                f"<em>trust unanimous verdicts, send the split ones for more scrutiny</em> — "
                f"concentrates a large share of the mistakes into a small, flaggable slice."
            )
        gate += "</p>"
    return note + gate


STYLE = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Source Serif 4', 'Georgia', serif; background: #f9f7f4; color: #2b2b2b;
  line-height: 1.85; font-size: 19px; -webkit-font-smoothing: antialiased; }
.article { max-width: 860px; margin: 0 auto; padding: 60px 24px 40px; }
h1 { font-size: 32px; font-weight: 700; color: #1a1a1a; line-height: 1.25; margin: 0 0 12px; letter-spacing: -0.3px; }
h2 { font-size: 22px; font-weight: 700; color: #1a1a1a; margin: 48px 0 16px; }
h3 { font-size: 18px; font-weight: 600; color: #2b2b2b; margin: 32px 0 10px; }
p { margin: 0 0 16px; }
ul, ol { margin: 0 0 16px 24px; }
li { margin-bottom: 8px; }
hr { border: none; border-top: 1px solid #e8e4de; margin: 32px 0; }
.subtitle { font-family: 'Inter', system-ui, sans-serif; font-size: 15px; color: #888; margin-bottom: 32px; line-height: 1.6; }
table { width: 100%; border-collapse: collapse; margin: 16px 0; font-family: 'Inter', system-ui, sans-serif;
  font-size: 13px; background: #fff; border-radius: 8px; overflow: hidden; border: 1px solid #e8e4de;
  box-shadow: 0 1px 3px rgba(40,36,32,0.04); }
th { text-align: left; padding: 12px 16px; border-bottom: 2px solid #e8e4de; color: #555; font-weight: 600;
  font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; background: #faf9f6; }
td { padding: 10px 16px; border-bottom: 1px solid #f0ede8; color: #2b2b2b; vertical-align: top; line-height: 1.5; }
tr:last-child td { border-bottom: none; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.win { color: #5b7a5e; font-weight: 600; }
.note { background: #fff; border: 1px solid #e8e4de; border-left: 4px solid #b8b3aa; border-radius: 6px;
  padding: 16px 20px; margin: 16px 0; font-size: 15px; line-height: 1.7; box-shadow: 0 1px 3px rgba(40,36,32,0.04); }
.note.accent { border-left-color: #5b7a5e; }
.note.warn { border-left-color: #c08a5a; }
.note strong { color: #1a1a1a; }
.note code, p code, td code, li code { background: #f0ede8; padding: 1px 5px; border-radius: 3px; font-size: 13px; color: #555; }
.formula { background: #faf9f6; border: 1px solid #e8e4de; border-radius: 6px; padding: 12px 20px; margin: 12px 0;
  font-family: 'Courier New', monospace; font-size: 15px; color: #2b2b2b; text-align: center; }
.figure { background: #fff; border: 1px solid #e8e4de; border-radius: 8px; padding: 20px; margin: 20px 0;
  box-shadow: 0 1px 3px rgba(40,36,32,0.04); }
.figure img { width: 100%; height: auto; display: block; border-radius: 4px; }
.figure figcaption { font-family: 'Inter', system-ui, sans-serif; font-size: 13px; color: #777; margin-top: 12px; line-height: 1.6; }
.figure figcaption strong { color: #2b2b2b; }
.meta-stamp { font-family: 'Inter', system-ui, sans-serif; font-size: 12px; color: #aaa; margin-top: 48px;
  padding-top: 16px; border-top: 1px solid #e8e4de; }
.meta-stamp code { background: #f0ede8; color: #555; padding: 1px 5px; border-radius: 3px; font-size: 11px; }
"""


def build() -> None:
    per_model = _load(RESULTS)
    combos = _load(RESULTS_COMBOS)

    if per_model is None and combos is None:
        print("No results yet. Run gather + analyze + combos first, then rebuild.")

    FIGS.mkdir(parents=True, exist_ok=True)
    drew = {}
    if per_model:
        drew["ladder"] = fig_ladder(per_model)
        drew["single_vs_temp"] = fig_single_vs_temp(per_model)
    if combos:
        drew["panels"] = fig_panels(combos, per_model)
        drew["calibration"] = fig_calibration(combos, per_model)
        drew["risk_coverage"] = fig_risk_coverage(combos)

    n = (per_model or combos or {}).get("config", {}).get("n_claims", "?")

    # Which arms actually have data — panel-only (--k 0) runs have no
    # temperature/mixture arm, so those sections/columns are omitted entirely
    # rather than rendered as broken "figure not found" boxes.
    has_temp = bool(per_model) and any(
        r and r.get("temperature") for r in per_model["results"].values()
    )
    models_line = ", ".join(
        _short(k) for k, _ in LADDER
        if (per_model and per_model["results"].get(k))
    ) or "gemma4:31b, gpt-oss:20b, qwen3.6:35b, qwen3.8:27b, muse-glimmer:30b, ornith-1.5:35b"

    # Sections are assembled in a list and numbered dynamically, so dropping the
    # single-vs-temperature section under --k 0 doesn't leave a gap in the count.
    sections: list[str] = []

    def add(title: str, body: str, anchor: str = "") -> None:
        idx = len(sections) + 2  # "1 · The method" is fixed above
        a = f' id="{anchor}"' if anchor else ""
        sections.append(f'<h2{a}>{idx} · {title}</h2>\n{body}')

    if per_model:
        body = (
            f"<p>Each model judged {n} balanced SciFact claims once at temperature 0. "
            "The detector is the Shannon entropy of that single 0–100 histogram: high "
            "entropy should mark the judgments the model got <em>wrong</em>. AUROC 0.5 = "
            "the entropy is noise; a CI clear of 0.5 (green) = a usable warning light.</p>"
            + figure_html(FIGS / "mm_fig1_ladder.png",
                "<strong>Fig 1.</strong> Left: single-call entropy AUROC per model, with "
                "95% bootstrap CIs. Right: one-hot degeneracy (fraction of histograms with "
                "a class ≥ 0.95) against accuracy — a model that never hedges cannot carry "
                "an entropy signal.")
            + _metrics_note_single()
            + per_model_table(per_model, has_temp)
        )
        add("Each model on its own — is its entropy a wrong-answer detector?", body, "single")

        if has_temp and drew.get("single_vs_temp"):
            body = (
                "<p>The practically attractive claim is that a single deterministic call is "
                "as good as pooling several sampled ones — no per-site prompt engineering. "
                "If the S1 and S2 bars overlap within their CIs, the cheap path wins.</p>"
                + figure_html(FIGS / "mm_fig2_single_vs_temp.png",
                    "<strong>Fig 2.</strong> Single-call entropy (S1) vs the temperature "
                    "band (S2), per model.")
            )
            add("One call or several? (H2)", body, "single-vs-temp")

    if combos:
        n_panel = len(_panel_members(combos))
        n_singles = sum(1 for k, _ in LADDER
                        if per_model and per_model["results"].get(k))
        n_panels = len(combos["results"])
        body = (
            _decomposition_explainer(n_panel)
            + figure_html(FIGS / "mm_fig3_panels.png",
                "<strong>Fig 3.</strong> <strong>Left:</strong> error-detection AUROC of the "
                "entropy of each configuration's final answer — the "
                f"{_nword(n_singles) if n_singles else ''} individual models then the "
                f"{_nword(n_panels) if n_panels else ''} panels — with 95% confidence "
                "intervals; the dashed line marks "
                "chance (0.5). This is the one signal defined identically for a single model "
                "(its own histogram) and a panel (the pooled histogram), so it is a fair "
                "individual-vs-combined comparison. <strong>Right:</strong> the vote gate — "
                "accuracy when the models agree vs when at least one dissents, with the share "
                "of claims in each bucket (defined only for panels).")
            + _metrics_note_panel()
            + comparison_table(per_model, combos)
            + _individual_vs_panel(per_model, combos)
            + _decomposition_interpretation(combos)
        )
        add("Reading the histograms: individual models vs a panel", body, "decomp")

        body = (
            "<p>For a training signal, error-detection is not enough — the confidence has to "
            "be <em>calibrated</em>: when a judge says 0.8, it must be right about 80% of "
            "the time, or every derived soft-label / reward is biased. Below, the individual "
            "models and the panels are shown on the same axes, so we can see whether combining "
            "improves calibration.</p>"
            + _metrics_note_calibration()
            + figure_html(FIGS / "mm_fig4_calibration.png",
                "<strong>Fig 4.</strong> <strong>Left:</strong> reliability curves — stated "
                "confidence vs empirical accuracy; the diagonal is perfect. Dashed lines are "
                "individual models, solid lines are panels (ECE in parentheses). "
                "<strong>Right:</strong> expected calibration error for every configuration, "
                "individual models then panels.")
            + _calibration_interpretation(per_model, combos)
            + "<h3>The filter view</h3><p>If we keep only the panel judgments where the "
            "members agreed (lowest disagreement first), how clean is the retained set? "
            "That retained set is the training data one would actually harvest.</p>"
            + figure_html(FIGS / "mm_fig5_risk_coverage.png",
                "<strong>Fig 5.</strong> Accuracy on the kept fraction as we drop the "
                "highest-disagreement panel judgments first. Rising-to-the-right = "
                "disagreement is a usable gate for selecting clean training pairs.")
        )
        add("Is the confidence a usable training signal? (calibration)", body, "calibration")

    reading_idx = len(sections) + 2
    body_sections = "\n".join(sections)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Local-model confidence as a training signal</title>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,400;0,600;0,700;1,400&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>{STYLE}</style>
</head>
<body>
<div class="article">
  <h1>Can a panel of local models be a calibrated judge?</h1>
  <p class="subtitle">Multi-model confidence revisit — {models_line}<br>
  SciFact evidence judgment · {n} balanced claims · engineered reasoning-first schema · all local, zero cloud spend</p>

  <p><strong>The question.</strong> Can these local models — alone or combined — score a
  judgment reliably enough, and report a confidence honest enough, that the confidence
  itself becomes a training signal? We test the mechanism on a labelled proxy
  (SciFact SUPPORT / CONTRADICT / NOINFO) where calibration can actually be checked.</p>

  <h2 id="method">1 · The method in one screen</h2>
  <p>Every model returns a full 0–100 split over the three classes, not a label. Two
  uncertainty signals fall out:</p>
  <div class="formula">single-model: entropy of one histogram &nbsp;·&nbsp; panel: std of the members' probability for the winning class</div>
  <p>Each is graded by <strong>error-detection AUROC</strong> (does the signal rank wrong
  judgments above right ones — no ground truth needed at inference) and, for the panel's
  soft-label value, by <strong>ECE + a reliability curve</strong> (is the number calibrated).
  A low signal never <em>proves</em> correctness; it only <em>fails to raise</em> doubt.</p>

  <h3>Two questions, kept apart</h3>
  <p><strong>Accuracy</strong> asks whether the judgment was right. <strong>Error-detection
  AUROC</strong> asks whether the judge's doubt ran higher on the ones it got wrong. Those are
  different properties and they need not move together: a model can be accurate but oblivious
  (just as certain when it is wrong), or mediocre but self-aware (hedging on precisely its own
  mistakes). Only the second is usable as a gate — it is what decides which judgments you can
  keep <em>without checking them</em>.</p>
  <div class="note accent"><strong>How to read any AUROC here.</strong> Pick one judgment the
  system got wrong and one it got right, at random: the AUROC is the probability that the wrong
  one carries the higher doubt score. <strong>0.5</strong> means the doubt score is noise;
  <strong>1.0</strong> means every wrong answer is doubted more than every right one. It grades
  the <em>ranking</em>, so there is no threshold to tune, and a detector that never fires cannot
  score well by hiding in the majority class. A <span class="win">★</span> marks a 95% bootstrap
  interval clear of 0.5; overlapping intervals mean a difference is not established, however far
  apart the point estimates look — with this few wrong answers, that is the common case.</div>
  <p class="subtitle" style="margin-bottom:0">Every other metric is defined in the section that
  uses it, immediately above the table it appears in. No outside reading required.</p>

  {body_sections}

  <h2 id="reading">{reading_idx} · How to read this for a judge</h2>
  <h3>The decision this informs</h3>
  <p>If a cross-family panel's disagreement clears chance AND its pooled confidence is
  calibrated (low ECE, reliability on the diagonal), then: pool the panel for the label,
  keep the low-disagreement fraction as clean training pairs, and weight by the calibrated
  confidence. If only error-detection holds but calibration is poor, the disagreement is
  still usable as a <em>gate</em> (keep / discard) but the confidence is not usable as a
  <em>weight</em> without a post-hoc calibration step.</p>
  <h3>The load-bearing caveat</h3>
  <p>SciFact evidence judgment is a labelled <em>proxy</em>. It validates the mechanism and
  lets us measure calibration honestly — but scoring real agentic-system output is a
  different distribution. Before deriving a training signal there, this same calibration
  check must be repeated on a small hand-labelled set of actual agentic outputs. A judge
  calibrated here is not automatically calibrated there.</p>
  <h3>Why these metrics and not the usual ones</h3>
  <p>Brier score and log-likelihood are computed and stored but never led with: both fold
  accuracy and calibration into a single number, and how those two come apart is the entire
  question here. Precision, recall and F1 are absent for a related reason — they need a fixed
  alarm threshold, and choosing one would bake in the deployment decision this experiment
  exists to inform.</p>

  <div class="meta-stamp">
  Generated by <code>multimodel_report.py</code> from <code>results/multimodel.json</code> +
  <code>results/multimodel_combos.json</code>. Regenerate the JSONs with
  <code>multimodel_experiment.py</code>, then rebuild. Design: <code>MULTIMODEL_PLAN.md</code>.
  </div>
</div>
</body>
</html>"""

    OUT.write_text(html)
    print(f"wrote {OUT}  (figures drawn: {', '.join(k for k, v in drew.items() if v) or 'none'})")


if __name__ == "__main__":
    build()
