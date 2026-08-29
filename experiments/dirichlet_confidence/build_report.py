#!/usr/bin/env python3
"""Build report.html — a styled, self-contained handover document.

Collects the dirichlet_confidence reliability experiments into one shareable
file matching the editorial style of experiments/interaction_finder/disco_flow.html
(Source Serif 4 body, Inter tables, warm palette). Figures are base64-embedded so
the output is a single file you can email or drop in a shared drive.

All numbers below are the final results from the committed runs:
  • figures/                  — fig1..fig5 (make_figures.py)
  • results/baselines.json    — compare_baselines.py
  • results/ablation_1v5.json — ablation_1v5.py
  • console output            — call_variability.py
They are transcribed here so the document is self-contained; re-run the source
scripts to regenerate, then rebuild this report.

    uv run python -m experiments.dirichlet_confidence.build_report
"""

from __future__ import annotations

import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
OUT = HERE / "report.html"


def img(path: Path) -> str:
    if not path.exists():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def figure(path: Path, caption_html: str) -> str:
    uri = img(path)
    if not uri:
        return (f'<div class="note">Figure <code>{path.name}</code> not found — '
                f'regenerate with <code>make_figures.py</code>, then rebuild.</div>')
    return (f'<figure class="figure">\n  <img src="{uri}" alt="{path.stem}">\n'
            f'  <figcaption>{caption_html}</figcaption>\n</figure>')


# CSS copied verbatim from experiments/interaction_finder/build_disco_flow.py so
# the two documents are visually identical.
STYLE = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Source Serif 4', 'Georgia', 'Times New Roman', serif;
  background: #f9f7f4; color: #2b2b2b; line-height: 1.85;
  font-size: 19px; -webkit-font-smoothing: antialiased;
}
.article { max-width: 860px; margin: 0 auto; padding: 60px 24px 40px; }
h1 { font-family: 'Source Serif 4', Georgia, serif; font-size: 32px; font-weight: 700;
     color: #1a1a1a; line-height: 1.25; margin: 0 0 12px; letter-spacing: -0.3px; }
h2 { font-family: 'Source Serif 4', Georgia, serif; font-size: 22px; font-weight: 700;
     color: #1a1a1a; margin: 48px 0 16px; }
h3 { font-family: 'Source Serif 4', Georgia, serif; font-size: 18px; font-weight: 600;
     color: #2b2b2b; margin: 32px 0 10px; }
p { margin: 0 0 16px; }
ul, ol { margin: 0 0 16px 24px; }
li { margin-bottom: 8px; }
hr { border: none; border-top: 1px solid #e8e4de; margin: 32px 0; }
.subtitle { font-family: 'Inter', system-ui, sans-serif; font-size: 15px; color: #888;
            margin-bottom: 32px; line-height: 1.6; }
table { width: 100%; border-collapse: collapse; margin: 16px 0;
  font-family: 'Inter', system-ui, sans-serif; font-size: 13px; background: #fff;
  border-radius: 8px; overflow: hidden; border: 1px solid #e8e4de;
  box-shadow: 0 1px 3px rgba(40,36,32,0.04); }
th { text-align: left; padding: 12px 16px; border-bottom: 2px solid #e8e4de; color: #555;
  font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; background: #faf9f6; }
td { padding: 10px 16px; border-bottom: 1px solid #f0ede8; color: #2b2b2b; vertical-align: top; line-height: 1.5; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: rgba(0,0,0,0.01); }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.win { color: #5b7a5e; font-weight: 600; }
.lose { color: #a8604f; font-weight: 600; }
.tie { color: #8a8378; }
.note { background: #fff; border: 1px solid #e8e4de; border-left: 4px solid #b8b3aa;
  border-radius: 6px; padding: 16px 20px; margin: 16px 0; font-size: 15px; line-height: 1.7;
  box-shadow: 0 1px 3px rgba(40,36,32,0.04); }
.note.accent { border-left-color: #5b7a5e; }
.note.warn { border-left-color: #c08a5a; }
.note strong { color: #1a1a1a; }
.note code, p code, td code, li code { background: #f0ede8; padding: 1px 5px;
  border-radius: 3px; font-size: 13px; color: #555; }
.formula { background: #faf9f6; border: 1px solid #e8e4de; border-radius: 6px;
  padding: 12px 20px; margin: 12px 0; font-family: 'Courier New', monospace;
  font-size: 15px; color: #2b2b2b; text-align: center; }
.figure { background: #fff; border: 1px solid #e8e4de; border-radius: 8px; padding: 20px;
  margin: 20px 0; box-shadow: 0 1px 3px rgba(40,36,32,0.04); }
.figure img { width: 100%; height: auto; display: block; border-radius: 4px; }
.figure figcaption { font-family: 'Inter', system-ui, sans-serif; font-size: 13px; color: #777;
  margin-top: 12px; line-height: 1.6; }
.figure figcaption strong { color: #2b2b2b; }
.toc { background: #fff; border: 1px solid #e8e4de; border-radius: 8px; padding: 20px 24px;
  margin: 24px 0 40px; font-family: 'Inter', system-ui, sans-serif; font-size: 14px;
  box-shadow: 0 1px 3px rgba(40,36,32,0.04); }
.toc ol { margin: 0 0 0 24px; }
.toc li { margin: 4px 0; color: #555; }
.toc a { color: #2b2b2b; text-decoration: none; border-bottom: 1px dotted #b8b3aa; }
.toc a:hover { color: #6d8f70; border-bottom-color: #6d8f70; }
.references { font-family: 'Inter', system-ui, sans-serif; font-size: 13px; color: #555; line-height: 1.6; }
.references ol { margin: 0 0 0 20px; }
.references li { margin-bottom: 10px; }
.references em { color: #2b2b2b; font-style: italic; }
.references a { color: #5b7a5e; text-decoration: none; border-bottom: 1px dotted #b8b3aa; }
.references a:hover { border-bottom-color: #5b7a5e; }
.meta-stamp { font-family: 'Inter', system-ui, sans-serif; font-size: 12px; color: #aaa;
  margin-top: 48px; padding-top: 16px; border-top: 1px solid #e8e4de; }
.meta-stamp code { background: #f0ede8; color: #555; padding: 1px 5px; border-radius: 3px; font-size: 11px; }
"""


# ---------------------------------------------------------------------------
# Result tables (transcribed from the committed runs — see module docstring)
# ---------------------------------------------------------------------------

def scoreboard_table() -> str:
    # AUROC point estimates; CIs ≈ ±0.08–0.11. best-in-row marked .win
    rows = [
        ("nano · phrasing", "0.559", ("0.734", True), "0.726", "0.566", "0.575"),
        ("nano · temperature", "0.554", "0.640", ("0.706", True), "0.591", "0.570"),
        ("mini · phrasing", "0.707", "0.682", ("0.788", True), "0.771", "0.760"),
        ("mini · temperature", "0.547", ("0.772", True), "0.720", "0.665", "0.581"),
        ("4o-mini · phrasing", "0.616", "0.697", ("0.767", True), "0.629", "0.629"),
        ("4o-mini · temperature", "0.684", "0.702", ("0.795", True), "0.735", "0.707"),
    ]
    body = []
    for cond, *cells in rows:
        tds = [f"<td>{cond}</td>"]
        for c in cells:
            if isinstance(c, tuple):
                tds.append(f'<td class="num win">{c[0]}</td>')
            else:
                tds.append(f'<td class="num">{c}</td>')
        body.append("<tr>" + "".join(tds) + "</tr>")
    return "\n".join(body)


def ablation_table() -> str:
    rows = [
        ("predictive entropy − 1&#8209;call entropy", "does averaging 5 calls beat 1?", [
            ("nano · phr", "&minus;0.008", "[&minus;0.057, +0.044]", "tie"),
            ("nano · tmp", "+0.066", "[+0.001, +0.138]", "5&#8209;call better"),
            ("mini · phr", "+0.106", "[+0.022, +0.196]", "5&#8209;call better"),
            ("mini · tmp", "&minus;0.052", "[&minus;0.097, &minus;0.011]", "1&#8209;call better"),
            ("4o · phr", "+0.071", "[+0.007, +0.132]", "5&#8209;call better"),
            ("4o · tmp", "+0.094", "[+0.029, +0.165]", "5&#8209;call better"),
        ]),
        ("mutual information − 1&#8209;call entropy", "does pure disagreement add over 1 call?", [
            ("nano · phr", "&minus;0.158", "[&minus;0.290, &minus;0.036]", "1&#8209;call better"),
            ("nano · tmp", "&minus;0.070", "[&minus;0.199, +0.071]", "tie"),
            ("mini · phr", "+0.079", "[&minus;0.045, +0.197]", "tie"),
            ("mini · tmp", "&minus;0.191", "[&minus;0.289, &minus;0.097]", "1&#8209;call better"),
            ("4o · phr", "&minus;0.068", "[&minus;0.208, +0.078]", "tie"),
            ("4o · tmp", "+0.005", "[&minus;0.117, +0.125]", "tie"),
        ]),
    ]
    out = []
    for title, q, sub in rows:
        out.append(f'<tr><td colspan="4"><strong>{title}</strong> — <em>{q}</em></td></tr>')
        for cond, delta, ci, verdict in sub:
            cls = ("win" if "5&#8209;call" in verdict else
                   "lose" if "1&#8209;call" in verdict else "tie")
            out.append(f'<tr><td style="padding-left:28px">{cond}</td>'
                       f'<td class="num">{delta}</td><td class="num">{ci}</td>'
                       f'<td class="{cls}">{verdict}</td></tr>')
    return "\n".join(out)


def variability_table() -> str:
    rows = [
        ("nano", "phrasing", "0.111", "79%"), ("nano", "temperature", "0.133", "75%"),
        ("mini", "phrasing", "0.180", "59%"), ("mini", "temperature", "0.074", "87%"),
        ("4o-mini", "phrasing", "0.082", "85%"), ("4o-mini", "temperature", "0.099", "82%"),
    ]
    return "\n".join(
        f"<tr><td>{m}</td><td>{a}</td><td class='num'>{tv}</td><td class='num'>{ag}</td></tr>"
        for m, a, tv, ag in rows)


def residual_table() -> str:
    # flip-error rate among kept claims at 50% coverage, 1-call vs 5-call gate.
    rows = [
        ("nano", "2.9%", "0.0%", "2.9%", "13.0% → 11.6%"),
        ("mini", "15.3%", "0.0%", "0.0%", "1.4% → 9.7%"),
        ("4o-mini", "4.2%", "0.0%", "6.9%", "13.9% → 12.5%"),
    ]
    return "\n".join(
        f"<tr><td>{m}</td><td class='num lose'>{p1}</td><td class='num win'>{p5}</td>"
        f"<td class='num'>{t1}</td><td class='num'>{st}</td></tr>"
        for m, p1, p5, t1, st in rows)


HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Confidence Reliability — Experiment Report</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400&display=swap" rel="stylesheet">
<style>{STYLE}</style>
</head>
<body>
<div class="article">

<h1>Can an LLM tell us when it is wrong?</h1>
<p class="subtitle">A handover report on the <code>dirichlet_confidence</code> experiments — whether sampling a model multiple times and measuring its self-disagreement gives a reliable signal of when its answer is wrong, and how that compares to simply reading the model's stated confidence once. Built for a PhD project at the intersection of reliability, reproducibility and observability of agentic AI. All figures and numbers come from committed runs over 150 SciFact claims; no model was queried to build this document.</p>

<div class="toc">
<strong>Contents</strong>
<ol>
  <li><a href="#question">The question</a></li>
  <li><a href="#setup">Setup: the data and what we elicited</a></li>
  <li><a href="#map">The experiments at a glance</a></li>
  <li><a href="#exp1">Experiment 1 — Does self-disagreement flag wrong answers?</a></li>
  <li><a href="#exp2">Experiment 2 — The band vs. standard baselines</a></li>
  <li><a href="#exp3">Experiment 3 — One call vs. five calls</a></li>
  <li><a href="#exp4">Experiment 4 — How much do the five calls actually vary?</a></li>
  <li><a href="#exp5">Experiment 5 — What kind of error survives the gate?</a></li>
  <li><a href="#conclusions">Conclusions</a></li>
  <li><a href="#novelty">Novelty &amp; prior work</a></li>
  <li><a href="#next">Recommended next steps</a></li>
  <li><a href="#files">File map &amp; how to reproduce</a></li>
  <li><a href="#references">References</a></li>
</ol>
</div>

<h2 id="question">1. The question</h2>
<p>Agentic pipelines lean on a hopeful assumption: that a model knows when it is unsure, so we can <em>abstain</em> or <em>escalate</em> on the shaky answers and trust the rest. The original idea behind these experiments was richer — ask the same question several different ways, treat each answer as a probability distribution over the options, and pool them as a <strong>mixture of Dirichlet distributions</strong>. The hope was that the <em>spread</em> between those answers (how much the model disagrees with itself, which we call the <strong>band</strong>) would be a strong, principled detector of wrong answers, and a route toward more reproducible AI.</p>
<p>We set out to test that on a clean, labelled proxy for the most load-bearing micro-decision an agent makes: <strong>judging a claim against evidence</strong>. The headline result is not the one we expected — and the honest negative result, plus the one targeted finding that survived it, is what this document is for.</p>

<h2 id="setup">2. Setup: the data and what we elicited</h2>
<p>The task is <strong>SciFact</strong> scientific-claim verification: given a claim and an abstract, decide <code>SUPPORT</code> / <code>CONTRADICT</code> / <code>NO&nbsp;INFO</code>, scored against gold labels. We used <strong>150 balanced claims</strong> and <strong>three models</strong> spanning sizes: <code>gpt-5.4-nano</code> (smallest), <code>gpt-5.4-mini</code>, and <code>gpt-4o-mini</code>.</p>
<p>Every model call returns a small <strong>histogram</strong> — 100 points split over the three options, e.g. <code>[95, 0, 5]</code> — i.e. the model's own verbalized confidence. We collected <strong>five histograms per claim</strong>, in two "arms" that create disagreement differently:</p>
<ul>
  <li><strong>Phrasing arm</strong> — the same question reworded five ways, at temperature 0 (deterministic). Variation comes only from <em>how you ask</em>.</li>
  <li><strong>Temperature arm</strong> — one fixed wording, sampled five times at temperature 0.7. Variation comes only from <em>sampling randomness</em>.</li>
</ul>
<p>From the five histograms per claim we compute several candidate uncertainty scores. The two key entropy-based ones:</p>
<div class="formula">verbalized entropy (1 call) = H(one histogram) &nbsp;·&nbsp; predictive entropy (5 calls) = H(mean of the five)</div>
<p>where <code>H</code> is Shannon entropy (0 = all points on one option, certain; high = points spread evenly, unsure). The <strong>band</strong> is the standard deviation, across the five histograms, of the winning option's probability; <strong>mutual information</strong> (the principled "disagreement" term, a.k.a. BALD) is predictive entropy minus the average single-call entropy. Every score is an <em>alarm score</em>: higher should mean "more likely wrong". We grade them by <strong>error-detection AUROC</strong> (0.5 = useless, 1.0 = perfect ranking of wrong above right) and by <strong>risk–coverage</strong> (accuracy on the answers you keep as you discard the most-uncertain ones).</p>

<h2 id="map">3. The experiments at a glance</h2>
<table>
<thead><tr><th>#</th><th>Experiment</th><th>Question</th><th>One-line answer</th></tr></thead>
<tbody>
<tr><td>1</td><td>Band as detector</td><td>Does self-disagreement flag wrong answers?</td><td>Weakly, and only for mid-size models.</td></tr>
<tr><td>2</td><td>Band vs. baselines</td><td>Is the band better than standard scores?</td><td><span class="lose">No</span> — plain predictive entropy ties or beats it everywhere.</td></tr>
<tr><td>3</td><td>1 call vs. 5 calls</td><td>Do you even need to ask five times?</td><td>Mostly no; the disagreement term adds nothing.</td></tr>
<tr><td>4</td><td>Call variability</td><td>Do the five calls actually differ?</td><td>Yes, modestly; rewording &gt; resampling for the brittle model.</td></tr>
<tr><td>5</td><td>Residual composition</td><td>What errors survive a confidence gate?</td><td><span class="win">Confident flips</span> — invisible to 1 call, removed by 5.</td></tr>
</tbody>
</table>

<h2 id="exp1">4. Experiment 1 — Does self-disagreement flag wrong answers?</h2>
<p>We first asked whether the band detects errors at all, with bootstrap confidence intervals over the 150 claims. It does — but the effect is modest and strongly model-dependent: useful for the two mid-size models (AUROC 0.63–0.77), near the coin-flip line for the smallest.</p>
{figure(FIGS / "fig1_band_error_detection_scorecard.png",
        "<strong>Does a model's self-disagreement flag its wrong answers?</strong> Error-detection AUROC for the band, "
        "per model and per way of creating disagreement. 0.50 = the band tells you nothing; the shaded floor at 0.60 is "
        "a rule-of-thumb for a usable warning light. The band clears the bar for the mid-size models but not for nano.")}
{figure(FIGS / "fig2_band_separation_right_vs_wrong.png",
        "<strong>Why it works (when it does): wrong answers carry a bigger band.</strong> Each dot is one claim; height is "
        "how much the five re-asks disagreed. Red = the model's answer was wrong, blue = right. The red boxes sit higher — "
        "the model wobbles more when it is about to be wrong — most clearly for the mid-size models.")}
{figure(FIGS / "fig3_risk_coverage.png",
        "<strong>The payoff: discard the answers the model disagreed with itself on.</strong> Walking from keep-everything "
        "(right) leftward, dropping the highest-band claims first; accuracy on what you keep climbs for the mid-size models, "
        "stays flat for nano. This is the band used as an abstention gate.")}
<p>So the band is a real but weak signal that fails entirely on the smallest model — a model that is <em>confidently wrong</em> never disagrees with itself. That capability dependence motivated everything that followed.</p>

<h2 id="exp2">5. Experiment 2 — The band vs. standard baselines</h2>
<p>The decisive test: put the band against the established sampling-based uncertainty scores on the same claims, with paired bootstrap comparisons. We compared <strong>self-consistency / semantic entropy</strong> (vote-based — on a closed 3-class task semantic entropy collapses to vote entropy), <strong>verbalized entropy</strong> from a single call, <strong>predictive entropy</strong> from five calls, the <strong>band</strong>, and <strong>mutual information</strong>.</p>
<table>
<thead><tr><th>Model · arm</th><th class="num">vote entropy</th><th class="num">1&#8209;call entropy</th><th class="num">predictive entropy</th><th class="num">band (ours)</th><th class="num">mutual info</th></tr></thead>
<tbody>
{scoreboard_table()}
</tbody>
</table>
<p style="font-family:'Inter',sans-serif;font-size:13px;color:#777">Error-detection AUROC (point estimates; 95% CIs ≈ ±0.08–0.11). Best in each row in green.</p>
<div class="note warn">
<strong>The band did not win.</strong> Plain <em>predictive entropy</em> — the most basic thing you can do (average the five histograms, measure flatness) — is as good as or better than the band in all six conditions, and significantly better for the two weaker models. The band never significantly beats it. The vote-based scores (self-consistency, semantic entropy) are weakest of all: throwing away the within-histogram probabilities and keeping only the argmax costs you.
</div>
{figure(FIGS / "fig4_payoff_by_score.png",
        "<strong>Payoff curves by score.</strong> Same prediction for every line, so all start together and a line that climbs "
        "faster is a better error-detector. Solid = confidence-based scores (incl. a single call); dashed = disagreement-based "
        "(band, mutual information). The confidence-based lines rise highest; the disagreement lines lag.")}
<p>The lesson: the abstention signal lives in the model's verbalized confidence, and that is readable in one call. Repeating the question and measuring disagreement buys little.</p>

<h2 id="exp3">6. Experiment 3 — One call vs. five calls</h2>
<p>If averaging verbalized confidence is what wins, do you even need five calls? We compared a single call's entropy against the five-call predictive entropy and the pure-disagreement mutual-information term, paired-bootstrap.</p>
<table>
<thead><tr><th>Comparison · condition</th><th class="num">&Delta;AUROC</th><th class="num">95% CI</th><th>verdict</th></tr></thead>
<tbody>
{ablation_table()}
</tbody>
</table>
<div class="note">
<strong>Two findings.</strong> (1) Averaging five calls beats one call only by a small, inconsistent margin (significant in 4/6 conditions, ~+0.07–0.11, and it <em>reversed</em> once). (2) The pure disagreement term (mutual information) <strong>never</strong> beats a single call and is significantly <em>worse</em> twice. So the premise — that cross-call disagreement flags errors — is not supported on aggregate metrics.
</div>

<h2 id="exp4">7. Experiment 4 — How much do the five calls actually vary?</h2>
<p>A natural worry: maybe disagreement adds nothing because the five calls are near-identical. We measured it. TV distance = the fraction of the 100 points that moves between two calls (0 = identical); "agree" = the share of claims where all five calls pick the same top answer.</p>
<table>
<thead><tr><th>Model</th><th>Arm</th><th class="num">TV distance</th><th class="num">all 5 agree</th></tr></thead>
<tbody>
{variability_table()}
</tbody>
</table>
<p>The calls genuinely differ — 7–18% of the probability mass shifts between calls — but most claims (75–87%) keep the same top answer. The wobble is mostly in the <em>confidence</em>, not the verdict. Note the asymmetry: <code>mini</code> is far more sensitive to <em>rewording</em> (TV 0.180, only 59% agree) than to <em>resampling</em> (TV 0.074, 87% agree). That phrasing-vs-sampling split is the well-known epistemic-vs-aleatoric decomposition.</p>
<div class="note">
<strong>The vivid case</strong> (mini, phrasing arm, gold = CONTRADICT — the same claim, reworded five ways):
<div class="formula" style="text-align:left">wording 0:&nbsp; [&nbsp;&nbsp;0, 100,&nbsp;&nbsp;0]&nbsp;&nbsp;→ confident CONTRADICT&nbsp;&nbsp;(correct)
wording 1:&nbsp; [&nbsp;96,&nbsp;&nbsp;&nbsp;1,&nbsp;&nbsp;3]&nbsp;&nbsp;→ confident SUPPORT&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(WRONG)
wording 2:&nbsp; [&nbsp;95,&nbsp;&nbsp;&nbsp;2,&nbsp;&nbsp;3]&nbsp;&nbsp;→ confident SUPPORT&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(WRONG)
wording 3:&nbsp; [&nbsp;&nbsp;4,&nbsp;&nbsp;94,&nbsp;&nbsp;2]&nbsp;&nbsp;→ confident CONTRADICT&nbsp;&nbsp;(correct)
wording 4:&nbsp; [&nbsp;95,&nbsp;&nbsp;&nbsp;0,&nbsp;&nbsp;5]&nbsp;&nbsp;→ confident SUPPORT&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(WRONG)</div>
Merely rewording flips the model between confidently right and confidently wrong. Each individual call <em>looks</em> confident — they just disagree with each other. This is the failure mode Experiment 5 is built around.
</div>

<h2 id="exp5">8. Experiment 5 — What kind of error survives the gate?</h2>
<p>This is the one place where the multi-call idea earns its keep. We split errors by behaviour under rewording: <strong>answer-flip</strong> errors (the top answer changes across wordings, yet each call is confident) versus <strong>stable</strong> errors (consistently wrong). Then, among the claims a confidence gate <em>keeps</em>, we measure the leftover errors under two gates: a <strong>1-call gate</strong> (single-histogram entropy) and a <strong>5-call gate</strong> (predictive entropy).</p>
<p>The prediction: a 1-call gate is <em>structurally blind</em> to confident flips (each call looks sure), while a 5-call gate is not (averaging disagreeing-but-confident answers raises entropy). The data confirm it — flip-errors among kept claims at 50% coverage:</p>
<table>
<thead><tr><th>Model</th><th class="num">flip-errors · 1&#8209;call gate</th><th class="num">flip-errors · 5&#8209;call gate</th><th class="num">flip-errors · 1&#8209;call (temperature arm)</th><th class="num">stable-errors (1→5 call)</th></tr></thead>
<tbody>
{residual_table()}
</tbody>
</table>
{figure(FIGS / "fig5_residual_phrasing.png",
        "<strong>Residual error composition (phrasing arm).</strong> Among kept claims, the leftover errors split into "
        "answer-flip (red) and stable (blue), under a 1-call gate (solid) and a 5-call predictive-entropy gate (dashed). "
        "The red solid line stays elevated while the red dashed collapses to zero — confident flips survive a single call "
        "but are removed by five. Clearest for mini, the most phrasing-brittle model.")}
{figure(FIGS / "fig5_residual_temperature.png",
        "<strong>Control: the temperature arm.</strong> Same analysis, but flips now come from resampling, not rewording. For "
        "mini the effect vanishes (resampling produces no confident flips) — confirming its residual is a <em>paraphrase</em> "
        "phenomenon. For nano and 4o-mini the picture is mixed: they are brittle to both axes.")}
<div class="note accent">
<strong>The surviving result.</strong> A 5-call gate drives confident-flip errors to zero where a 1-call gate leaves them (15.3% for mini). It does <em>not</em> reduce total error — it <strong>reallocates</strong> it, trading silent, irreproducible paraphrase-flips for stable, reproducible errors (mini's stable errors rise 1.4% → 9.7%). For reproducible / auditable AI you may prefer that residual even at equal error rate: a consistently-wrong answer is at least repeatable; a paraphrase-dependent flip is a silent landmine.
</div>

<h2 id="conclusions">9. Conclusions</h2>
<ol>
  <li><strong>The abstention signal is the model's verbalized confidence, readable in one call.</strong> A single histogram's entropy rivals or beats every multi-call method we tried (self-consistency, semantic entropy, the Dirichlet band, mutual information), at 1/5 the cost.</li>
  <li><strong>The disagreement band is not a better detector.</strong> On aggregate AUROC it ties or loses to plain predictive entropy everywhere, and the pure disagreement term adds nothing over a single call. The original mixture-of-Dirichlets hypothesis is not supported on this task.</li>
  <li><strong>There is a capability floor.</strong> Every self-knowledge signal degrades on the smallest model, and disagreement degrades fastest — a confidently-wrong model does not disagree with itself.</li>
  <li><strong>But disagreement catches a distinct, dangerous residual.</strong> Confident answer-flips under rewording are invisible to single-call confidence and removed by a multi-call predictive-entropy gate. Reframed as a <em>reproducibility</em> rather than accuracy concern, this is the one place the extra calls are worth paying for.</li>
</ol>
<div class="note warn">
<strong>Caveats, loudly.</strong> Closed-set 3-class task; n = 150 (≈36–52 errors per cell, so CIs are wide and several comparisons are "no difference"); OpenAI models with verbalizable confidence only; phrasing arm is the main analysis. Crucially, SciFact is the regime where semantic entropy and disagreement methods are <em>weakest by design</em> (no free text to cluster) and verbalized confidence is easiest — so none of this generalises to free-text generation without a dedicated experiment. A low entropy never certifies correctness; the signal only ever raises suspicion.
</div>

<h2 id="novelty">10. Novelty &amp; prior work</h2>
<p>A web literature scan (June 2026) found the field crowded and partly ahead of us. The mechanistic core is largely taken:</p>
<ul>
  <li><strong>"Entropy alone is insufficient for safe selective prediction"</strong><a class="references" href="#references"></a> (arXiv 2603.21172) already shows entropy-based selective prediction has a "risk floor" because low-entropy confidently-wrong answers survive the gate — our residual finding. It uses a trained activation probe, resampling (not paraphrasing), and no reproducibility framing. <em>This is the single closest paper a reviewer would cite against the work.</em></li>
  <li>The <strong>phrasing-vs-temperature decomposition is the established epistemic/aleatoric split</strong> (arXiv 2311.08718, "input clarification ensembling"; 2510.17028, "Mapping from Meaning"). Our variance decomposition is not novel.</li>
  <li><strong>Paraphrase-consistency hallucination detection</strong> exists (arXiv 2606.08158), and the unreliability of verbalized confidence is well documented.</li>
</ul>
<p>What appears genuinely <strong>open</strong> is the bridge: connecting selective-prediction <em>residuals</em> to the <strong>reproducibility / FAIR</strong> agenda — the normative argument that you should prefer a reproducible residual over an irreproducible one even at equal error rate, with the paraphrase-flip specificity that distinguishes it from generic "confidently wrong". The agent-reproducibility literature (arXiv 2605.28840, 2601.15322; the DEAL reporting checklist) lives separately from the uncertainty-quantification literature, and nobody seems to have joined them. That intersection — which is precisely the FAIR × reproducibility × observability target — is the defensible angle, though it is a framing contribution on top of mostly-known mechanics.</p>

<h2 id="next">11. Recommended next steps</h2>
<ol>
  <li><strong>Read the two closest papers in full first</strong> (2603.21172 and 2311.08718). They largely determine what is left to claim. Do this before any new compute.</li>
  <li><strong>The decisive experiment is a free-text generation task</strong> (short-answer QA: TriviaQA / PopQA style) with proper semantic-entropy clustering (NLI-based), real n, and bootstrap CIs. SciFact is the regime where verbalized confidence is easiest and disagreement weakest; the contrast lives in free text. This costs real API spend — plan it, get sign-off, then run.</li>
  <li><strong>Sharpen the metric.</strong> Aggregate AUROC undervalues disagreement methods because their unique catch (confident flips) is rare. A flip-targeted metric — detection rate on the silent-error subset specifically — is the right yardstick and the methodological point.</li>
  <li><strong>Frame around reproducibility, not accuracy.</strong> "Reproducible vs. irreproducible residuals" is the open seam and the thesis-aligned story.</li>
</ol>

<h2 id="files">12. File map &amp; how to reproduce</h2>
<p>Everything lives in <code>experiments/dirichlet_confidence/</code>. All analysis scripts are pure re-analysis of the cached model answers — no API calls, no cost.</p>
<table>
<thead><tr><th>File</th><th>What it is</th></tr></thead>
<tbody>
<tr><td><code>band_experiment.py</code></td><td>Experiment 1 — band as error-detector; gathers (or reuses) the cached histograms and scores the band.</td></tr>
<tr><td><code>compare_baselines.py</code></td><td>Experiment 2 — band vs. self-consistency / semantic entropy / predictive entropy, with paired bootstrap. → <code>results/baselines.json</code></td></tr>
<tr><td><code>ablation_1v5.py</code></td><td>Experiment 3 — one call vs. five; the 1-call / predictive / mutual-information comparison. → <code>results/ablation_1v5.json</code></td></tr>
<tr><td><code>call_variability.py</code></td><td>Experiment 4 — how much the five histograms vary (TV distance, agreement, worked examples).</td></tr>
<tr><td><code>make_figures.py</code></td><td>Renders all figures (fig1–fig5) incl. Experiment 5's residual composition, via <code>andamentum.figures</code>.</td></tr>
<tr><td><code>mixture.py</code></td><td>The mixture-of-Dirichlets maths: smoothing, concentration fitting, variance decomposition.</td></tr>
<tr><td><code>metrics.py</code> · <code>phrasings.py</code> · <code>scifact.py</code></td><td>AUROC / risk-coverage / ECE; the five prompt templates; the SciFact loader.</td></tr>
<tr><td><code>cache/band_experiment*.jsonl</code></td><td>The cached model answers — 150 claims × 5 calls × 2 arms × 3 models. The substrate every analysis reads.</td></tr>
<tr><td><code>build_report.py</code></td><td>This document.</td></tr>
</tbody>
</table>
<p>To regenerate: <code>uv run python -m experiments.dirichlet_confidence.make_figures</code> (figures), then the analysis scripts as needed, then <code>uv run python -m experiments.dirichlet_confidence.build_report</code> to rebuild this page. See <code>FINDINGS.md</code> and <code>README.md</code> in the same folder for the original narrative.</p>

<h2 id="references">13. References</h2>
<div class="references">
<ol>
  <li><em>Entropy Alone is Insufficient for Safe Selective Prediction in LLMs.</em> arXiv:2603.21172. <a href="https://arxiv.org/abs/2603.21172">arxiv.org/abs/2603.21172</a> — the closest prior work: entropy's "risk floor" and confidently-wrong residual.</li>
  <li>Hou et al. <em>Decomposing Uncertainty for Large Language Models through Input Clarification Ensembling.</em> arXiv:2311.08718. <a href="https://arxiv.org/abs/2311.08718">arxiv.org/abs/2311.08718</a> — the epistemic (paraphrase) vs. aleatoric (sampling) decomposition.</li>
  <li><em>Mapping from Meaning: Addressing the Miscalibration of Prompt-Sensitive Language Models.</em> arXiv:2510.17028. <a href="https://arxiv.org/abs/2510.17028">arxiv.org/abs/2510.17028</a></li>
  <li><em>Constrained Paraphrase Consistency for LLM Hallucination Detection.</em> arXiv:2606.08158. <a href="https://arxiv.org/abs/2606.08158">arxiv.org/abs/2606.08158</a></li>
  <li>Kuhn, Gal, Farquhar. <em>Semantic Uncertainty</em> (ICLR 2023); Farquhar et al., <em>Detecting hallucinations using semantic entropy</em>, Nature 2024. Wang et al., <em>Self-Consistency</em> (ICLR 2023). Kadavath et al., <em>Language Models (Mostly) Know What They Know</em> (2022). Lin/Hilton/Evans (2022); Tian et al. (2023) — verbalized confidence and calibration.</li>
  <li><em>How Consistent Are LLM Agents? Measuring Behavioral Reproducibility in Multi-Step Tool-Calling Pipelines.</em> arXiv:2605.28840. · <em>Replayable Financial Agents.</em> arXiv:2601.15322 — the (separate) agent-reproducibility literature this work could bridge to.</li>
</ol>
</div>

<p class="meta-stamp">
<code>dirichlet_confidence</code> reliability experiments · 150 SciFact claims · 3 models · figures embedded from <code>figures/</code> · numbers from <code>results/*.json</code> and committed runs. Built with <code>build_report.py</code>; style shared with <code>disco_flow.html</code>. Pure re-analysis — no model was queried to produce this report.
</p>

</div>
</body>
</html>
"""


def main() -> None:
    OUT.write_text(HTML, encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT}  ({kb:.0f} KB, figures embedded)")


if __name__ == "__main__":
    main()
