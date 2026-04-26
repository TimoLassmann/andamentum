// benchmarks/chunker/app/static/app.js — vanilla JS, no build step.

const $ = (id) => document.getElementById(id);
let lastResult = null;

$("go").addEventListener("click", async () => {
  const text = $("src").value;
  if (!text.trim()) { alert("paste some text first"); return; }
  $("go").disabled = true;
  $("dl").disabled = true;
  $("err").innerHTML = "";
  $("render").innerHTML = "running… (this can take a while on local models)";
  $("stats").textContent = "";

  try {
    const res = await fetch("/api/chunk", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        model: $("model").value,
        domain: $("domain").value,
        window_size: parseInt($("window").value, 10),
        lookahead: parseInt($("lookahead").value, 10),
      }),
    });
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(`HTTP ${res.status}: ${detail}`);
    }
    lastResult = await res.json();
    renderResult(text, lastResult);
    $("dl").disabled = false;
  } catch (e) {
    $("err").innerHTML = `<div class="err">${escapeHtml(e.message)}</div>`;
    $("render").innerHTML = "";
  } finally {
    $("go").disabled = false;
  }
});

$("dl").addEventListener("click", () => {
  if (!lastResult) return;
  const truth = {
    convention: "DRAFT — describe what counts as a unit for this case",
    expected_f1_floor: 0.65,
    boundary_tolerance_chars: 50,
    domain: $("domain").value,
    units: lastResult.units.map(u => ({
      title: u.title,
      start_anchor: firstWords(u.text, 8),
      end_anchor: lastWords(u.text, 8),
    })),
  };
  const blob = new Blob([JSON.stringify(truth, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "draft.truth.json";
  a.click();
  URL.revokeObjectURL(url);
});

function renderResult(source, result) {
  const spans = [];
  for (let i = 0; i < result.units.length; i++) {
    const u = result.units[i];
    spans.push({ start: u.source_start, end: u.source_end, kind: "unit", idx: i, title: u.title });
  }
  for (const g of result.gaps) {
    spans.push({ start: g.source_start, end: g.source_end, kind: "gap" });
  }
  spans.sort((a, b) => a.start - b.start);

  let html = "";
  let cursor = 0;
  for (const s of spans) {
    if (s.start > cursor) {
      html += escapeHtml(source.slice(cursor, s.start));
    }
    const text = escapeHtml(source.slice(s.start, s.end));
    if (s.kind === "unit") {
      html += `<span class="unit u${s.idx % 8}" title="${escapeHtml(s.title)} (${s.start}-${s.end})">${text}</span>`;
    } else {
      html += `<span class="gap" title="gap ${s.start}-${s.end}">${text}</span>`;
    }
    cursor = s.end;
  }
  if (cursor < source.length) {
    html += escapeHtml(source.slice(cursor));
  }
  $("render").innerHTML = html;
  $("stats").textContent =
    `coverage=${(result.coverage * 100).toFixed(1)}%   gaps=${(result.gap_fraction * 100).toFixed(1)}%   ` +
    `units=${result.units.length}   model calls=${result.model_calls}`;
}

function firstWords(s, n) {
  return s.trim().split(/\s+/).slice(0, n).join(" ");
}
function lastWords(s, n) {
  return s.trim().split(/\s+/).slice(-n).join(" ");
}
function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
