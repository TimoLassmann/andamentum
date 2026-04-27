// benchmarks/chunker/app/static/app.js — vanilla JS, no build step.
//
// State model:
//   editorState.source     — the source text the chunker ran on (frozen at chunk time)
//   editorState.units      — array of editable unit drafts (the source of truth for the UI)
//   editorState.focusedIdx — which unit card is currently focused (or null)
//
// Each unit draft: { title, start_anchor, end_anchor, kind,
//                    startMatch, endMatch, source_start, source_end }
// The startMatch/endMatch fields hold MatchAnchorResponse objects from /api/match-anchor;
// source_start/source_end are derived ints (or null when either anchor doesn't resolve).

const $ = (id) => document.getElementById(id);

const editorState = {
  source: "",
  units: [],
  focusedIdx: null,
};

// Debounce per-unit re-validation so typing doesn't flood the server.
const _validateTimers = new Map();

// ---------- Chunk button --------------------------------------------------

$("go").addEventListener("click", async () => {
  const text = $("src").value;
  if (!text.trim()) {
    alert("Paste some text first.");
    return;
  }
  $("go").disabled = true;
  $("err").textContent = "";
  $("stats").textContent = "running… (this can take a while on local models)";

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
    const result = await res.json();
    loadFromChunkResult(text, result);
  } catch (e) {
    $("err").textContent = e.message;
    $("stats").textContent = "";
  } finally {
    $("go").disabled = false;
  }
});

$("reset").addEventListener("click", () => {
  if (!confirm("Discard all edits and reload from the chunker output?")) return;
  if (!editorState._lastResult) return;
  loadFromChunkResult(editorState.source, editorState._lastResult);
});

$("add-unit").addEventListener("click", () => {
  editorState.units.push({
    title: "New unit",
    start_anchor: "",
    end_anchor: "",
    kind: "",
    startMatch: null,
    endMatch: null,
    source_start: null,
    source_end: null,
  });
  editorState.focusedIdx = editorState.units.length - 1;
  renderAll();
});

// ---------- Loading chunker results into editable state ------------------

function loadFromChunkResult(source, result) {
  editorState.source = source;
  editorState._lastResult = result;
  editorState.units = result.units.map((u) => ({
    title: u.title,
    start_anchor: firstWords(u.text, 8),
    end_anchor: lastWords(u.text, 8),
    kind: u.kind || "",
    startMatch: { found: true, start: u.source_start, end: -1, method: u.anchor_match_method },
    endMatch: { found: true, start: -1, end: u.source_end, method: u.anchor_match_method },
    source_start: u.source_start,
    source_end: u.source_end,
  }));
  editorState.focusedIdx = null;
  $("editor").classList.remove("hidden");
  $("case-meta").classList.remove("hidden");
  $("reset").disabled = false;
  $("stats").textContent =
    `coverage=${(result.coverage * 100).toFixed(1)}%   ` +
    `gaps=${(result.gap_fraction * 100).toFixed(1)}%   ` +
    `units=${result.units.length}   model calls=${result.model_calls}`;
  // Re-validate from the editor's perspective so anchors can be tweaked freely.
  for (let i = 0; i < editorState.units.length; i++) revalidateUnit(i);
  renderAll();
}

// ---------- Per-unit anchor validation -----------------------------------

async function revalidateUnit(idx) {
  // Cursor for THIS unit = end of the previous unit (or 0 for the first).
  const u = editorState.units[idx];
  if (!u) return;
  const prev = idx > 0 ? editorState.units[idx - 1] : null;
  const cursor = prev && prev.source_end != null ? prev.source_end : 0;

  if (!u.start_anchor) {
    u.startMatch = { found: false };
    u.source_start = null;
  } else {
    u.startMatch = await matchAnchor(u.start_anchor, cursor);
    u.source_start = u.startMatch.found ? u.startMatch.start : null;
  }

  const endCursor = u.startMatch && u.startMatch.found ? u.startMatch.end : cursor;
  if (!u.end_anchor) {
    u.endMatch = { found: false };
    u.source_end = null;
  } else {
    u.endMatch = await matchAnchor(u.end_anchor, endCursor);
    u.source_end = u.endMatch.found ? u.endMatch.end : null;
  }

  renderAll();
}

async function matchAnchor(anchor, search_from) {
  try {
    const res = await fetch("/api/match-anchor", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: editorState.source, anchor, search_from }),
    });
    if (!res.ok) return { found: false };
    return await res.json();
  } catch (_e) {
    return { found: false };
  }
}

function scheduleRevalidate(idx) {
  if (_validateTimers.has(idx)) clearTimeout(_validateTimers.get(idx));
  _validateTimers.set(
    idx,
    setTimeout(() => {
      _validateTimers.delete(idx);
      // Revalidate this unit AND every later unit, since the cursor may shift.
      (async () => {
        for (let i = idx; i < editorState.units.length; i++) {
          await revalidateUnit(i);
        }
      })();
    }, 250),
  );
}

// ---------- Rendering ----------------------------------------------------

function renderAll() {
  renderPreview();
  renderUnitCards();
  // Place handles on the next animation frame so layout is settled.
  requestAnimationFrame(placeBoundaryHandles);
}

function renderPreview() {
  const source = editorState.source;
  const units = editorState.units;
  // Build sorted spans of resolved units (skip unresolved ones).
  const spans = [];
  for (let i = 0; i < units.length; i++) {
    const u = units[i];
    if (u.source_start != null && u.source_end != null && u.source_end > u.source_start) {
      spans.push({ start: u.source_start, end: u.source_end, idx: i, title: u.title });
    }
  }
  spans.sort((a, b) => a.start - b.start);

  let html = "";
  let cursor = 0;
  for (const s of spans) {
    if (s.start > cursor) {
      html += `<span class="gap" data-source-start="${cursor}" data-source-end="${s.start}" title="gap ${cursor}-${s.start}">${escapeHtml(source.slice(cursor, s.start))}</span>`;
    }
    const cls = `unit u${s.idx % 8}` + (s.idx === editorState.focusedIdx ? " focused" : "");
    html += `<span class="${cls}" data-idx="${s.idx}" data-source-start="${s.start}" data-source-end="${s.end}" title="${escapeAttr(s.title)} (${s.start}-${s.end})">${escapeHtml(source.slice(s.start, s.end))}</span>`;
    cursor = Math.max(cursor, s.end);
  }
  if (cursor < source.length) {
    html += `<span class="gap" data-source-start="${cursor}" data-source-end="${source.length}" title="gap ${cursor}-${source.length}">${escapeHtml(source.slice(cursor))}</span>`;
  }
  $("preview").innerHTML = html;

  for (const span of $("preview").querySelectorAll(".unit")) {
    span.addEventListener("click", (e) => {
      // Don't steal focus during a drag-select — only on a true click.
      if (window.getSelection().toString().length > 0) return;
      editorState.focusedIdx = parseInt(span.dataset.idx, 10);
      renderAll();
      const card = document.querySelector(`.unit-card[data-idx="${editorState.focusedIdx}"]`);
      if (card) card.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
  }
}

function renderUnitCards() {
  const container = $("units");
  container.innerHTML = "";
  const units = editorState.units;
  const swatchClasses = ["u0", "u1", "u2", "u3", "u4", "u5", "u6", "u7"];

  units.forEach((u, idx) => {
    // "Insert above" affordance — empty unit gets spliced in at this index.
    const above = document.createElement("div");
    above.className = "insert-above";
    const aboveBtn = document.createElement("button");
    aboveBtn.textContent = "+ insert unit above";
    aboveBtn.addEventListener("click", () => insertEmptyUnitAt(idx));
    above.appendChild(aboveBtn);
    container.appendChild(above);

    const card = document.createElement("div");
    card.className = "unit-card" + (idx === editorState.focusedIdx ? " focused" : "");
    card.dataset.idx = String(idx);
    card.innerHTML = `
      <div class="card-head">
        <span class="swatch ${swatchClasses[idx % 8]}"></span>
        <strong style="color:#666; font-size: 11px;">${idx + 1}.</strong>
        <input type="text" data-field="title" value="${escapeAttr(u.title)}" placeholder="title">
      </div>
      <div class="field">
        <label>start_anchor</label>
        <input type="text" data-field="start_anchor" value="${escapeAttr(u.start_anchor)}">
        ${anchorStatusHtml(u.startMatch)}
      </div>
      <div class="field">
        <label>end_anchor</label>
        <input type="text" data-field="end_anchor" value="${escapeAttr(u.end_anchor)}">
        ${anchorStatusHtml(u.endMatch)}
      </div>
      <div class="field">
        <label>kind</label>
        <input type="text" data-field="kind" value="${escapeAttr(u.kind)}" placeholder="(optional)">
        <span></span>
      </div>
      <div class="actions">
        <button class="ghost" data-action="up" ${idx === 0 ? "disabled" : ""}>↑</button>
        <button class="ghost" data-action="down" ${idx === units.length - 1 ? "disabled" : ""}>↓</button>
        <button class="danger" data-action="delete">delete</button>
      </div>
    `;

    card.addEventListener("click", (e) => {
      // Don't steal focus when interacting with form controls.
      if (e.target.closest("input, button")) return;
      editorState.focusedIdx = idx;
      renderAll();
    });

    for (const input of card.querySelectorAll("input[data-field]")) {
      input.addEventListener("input", () => {
        const field = input.dataset.field;
        editorState.units[idx][field] = input.value;
        if (field === "start_anchor" || field === "end_anchor") {
          scheduleRevalidate(idx);
        }
      });
      // Re-render preview on focus so the highlighted span updates immediately
      // (even if the anchor check is still pending — no flicker mid-typing).
      input.addEventListener("focus", () => {
        editorState.focusedIdx = idx;
        renderPreview();
      });
    }

    for (const btn of card.querySelectorAll("button[data-action]")) {
      btn.addEventListener("click", () => {
        const action = btn.dataset.action;
        if (action === "delete") {
          if (!confirm(`Delete unit "${u.title}"?`)) return;
          editorState.units.splice(idx, 1);
          if (editorState.focusedIdx === idx) editorState.focusedIdx = null;
          // Cascade re-validate the rest (cursor positions shift).
          revalidateFrom(idx);
        } else if (action === "up" && idx > 0) {
          [editorState.units[idx - 1], editorState.units[idx]] =
            [editorState.units[idx], editorState.units[idx - 1]];
          revalidateFrom(idx - 1);
        } else if (action === "down" && idx < units.length - 1) {
          [editorState.units[idx], editorState.units[idx + 1]] =
            [editorState.units[idx + 1], editorState.units[idx]];
          revalidateFrom(idx);
        }
        renderAll();
      });
    }

    container.appendChild(card);
  });
}

async function revalidateFrom(startIdx) {
  for (let i = startIdx; i < editorState.units.length; i++) {
    await revalidateUnit(i);
  }
}

function anchorStatusHtml(match) {
  if (!match) return `<span class="status idle">—</span>`;
  if (!match.found) return `<span class="status bad">✗</span>`;
  return `<span class="status ok" title="${escapeAttr(match.method || "")}">✓ ${match.method ? match.method[0] : ""}</span>`;
}

// ---------- Save ----------------------------------------------------------

$("save").addEventListener("click", async () => {
  const status = $("save-status");
  status.className = "";
  status.textContent = "";

  const name = $("case-name").value.trim();
  if (!name) { status.className = "bad"; status.textContent = "case name required"; return; }
  const convention = $("case-convention").value.trim();
  if (!convention) { status.className = "bad"; status.textContent = "convention required"; return; }
  if (editorState.units.length === 0) {
    status.className = "bad"; status.textContent = "need at least one unit"; return;
  }

  // Reject if any anchor is unresolved — server will too, but fail fast.
  const bad = editorState.units.findIndex(
    (u) => !u.startMatch || !u.startMatch.found || !u.endMatch || !u.endMatch.found,
  );
  if (bad >= 0) {
    status.className = "bad";
    status.textContent = `unit ${bad + 1} has an unresolved anchor — fix it before saving`;
    return;
  }

  $("save").disabled = true;
  try {
    const res = await fetch("/api/save-case", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        extension: $("case-ext").value,
        text: editorState.source,
        convention,
        expected_f1_floor: parseFloat($("case-floor").value),
        boundary_tolerance_chars: parseInt($("case-tol").value, 10),
        domain: $("domain").value,
        units: editorState.units.map((u) => ({
          title: u.title,
          start_anchor: u.start_anchor,
          end_anchor: u.end_anchor,
          ...(u.kind ? { kind: u.kind } : {}),
        })),
        overwrite: $("case-overwrite").checked,
      }),
    });
    if (!res.ok) {
      let detail = await res.text();
      try { detail = JSON.parse(detail).detail || detail; } catch (_) {}
      throw new Error(`HTTP ${res.status}: ${detail}`);
    }
    const data = await res.json();
    status.className = "ok";
    status.textContent = `saved → ${data.input_path}  +  ${data.truth_path}`;
  } catch (e) {
    status.className = "bad";
    status.textContent = e.message;
  } finally {
    $("save").disabled = false;
  }
});

// ---------- Helpers ------------------------------------------------------

function firstWords(s, n) { return s.trim().split(/\s+/).slice(0, n).join(" "); }
function lastWords(s, n)  { return s.trim().split(/\s+/).slice(-n).join(" "); }

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function escapeAttr(s) {
  return escapeHtml(s).replace(/"/g, "&quot;");
}

// ---------- SearXNG status + Search + Fetch ------------------------------

async function refreshSearxngStatus() {
  setPill("unknown", "");
  try {
    const r = await fetch("/api/searxng-status");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    setPill(data.state, data.message);
  } catch (e) {
    setPill("error", e.message);
  }
}

function setPill(state, msg) {
  const pill = $("searxng-pill");
  pill.className = `pill ${state}`;
  pill.textContent = state;
  $("search-status-msg").textContent = msg || "";
  $("searxng-start").style.display =
    (state === "stopped" || state === "podman-missing") ? "" : "none";
}

$("searxng-refresh").addEventListener("click", refreshSearxngStatus);

$("searxng-start").addEventListener("click", async () => {
  $("searxng-start").disabled = true;
  setPill("starting", "pulling image and starting container (can take 30-60s)…");
  try {
    const r = await fetch("/api/searxng-start", { method: "POST" });
    if (!r.ok) {
      let detail = await r.text();
      try { detail = JSON.parse(detail).detail || detail; } catch (_) {}
      throw new Error(detail);
    }
    await refreshSearxngStatus();
  } catch (e) {
    setPill("error", e.message);
  } finally {
    $("searxng-start").disabled = false;
  }
});

$("search-go").addEventListener("click", async () => {
  const q = $("search-query").value.trim();
  if (!q) { setSearchMsg("type a query first", "bad"); return; }
  $("search-go").disabled = true;
  $("search-results").innerHTML = "";
  setSearchMsg("searching…", "");
  try {
    const r = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q, max_results: parseInt($("search-max").value, 10) }),
    });
    if (!r.ok) {
      let detail = await r.text();
      try { detail = JSON.parse(detail).detail || detail; } catch (_) {}
      throw new Error(detail);
    }
    const data = await r.json();
    if (data.hits.length === 0) {
      setSearchMsg("no hits", "");
    } else {
      setSearchMsg(`${data.hits.length} hits — click one to load its content`, "ok");
      renderHits(data.hits);
    }
  } catch (e) {
    setSearchMsg(e.message, "bad");
    refreshSearxngStatus();  // a 503 usually means status flipped
  } finally {
    $("search-go").disabled = false;
  }
});

$("search-query").addEventListener("keydown", (e) => {
  if (e.key === "Enter") $("search-go").click();
});

$("fetch-go").addEventListener("click", async () => {
  const url = $("fetch-url").value.trim();
  if (!url) { setSearchMsg("paste a URL first", "bad"); return; }
  await loadFromUrl(url);
});

$("fetch-url").addEventListener("keydown", (e) => {
  if (e.key === "Enter") $("fetch-go").click();
});

function renderHits(hits) {
  const container = $("search-results");
  container.innerHTML = "";
  for (const h of hits) {
    const div = document.createElement("div");
    div.className = "search-hit";
    div.innerHTML = `
      <div class="title">${escapeHtml(h.title)}</div>
      <div class="url">${escapeHtml(h.url)} · ${escapeHtml(h.domain)}</div>
      <div class="snippet">${escapeHtml(h.snippet)}</div>
    `;
    div.addEventListener("click", () => loadFromUrl(h.url));
    container.appendChild(div);
  }
}

async function loadFromUrl(url) {
  $("fetch-go").disabled = true;
  $("search-go").disabled = true;
  setSearchMsg(`fetching ${url}…`, "");
  clearPdfLinks();
  try {
    const r = await fetch("/api/fetch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    if (!r.ok) {
      let detail = await r.text();
      try { detail = JSON.parse(detail).detail || detail; } catch (_) {}
      throw new Error(detail);
    }
    const data = await r.json();
    $("src").value = data.markdown;
    const rewrote = data.final_url !== data.requested_url
      ? ` (rewrote → ${data.final_url})`
      : "";
    const trunc = data.truncated
      ? ` (truncated from ${data.original_length.toLocaleString()} chars)`
      : "";
    const kind = data.is_pdf ? " · 📄 PDF" : "";
    setSearchMsg(
      `loaded "${data.title}"${kind}${rewrote} · ${data.word_count.toLocaleString()} words${trunc} — click "Chunk it" below`,
      "ok",
    );
    if (!data.is_pdf && data.pdf_links && data.pdf_links.length > 0) {
      renderPdfLinks(data.pdf_links);
    }
    $("src").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (e) {
    setSearchMsg(e.message, "bad");
  } finally {
    $("fetch-go").disabled = false;
    $("search-go").disabled = false;
  }
}

function renderPdfLinks(links) {
  const msg = $("search-msg");
  const wrap = document.createElement("div");
  wrap.id = "pdf-link-buttons";
  wrap.style.marginTop = "6px";
  wrap.innerHTML = `<span style="color: var(--muted); font-size: 12px; margin-right: 6px;">PDF link${links.length > 1 ? "s" : ""} on this page:</span>`;
  for (const link of links) {
    const btn = document.createElement("button");
    btn.className = "ghost";
    btn.style.fontSize = "11px";
    btn.style.padding = "3px 8px";
    btn.style.marginRight = "6px";
    btn.textContent = `📄 ${link.label}`;
    btn.title = link.url;
    btn.addEventListener("click", () => loadFromUrl(link.url));
    wrap.appendChild(btn);
  }
  msg.appendChild(wrap);
}

function clearPdfLinks() {
  const old = document.getElementById("pdf-link-buttons");
  if (old) old.remove();
}

function setSearchMsg(msg, cls) {
  const el = $("search-msg");
  el.className = cls || "";
  el.textContent = msg;
}

// Auto-poll status on first load.
refreshSearxngStatus();

// ---------- Selection-driven boundary editing ----------------------------
//
// Every span in #preview carries data-source-start / data-source-end attrs
// (added by renderPreview). On a non-collapsed selection inside the preview,
// we walk up from the selection's start/end nodes to the nearest such span,
// add the within-span text offset, and end up with absolute source-string
// offsets we can use to set anchors or build new units.

const ANCHOR_WORDS = 8;  // length of auto-extracted anchor strings

function _ancestorSpan(node) {
  while (node && node !== document.body) {
    if (node.dataset && node.dataset.sourceStart !== undefined) return node;
    node = node.parentElement || node.parentNode;
  }
  return null;
}

function _nodeOffsetWithinSpan(span, container, containerOffset) {
  // Sum the text length of every text node before `container` within `span`,
  // then add `containerOffset` (which is char-offset for text nodes, or
  // child-index for element nodes).
  if (container === span) {
    // Selection landed on the span element itself — use child-index semantics
    let acc = 0;
    for (let i = 0; i < containerOffset; i++) {
      const child = span.childNodes[i];
      if (child) acc += (child.textContent || "").length;
    }
    return acc;
  }
  let acc = 0;
  let found = false;
  function walk(node) {
    if (found) return;
    if (node === container) {
      acc += containerOffset;
      found = true;
      return;
    }
    if (node.nodeType === Node.TEXT_NODE) {
      acc += node.textContent.length;
      return;
    }
    for (const child of node.childNodes) walk(child);
  }
  walk(span);
  return acc;
}

function mapSelectionToSourceOffsets(sel) {
  if (!sel || sel.isCollapsed || sel.rangeCount === 0) return null;
  const range = sel.getRangeAt(0);
  const previewEl = $("preview");
  if (!previewEl.contains(range.commonAncestorContainer)) return null;

  const startSpan = _ancestorSpan(range.startContainer);
  const endSpan = _ancestorSpan(range.endContainer);
  if (!startSpan || !endSpan) return null;

  const startBase = parseInt(startSpan.dataset.sourceStart, 10);
  const endBase = parseInt(endSpan.dataset.sourceStart, 10);
  const startInSpan = _nodeOffsetWithinSpan(startSpan, range.startContainer, range.startOffset);
  const endInSpan = _nodeOffsetWithinSpan(endSpan, range.endContainer, range.endOffset);
  const start = startBase + startInSpan;
  const end = endBase + endInSpan;
  if (end <= start) return null;
  return { start, end, text: editorState.source.slice(start, end) };
}

function findContainingUnitIdx(sel) {
  // Returns idx of the unit whose [source_start, source_end] fully contains sel,
  // or null if no unit contains it (selection is in a gap or spans multiple units).
  for (let i = 0; i < editorState.units.length; i++) {
    const u = editorState.units[i];
    if (u.source_start == null || u.source_end == null) continue;
    if (sel.start >= u.source_start && sel.end <= u.source_end) return i;
  }
  return null;
}

function insertUnitInSourceOrder(newUnit) {
  let insertIdx = editorState.units.length;
  for (let i = 0; i < editorState.units.length; i++) {
    const u = editorState.units[i];
    if (u.source_start != null && newUnit.source_start != null
        && u.source_start > newUnit.source_start) {
      insertIdx = i;
      break;
    }
  }
  editorState.units.splice(insertIdx, 0, newUnit);
  return insertIdx;
}

function insertEmptyUnitAt(idx) {
  editorState.units.splice(idx, 0, {
    title: "New unit",
    start_anchor: "",
    end_anchor: "",
    kind: "",
    startMatch: null,
    endMatch: null,
    source_start: null,
    source_end: null,
  });
  editorState.focusedIdx = idx;
  hideSelectionToolbar();
  renderAll();
  // Scroll the new card into view so it's obvious it appeared.
  setTimeout(() => {
    const card = document.querySelector(`.unit-card[data-idx="${idx}"]`);
    if (card) card.scrollIntoView({ behavior: "smooth", block: "center" });
  }, 0);
}

// Toolbar -----------------------------------------------------------------

function showSelectionToolbar(sel) {
  const tb = $("sel-toolbar");
  const containingIdx = findContainingUnitIdx(sel);
  const focusedIdx = editorState.focusedIdx;
  // Prefer the unit you're already focused on if the selection is inside it;
  // otherwise default to the unit the selection is contained in.
  let targetIdx = null;
  if (focusedIdx != null) {
    const u = editorState.units[focusedIdx];
    if (u && u.source_start != null && u.source_end != null
        && sel.start >= u.source_start && sel.end <= u.source_end) {
      targetIdx = focusedIdx;
    }
  }
  if (targetIdx == null) targetIdx = containingIdx;

  tb.innerHTML = "";
  const targetLabel = targetIdx != null ? `unit ${targetIdx + 1}` : null;

  if (targetLabel) {
    const setStart = document.createElement("button");
    setStart.textContent = `↦ start of ${targetLabel}`;
    setStart.title = "Set the first ~8 words of selection as this unit's start_anchor";
    setStart.addEventListener("click", () => actionSetStart(targetIdx, sel));
    tb.appendChild(setStart);

    const setEnd = document.createElement("button");
    setEnd.textContent = `↤ end of ${targetLabel}`;
    setEnd.title = "Set the last ~8 words of selection as this unit's end_anchor";
    setEnd.addEventListener("click", () => actionSetEnd(targetIdx, sel));
    tb.appendChild(setEnd);

    const sep = document.createElement("span");
    sep.className = "sep";
    tb.appendChild(sep);

    const split = document.createElement("button");
    split.textContent = `✂ split ${targetLabel}`;
    split.title = "Split this unit at the selection — selection becomes a new unit";
    split.addEventListener("click", () => actionSplit(targetIdx, sel));
    tb.appendChild(split);
  }

  const newUnit = document.createElement("button");
  newUnit.textContent = `+ new unit from selection`;
  newUnit.title = "Create a new unit from the selected text";
  newUnit.addEventListener("click", () => actionNewUnit(sel));
  tb.appendChild(newUnit);

  // Position above the selection rect (or below if it'd go off-screen).
  const rect = window.getSelection().getRangeAt(0).getBoundingClientRect();
  tb.style.display = "block";
  // Render once to measure size.
  const tbRect = tb.getBoundingClientRect();
  let top = rect.top - tbRect.height - 8;
  if (top < 8) top = rect.bottom + 8;
  let left = rect.left + (rect.width / 2) - (tbRect.width / 2);
  left = Math.max(8, Math.min(left, window.innerWidth - tbRect.width - 8));
  tb.style.top = `${top}px`;
  tb.style.left = `${left}px`;
}

function hideSelectionToolbar() {
  const tb = $("sel-toolbar");
  if (tb) tb.style.display = "none";
}

function _firstWords(text, n) {
  return text.trim().split(/\s+/).slice(0, n).join(" ");
}
function _lastWords(text, n) {
  return text.trim().split(/\s+/).slice(-n).join(" ");
}

function actionSetStart(idx, sel) {
  editorState.units[idx].start_anchor = _firstWords(sel.text, ANCHOR_WORDS);
  editorState.focusedIdx = idx;
  hideSelectionToolbar();
  window.getSelection().removeAllRanges();
  revalidateFrom(idx);
  renderAll();
}

function actionSetEnd(idx, sel) {
  editorState.units[idx].end_anchor = _lastWords(sel.text, ANCHOR_WORDS);
  editorState.focusedIdx = idx;
  hideSelectionToolbar();
  window.getSelection().removeAllRanges();
  revalidateFrom(idx);
  renderAll();
}

function actionNewUnit(sel) {
  const newUnit = {
    title: "New unit",
    start_anchor: _firstWords(sel.text, ANCHOR_WORDS),
    end_anchor: _lastWords(sel.text, ANCHOR_WORDS),
    kind: "",
    startMatch: null,
    endMatch: null,
    source_start: sel.start,  // tentative — revalidation will recompute
    source_end: sel.end,
  };
  const insertedAt = insertUnitInSourceOrder(newUnit);
  editorState.focusedIdx = insertedAt;
  hideSelectionToolbar();
  window.getSelection().removeAllRanges();
  // Revalidate from the insertion point onwards so cursors cascade.
  revalidateFrom(insertedAt);
  renderAll();
}

function actionSplit(idx, sel) {
  // Split unit `idx` at the selection: selection becomes a new unit; original
  // unit's end_anchor is shortened to the words just before the selection.
  const original = editorState.units[idx];
  if (original.source_start == null || original.source_end == null) {
    hideSelectionToolbar();
    return;
  }
  const before = editorState.source.slice(original.source_start, sel.start);
  const after = editorState.source.slice(sel.end, original.source_end);

  if (before.trim().length === 0 || after.trim().length === 0) {
    // Selection touches a boundary — split would create an empty fragment.
    // Easier and safer: just let the user use "set start" or "set end" instead.
    actionNewUnit(sel);
    return;
  }

  // Shrink the original unit so it ends just before the selection.
  original.end_anchor = _lastWords(before, ANCHOR_WORDS);

  // Build a new unit FROM the selection, ending where the original ended.
  const newUnit = {
    title: original.title + " (split)",
    start_anchor: _firstWords(sel.text, ANCHOR_WORDS),
    end_anchor: _lastWords(after, ANCHOR_WORDS),
    kind: original.kind,
    startMatch: null,
    endMatch: null,
    source_start: sel.start,
    source_end: original.source_end,
  };
  editorState.units.splice(idx + 1, 0, newUnit);
  editorState.focusedIdx = idx + 1;
  hideSelectionToolbar();
  window.getSelection().removeAllRanges();
  revalidateFrom(idx);
  renderAll();
}

// Wire selection events ---------------------------------------------------

function _handleSelection(e) {
  // Clicks on the toolbar must NOT clear the selection or hide the toolbar.
  if (e && e.target && e.target.closest && e.target.closest("#sel-toolbar")) return;
  const sel = window.getSelection();
  const offsets = mapSelectionToSourceOffsets(sel);
  if (!offsets) {
    hideSelectionToolbar();
    return;
  }
  showSelectionToolbar(offsets);
}

document.addEventListener("mouseup", _handleSelection);
document.addEventListener("keyup", (e) => {
  // Only react to Shift+arrow-style selection changes inside the preview
  if (e.shiftKey || e.key === "ArrowLeft" || e.key === "ArrowRight"
      || e.key === "ArrowUp" || e.key === "ArrowDown") {
    _handleSelection(e);
  }
});
// Hide on Escape and on scroll inside the preview pane.
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    hideSelectionToolbar();
    window.getSelection().removeAllRanges();
  }
});

// ---------- Draggable boundary handles ----------------------------------

let dragState = null;

function _firstTextNode(node) {
  if (node.nodeType === Node.TEXT_NODE) return node;
  for (const child of node.childNodes) {
    const t = _firstTextNode(child);
    if (t) return t;
  }
  return null;
}
function _lastTextNode(node) {
  if (node.nodeType === Node.TEXT_NODE) return node;
  for (let i = node.childNodes.length - 1; i >= 0; i--) {
    const t = _lastTextNode(node.childNodes[i]);
    if (t) return t;
  }
  return null;
}

function placeBoundaryHandles() {
  const preview = $("preview");
  // Strip old handles
  for (const h of preview.querySelectorAll(".boundary-handle")) h.remove();
  if (!editorState.source) return;

  const previewRect = preview.getBoundingClientRect();
  const scrollLeft = preview.scrollLeft;
  const scrollTop = preview.scrollTop;

  for (const span of preview.querySelectorAll(".unit")) {
    const idx = parseInt(span.dataset.idx, 10);
    const firstText = _firstTextNode(span);
    const lastText = _lastTextNode(span);
    if (!firstText || !lastText || lastText.length === 0) continue;

    // Bounding rect of the span's first character → top-left of unit
    const firstR = document.createRange();
    firstR.setStart(firstText, 0);
    firstR.setEnd(firstText, Math.min(1, firstText.length));
    const fr = firstR.getBoundingClientRect();

    // Bounding rect of the span's last character → bottom-right of unit
    const lastR = document.createRange();
    lastR.setStart(lastText, Math.max(0, lastText.length - 1));
    lastR.setEnd(lastText, lastText.length);
    const lr = lastR.getBoundingClientRect();

    // Skip if the rects are degenerate (e.g. unit is collapsed/zero-width).
    if (fr.height === 0 || lr.height === 0) continue;

    const startH = document.createElement("div");
    startH.className = "boundary-handle start";
    startH.dataset.idx = String(idx);
    startH.dataset.side = "start";
    startH.style.left =
      (fr.left - previewRect.left + scrollLeft - 3) + "px";
    startH.style.top = (fr.top - previewRect.top + scrollTop) + "px";
    startH.style.height = fr.height + "px";
    startH.title = `unit ${idx + 1} — drag to move start`;
    preview.appendChild(startH);

    const endH = document.createElement("div");
    endH.className = "boundary-handle end";
    endH.dataset.idx = String(idx);
    endH.dataset.side = "end";
    endH.style.left =
      (lr.right - previewRect.left + scrollLeft - 3) + "px";
    endH.style.top = (lr.top - previewRect.top + scrollTop) + "px";
    endH.style.height = lr.height + "px";
    endH.title = `unit ${idx + 1} — drag to move end`;
    preview.appendChild(endH);
  }
}

window.addEventListener("resize", () => requestAnimationFrame(placeBoundaryHandles));
// Reposition handles when the preview pane scrolls, so they stay glued to text.
function _attachPreviewScroll() {
  const preview = $("preview");
  if (preview && !preview._scrollWired) {
    preview.addEventListener("scroll", () => {
      // Handles are children of #preview so they scroll naturally; this is
      // only needed if we ever switch to body-attached handles. Cheap no-op.
    });
    preview._scrollWired = true;
  }
}
_attachPreviewScroll();

// --- Drag flow -----------------------------------------------------------

document.addEventListener("mousedown", (e) => {
  const handle = e.target.closest(".boundary-handle");
  if (!handle) return;
  e.preventDefault();
  e.stopPropagation();
  // Selection toolbar would also fire on mouseup — kill any active selection
  // so the toolbar doesn't pop up when the user is just dragging a handle.
  window.getSelection().removeAllRanges();
  hideSelectionToolbar();

  dragState = {
    idx: parseInt(handle.dataset.idx, 10),
    side: handle.dataset.side,
    handle,
  };
  handle.classList.add("active");
  document.body.style.cursor = "ew-resize";
});

document.addEventListener("mousemove", (e) => {
  if (!dragState) return;
  const offset = mousePointToSourceOffset(e.clientX, e.clientY);
  if (offset == null) {
    hideDragGhost();
    return;
  }
  const snapped = snapToWordBoundary(editorState.source, offset, dragState.side);
  showDragGhost(e.clientX, e.clientY, snapped);
});

document.addEventListener("mouseup", (e) => {
  if (!dragState) return;
  const offset = mousePointToSourceOffset(e.clientX, e.clientY);
  if (offset != null) {
    const snapped = snapToWordBoundary(
      editorState.source, offset, dragState.side,
    );
    commitBoundary(dragState.idx, dragState.side, snapped);
  }
  dragState.handle.classList.remove("active");
  dragState = null;
  document.body.style.cursor = "";
  hideDragGhost();
});

function mousePointToSourceOffset(x, y) {
  // Modern path
  let containerNode, offsetInNode;
  if (document.caretPositionFromPoint) {
    const pos = document.caretPositionFromPoint(x, y);
    if (!pos) return null;
    containerNode = pos.offsetNode;
    offsetInNode = pos.offset;
  } else if (document.caretRangeFromPoint) {
    const range = document.caretRangeFromPoint(x, y);
    if (!range) return null;
    containerNode = range.startContainer;
    offsetInNode = range.startOffset;
  } else {
    return null;
  }
  const span = _ancestorSpan(containerNode);
  if (!span) return null;
  const base = parseInt(span.dataset.sourceStart, 10);
  return base + _nodeOffsetWithinSpan(span, containerNode, offsetInNode);
}

function snapToWordBoundary(source, offset, side) {
  offset = Math.max(0, Math.min(offset, source.length));
  if (side === "start") {
    // Move forward past whitespace so the unit begins at a word.
    while (offset < source.length && /\s/.test(source[offset])) offset++;
  } else {
    // Move backward past whitespace so the unit ends at the last word.
    while (offset > 0 && /\s/.test(source[offset - 1])) offset--;
  }
  return offset;
}

function commitBoundary(idx, side, offset) {
  const u = editorState.units[idx];
  if (!u) return;
  const source = editorState.source;
  if (side === "start") {
    // start_anchor = first ANCHOR_WORDS words from `offset`
    const slice = source.slice(offset, Math.min(source.length, offset + 500));
    u.start_anchor = slice.trim().split(/\s+/).slice(0, ANCHOR_WORDS).join(" ");
  } else {
    // end_anchor = last ANCHOR_WORDS words ending at `offset`
    const slice = source.slice(Math.max(0, offset - 500), offset);
    u.end_anchor = slice.trim().split(/\s+/).slice(-ANCHOR_WORDS).join(" ");
  }
  editorState.focusedIdx = idx;
  // Cascade: subsequent units' validation cursors shift when this one moves.
  revalidateFrom(idx);
  renderAll();
}

// --- Drag ghost line + tooltip ------------------------------------------

function showDragGhost(mouseX, mouseY, snappedOffset) {
  // Ghost = a vertical line at the snapped character's screen X.
  const rect = offsetToScreenRect(snappedOffset);
  const ghost = $("drag-ghost");
  const tip = $("drag-tooltip");
  if (rect) {
    ghost.style.left = rect.left + "px";
    ghost.style.top = rect.top + "px";
    ghost.style.height = rect.height + "px";
    ghost.style.display = "block";
  } else {
    // Fallback: line at mouse X, full preview height
    const previewRect = $("preview").getBoundingClientRect();
    ghost.style.left = mouseX + "px";
    ghost.style.top = previewRect.top + "px";
    ghost.style.height = previewRect.height + "px";
    ghost.style.display = "block";
  }
  // Tooltip with a snippet of context so the user knows what they're snapping to
  const source = editorState.source;
  const before = source.slice(Math.max(0, snappedOffset - 30), snappedOffset);
  const after = source.slice(snappedOffset, snappedOffset + 30);
  tip.textContent = `…${before}│${after}…`;
  tip.style.display = "block";
  tip.style.left = (mouseX + 12) + "px";
  tip.style.top = (mouseY - 28) + "px";
}

function hideDragGhost() {
  $("drag-ghost").style.display = "none";
  $("drag-tooltip").style.display = "none";
}

function offsetToScreenRect(offset) {
  // Find the preview span that covers `offset`, then build a Range at the
  // exact char and return its bounding rect (in viewport coords).
  const preview = $("preview");
  for (const span of preview.querySelectorAll("[data-source-start]")) {
    const sStart = parseInt(span.dataset.sourceStart, 10);
    const sEnd = parseInt(span.dataset.sourceEnd, 10);
    if (offset < sStart || offset > sEnd) continue;
    const within = offset - sStart;
    const text = _firstTextNode(span);  // single text node for unit/gap spans
    if (!text) continue;
    const rng = document.createRange();
    const at = Math.min(within, text.length);
    // Use a 1-char range so the rect has non-zero width; clamp to span length
    rng.setStart(text, Math.max(0, at - 1 >= 0 ? at - 1 : 0));
    rng.setEnd(text, Math.min(text.length, at));
    const r = rng.getBoundingClientRect();
    if (r.width > 0 || r.height > 0) return r;
  }
  return null;
}
