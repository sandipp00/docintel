"use strict";

const $ = (id) => document.getElementById(id);
const state = { docs: [], current: null, cache: {}, threads: {} };

const TYPES = {
  invoice: { icon: "🧾", color: "var(--t-invoice)" }, resume: { icon: "👤", color: "var(--t-resume)" },
  contract: { icon: "📜", color: "var(--t-contract)" }, report: { icon: "📊", color: "var(--t-report)" },
  other: { icon: "📁", color: "var(--t-other)" },
};
const typeOf = (label) => TYPES[label] ?? TYPES.other;

// Document text is untrusted: everything is rendered with textContent, never innerHTML.
function el(tag, props = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") n.className = v; else if (k === "text") n.textContent = v;
    else if (k === "style") n.setAttribute("style", v); else n[k] = v;
  }
  for (const kid of kids) if (kid != null) n.append(kid);
  return n;
}
const pct = (x) => (x == null ? "—" : Math.round(x * 100) + "%");
const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;

// ---------------------------------------------------------------- api + feedback
async function api(path, opts = {}) {
  const res = await fetch("/api" + path, opts);
  if (res.status === 204) return null;
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = body.detail;
    throw new Error(typeof d === "string" ? d : res.status === 429 ? "Too many requests. Please wait a moment and try again."
                    : Array.isArray(d) ? "That input isn't valid." : `Something went wrong (${res.status}).`);
  }
  return body;
}
const post = (path, data) => api(path, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data ?? {}) });

function toast(msg, kind = "info") {
  const t = el("div", { class: "toast " + kind, text: msg });
  $("toasts").append(t);
  setTimeout(() => t.remove(), kind === "error" ? 7000 : 3500);
}
function busy(msg) { $("busy-text").textContent = msg ?? ""; $("busy").hidden = !msg; }
const skeleton = () => el("div", { class: "skeleton" }, el("i"), el("i"), el("i"), el("i"));

// ---------------------------------------------------------------- theme
function applyTheme(t) { document.documentElement.dataset.theme = t; }
(function initTheme() {
  let t = null;
  try { t = localStorage.getItem("theme"); } catch { /* storage may be blocked */ }
  applyTheme(t ?? (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
})();
$("theme").onclick = () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  applyTheme(next);
  try { localStorage.setItem("theme", next); } catch { /* ignore */ }
};

// ---------------------------------------------------------------- views + document list
function showView() {
  const has = state.docs.length > 0;
  $("hero").hidden = has; $("workspace").hidden = !has;
  $("detail").hidden = !state.current; $("pick").hidden = !!state.current;
}

function renderDocs() {
  const ul = $("docs"); ul.replaceChildren();
  for (const d of state.docs) {
    const tp = typeOf(d.classification.label);
    const li = el("li", { class: d.doc_id === state.current?.doc_id ? "active" : "", tabIndex: 0, style: `--tc:${tp.color}` },
      el("div", { class: "type-icon", text: tp.icon }),
      el("div", {}, el("div", { class: "nm", text: d.filename, title: d.filename }),
        el("div", { class: "sub", text: `${d.classification.label} · ${pct(d.classification.confidence)} · ${plural(d.n_pages, "page")}` })));
    li.onclick = () => select(d.doc_id);
    li.onkeydown = (e) => { if (e.key === "Enter") select(d.doc_id); };
    ul.append(li);
  }
  showView();
}

async function refresh() { state.docs = await api("/documents"); renderDocs(); }

async function upload(files) {
  files = [...files];
  if (!files.length) return;
  busy(`Reading ${plural(files.length, "file")}… scanned pages can take a few seconds.`);
  try {
    const form = new FormData();
    for (const f of files) form.append("files", f);
    const added = await api("/upload", { method: "POST", body: form });
    await refresh();
    toast(`Added ${plural(added.length, "document")}.`);
    select(added[0].doc_id);
  } catch (e) { toast(e.message, "error"); } finally { busy(null); }
}

async function loadSample(name) {
  busy(`Loading the ${name} sample…`);
  try {
    const d = await post(`/samples/${encodeURIComponent(name)}`);
    await refresh();
    select(d.doc_id);
  } catch (e) { toast(e.message, "error"); } finally { busy(null); }
}

// ---------------------------------------------------------------- selecting a document
function select(id) {
  const c = state.docs.find((d) => d.doc_id === id);
  if (!c) return;
  state.current = c; state.cache = {};
  renderDocs();
  const tp = typeOf(c.classification.label);
  $("detail").style.setProperty("--tc", tp.color);
  $("d-icon").textContent = tp.icon;
  $("d-icon").style.setProperty("--tc", tp.color);
  $("d-name").textContent = c.filename;
  const chips = $("d-chips"); chips.replaceChildren(
    el("span", { class: "chip type", style: `--tc:${tp.color}`, text: `${c.classification.label} · ${pct(c.classification.confidence)}` }),
    el("span", { class: "chip", text: plural(c.n_pages, "page") }),
    el("span", { class: "chip", text: c.method === "digital" || c.method === "docx" ? "digital text" : "read with OCR" +
      (c.ocr_confidence != null ? ` (${pct(c.ocr_confidence)})` : "") }));
  const sg = $("suggest"); sg.replaceChildren();
  for (const q of c.suggested_questions) {
    sg.append(el("button", { type: "button", text: q, onclick: () => { $("q").value = q; ask(); } }));
  }
  $("thread").replaceChildren(...(state.threads[id] ?? []));
  showTab(document.querySelector(".seg .active").dataset.tab);
  if (matchMedia("(max-width: 800px)").matches) $("detail").scrollIntoView({ behavior: "smooth" });
}

function showTab(name) {
  document.querySelectorAll(".seg button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) => (t.hidden = t.id !== "tab-" + name));
  const id = state.current?.doc_id;
  if (!id) return;
  if (name === "fields") loadFields(id);
  if (name === "summary") loadSummary(id);
  if (name === "text") loadText(id);
  if (name === "ask") setTimeout(() => $("q").focus(), 0);
}
const stale = (id) => state.current?.doc_id !== id;
const failed = (box, e) => box.replaceChildren(el("div", { class: "note bad", text: e.message }));

// ---------------------------------------------------------------- fields
const MONEY = new Set(["subtotal", "tax", "total", "amount", "unit_price"]);

function money(v, currency) {
  if (typeof v !== "number") return String(v);
  try {
    if (/^[A-Za-z]{3}$/.test(currency ?? "")) return new Intl.NumberFormat(undefined, { style: "currency", currency: currency.toUpperCase() }).format(v);
  } catch { /* unknown currency code: fall through */ }
  const n = v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  return currency ? `${currency} ${n}` : n;
}
const isEmpty = (v) => v == null || v === "" || (Array.isArray(v) && !v.length);
const label = (k) => k.replace(/_/g, " ");

function plainValue(k, v, currency) {
  if (isEmpty(v)) return "—";
  if (MONEY.has(k)) return money(v, currency);
  if (Array.isArray(v)) return v.map((x) => (typeof x === "object" ? Object.values(x).filter((y) => y != null).join(" · ") : x)).join(", ");
  return String(v);
}

function table(rows, currency) {
  const cols = [...new Set(rows.flatMap((r) => Object.keys(r)))];
  const t = el("table");
  t.append(el("tr", {}, ...cols.map((c) => el("th", { class: MONEY.has(c) || c === "quantity" ? "num" : "", text: label(c) }))));
  for (const r of rows) {
    t.append(el("tr", {}, ...cols.map((c) => el("td", {
      class: MONEY.has(c) || c === "quantity" ? "num" : "", text: r[c] == null ? "—" : MONEY.has(c) ? money(r[c], currency) : String(r[c]) }))));
  }
  return el("div", { class: "tbl-wrap" }, t);
}

function fieldCard(k, v, r) {
  const currency = r.final.currency;
  const cmp = r.comparison?.[k];
  const tag = !r.llm || !cmp || cmp === "both_missing" ? null
    : cmp === "match" ? el("span", { class: "tag ok", text: "✓ verified" })
    : cmp === "mismatch" ? el("span", { class: "tag bad", text: "⚠ methods differ" })
    : el("span", { class: "tag single", text: "one method" });
  const card = el("div", { class: "field" }, el("div", { class: "lab" }, el("span", { text: label(k) }), tag));
  if (isEmpty(v)) {
    card.append(el("div", { class: "val empty", text: "Not found" }));
  } else if (Array.isArray(v) && typeof v[0] === "object") {
    card.classList.add("wide"); card.append(table(v, currency));
  } else if (Array.isArray(v)) {
    card.classList.add(v.length > 4 ? "wide" : "x"); card.append(el("div", { class: "pills" }, ...v.map((x) => el("span", { text: x }))));
  } else {
    card.append(el("div", { class: "val", text: plainValue(k, v, currency) }));
  }
  if (cmp === "mismatch") {
    card.append(el("div", { class: "alt", text: `Rule-based: ${plainValue(k, r.baseline[k], currency)} · AI: ${plainValue(k, r.llm?.[k], currency)}` }));
  }
  return card;
}

async function loadFields(id) {
  const box = $("tab-fields");
  box.replaceChildren(skeleton());
  try {
    const r = (state.cache.fields ??= await post(`/documents/${id}/extract`, { use_llm: true }));
    if (stale(id)) return;
    box.replaceChildren();
    if (!Object.keys(r.final).length) {
      box.append(el("div", { class: "note", text: r.llm_note ?? "Field extraction applies to invoices, resumes and contracts." }));
      return;
    }
    if (r.llm_status !== "ok") {
      const why = { rate_limited: "the AI service is busy", unavailable: "no AI service is connected",
                    invalid_output: "the AI reply couldn't be used", skipped: "AI check was skipped" }[r.llm_status] ?? r.llm_status;
      box.append(el("div", { class: "note", text: `ℹ️ Showing rule-based results only (${why}).` }));
    }
    if (r.security_flags.length) {
      box.append(el("div", { class: "note bad", text: `🛡️ ${plural(r.security_flags.length, "line")} looked like an instruction aimed at the AI and ${r.security_flags.length === 1 ? "was" : "were"} withheld from it.` }));
    }
    const col = r.confidence >= 0.8 ? "var(--ok)" : r.confidence >= 0.5 ? "var(--warn)" : "var(--bad)";
    box.append(el("div", { class: "score" },
      el("div", { class: "ring", style: `--p:${Math.round(r.confidence * 100)};--rc:${col}` }, el("span", { text: pct(r.confidence) })),
      el("div", {}, el("b", { text: "Extraction confidence" }),
        el("span", { class: "muted", text: (r.llm_status === "ok" ? "Two methods cross-checked each field. " : "Single method, so confidence is capped. ") +
          `${pct(r.field_success_rate)} of fields found.` }))));
    box.append(el("div", { class: "fields" }, ...Object.entries(r.final).map(([k, v]) => fieldCard(k, v, r))));
  } catch (e) { failed(box, e); }
}

// ---------------------------------------------------------------- summary
async function loadSummary(id) {
  const box = $("tab-summary");
  box.replaceChildren(skeleton());
  try {
    const s = (state.cache.summary ??= await post(`/documents/${id}/summary`));
    if (stale(id)) return;
    box.replaceChildren(el("div", { class: "callout", text: s.summary }));
    if (s.key_points.length) box.append(el("ul", { class: "points" }, ...s.key_points.map((p) => el("li", { text: p }))));
    box.append(el("p", { class: "src-note", text: s.method === "llm" ? "✨ Written by AI and checked against the document."
      : "Built by picking the most representative sentences (no AI used)." }));
  } catch (e) { failed(box, e); }
}

// ---------------------------------------------------------------- text
async function loadText(id) {
  const pre = $("text");
  pre.textContent = "Loading…";
  try {
    const detail = (state.cache.detail ??= await api(`/documents/${id}`));
    const text = detail.pages.map((p) => p.text).join("\f");
    let marks = [];
    try {
      marks = (state.cache.fields ??= await post(`/documents/${id}/extract`, { use_llm: true })).highlights;
    } catch { /* highlights are optional */ }
    if (stale(id)) return;
    pre.replaceChildren();
    const gap = (s) => s.replace(/\f/g, "\n\n──────── page break ────────\n\n");
    let pos = 0;
    for (const m of marks) {
      if (m.start < pos) continue;
      pre.append(gap(text.slice(pos, m.start)), el("mark", { text: text.slice(m.start, m.end), title: label(m.field) }));
      pos = m.end;
    }
    pre.append(gap(text.slice(pos)));
  } catch (e) { pre.textContent = e.message; }
}

// ---------------------------------------------------------------- ask (chat)
function addMsg(id, node) {
  (state.threads[id] ??= []).push(node);
  if (state.current?.doc_id === id) { $("thread").append(node); node.scrollIntoView({ block: "nearest", behavior: "smooth" }); }
}

async function ask(ev) {
  ev?.preventDefault();
  const question = $("q").value.trim();
  const id = state.current?.doc_id;
  if (question.length < 3 || !id) return;
  $("q").value = "";
  const btn = document.querySelector(".send");
  btn.disabled = true;
  addMsg(id, el("div", { class: "msg user", text: question }));
  const bot = el("div", { class: "msg bot" }, el("div", { class: "dots" }, el("i"), el("i"), el("i")));
  addMsg(id, bot);
  try {
    const a = await post("/ask", { question, doc_ids: $("ask-all").checked ? null : [id] });
    bot.replaceChildren(el("div", { text: a.answer }));
    if (!a.found) bot.classList.add("refused");
    if (a.note) bot.append(el("div", { class: "meta", text: a.note }));
    if (a.citations.length) {
      bot.append(el("div", { class: "cites" }, ...a.citations.map((c) =>
        el("div", { class: "cite" }, el("b", { text: `${c.file} · page ${c.page}` }), el("div", { text: `“${c.text.slice(0, 260)}${c.text.length > 260 ? "…" : ""}”` })))));
    }
    if (a.support != null) bot.append(el("div", { class: "meta", text: `Backed by the source: ${pct(a.support)}` }));
  } catch (e) {
    bot.replaceChildren(el("div", { text: e.message }));
    bot.classList.add("refused");
  } finally { btn.disabled = false; $("q").focus(); }
}

// ---------------------------------------------------------------- wiring
const fileInput = $("file");
fileInput.onchange = (e) => { upload(e.target.files); e.target.value = ""; };
const openPicker = (e) => { if (e.type === "click" || e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } };
$("drop").onkeydown = openPicker;
$("drop-mini").onclick = openPicker; $("drop-mini").onkeydown = openPicker;

let dragDepth = 0;
addEventListener("dragenter", (e) => { if (e.dataTransfer?.types?.includes("Files")) { dragDepth++; $("drop-overlay").hidden = false; } });
addEventListener("dragleave", () => { dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) $("drop-overlay").hidden = true; });
addEventListener("dragover", (e) => e.preventDefault());
addEventListener("drop", (e) => { e.preventDefault(); dragDepth = 0; $("drop-overlay").hidden = true; upload(e.dataTransfer?.files ?? []); });

document.querySelectorAll(".seg button").forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));
$("ask-form").onsubmit = ask;
$("d-delete").onclick = async () => {
  const id = state.current?.doc_id;
  if (!id || !confirm("Delete this document? This can't be undone.")) return;
  try { await api(`/documents/${id}`, { method: "DELETE" }); } catch (e) { toast(e.message, "error"); }
  delete state.threads[id];
  state.current = null;
  await refresh();
  if (state.docs.length) select(state.docs[0].doc_id);
};

function renderSamples(names) {
  for (const host of [$("sample-list"), $("sample-mini")]) {
    host.replaceChildren(...names.map((n) => el("button", { class: "sample-btn", type: "button", text: n, onclick: () => loadSample(n) })));
  }
  $("sample-mini").prepend(el("span", { text: "Samples: " }));
}

(async () => {
  try {
    const h = await api("/health");
    const b = $("llm-badge");
    b.textContent = h.llm_enabled ? `● AI on · ${h.llm_provider}` : "○ AI off · rule-based";
    b.classList.add(h.llm_enabled ? "on" : "off");
    renderSamples(await api("/samples").catch(() => []));
    await refresh();
    if (state.docs.length) select(state.docs[0].doc_id);
  } catch { toast("Could not reach the server. Is it running?", "error"); }
})();
