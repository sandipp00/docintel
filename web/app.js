"use strict";

const $ = (id) => document.getElementById(id);
const state = { docs: [], current: null, cache: {} };

// Document text is untrusted: everything below is rendered with textContent, never innerHTML.
function el(tag, props = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") n.className = v; else if (k === "text") n.textContent = v; else n[k] = v;
  }
  for (const kid of kids) if (kid != null) n.append(kid);
  return n;
}

async function api(path, opts = {}) {
  const res = await fetch("/api" + path, opts);
  if (res.status === 204) return null;
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = body.detail;
    throw new Error(typeof d === "string" ? d : res.status === 429 ? "Too many requests; wait a moment."
                    : Array.isArray(d) ? "That input isn't valid." : `Request failed (${res.status}).`);
  }
  return body;
}
const post = (path, data) => api(path, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data ?? {}) });

function setStatus(msg, isError = false) {
  const s = $("upload-status"); s.textContent = msg; s.className = isError ? "error" : "";
}
const pct = (x) => (x == null ? "—" : Math.round(x * 100) + "%");

// ---------------------------------------------------------------- document list
function renderDocs() {
  const ul = $("docs"); ul.replaceChildren();
  for (const d of state.docs) {
    const li = el("li", { class: d.doc_id === state.current?.doc_id ? "active" : "" },
      el("div", { class: "name", text: d.filename }),
      el("span", { class: "tag", text: `${d.classification.label} · ${pct(d.classification.confidence)}` }),
      el("span", { class: "muted", text: `  ${d.n_pages} page${d.n_pages === 1 ? "" : "s"}` }));
    li.onclick = () => select(d.doc_id);
    ul.append(li);
  }
  $("empty").hidden = state.docs.length > 0 || !!state.current;
}

async function refresh() {
  state.docs = await api("/documents");
  renderDocs();
}

async function upload(files) {
  if (!files.length) return;
  const form = new FormData();
  for (const f of files) form.append("files", f);
  setStatus("Uploading and reading…");
  try {
    const added = await api("/upload", { method: "POST", body: form });
    setStatus(`Added ${added.length} document${added.length === 1 ? "" : "s"}.`);
    await refresh();
    select(added[0].doc_id);
  } catch (e) { setStatus(e.message, true); }
}

// ---------------------------------------------------------------- detail
async function select(id) {
  state.current = state.docs.find((d) => d.doc_id === id) ?? null;
  if (!state.current) return;
  renderDocs();
  $("detail").hidden = false; $("empty").hidden = true;
  const c = state.current;
  $("d-name").textContent = c.filename;
  $("d-meta").textContent = `${c.file_type.toUpperCase()} · ${c.n_pages} page(s) · ${c.n_chars} chars · ` +
    `read via ${c.method}${c.ocr_confidence != null ? ` (OCR confidence ${pct(c.ocr_confidence)})` : ""}`;
  $("answers").replaceChildren();
  const chips = $("chips"); chips.replaceChildren();
  for (const q of c.suggested_questions) {
    chips.append(el("button", { type: "button", text: q, onclick: () => { $("q").value = q; ask(); } }));
  }
  state.cache = {};
  showTab(document.querySelector(".tabs .active").dataset.tab);
}

function showTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) => (t.hidden = t.id !== "tab-" + name));
  const id = state.current?.doc_id;
  if (!id) return;
  if (name === "fields") loadFields(id);
  if (name === "summary") loadSummary(id);
  if (name === "text") loadText(id);
}

function loading(node, msg) { node.replaceChildren(el("p", { class: "muted", text: msg })); }
function failed(node, e) { node.replaceChildren(el("div", { class: "note bad", text: e.message })); }
const stale = (id) => state.current?.doc_id !== id;

function fmt(v) {
  if (v == null || v === "" || (Array.isArray(v) && !v.length)) return "—";
  if (Array.isArray(v)) {
    return v.map((x) => (typeof x === "object" ? Object.values(x).filter((y) => y != null).join(" · ") : x)).join("\n");
  }
  return String(v);
}

async function getFields(id) {
  return (state.cache.fields ??= await post(`/documents/${id}/extract`, { use_llm: true }));
}

async function loadFields(id) {
  const box = $("tab-fields");
  loading(box, "Extracting fields…");
  try {
    const r = await getFields(id);
    if (stale(id)) return;
    box.replaceChildren();
    if (!Object.keys(r.final).length) {
      box.append(el("p", { class: "muted", text: r.llm_note ?? "No fields are extracted for this document type." }));
      return;
    }
    if (r.llm_status !== "ok") {
      box.append(el("div", { class: "note", text: `Baseline extraction only (LLM ${r.llm_status.replace("_", " ")})` +
        (r.llm_note ? `: ${r.llm_note}` : ".") }));
    }
    if (r.security_flags.length) {
      box.append(el("div", { class: "note bad",
        text: `${r.security_flags.length} line(s) looked like prompt injection and were withheld from the LLM.` }));
    }
    const table = el("table");
    const head = el("tr", {}, el("th", { text: "Field" }), el("th", { text: "Value" }));
    if (r.llm) head.append(el("th", { text: "Rule-based" }), el("th", { text: "LLM" }), el("th", { text: "Agreement" }));
    table.append(head);
    for (const [k, v] of Object.entries(r.final)) {
      const cmp = r.comparison[k];
      const row = el("tr", {}, el("th", { text: k.replace(/_/g, " ") }), el("td", { class: "multi", text: fmt(v) }));
      if (r.llm) {
        row.append(el("td", { class: "multi", text: fmt(r.baseline[k]) }),
                   el("td", { class: "multi", text: fmt(r.llm[k]) }),
                   el("td", { class: cmp === "match" ? "cmp-match" : cmp === "mismatch" ? "cmp-mismatch" : "cmp-other",
                              text: (cmp ?? "").replace(/_/g, " ") }));
      }
      table.append(row);
    }
    box.append(table, el("p", { class: "muted",
      text: `Confidence ${pct(r.confidence)} · ${pct(r.field_success_rate)} of fields found` }));
  } catch (e) { failed(box, e); }
}

async function loadSummary(id) {
  const box = $("tab-summary");
  loading(box, "Summarising…");
  try {
    const s = (state.cache.summary ??= await post(`/documents/${id}/summary`));
    if (stale(id)) return;
    box.replaceChildren(el("p", { text: s.summary }));
    if (s.key_points.length) box.append(el("ul", { class: "points" }, ...s.key_points.map((p) => el("li", { text: p }))));
    box.append(el("p", { class: "muted",
      text: s.method === "llm" ? "Generated by the language model." : "Extractive summary (no LLM used)." }));
  } catch (e) { failed(box, e); }
}

async function loadText(id) {
  const pre = $("text");
  pre.textContent = "Loading…";
  try {
    const detail = (state.cache.detail ??= await api(`/documents/${id}`));
    const text = detail.pages.map((p) => p.text).join("\f");
    let marks = [];
    try { marks = (await getFields(id)).highlights; } catch { /* highlights are optional */ }
    if (stale(id)) return;
    pre.replaceChildren();
    const gap = (s) => s.replace(/\f/g, "\n\n──────── page break ────────\n\n");
    let pos = 0;
    for (const m of marks) {
      if (m.start < pos) continue;
      pre.append(gap(text.slice(pos, m.start)), el("mark", { text: text.slice(m.start, m.end), title: m.field }));
      pos = m.end;
    }
    pre.append(gap(text.slice(pos)));
  } catch (e) { pre.textContent = e.message; }
}

// ---------------------------------------------------------------- ask
async function ask(ev) {
  ev?.preventDefault();
  const question = $("q").value.trim();
  if (question.length < 3 || !state.current) return;
  const btn = document.querySelector("#ask-form button");
  btn.disabled = true;
  const body = { question, doc_ids: $("ask-all").checked ? null : [state.current.doc_id] };
  const card = el("div", { class: "answer" }, el("div", { class: "q", text: question }),
    el("p", { class: "muted", text: "Thinking…" }));
  $("answers").prepend(card);
  try {
    const a = await post("/ask", body);
    card.replaceChildren(el("div", { class: "q", text: question }), el("p", { text: a.answer }));
    if (!a.found) card.classList.add("refused");
    if (a.note) card.append(el("p", { class: "muted", text: a.note }));
    for (const c of a.citations) {
      card.append(el("blockquote", { text: `“${c.text.slice(0, 300)}” — ${c.file}, page ${c.page}` }));
    }
    if (a.support != null) card.append(el("p", { class: "muted", text: `Support ${pct(a.support)}` }));
  } catch (e) {
    card.replaceChildren(el("div", { class: "q", text: question }), el("div", { class: "note bad", text: e.message }));
  } finally { btn.disabled = false; }
}

// ---------------------------------------------------------------- wiring
const drop = $("drop");
$("file").onchange = (e) => { upload([...e.target.files]); e.target.value = ""; };
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); upload([...e.dataTransfer.files]); };
drop.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } };
document.querySelectorAll(".tabs button").forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));
$("ask-form").onsubmit = ask;
$("d-delete").onclick = async () => {
  const id = state.current?.doc_id;
  if (!id || !confirm("Delete this document?")) return;
  try {
    await api(`/documents/${id}`, { method: "DELETE" });
  } catch (e) { setStatus(e.message, true); }
  state.current = null; $("detail").hidden = true;
  await refresh();
};

(async () => {
  try {
    const h = await api("/health");
    const b = $("llm-badge");
    b.textContent = h.llm_enabled ? `LLM: ${h.llm_provider}` : "LLM off (rule-based only)";
    b.classList.add(h.llm_enabled ? "on" : "off");
    await refresh();
  } catch (e) { setStatus("Could not reach the server.", true); }
})();
