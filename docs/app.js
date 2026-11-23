(function () {
  "use strict";

  const EXAMPLES = [
    ["Exact ID", "What does AS-5 require?", "--c-prefilter"],
    ["Paraphrase", "How do we stop attackers from guessing passwords over and over on the login page?", "--c-semantic"],
    ["Crosswalk", "Which IM8 Reform controls reference CCM BCR-09?", "--c-bm25"],
    ["Agency-defined", "What is the minimum password length required?", "--c-fusion"],
    ["Out of scope", "How does the EU AI Act classify high-risk AI systems?", "--c-rerank"],
  ];
  // Stage order on the signal line. "crosswalk" is inserted only when the API reports it.
  const STAGES = {
    prefilter: ["Pre-filter", "Regex rules, no model involved", "--c-prefilter"],
    crosswalk: ["Crosswalk", "Structured reference index lookup", "--c-crosswalk"],
    semantic: ["Semantic", "bge-small embeddings, cosine similarity", "--c-semantic"],
    bm25: ["BM25", "Keyword match over the same 137 controls", "--c-bm25"],
    fusion: ["Fusion", "Reciprocal rank fusion, k = 60", "--c-fusion"],
    rerank: ["Rerank", "A cross-encoder re-reads the top 20", "--c-rerank"],
    guardrail: ["Guardrail", "The refusal is decided in code, before any model call", "--c-guardrail"],
    answer: ["Answer", "Grounded in the retrieved controls", "--c-answer"],
  };
  const BASE_ORDER = ["prefilter", "semantic", "bm25", "fusion", "rerank", "guardrail", "answer"];
  const SLOPE_COLS = ["semantic", "bm25", "fusion", "rerank"];

  const local = ["localhost", "127.0.0.1", ""].includes(location.hostname);
  const params = new URLSearchParams(location.search);
  const API = (params.get("api") || (local ? "" : window.AEGIS_API_BASE || "")).replace(/\/$/, "");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  const $ = (id) => document.getElementById(id);
  function el(tag, attrs, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === "class") n.className = v; else if (k === "text") n.textContent = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v); else n.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid != null) n.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    return n;
  }
  const svgEl = (tag, attrs) => { const n = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v); return n; };
  const fmt = (x, d = 3) => (x == null || Number.isNaN(x) ? "n/a" : Number(x).toFixed(d));
  // The model often emits non-breaking hyphens (BR‑4); normalise so IDs and chips match.
  const clean = (s) => String(s).normalize("NFKC").replace(/[‐-―−]/g, "-");
  const stageVar = (name) => `var(${STAGES[name][2]})`;

  // ---------- theme ----------
  const themeBtn = $("theme");
  function setTheme(t, save) {
    document.documentElement.dataset.theme = t;
    themeBtn.textContent = t === "dark" ? "Light" : "Dark";
    themeBtn.setAttribute("aria-pressed", String(t === "dark"));
    themeBtn.setAttribute("aria-label", t === "dark" ? "Switch to light theme" : "Switch to dark theme");
    if (save) { try { localStorage.setItem("aegis-theme", t); } catch (e) { /* private mode: ignore */ } }
  }
  setTheme(document.documentElement.dataset.theme === "dark" ? "dark" : "light", false);
  themeBtn.addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark", true));

  // ---------- health polling ("waking up") ----------
  const status = $("status");
  let ready = false;
  async function waitForApi() {
    const started = Date.now();
    status.className = "status";
    status.replaceChildren("Waking up the server (free hosting, ~1 minute)…", el("span", { class: "bar" }));
    while (Date.now() - started < 4 * 60 * 1000) {
      try {
        const ctl = new AbortController();
        const t = setTimeout(() => ctl.abort(), 10000);
        const r = await fetch(API + "/health", { signal: ctl.signal });
        clearTimeout(t);
        if (r.ok) {
          const h = await r.json();
          ready = true;
          status.className = "status ready";
          status.textContent = `Server ready · retrieval backend: ${h.backend} · model: ${h.model}`;
          return;
        }
      } catch (e) { /* still waking: retry */ }
      await new Promise((res) => setTimeout(res, 4000));
    }
    status.className = "status fail";
    status.replaceChildren("The server did not respond. ", el("button", { type: "button", onclick: waitForApi }, "Retry"));
  }

  // ---------- highlight (a control's path through the stages) ----------
  const run = $("run");
  let pinned = null, defaultHl = null;
  function highlight(id) {
    const target = id || pinned || defaultHl;
    for (const n of run.querySelectorAll("[data-id]")) {
      const on = !!target && n.dataset.id === target;
      n.classList.toggle("hl", on);
      n.classList.toggle("dim", !!target && !on && !!n.closest(".slope"));   // only the slopegraph fades; tables and bars stay legible
    }
  }
  run.addEventListener("pointerover", (e) => { const n = e.target.closest("[data-id]"); if (n && e.pointerType !== "touch") highlight(n.dataset.id); });
  run.addEventListener("pointerout", (e) => { if (e.target.closest("[data-id]") && e.pointerType !== "touch") highlight(null); });
  run.addEventListener("focusin", (e) => { const n = e.target.closest("[data-id]"); if (n) highlight(n.dataset.id); });
  run.addEventListener("focusout", () => highlight(null));
  run.addEventListener("click", (e) => {
    const n = e.target.closest("[data-id]");
    if (!n || n.closest(".answer")) return;
    pinned = pinned === n.dataset.id ? null : n.dataset.id;
    highlight(null);
  });

  // ---------- stage scaffolding ----------
  let state;   // per-run: { nodes: {name: li}, order: [], touched: Set, open: name|null, data: {} }
  function newRun() {
    state = { nodes: {}, order: [], touched: new Set(), open: null, data: {}, got: 0 };
    pinned = null; defaultHl = null;
    $("line").replaceChildren();
    for (const name of BASE_ORDER) addStage(name, null, "pending");
  }
  function addStage(name, beforeName, mode) {
    const [title, desc] = STAGES[name];
    const li = el("li", { class: "stage " + mode, style: `--stage: ${stageVar(name)}`, "data-stage": name });
    const head = el("button", { class: "head", type: "button", "aria-expanded": "false", "aria-controls": "d-" + name, disabled: mode === "pending" },
      el("span", { class: "name", text: title }),
      el("span", { class: "sum" }),
      el("span", { class: "desc", text: desc }),
      el("span", { class: "ph" }));
    head.addEventListener("click", () => { state.touched.add(name); setOpen(name, !li.classList.contains("open")); });
    const inner = el("div", { class: "inner" }, el("div", { class: "detail", id: "d-" + name }));
    inner.setAttribute("inert", "");
    li.append(el("span", { class: "node", "aria-hidden": "true" }), head, el("div", { class: "collapse" }, inner));
    state.nodes[name] = li;
    const line = $("line");
    const ref = beforeName && state.nodes[beforeName];
    if (ref) { line.insertBefore(li, ref); state.order.splice(state.order.indexOf(beforeName), 0, name); } else { line.append(li); state.order.push(name); }
    return li;
  }
  function setOpen(name, open) {
    const li = state.nodes[name];
    if (!li) return;
    li.classList.toggle("open", open);
    li.querySelector(".head").setAttribute("aria-expanded", String(open));
    const inner = li.querySelector(".inner");
    if (open) inner.removeAttribute("inert"); else inner.setAttribute("inert", "");
  }
  function activate(name, summary) {
    let li = state.nodes[name];
    if (!li) li = addStage(name, "semantic", "pending");           // the optional crosswalk stage
    li.classList.remove("pending", "skipped");
    li.querySelector(".head").disabled = false;
    li.querySelector(".sum").textContent = summary || "";
    li.classList.add("entering"); setTimeout(() => li.classList.remove("entering"), 400);
    const i = state.order.indexOf(name);                            // draw the segment leading into this stage
    for (let j = i - 1; j >= 0; j--) { const prev = state.nodes[state.order[j]]; if (!prev.classList.contains("pending") && !prev.classList.contains("skipped")) { prev.classList.add("next"); break; } }
    if (state.open && !state.touched.has(state.open) && state.open !== name) setOpen(state.open, false);   // latest stage open by default
    state.open = name; setOpen(name, true);
    return li.querySelector(".detail");
  }

  // ---------- stage renderers ----------
  function rankTable(name, hits, withRanks) {
    const head = ["#", "Control", "Title", name === "bm25" ? "BM25 score" : name === "fusion" ? "RRF score" : "Cosine"];
    if (withRanks) head.push("Sem", "BM25");
    return el("div", { class: "tablewrap", style: "margin:.25rem 0 0" }, el("table", { class: "rank" },
      el("thead", {}, el("tr", {}, head.map((h, i) => el("th", { scope: "col", class: i >= 3 ? "num" : "", text: h })))),
      el("tbody", {}, hits.map((h, i) => el("tr", { "data-id": h.control_id },
        el("td", { text: i + 1 }), el("td", {}, el("button", { class: "id", type: "button", "data-id": h.control_id, text: h.control_id })),
        el("td", { text: h.title }), el("td", { class: "num", text: fmt(h.score, name === "bm25" ? 2 : 4) }),
        withRanks ? [el("td", { class: "num", text: h.semantic_rank ?? "–" }), el("td", { class: "num", text: h.bm25_rank ?? "–" })] : null)))));
  }
  function bars(hits) {
    const sc = hits.map((h) => h.score), lo = Math.min(0, ...sc), hi = Math.max(0, ...sc), span = hi - lo || 1;
    const pos = (v) => ((v - lo) / span) * 100, z = pos(0);
    const rows = hits.map((h) => {
      const a = Math.min(z, pos(h.score)), w = Math.abs(pos(h.score) - z);
      return el("div", { class: "brow", "data-id": h.control_id },
        el("button", { class: "id", type: "button", "data-id": h.control_id, text: h.control_id }),
        el("div", { class: "track" }, el("div", { class: "fillbar", style: `left:${a}%;width:${Math.max(w, .6)}%` }), el("div", { class: "zero", style: `left:${z}%` })),
        el("span", { class: "num", text: fmt(h.score, 2) }));
    });
    const axis = el("div", { class: "axis" }, el("span"), el("div", { class: "zl" }, el("span", { style: `left:${z}%`, text: "0" })), el("span"));
    return el("div", { class: "bars" }, rows, axis,
      el("p", { class: "caption", text: "Raw cross-encoder scores (logits). Higher is more relevant; negative values are normal and extend left of the zero line." }));
  }

  // rank movement: a slopegraph on wide screens, a ranked list on narrow ones
  function slopegraph() {
    const lists = SLOPE_COLS.map((c) => (state.data[c] || []).slice(0, 8));
    const host = $("slope-host");
    if (!host || lists.some((l) => !l.length)) return;
    host.replaceChildren();
    const W = Math.max(host.clientWidth, 560), rowH = 30, headH = 34, labW = 64, padL = 4;
    const N = Math.max(...lists.map((l) => l.length)), H = headH + N * rowH + 6;
    const step = (W - labW - padL * 2) / (SLOPE_COLS.length - 1);
    const x = (c) => padL + c * step, y = (r) => headH + r * rowH + rowH / 2;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": "Rank movement of the top controls through semantic, BM25, fusion and rerank" });
    const links = svgEl("g", {}), nodes = svgEl("g", {});
    svg.append(links, nodes);
    SLOPE_COLS.forEach((c, ci) => {
      nodes.append(svgEl("rect", { x: x(ci), y: 8, width: 9, height: 9, fill: stageVar(c), stroke: "var(--rule-strong)" }));
      const t = svgEl("text", { x: x(ci) + 15, y: 16, class: "col" }); t.textContent = STAGES[c][0]; nodes.append(t);
      lists[ci].forEach((h, r) => {
        const g = svgEl("g", { class: "nd", "data-id": h.control_id, tabindex: "0", role: "button", "aria-label": `${h.control_id}, rank ${r + 1} in ${STAGES[c][0]}` });
        g.append(svgEl("rect", { x: x(ci) - 2, y: y(r) - rowH / 2 + 2, width: labW + 4, height: rowH - 4, fill: "transparent" }));
        const rk = svgEl("text", { x: x(ci), y: y(r) + 4, class: "rk" }); rk.textContent = String(r + 1); g.append(rk);
        const lb = svgEl("text", { x: x(ci) + 16, y: y(r) + 4, class: "lab" }); lb.textContent = h.control_id; g.append(lb);
        g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pinned = pinned === h.control_id ? null : h.control_id; highlight(null); } });
        nodes.append(g);
      });
    });
    for (let ci = 0; ci < SLOPE_COLS.length - 1; ci++) {
      lists[ci].forEach((h, r) => {
        const r2 = lists[ci + 1].findIndex((o) => o.control_id === h.control_id);
        if (r2 < 0) return;
        const x1 = x(ci) + labW, x2 = x(ci + 1) - 4, xm = (x1 + x2) / 2;
        links.append(svgEl("path", { class: "link", "data-id": h.control_id, d: `M${x1},${y(r)} C${xm},${y(r)} ${xm},${y(r2)} ${x2},${y(r2)}`, stroke: stageVar(SLOPE_COLS[ci + 1]) }));
      });
    }
    host.append(svg);
    // narrow screens: the same movement as a ranked list
    const ids = [...new Set([...lists[3], ...lists[2], ...lists[0], ...lists[1]].map((h) => h.control_id))];
    const rk = (c, id) => { const i = lists[SLOPE_COLS.indexOf(c)].findIndex((h) => h.control_id === id); return i < 0 ? "–" : String(i + 1); };
    host.append(el("table", { class: "rankpath" },
      el("thead", {}, el("tr", {}, el("th", { scope: "col", text: "Control" }), SLOPE_COLS.map((c) => el("th", { scope: "col", class: "num", text: STAGES[c][0] })))),
      el("tbody", {}, ids.map((id) => el("tr", { "data-id": id }, el("td", {}, el("button", { class: "id", type: "button", "data-id": id, text: id })),
        SLOPE_COLS.map((c) => el("td", { class: "num", text: rk(c, id) })))))));
    highlight(null);
  }
  let resizeT;
  window.addEventListener("resize", () => { clearTimeout(resizeT); resizeT = setTimeout(() => { if (state && state.data.rerank) slopegraph(); }, 150); });

  function guardrailBody(d) {
    const kids = [];
    if (d.score == null) {
      kids.push(el("p", { class: "verdict refuse", text: "Refused before retrieval." }),
        el("p", { class: "vmeta", text: `Reason: ${d.reason || "pre-filter"}. No search ran and no model was called.` }));
      return kids;
    }
    const thr = d.threshold, sc = d.score;
    const lo = Math.min(0, sc * 1.15), hi = Math.max(thr * 1.8, sc * 1.15, 1), span = hi - lo;
    const p = (v) => ((v - lo) / span) * 100;
    kids.push(el("p", { class: "verdict " + (d.refuse ? "refuse" : "pass"), text: d.refuse ? "Refused before the model was called." : "Answered." }),
      el("p", { class: "vmeta" }, `${d.feature} = ${fmt(sc, 2)} · threshold ${fmt(thr, 2)}`,
        d.bypassed_threshold ? ` · threshold skipped: structural lookup (${d.bypass_reason})` : "", d.reason ? ` · reason: ${d.reason}` : ""));
    kids.push(el("div", { class: "gauge", role: "img", "aria-label": `Score ${fmt(sc, 2)} against threshold ${fmt(thr, 2)}: ${d.refuse ? "refused" : "passed"}` },
      el("div", { class: "zr", style: `width:${p(thr)}%`, text: "refuse" }), el("div", { class: "za", style: `width:${100 - p(thr)}%`, text: "answer" }),
      el("div", { class: "tick", style: `left:${p(thr)}%` }, el("span", { text: `threshold ${fmt(thr, 2)}` })),
      el("div", { class: "score", style: `left:${p(sc)}%` }, el("span", { text: `score ${fmt(sc, 2)}` }))));
    kids.push(el("p", { class: "caption", text: "The refusal is decided in code from this number, never by the language model." }));
    return kids;
  }

  function answerBody(text, hits, refused, reason, error, cited) {
    const host = el("div", { class: "answer" });
    host.append(el("p", { class: "kick", text: refused ? `Refused${reason ? " · " + reason : ""}` : "Answer" }));
    const box = el("p", { class: "text" });
    const byId = Object.fromEntries((hits || []).map((h) => [h.control_id, h]));
    const panel = el("div", {});
    for (const part of clean(text || "").split(/(\[[A-Z]{2}-\d+\])/)) {
      const m = /^\[([A-Z]{2}-\d+)\]$/.exec(part);
      if (!m) { box.append(part); continue; }
      const id = m[1], src = byId[id], pid = "src-" + id;
      const chip = el("button", { class: "chip" + (src ? "" : " invalid"), type: "button", "aria-expanded": "false", "aria-controls": pid, title: src ? "Show the control text" : "Not among the retrieved controls", text: `[${id}]` });
      chip.addEventListener("click", () => {
        const open = chip.getAttribute("aria-expanded") === "true";
        chip.setAttribute("aria-expanded", String(!open));
        const ex = panel.querySelector("#" + pid);
        if (open && ex) { ex.classList.remove("open"); setTimeout(() => ex.remove(), reduced ? 0 : 220); }
        else if (!open && !ex) {
          const w = el("div", { class: "src", id: pid }, el("div", {}, el("p", { text: src ? src.content : `${id} was cited but is not among the retrieved controls.` })));
          panel.append(w); requestAnimationFrame(() => requestAnimationFrame(() => w.classList.add("open")));
        }
      });
      box.append(chip);
    }
    host.append(box, panel);
    if (error) host.append(el("div", { class: "notice" + (error.kind === "quota" ? "" : " err"), text: error.message }));
    if (hits && hits.length && !refused) host.append(el("p", { class: "retrieved", text: "Retrieved: " + hits.map((h) => h.control_id).join(", ") + ". Select a citation to read the control." }));
    if (cited && cited.length) host.append(el("div", { class: "notice err", text: "Citations not in the retrieved set: " + cited.join(", ") }));
    return host;
  }

  // ---------- one event ----------
  function onEvent(name, data) {
    state.got++;
    if (name === "prefilter") { activate("prefilter", data.blocked ? "blocked" : "clear").replaceChildren(
      el("p", { class: "verdict " + (data.blocked ? "refuse" : "pass"), text: data.blocked ? "Blocked." : "Clear." }),
      el("p", { class: "vmeta", text: data.blocked ? `Rule matched: ${data.reason}` : "No pre-filter rule matched this question." })); }
    else if (name === "crosswalk") { activate("crosswalk", `${(data.controls || []).length} controls`).replaceChildren(
      el("p", { text: data.summary }), el("p", { class: "vmeta" }, "Controls: ", (data.controls || []).map((c) => el("button", { class: "id", type: "button", "data-id": c, text: c + " " })))); }
    else if (["semantic", "bm25", "fusion"].includes(name)) {
      state.data[name] = data.hits;
      activate(name, data.hits.length ? `top: ${data.hits[0].control_id}` : "no hits").replaceChildren(rankTable(name, data.hits, name === "fusion"));
    } else if (name === "rerank") {
      state.data.rerank = data.hits;
      defaultHl = data.hits.length ? data.hits[0].control_id : null;
      const d = activate("rerank", data.hits.length ? `top: ${data.hits[0].control_id}` : "");
      d.replaceChildren(el("h3", { text: "Rerank scores" }), bars(data.hits));
      const li = state.nodes.rerank;
      const old = li.querySelector(".slopewrap"); if (old) old.remove();
      li.append(el("div", { class: "slopewrap" }, el("h3", { text: "Rank movement, top 8" }), el("p", { class: "caption", text: "Each line follows one control from one ranking to the next. Hover, focus or tap an ID to follow it." }), el("div", { class: "slope", id: "slope-host" })));
      requestAnimationFrame(slopegraph);
    } else if (name === "guardrail") {
      state.nodes.guardrail.style.setProperty("--stage", data.refuse ? "var(--refuse)" : "var(--c-guardrail)");
      activate("guardrail", data.refuse ? "refused" : "passed").replaceChildren(...guardrailBody(data));
    } else if (name === "answer") {
      state.answerText = data.text;
      activate("answer", `${(data.cited_ids || []).length} citations`).replaceChildren(answerBody(data.text, [], false, null, null, null));
    }
  }
  function finish(result, error) {
    const refused = !!result.refused, text = result.answer || state.answerText || "";
    const li = state.nodes.answer;
    const detail = activate("answer", refused ? "refused" : error ? "retrieval only" : `${(result.cited_ids || []).length} citations`);
    detail.replaceChildren(answerBody(text, result.hits || [], refused, result.refusal_reason, error, result.invalid_citations));
    li.scrollIntoView({ block: "nearest", behavior: reduced ? "auto" : "smooth" });
    for (const name of state.order) {                      // stages that never ran
      const n = state.nodes[name];
      if (n.classList.contains("pending")) {
        n.classList.replace("pending", "skipped");
        n.querySelector(".sum").textContent = "not run";
        n.querySelector(".ph").remove();
      }
    }
  }

  // ---------- run a question ----------
  async function ask(question) {
    if (!ready) await waitForApi();
    run.hidden = false;
    newRun();
    $("go").disabled = true;
    const result = {};
    let error = null, buf = "";
    try {
      const resp = await fetch(`${API}/ask/stream?` + new URLSearchParams({ question }));
      if (resp.status === 429) {
        const j = await resp.json().catch(() => ({}));
        run.hidden = true;
        status.className = "status fail";
        status.textContent = j.message || "Too many requests. Please wait and try again.";
        return;
      }
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      const reader = resp.body.getReader(), dec = new TextDecoder();
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf("\n\n")) >= 0) {
          const block = buf.slice(0, i); buf = buf.slice(i + 2);
          const name = (/^event: (.*)$/m.exec(block) || [])[1];
          const data = JSON.parse((/^data: (.*)$/m.exec(block) || [, "{}"])[1]);
          if (name === "result") Object.assign(result, data);
          else if (name === "error") error = data;
          else if (name !== "done") onEvent(name, data);
        }
      }
      finish(result, error);
    } catch (e) {
      const dropped = state && state.got > 0;
      finish(result, { kind: "llm_error", message: dropped ? "The connection dropped mid-answer. Please try again." : "Could not reach the server. It may be waking up; please try again in a minute." });
      ready = false; waitForApi();
    } finally { $("go").disabled = false; }
  }

  // ---------- init ----------
  const ex = $("examples");
  for (const [label, q, v] of EXAMPLES) {
    ex.append(el("li", {}, el("button", { class: "ex", type: "button", onclick: () => { $("q").value = q; ask(q); } },
      el("span", { class: "t", style: `--stage: var(${v})` }, el("i", { "aria-hidden": "true" }), label), el("span", { class: "q", text: q }))));
  }
  $("form").addEventListener("submit", (e) => { e.preventDefault(); const q = $("q").value.trim(); if (q.length >= 3) ask(q); });
  waitForApi();
})();
