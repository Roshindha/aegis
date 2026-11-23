(function () {
  "use strict";
  const EXAMPLES = [
    ["Exact ID", "What does AS-5 require?"],
    ["Paraphrase", "How do we stop attackers from guessing passwords over and over on the login page?"],
    ["Crosswalk", "Which IM8 Reform controls reference CCM BCR-09?"],
    ["Agency-defined value", "What is the minimum password length required?"],
    ["Out of scope", "How does the EU AI Act classify high-risk AI systems?"],
  ];
  const STAGE_TITLES = {
    prefilter: "1. Pre-filter (regex rules)", crosswalk: "Crosswalk index lookup",
    semantic: "2a. Semantic search (bge-small)", bm25: "2b. BM25 keyword search",
    fusion: "3. Fusion (RRF, k=60)", rerank: "4. Cross-encoder rerank",
    guardrail: "5. Guardrail", answer: "6. Answer",
  };
  const local = ["localhost", "127.0.0.1", ""].includes(location.hostname);
  const params = new URLSearchParams(location.search);
  const API = (params.get("api") || (local ? "" : window.AEGIS_API_BASE || "")).replace(/\/$/, "");

  const $ = (id) => document.getElementById(id);
  function el(tag, attrs, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") n.className = v; else if (k === "text") n.textContent = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v); else n.setAttribute(k, v);
    }
    for (const kid of kids.flat()) if (kid != null) n.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    return n;
  }
  const fmt = (x, d = 3) => (x == null ? "n/a" : Number(x).toFixed(d));

  // ---------- wake-up ----------
  const status = $("status");
  let ready = false;
  async function waitForApi() {
    const started = Date.now();
    status.className = "status waking";
    status.textContent = "Waking up the server (free hosting, ~1 minute)…";
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
    status.textContent = "";
    status.append("The server did not respond. ", el("button", { class: "ex", onclick: waitForApi, type: "button" }, "Retry"));
  }

  // ---------- rendering ----------
  function rankTable(hits) {
    return el("table", {},
      el("thead", {}, el("tr", {}, ["#", "Control", "Title", "Score", "Sem rank", "BM25 rank"].map((h) => el("th", { text: h })))),
      el("tbody", {}, hits.map((h, i) => el("tr", {},
        el("td", { class: "num", text: i + 1 }), el("td", {}, el("b", { text: h.control_id })),
        el("td", { text: h.title }), el("td", { class: "num", text: fmt(h.score, 4) }),
        el("td", { class: "num", text: h.semantic_rank ?? "–" }), el("td", { class: "num", text: h.bm25_rank ?? "–" })))));
  }
  function rerankBars(hits) {
    const s = hits.map((h) => h.score), lo = Math.min(...s), hi = Math.max(...s), span = hi - lo || 1;
    return el("div", {}, hits.map((h) => el("div", { class: "barrow" },
      el("b", { text: h.control_id }),
      el("div", {}, el("div", { class: "bar", style: `width:${Math.max(3, ((h.score - lo) / span) * 100)}%` })),
      el("span", { class: "num", text: fmt(h.score, 2) }))),
      el("p", { class: "muted", text: "Raw cross-encoder scores (logits): higher is more relevant; negative values are normal. Bars are scaled within these results." }));
  }
  function guardrailBody(d) {
    const kids = [];
    kids.push(el("p", {}, el("span", { class: "badge " + (d.refuse ? "bad" : "ok"), text: d.refuse ? "REFUSED" : "PASSED" }),
      d.reason ? ` reason: ${d.reason}` : ""));
    if (d.score == null) {
      kids.push(el("p", { class: "muted", text: "Refused by the pre-filter: no retrieval was run and no model was called." }));
    } else {
      const thr = d.threshold, sc = d.score, max = Math.max(thr * 2, sc * 1.2, 1);
      kids.push(el("p", {}, `Signal `, el("code", { text: d.feature }), ` = ${fmt(sc, 2)} vs threshold ${fmt(thr, 2)}`,
        d.bypassed_threshold ? ` (threshold skipped: structural lookup, ${d.bypass_reason})` : ""));
      kids.push(el("div", { class: "meter", role: "img", "aria-label": `score ${fmt(sc, 2)} against threshold ${fmt(thr, 2)}` },
        el("div", { class: "fill", style: `width:${Math.min(100, (Math.max(sc, 0) / max) * 100)}%` }),
        el("div", { class: "mark", style: `left:${(thr / max) * 100}%` }),
        el("span", { class: "lab", style: `left:${(thr / max) * 100}%`, text: "threshold" })));
    }
    kids.push(el("p", { class: "muted", text: "The refusal is decided in code, before any LLM call." }));
    return kids;
  }
  function stagePanel(name, data) {
    let body, summary = "";
    if (name === "prefilter") {
      body = el("p", {}, el("span", { class: "badge " + (data.blocked ? "bad" : "ok"), text: data.blocked ? "BLOCKED" : "CLEAR" }), data.reason ? ` ${data.reason}` : " no rule matched");
      summary = data.blocked ? "blocked" : "clear";
    } else if (name === "crosswalk") {
      body = el("div", {}, el("p", { text: data.summary }), el("p", {}, "Controls: ", (data.controls || []).join(", ") || "none"));
      summary = (data.controls || []).length + " controls";
    } else if (["semantic", "bm25", "fusion"].includes(name)) {
      body = rankTable(data.hits); summary = data.hits.length ? "top: " + data.hits[0].control_id : "no hits";
    } else if (name === "rerank") {
      body = rerankBars(data.hits); summary = data.hits.length ? "top: " + data.hits[0].control_id : "";
    } else if (name === "guardrail") {
      body = guardrailBody(data); summary = data.refuse ? "refused" : "passed";
    } else if (name === "answer") {
      body = el("p", { class: "answer-text", text: data.text }); summary = (data.cited_ids || []).length + " citations";
    } else return null;
    const d = el("details", { open: "" }, el("summary", {}, STAGE_TITLES[name] || name, el("span", { class: "muted", text: summary })), el("div", { class: "body" }, body));
    return d;
  }
  function answerWithChips(text, hits, host) {
    const byId = Object.fromEntries(hits.map((h) => [h.control_id, h]));
    const detail = el("div", {});
    const parts = text.split(/(\[[A-Z]{2}-\d+\])/);
    const box = el("div", { class: "answer-text" });
    for (const p of parts) {
      const m = /^\[([A-Z]{2}-\d+)\]$/.exec(p);
      if (!m) { box.append(p); continue; }
      const id = m[1], src = byId[id];
      const chip = el("button", { class: "chip" + (src ? "" : " invalid"), type: "button", "aria-expanded": "false",
        title: src ? "Show control text" : "Not in the retrieved controls", text: id });
      chip.addEventListener("click", () => {
        const open = chip.getAttribute("aria-expanded") === "true";
        chip.setAttribute("aria-expanded", String(!open));
        const existing = detail.querySelector(`[data-id="${id}"]`);
        if (open && existing) existing.remove();
        else if (!open) detail.append(el("div", { class: "source", "data-id": id, text: src ? src.content : `${id} was cited but is not among the retrieved controls.` }));
      });
      box.append(chip);
    }
    host.append(box, detail);
  }

  // ---------- run ----------
  async function ask(question) {
    if (!ready) await waitForApi();
    const card = $("answer-card"), stages = $("stages");
    card.hidden = false; $("pipeline").hidden = false;
    card.replaceChildren(el("p", { class: "muted", text: "Running the pipeline…" }));
    stages.replaceChildren();
    $("go").disabled = true;
    const result = { hits: [] };
    let partial = "", error = null, got = false;
    try {
      const resp = await fetch(`${API}/ask/stream?` + new URLSearchParams({ question }));
      if (resp.status === 429) {
        const j = await resp.json().catch(() => ({}));
        card.replaceChildren(el("div", { class: "banner" }, j.message || "Too many requests. Please wait and try again."));
        return;
      }
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      const reader = resp.body.getReader(), dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf("\n\n")) >= 0) {
          const block = buf.slice(0, i); buf = buf.slice(i + 2);
          const name = (/^event: (.*)$/m.exec(block) || [])[1];
          const data = JSON.parse((/^data: (.*)$/m.exec(block) || [, "{}"])[1]);
          got = true;
          if (name === "result") Object.assign(result, data);
          else if (name === "error") error = data;
          else if (name === "answer") { partial = data.text; const p = stagePanel("answer", data); if (p) stages.append(p); }
          else if (name !== "done") { const p = stagePanel(name, data); if (p) stages.append(p); }
        }
      }
      renderCard(card, result, partial, error);
    } catch (e) {
      card.replaceChildren(el("div", { class: "banner err" }, got ? "The connection dropped mid-answer. Please try again." : "Could not reach the server. It may be waking up; please try again in a minute."));
      ready = false; waitForApi();
    } finally { $("go").disabled = false; }
  }
  function renderCard(card, r, partial, error) {
    card.replaceChildren();
    if (r.question === undefined && !partial && !error) { card.append(el("div", { class: "banner err", text: "No result returned." })); return; }
    card.append(el("span", { class: "badge " + (r.refused ? "bad" : "ok"), text: r.refused ? "Refused" : "Answered" }),
      r.refused ? el("span", { class: "muted", text: `  (${r.refusal_reason || "no relevant control"})` }) : "");
    if (error) card.append(el("div", { class: "banner" + (error.kind === "quota" ? "" : " err"), text: error.message }));
    const text = r.answer || partial;
    if (text) answerWithChips(text, r.hits || [], card);
    if (r.hits && r.hits.length && !r.refused) {
      card.append(el("p", { class: "muted", text: "Retrieved controls: " + r.hits.map((h) => h.control_id).join(", ") + ". Click a citation to read the control." }));
    }
    if (r.invalid_citations && r.invalid_citations.length) card.append(el("div", { class: "banner err", text: "Invalid citations (not in retrieved set): " + r.invalid_citations.join(", ") }));
  }

  // ---------- init ----------
  const ex = $("examples");
  for (const [label, q] of EXAMPLES) {
    ex.append(el("button", { class: "ex", type: "button", onclick: () => { $("q").value = q; ask(q); } }, el("small", { text: label }), q));
  }
  $("form").addEventListener("submit", (e) => { e.preventDefault(); const q = $("q").value.trim(); if (q.length >= 3) ask(q); });
  waitForApi();
})();
