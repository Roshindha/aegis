"""Fill the README results block from committed eval result files (no hand-copied numbers).

    python -m eval.update_readme eval/results/final.json eval/results/baseline-pre-crosswalk.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def compose(final: dict, baseline: dict | None, final_name: str, baseline_name: str | None) -> str:
    ret = final["retrieval"]
    L = [f"Run `{final_name}` · date {final['date']} · commit `{final['commit']}` · retrieval backend `{final['backend']}`"
         + (f" · LLM `{final['model']}`" if final.get("model") else ""),
         "",
         f"**{final['n_questions']} benchmark questions, {final['n_unverified']} of them human-unverified.** "
         "All curated questions were drafted with AI assistance and have not yet been human-verified; "
         "every metric below carries that caveat.", "",
         "### Retrieval ablation", "",
         "Answerable questions that have gold controls. Set-recall = fraction of gold controls found; hit = at least one found.", "",
         "| method | n | unverified | MRR | set-recall@5 | set-recall@20 | hit@5 | hit@20 |", "|---|---|---|---|---|---|---|---|"]
    for m, d in ret.items():
        o = d["overall"]
        L.append(f"| {m} | {o['n']} | {o['n_unverified']} | {o['mrr']:.3f} | {o['set_recall@5']:.3f} | {o['set_recall@20']:.3f} | {o['hit@5']:.3f} | {o['hit@20']:.3f} |")
    cats = sorted({c for d in ret.values() for c in d["by_category"]})
    L += ["", "MRR by category:", "", "| category | n | " + " | ".join(ret) + " |", "|---|---|" + "---|" * len(ret)]
    for c in cats:
        n = next(d["by_category"][c]["n"] for d in ret.values() if c in d["by_category"])
        L.append(f"| {c} | {n} | " + " | ".join(f"{d['by_category'][c]['mrr']:.3f}" if c in d["by_category"] else "-" for d in ret.values()) + " |")

    if baseline:
        b = baseline["retrieval"]["hybrid_rerank"]["by_category"]
        a = ret.get("hybrid_rerank_crosswalk", {}).get("by_category", {})
        L += ["", f"### Crosswalk: baseline vs after (baseline run `{baseline_name}`)", "",
              "| category | n | metric | baseline (hybrid+rerank) | with crosswalk mode |", "|---|---|---|---|---|"]
        for c in ("crosswalk", "crosswalk_reverse"):
            if c in b and c in a:
                for k in ("set_recall@5", "set_recall@20", "hit@5", "mrr"):
                    L.append(f"| {c} | {b[c]['n']} | {k} | {b[c][k]:.3f} | {a[c][k]:.3f} |")
        bg, ag = baseline.get("generation"), final.get("generation")
        if bg and ag:
            L += ["", "End-to-end answer correctness on those questions (guardrail config):", "",
                  "| category | baseline correct | with crosswalk correct | n answered (baseline / after) |", "|---|---|---|---|"]
            for c in ("crosswalk", "crosswalk_reverse"):
                x, y = bg["configs"]["guardrail"]["by_category"].get(c), ag["configs"]["guardrail"]["by_category"].get(c)
                if x and y:
                    L.append(f"| {c} | {_pct(x['correct'])} | {_pct(y['correct'])} | {x['n']} / {y['n']} |")
        L += ["", "Caveat: crosswalk gold labels come from the same `references` data the index is built from, so the "
                  "post-crosswalk numbers measure query detection and routing, not independent truth."]

    L += ["", "### Guardrail (abstention; positive class = should refuse)", "",
          "Threshold configs use held-out thresholds (5-fold cross-validation). The regex pre-filter was written after "
          "seeing the benchmark's out-of-scope questions, so its numbers are in-sample and optimistic.", "",
          "| config | n | unverified | precision | recall | F1 | false-refusal rate on answerable |", "|---|---|---|---|---|---|---|"]
    for cfg, g in final["guardrail"].items():
        L.append(f"| {cfg} | {g['n']} | {g['n_unverified']} | {g['precision']:.3f} | {g['recall']:.3f} | {g['f1']:.3f} | {g['false_refusal_rate']:.3f} |")
    t = final.get("threshold")
    if t:
        L += ["", f"Shipped threshold: feature `{t['feature']}` ≥ {t['threshold']}. Candidates (held-out F1): "
              + ", ".join(f"`{f}` {c['cv_heldout']['f1']:.3f}" for f, c in t["candidates"].items()) + "."]

    gen = final.get("generation")
    if gen:
        L += ["", f"### Generation ({gen['n_answered']}/{gen['n_total']} questions answered"
              + ("" if gen["complete"] else "; INCOMPLETE") + ")", "",
              "| config | n answerable | unverified | answer correct | false refusals | answers with all-valid citations | cited-ID validity | out-of-scope refused | abstention P / R / F1 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for cfg, c in gen["configs"].items():
            x, ab = c["answerable"], c["abstention"]
            L.append(f"| {cfg} | {x['n']} | {x['n_unverified']} | {_pct(x['correct'])} | {_pct(x['false_refusal'])} | {_pct(x['citations_valid'])} | "
                     f"{_pct(c['cited_id_validity'])} | {_pct(c['out_of_scope_correct']['correct'])} | {ab['precision']:.2f} / {ab['recall']:.2f} / {ab['f1']:.2f} |")
        L += ["", "Answer correctness by category (guardrail config):", "", "| category | n | unverified | correct |", "|---|---|---|---|"]
        for cat, x in gen["configs"]["guardrail"]["by_category"].items():
            L.append(f"| {cat} | {x['n']} | {x['n_unverified']} | {_pct(x['correct'])} |")
    rg = final.get("ragas")
    if rg:
        L += ["", f"### RAGAS (sample of {rg['n']} answers; judge `{rg.get('judge')}`)", "",
              f"faithfulness {rg['faithfulness']:.3f} · answer relevancy {rg['answer_relevancy']:.3f}"]
    return "\n".join(L)


def update(readme: Path, block: str) -> None:
    text = readme.read_text(encoding="utf-8")
    pat = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    new = f"{START}\n{block}\n{END}"
    if not pat.search(text):
        raise SystemExit(f"README is missing the {START} ... {END} markers")
    readme.write_text(pat.sub(lambda _: new, text), encoding="utf-8")


def main() -> None:
    fp = Path(sys.argv[1])
    bp = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    final = json.loads(fp.read_text())
    base = json.loads(bp.read_text()) if bp else None
    update(Path("README.md"), compose(final, base, fp.name, bp.name if bp else None))
    print("README results block updated from", fp)


if __name__ == "__main__":
    main()
