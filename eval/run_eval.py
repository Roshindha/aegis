"""Run the full evaluation and write eval/results/<date>.json and .md.

    python -m eval.run_eval --retrieval-only      # no API calls, no key needed
    python -m eval.run_eval                       # + generation (Groq), cached on disk
    python -m eval.run_eval --ragas 20            # + RAGAS on a sample of 20 answers

Every number in the README must come from a results file written by this script.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from datetime import date
from pathlib import Path
from typing import Callable

from aegis import crosswalk, guardrail
from aegis.generate import DEFAULT_MODEL
from aegis.guardrail import REFUSAL_TEXT
from aegis.pipeline import CANDIDATES
from aegis.rerank import rerank
from aegis.retrieve import Hit, bm25_search, hybrid_search, rrf_fuse, semantic_search
from eval import metrics as M
from eval.calibrate import heldout_thresholds
from eval.llm import DailyLimitReached, load_cache, make_cached_answerer
from eval.validate import load_questions, validate
from aegis.corpus import load_chunks

RESULTS = Path("eval/results")
RET_KEYS = ["mrr", "set_recall@5", "set_recall@20", "hit@5", "hit@20"]
GEN_KEYS = ["correct", "false_refusal", "has_citation", "citations_valid", "gold_cited_recall"]
SEED = 5


def build_retrievers() -> dict[str, Callable[[str], list[Hit]]]:
    """method name -> function(question) -> top-20 hits, best first."""
    def sem(q): return semantic_search(q, 20)
    def bm(q): return bm25_search(q, 20)
    def hyb(q): return rrf_fuse([semantic_search(q, CANDIDATES), bm25_search(q, CANDIDATES)], k=20)
    def hyb_rr(q): return rerank(q, hyb(q), k=20)
    retrievers = {"semantic": sem, "bm25": bm, "hybrid": hyb, "hybrid_rerank": hyb_rr}
    retrievers["hybrid_rerank_crosswalk"] = lambda q: crosswalk.crosswalk_search(q, hyb_rr)
    return retrievers


# ---------- retrieval ----------

def eval_retrieval(questions: list[dict], ranked: dict[str, dict[str, list[str]]]) -> dict:
    """ranked[method][qid] = control IDs best-first. Answerable questions with gold IDs only."""
    out: dict = {}
    for method, per_q in ranked.items():
        rows = []
        for q in questions:
            if q["answerable"] and q["gold_control_ids"]:
                rows.append({**M.retrieval_metrics(per_q[q["qid"]], q["gold_control_ids"]),
                             "category": q["category"], "verified": q["verified"]})
        out[method] = {"overall": M.aggregate(rows, RET_KEYS),
                       "by_category": {c: M.aggregate(r, RET_KEYS) for c, r in sorted(M.group_by(rows, "category").items())}}
    return out


# ---------- guardrail (decision level; needs no LLM) ----------

def guardrail_decisions(questions: list[dict], signals: dict[str, dict[str, float]]) -> dict[str, dict[str, bool]]:
    """config -> qid -> refused?   signals[feature][qid] = relevance signal. Threshold configs
    use HELD-OUT thresholds (cross-validated), except the one explicitly labelled 'shipped'."""
    byp = {q["qid"]: guardrail.structural_bypass(q["question"]) is not None for q in questions}
    free = [q for q in questions if not byp[q["qid"]]]
    feature, final_t = guardrail.load_calibration()

    def heldout(feat: str) -> dict[str, float]:
        return dict(zip((q["qid"] for q in free), heldout_thresholds(
            [signals[feat][q["qid"]] for q in free], [not q["answerable"] for q in free])))

    held, held_rr = heldout(feature), heldout("rerank_top1")
    cfgs = ("none", "prefilter_only", "threshold_rerank_only_heldout", "threshold_only_heldout",
            "both_heldout", "both_shipped_threshold")
    out: dict[str, dict[str, bool]] = {k: {} for k in cfgs}
    for q in questions:
        qid, text = q["qid"], q["question"]
        pre = guardrail.prefilter(text).blocked
        free_q = not byp[qid]
        thr_h = free_q and signals[feature][qid] < held[qid] if free_q else False
        thr_rr = free_q and signals["rerank_top1"][qid] < held_rr[qid] if free_q else False
        thr_f = free_q and signals[feature][qid] < final_t
        out["none"][qid] = False
        out["prefilter_only"][qid] = pre
        out["threshold_rerank_only_heldout"][qid] = thr_rr
        out["threshold_only_heldout"][qid] = thr_h
        out["both_heldout"][qid] = pre or thr_h
        out["both_shipped_threshold"][qid] = pre or thr_f
    return out


def eval_guardrail(questions: list[dict], decisions: dict[str, dict[str, bool]]) -> dict:
    truth = [not q["answerable"] for q in questions]
    res = {}
    for cfg, d in decisions.items():
        pred = [d[q["qid"]] for q in questions]
        ans = [q for q in questions if q["answerable"]]
        res[cfg] = {**M.prf(pred, truth),
                    "false_refusal_rate": sum(d[q["qid"]] for q in ans) / len(ans),
                    "n": len(questions), "n_unverified": sum(not q["verified"] for q in questions)}
    return res


# ---------- generation ----------

def eval_generation(questions, hits_by_q, answerer, decisions_both, max_questions=None) -> dict:
    """One LLM call per question (cached). Two system configs share it:
    no_guardrail  -> abstains only if the LLM itself refuses;
    guardrail     -> abstains if the guardrail refuses (LLM never consulted) or the LLM refuses."""
    order = sorted(questions, key=lambda q: q["qid"])
    random.Random(SEED).shuffle(order)                  # partial runs are random samples, not one category
    if max_questions:
        order = order[:max_questions]
    rows: dict[str, list[dict]] = {"no_guardrail": [], "guardrail": []}
    answers: dict[str, dict] = {}
    complete = True
    for i, q in enumerate(order, 1):
        try:
            ans = answerer(q["question"], hits_by_q[q["qid"]][:crosswalk.context_size(hits_by_q[q["qid"]], 5)])
        except DailyLimitReached as e:
            print(f"\nGroq daily limit reached after {i - 1} questions: {str(e)[:160]}")
            complete = False
            break
        llm_refused = M.is_refusal(ans.text)
        answers[q["qid"]] = {"text": ans.text, "cited": ans.cited_ids, "invalid": ans.invalid_citations}
        base = {"qid": q["qid"], "category": q["category"], "verified": q["verified"]}
        rows["no_guardrail"].append({**base, **M.score_answer(q, ans.text, llm_refused, ans.cited_ids, ans.invalid_citations)})
        if decisions_both[q["qid"]]:
            rows["guardrail"].append({**base, **M.score_answer(q, REFUSAL_TEXT, True, [], [])})
        else:
            rows["guardrail"].append({**base, **M.score_answer(q, ans.text, llm_refused, ans.cited_ids, ans.invalid_citations)})
        if i % 10 == 0:
            print(f"  generated {i}/{len(order)}", flush=True)
    by_id = {q["qid"]: q for q in questions}
    out: dict = {"complete": complete, "n_answered": len(rows["guardrail"]), "n_total": len(questions), "configs": {}}
    for cfg, rs in rows.items():
        ans_rows = [r for r in rs if by_id[r["qid"]]["answerable"]]
        cited = sum(r.get("cited_n", 0) for r in ans_rows)
        invalid = sum(r.get("cited_invalid_n", 0) for r in ans_rows)
        pred = [r["abstained"] for r in rs]
        truth = [not by_id[r["qid"]]["answerable"] for r in rs]
        out["configs"][cfg] = {
            "answerable": M.aggregate(ans_rows, GEN_KEYS),
            "cited_id_validity": (1 - invalid / cited) if cited else None,
            "abstention": {**M.prf(pred, truth), "n": len(rs), "n_unverified": sum(not r["verified"] for r in rs)},
            "out_of_scope_correct": M.aggregate([r for r in rs if not by_id[r["qid"]]["answerable"]], ["correct"]),
            "by_category": {c: M.aggregate(r, ["correct"]) for c, r in sorted(M.group_by(rs, "category").items())},
        }
    out["answers"] = answers
    return out


# ---------- reporting ----------

def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def render_markdown(res: dict) -> str:
    L = [f"# Evaluation results ({res['date']})", "",
         f"- commit `{res['commit']}`, backend `{res['backend']}`, LLM `{res.get('model', 'n/a')}`",
         f"- {res['n_questions']} questions, **{res['n_unverified']} unverified** "
         f"(curated questions are AI-drafted; `verified` is set by a human)", ""]
    L += ["## Retrieval (answerable questions with gold controls)", "",
          "| method | n | unverified | MRR | set-recall@5 | set-recall@20 | hit@5 | hit@20 |", "|---|---|---|---|---|---|---|---|"]
    for m, d in res["retrieval"].items():
        o = d["overall"]
        L.append(f"| {m} | {o['n']} | {o['n_unverified']} | {o['mrr']:.3f} | {o['set_recall@5']:.3f} | {o['set_recall@20']:.3f} | {o['hit@5']:.3f} | {o['hit@20']:.3f} |")
    cats = sorted({c for d in res["retrieval"].values() for c in d["by_category"]})
    L += ["", "### Set-recall@20 by category", "", "| category | n | " + " | ".join(res["retrieval"]) + " |",
          "|---|---|" + "---|" * len(res["retrieval"])]
    for c in cats:
        n = next(d["by_category"][c]["n"] for d in res["retrieval"].values() if c in d["by_category"])
        L.append(f"| {c} | {n} | " + " | ".join(f"{d['by_category'][c]['set_recall@20']:.3f}" if c in d["by_category"] else "-" for d in res["retrieval"].values()) + " |")
    L += ["", "### Set-recall@5 by category", "", "| category | n | " + " | ".join(res["retrieval"]) + " |",
          "|---|---|" + "---|" * len(res["retrieval"])]
    for c in cats:
        n = next(d["by_category"][c]["n"] for d in res["retrieval"].values() if c in d["by_category"])
        L.append(f"| {c} | {n} | " + " | ".join(f"{d['by_category'][c]['set_recall@5']:.3f}" if c in d["by_category"] else "-" for d in res["retrieval"].values()) + " |")
    L += ["", "## Guardrail (abstention; positive class = should refuse)", "",
          f"Threshold feature: `{(res.get('threshold') or {}).get('feature')}` (chosen by held-out F1; `threshold_rerank_only` is the plain rerank-logit variant). "
          "Threshold configs use held-out (cross-validated) thresholds. The pre-filter was written after seeing the "
          "benchmark's out-of-scope questions, so its numbers are in-sample.", "",
          "| config | n | unverified | precision | recall | F1 | false-refusal rate (answerable) |", "|---|---|---|---|---|---|---|"]
    for cfg, g in res["guardrail"].items():
        L.append(f"| {cfg} | {g['n']} | {g['n_unverified']} | {g['precision']:.3f} | {g['recall']:.3f} | {g['f1']:.3f} | {g['false_refusal_rate']:.3f} |")
    gen = res.get("generation")
    if gen:
        L += ["", f"## Generation ({gen['n_answered']}/{gen['n_total']} questions answered"
                  + ("" if gen["complete"] else "; **incomplete: Groq daily limit hit, resume later**") + ")", "",
              "| config | n answerable | unverified | answer correct | false refusals | has citation | answers w/ all-valid citations | cited-ID validity | OOS correctly refused | abstention P / R / F1 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for cfg, c in gen["configs"].items():
            a, ab = c["answerable"], c["abstention"]
            L.append(f"| {cfg} | {a['n']} | {a['n_unverified']} | {_pct(a['correct'])} | {_pct(a['false_refusal'])} | {_pct(a['has_citation'])} | "
                     f"{_pct(a['citations_valid'])} | {_pct(c['cited_id_validity'])} | {_pct(c['out_of_scope_correct']['correct'])} | "
                     f"{ab['precision']:.2f} / {ab['recall']:.2f} / {ab['f1']:.2f} |")
        L += ["", "### Answer correctness by category (guardrail config)", "", "| category | n | unverified | correct |", "|---|---|---|---|"]
        for cat, a in gen["configs"]["guardrail"]["by_category"].items():
            L.append(f"| {cat} | {a['n']} | {a['n_unverified']} | {_pct(a['correct'])} |")
    rg = res.get("ragas")
    if rg:
        L += ["", f"## RAGAS (sample of {rg['n']} answers, judge: {rg.get('judge')})", "",
              f"- faithfulness: {rg['faithfulness']:.3f}", f"- answer relevancy: {rg['answer_relevancy']:.3f}"]
    return "\n".join(L) + "\n"


def git_state() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "aegis", "eval", "corpus"], text=True).strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:                                               # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--ragas", type=int, default=0, metavar="N", help="RAGAS on N sampled answers")
    ap.add_argument("--limit", type=int, default=None, help="max questions to send to the LLM")
    ap.add_argument("--questions", default="eval/questions.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    questions = load_questions(Path(args.questions))
    errs = validate(questions, load_chunks())
    if errs:
        print("\n".join(errs))
        return 1
    retrievers = build_retrievers()
    ranked: dict[str, dict[str, list[str]]] = {m: {} for m in retrievers}
    final_name = "hybrid_rerank_crosswalk" if "hybrid_rerank_crosswalk" in retrievers else "hybrid_rerank"
    final_hits: dict[str, list[Hit]] = {}
    for i, q in enumerate(questions, 1):
        for m, fn in retrievers.items():
            hits = fn(q["question"])
            ranked[m][q["qid"]] = [h.control_id for h in hits]
            if m == final_name:
                final_hits[q["qid"]] = hits
        if i % 25 == 0:
            print(f"  retrieved {i}/{len(questions)}", flush=True)

    signals: dict[str, dict[str, float]] = {f: {} for f in guardrail.FEATURES}
    base = retrievers["hybrid_rerank"]
    for q in questions:                                             # guardrail signals, as the pipeline computes them
        bm = bm25_search(q["question"], CANDIDATES)
        ranked_q = base(q["question"])[:5]
        for f in guardrail.FEATURES:
            v = guardrail.feature_value(f, ranked_q, bm)
            signals[f][q["qid"]] = float("-inf") if v is None else v

    decisions = guardrail_decisions(questions, signals)
    res = {"date": date.today().isoformat(), "commit": git_state(),
           "backend": os.environ.get("AEGIS_BACKEND", "local"), "n_questions": len(questions),
           "n_unverified": sum(not q["verified"] for q in questions),
           "final_system": final_name, "retrieval": eval_retrieval(questions, ranked),
           "guardrail": eval_guardrail(questions, decisions),
           "threshold": json.loads(guardrail.THRESHOLD_FILE.read_text()) if guardrail.THRESHOLD_FILE.exists() else None}

    if not args.retrieval_only:
        from aegis.generate import answer_question
        model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
        res["model"] = model
        cache = load_cache()
        answerer = make_cached_answerer(answer_question, cache, model)
        res["generation"] = eval_generation(questions, final_hits, answerer, decisions["both_shipped_threshold"], args.limit)
        if args.ragas:
            from eval.ragas_eval import run_ragas
            res["ragas"] = run_ragas(questions, final_hits, res["generation"]["answers"], args.ragas)

    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = Path(args.out) if args.out else RESULTS / (res["date"] + ("-retrieval" if args.retrieval_only else ""))
    stem.with_suffix(".json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    stem.with_suffix(".md").write_text(render_markdown(res), encoding="utf-8")
    print(render_markdown(res))
    print(f"Wrote {stem.with_suffix('.json')} and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
