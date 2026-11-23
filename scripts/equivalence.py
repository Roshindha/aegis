"""Compare the torch reference models with the ONNX serving models over the whole benchmark.

Reports, per question: query-embedding cosine (torch vs ONNX), max |rerank score difference| over the
same 20 candidates, and whether the final top-5 ranking (full pipeline: hybrid -> rerank) is identical.
Needs requirements-eval.txt (torch).   python -m scripts.equivalence [--out eval/results/onnx-equivalence.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import aegis.retrieve as rv
from aegis import embed as onnx_embed
from aegis import rerank as onnx_rerank
from aegis import torch_ref
from aegis.retrieve import hybrid_search

QUESTIONS = Path("eval/questions.jsonl")


def summarise(cosines, diffs, same_top5, same_top1, n) -> dict:
    return {
        "questions": n,
        "embedding_cosine_min": float(np.min(cosines)), "embedding_cosine_mean": float(np.mean(cosines)),
        "rerank_abs_diff_max": float(np.max(diffs)), "rerank_abs_diff_mean": float(np.mean(diffs)),
        "top5_identical": int(sum(same_top5)), "top1_identical": int(sum(same_top1)),
        "top5_differs_on": [q for q, ok in zip(range(n), same_top5) if not ok],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/onnx-equivalence.json")
    args = ap.parse_args()
    qs = [json.loads(l) for l in QUESTIONS.read_text().splitlines() if l.strip()]
    cosines, diffs, same5, same1, diff_qids = [], [], [], [], []
    for q in qs:
        text = q["question"]
        cosines.append(float(onnx_embed.embed_query(text) @ torch_ref.embed_query(text)))

        # Candidates come from the serving path (ONNX query embedding); both rerankers score the same 20.
        cands = hybrid_search(text, k=20)
        docs = [h.content for h in cands]
        t_scores = np.array(torch_ref.rerank_scores(text, docs))
        o_scores = np.array(onnx_rerank.get_reranker().score(text, docs))
        diffs.append(float(np.max(np.abs(t_scores - o_scores))))

        # Whole pipeline, torch end to end vs ONNX end to end: do the final top-5 control IDs match?
        onnx_top = [h.control_id for h in onnx_rerank.rerank(text, hybrid_search(text, k=20), k=5)]
        orig_q = rv.embed_query
        rv.embed_query = torch_ref.embed_query
        try:
            t_cands = hybrid_search(text, k=20)
        finally:
            rv.embed_query = orig_q
        t_sc = torch_ref.rerank_scores(text, [h.content for h in t_cands])
        torch_top = [h.control_id for _, h in sorted(zip(t_sc, t_cands), key=lambda p: -p[0])][:5]
        same5.append(onnx_top == torch_top)
        same1.append(onnx_top[:1] == torch_top[:1])
        if onnx_top != torch_top:
            diff_qids.append({"qid": q["qid"], "torch": torch_top, "onnx": onnx_top})

    res = summarise(cosines, diffs, same5, same1, len(qs))
    res["top5_differs_on"] = diff_qids
    Path(args.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "top5_differs_on"}, indent=2))
    print("rankings differ on:", [d["qid"] for d in diff_qids] or "none")


if __name__ == "__main__":
    main()
