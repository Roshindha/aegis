"""RAGAS faithfulness + answer relevancy on a sample of generated answers.

Judge = the same Groq model through Groq's OpenAI-compatible endpoint (so no extra key);
embeddings for answer relevancy = the local bge-small model. Because RAGAS pulls in many
dependencies (requirements-eval.txt), run_eval only WRITES the inputs file; this module
scores it and merges the numbers into the results JSON:

    python -m eval.run_eval --ragas 15 --out eval/results/final     # writes final.json + final-ragas-inputs.json
    pip install -r requirements-eval.txt
    python -m eval.ragas_eval eval/results/final.json

RAGAS's own LLM-judged scores are noisy with a 20B judge and ~15 samples; treat as indicative.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
from pathlib import Path
from typing import Awaitable, Callable

from eval.metrics import is_refusal

SEED = 11
JUDGE_BASE_URL = "https://api.groq.com/openai/v1"


def sample_inputs(questions: list[dict], contexts_by_q: dict[str, list[str]], answers: dict[str, dict],
                  n: int, seed: int = SEED) -> list[dict]:
    """A deterministic sample of answerable questions the LLM actually answered (not refusals)."""
    pool = [q for q in sorted(questions, key=lambda q: q["qid"])
            if q["answerable"] and q["qid"] in answers and not is_refusal(answers[q["qid"]]["text"])
            and answers[q["qid"]]["text"].strip()]
    random.Random(seed).shuffle(pool)
    return [{"qid": q["qid"], "category": q["category"], "verified": q["verified"], "question": q["question"],
             "answer": answers[q["qid"]]["text"], "contexts": contexts_by_q[q["qid"]]} for q in pool[:n]]


async def score_all(inputs: list[dict], scorer: Callable[[dict], Awaitable[dict]]) -> dict:
    """scorer(item) -> {"faithfulness": float, "answer_relevancy": float}. Failures are counted, not hidden."""
    ok, failed = [], []
    for i, item in enumerate(inputs, 1):
        try:
            ok.append(await scorer(item))
        except Exception as e:                                       # noqa: BLE001
            failed.append({"qid": item["qid"], "error": f"{type(e).__name__}: {str(e)[:200]}"})
        print(f"  ragas {i}/{len(inputs)}", flush=True)
    vals = lambda k: [r[k] for r in ok if r.get(k) is not None]
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    return {"n": len(ok), "n_requested": len(inputs), "n_failed": len(failed), "failures": failed[:5],
            "faithfulness": mean(vals("faithfulness")), "answer_relevancy": mean(vals("answer_relevancy")),
            "n_unverified": sum(not it["verified"] for it in inputs)}


def make_groq_scorer() -> Callable[[dict], Awaitable[dict]]:
    """Build real RAGAS metrics (imports ragas lazily)."""
    from openai import AsyncOpenAI, RateLimitError
    from ragas.embeddings.huggingface_provider import HuggingFaceEmbeddings
    from ragas.llms.base import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, Faithfulness

    from aegis.embed import MODEL_NAME
    from aegis.generate import DEFAULT_MODEL

    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    client = AsyncOpenAI(base_url=JUDGE_BASE_URL, api_key=os.environ["GROQ_API_KEY"], max_retries=6)
    llm = llm_factory(model, provider="openai", client=client)
    faith = Faithfulness(llm=llm)
    rel = AnswerRelevancy(llm=llm, embeddings=HuggingFaceEmbeddings(model=MODEL_NAME))

    async def retrying(coro_fn, tries=6):
        for t in range(tries):
            try:
                return await coro_fn()
            except RateLimitError:
                await asyncio.sleep(2 ** t * 3)
        raise RuntimeError("rate limited")

    async def scorer(item: dict) -> dict:
        f = await retrying(lambda: faith.ascore(user_input=item["question"], response=item["answer"],
                                                retrieved_contexts=item["contexts"]))
        r = await retrying(lambda: rel.ascore(user_input=item["question"], response=item["answer"]))
        return {"faithfulness": f.value, "answer_relevancy": r.value}
    return scorer


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv()
    results_path = Path(sys.argv[1])
    inputs = json.loads(results_path.with_name(results_path.stem + "-ragas-inputs.json").read_text())
    res = json.loads(results_path.read_text())
    out = asyncio.run(score_all(inputs, make_groq_scorer()))
    out["judge"] = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b") + " via Groq"
    res["ragas"] = out
    results_path.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    from eval.run_eval import render_markdown
    results_path.with_suffix(".md").write_text(render_markdown(res), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if out["n"] else 1


if __name__ == "__main__":
    sys.exit(main())
