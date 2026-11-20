import asyncio

from eval.ragas_eval import sample_inputs, score_all

QS = [{"qid": f"q{i}", "category": "paraphrase", "question": f"Q{i}?", "answerable": i != 3, "verified": i == 0}
      for i in range(6)]
ANS = {f"q{i}": {"text": "Use MFA [AC-2]."} for i in range(6)}
ANS["q1"] = {"text": "The provided IM8 controls do not answer this question."}
CTX = {f"q{i}": [f"ctx {i}"] for i in range(6)}


def test_sample_excludes_unanswerable_and_refusals_and_is_deterministic():
    a = sample_inputs(QS, CTX, ANS, 10)
    assert {x["qid"] for x in a} == {"q0", "q2", "q4", "q5"}
    assert a == sample_inputs(QS, CTX, ANS, 10)
    assert len(sample_inputs(QS, CTX, ANS, 2)) == 2


def test_score_all_means_and_counts_failures():
    items = sample_inputs(QS, CTX, ANS, 4)

    async def scorer(item):
        if item["qid"] == "q2":
            raise RuntimeError("judge failed")
        return {"faithfulness": 1.0, "answer_relevancy": 0.5}
    out = asyncio.run(score_all(items, scorer))
    assert out["n"] == 3 and out["n_failed"] == 1 and out["faithfulness"] == 1.0 and out["answer_relevancy"] == 0.5
