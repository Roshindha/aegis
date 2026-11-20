# Evaluation results (2026-11-20)

- commit `02729f5+dirty`, backend `local`, LLM `openai/gpt-oss-20b`
- 124 questions, **124 unverified** (curated questions are AI-drafted; `verified` is set by a human)

## Retrieval (answerable questions with gold controls)

| method | n | unverified | MRR | set-recall@5 | set-recall@20 | hit@5 | hit@20 |
|---|---|---|---|---|---|---|---|
| semantic | 98 | 98 | 0.577 | 0.638 | 0.762 | 0.704 | 0.796 |
| bm25 | 98 | 98 | 0.761 | 0.784 | 0.866 | 0.847 | 0.898 |
| hybrid | 98 | 98 | 0.735 | 0.768 | 0.877 | 0.827 | 0.908 |
| hybrid_rerank | 98 | 98 | 0.770 | 0.805 | 0.877 | 0.847 | 0.908 |
| hybrid_rerank_crosswalk | 98 | 98 | 0.937 | 0.951 | 0.999 | 0.990 | 1.000 |

### Set-recall@20 by category

| category | n | semantic | bm25 | hybrid | hybrid_rerank | hybrid_rerank_crosswalk |
|---|---|---|---|---|---|---|
| agency_param | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| applicability | 10 | 0.700 | 1.000 | 1.000 | 1.000 | 1.000 |
| crosswalk | 15 | 0.066 | 0.244 | 0.208 | 0.208 | 1.000 |
| crosswalk_reverse | 5 | 0.400 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_id | 10 | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 |
| group_list | 10 | 0.866 | 1.000 | 0.988 | 0.988 | 0.988 |
| multi_control | 7 | 1.000 | 0.881 | 1.000 | 1.000 | 1.000 |
| no_level | 13 | 0.923 | 1.000 | 1.000 | 1.000 | 1.000 |
| paraphrase | 22 | 1.000 | 0.955 | 1.000 | 1.000 | 1.000 |

### Set-recall@5 by category

| category | n | semantic | bm25 | hybrid | hybrid_rerank | hybrid_rerank_crosswalk |
|---|---|---|---|---|---|---|
| agency_param | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| applicability | 10 | 0.500 | 1.000 | 0.900 | 1.000 | 1.000 |
| crosswalk | 15 | 0.036 | 0.161 | 0.039 | 0.044 | 1.000 |
| crosswalk_reverse | 5 | 0.200 | 1.000 | 0.800 | 1.000 | 1.000 |
| exact_id | 10 | 0.500 | 1.000 | 0.900 | 1.000 | 1.000 |
| group_list | 10 | 0.534 | 0.629 | 0.647 | 0.652 | 0.652 |
| multi_control | 7 | 0.952 | 0.738 | 0.881 | 0.952 | 0.952 |
| no_level | 13 | 0.846 | 1.000 | 1.000 | 1.000 | 1.000 |
| paraphrase | 22 | 1.000 | 0.864 | 0.955 | 0.955 | 0.955 |

## Guardrail (abstention; positive class = should refuse)

Threshold feature: `bm25_top1` (chosen by held-out F1; `threshold_rerank_only` is the plain rerank-logit variant). Threshold configs use held-out (cross-validated) thresholds. The pre-filter was written after seeing the benchmark's out-of-scope questions, so its numbers are in-sample.

| config | n | unverified | precision | recall | F1 | false-refusal rate (answerable) |
|---|---|---|---|---|---|---|
| none | 124 | 124 | 0.000 | 0.000 | 0.000 | 0.000 |
| prefilter_only | 124 | 124 | 1.000 | 0.846 | 0.917 | 0.000 |
| threshold_rerank_only_heldout | 124 | 124 | 0.714 | 0.769 | 0.741 | 0.082 |
| threshold_only_heldout | 124 | 124 | 0.952 | 0.769 | 0.851 | 0.010 |
| both_heldout | 124 | 124 | 0.962 | 0.962 | 0.962 | 0.010 |
| both_shipped_threshold | 124 | 124 | 1.000 | 0.962 | 0.980 | 0.000 |

## Generation (124/124 questions answered)

| config | n answerable | unverified | answer correct | false refusals | has citation | answers w/ all-valid citations | cited-ID validity | OOS correctly refused | abstention P / R / F1 |
|---|---|---|---|---|---|---|---|---|---|
| no_guardrail | 98 | 98 | 87.8% | 4.1% | 68.1% | 100.0% | 100.0% | 96.2% | 0.86 / 0.96 / 0.91 |
| guardrail | 98 | 98 | 87.8% | 4.1% | 68.1% | 100.0% | 100.0% | 100.0% | 0.87 / 1.00 / 0.93 |

### Answer correctness by category (guardrail config)

| category | n | unverified | correct |
|---|---|---|---|
| agency_param | 6 | 6 | 100.0% |
| applicability | 10 | 10 | 100.0% |
| crosswalk | 15 | 15 | 100.0% |
| crosswalk_reverse | 5 | 5 | 100.0% |
| exact_id | 10 | 10 | 100.0% |
| group_list | 10 | 10 | 30.0% |
| multi_control | 7 | 7 | 85.7% |
| no_level | 13 | 13 | 76.9% |
| out_of_scope | 26 | 26 | 100.0% |
| paraphrase | 22 | 22 | 95.5% |

## RAGAS (sample of 6 answers, judge: openai/gpt-oss-20b via Groq)

- faithfulness: 0.722
- answer relevancy: 0.880
