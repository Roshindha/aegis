"""Calibrate the refusal threshold on the benchmark.

The guardrail refuses when the top cross-encoder score < t. We choose t to maximise
abstention F1 (positive class = "should refuse" = unanswerable question), but we do NOT
report the F1 measured on the questions used to choose t. Instead:

  * stratified K-fold cross-validation: for each fold, t is chosen on the other folds and
    applied to the held-out fold; the pooled held-out predictions give the reported F1;
  * the shipped threshold is then fitted on all questions (more data = more stable t).

Lookup-style questions (real control ID, group title, crosswalk reference) bypass the threshold at inference, so they are
excluded here: the threshold only ever decides the others.

    python -m eval.calibrate
"""
from __future__ import annotations

import json
import random
from datetime import date
from pathlib import Path

from aegis import guardrail
from aegis.pipeline import retrieve
from eval.validate import load_questions

OUT = Path("eval/threshold.json")
FOLDS = 5
SEED = 13


def f1_scores(scores: list[float], labels: list[bool], t: float) -> dict:
    """Abstention metrics when we refuse iff score < t. labels[i] True = should refuse."""
    pred = [s < t for s in scores]
    tp = sum(p and y for p, y in zip(pred, labels))
    fp = sum(p and not y for p, y in zip(pred, labels))
    fn = sum((not p) and y for p, y in zip(pred, labels))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def best_threshold(scores: list[float], labels: list[bool]) -> float:
    """The t maximising F1. Candidates are midpoints between neighbouring distinct scores;
    among ties we take the middle candidate (the most margin on both sides)."""
    uniq = sorted(set(scores))
    cands = [uniq[0] - 1.0] + [(a + b) / 2 for a, b in zip(uniq, uniq[1:])] + [uniq[-1] + 1.0]
    f = [f1_scores(scores, labels, t)["f1"] for t in cands]
    best = max(f)
    tied = [t for t, v in zip(cands, f) if v == best]
    return tied[len(tied) // 2]


def stratified_folds(labels: list[bool], k: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in (True, False):
        idx = [i for i, y in enumerate(labels) if y == cls]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % k].append(i)
    return folds


def cross_validate(scores: list[float], labels: list[bool], k: int = FOLDS, seed: int = SEED) -> dict:
    folds = stratified_folds(labels, k, seed)
    pooled_pred: list[bool] = []
    pooled_y: list[bool] = []
    per_fold_t = []
    for test in folds:
        test_set = set(test)
        tr = [i for i in range(len(scores)) if i not in test_set]
        t = best_threshold([scores[i] for i in tr], [labels[i] for i in tr])
        per_fold_t.append(round(t, 3))
        pooled_pred += [scores[i] < t for i in test]
        pooled_y += [labels[i] for i in test]
    tp = sum(p and y for p, y in zip(pooled_pred, pooled_y))
    fp = sum(p and not y for p, y in zip(pooled_pred, pooled_y))
    fn = sum((not p) and y for p, y in zip(pooled_pred, pooled_y))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "fold_thresholds": per_fold_t}


def top_scores(questions: list[dict]) -> list[float]:
    out = []
    for q in questions:
        hits = retrieve("hybrid_rerank", q["question"], k=5)
        out.append(hits[0].score if hits else float("-inf"))
    return out


def main() -> None:
    qs = [q for q in load_questions(Path("eval/questions.jsonl"))
          if guardrail.structural_bypass(q["question"]) is None]    # threshold never decides these
    scores = top_scores(qs)
    labels = [not q["answerable"] for q in qs]
    t = best_threshold(scores, labels)
    cv = cross_validate(scores, labels)
    in_sample = f1_scores(scores, labels, t)
    out = {
        "threshold": round(t, 4),
        "method": f"maximise abstention F1 over midpoints of observed top-1 rerank scores; "
                  f"reported F1 is from stratified {FOLDS}-fold CV (seed {SEED}); shipped t fitted on all",
        "n_questions": len(qs), "n_should_refuse": sum(labels),
        "n_excluded_bypass": len(load_questions(Path("eval/questions.jsonl"))) - len(qs),
        "n_unverified": sum(not q["verified"] for q in qs),
        "cv_heldout": {k: round(v, 4) if isinstance(v, float) else v for k, v in cv.items()},
        "in_sample": {k: round(v, 4) for k, v in in_sample.items()},
        "date": date.today().isoformat(),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
