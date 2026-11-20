from eval.update_readme import END, START, compose, update

AGG = {"n": 2, "n_unverified": 2, "mrr": 0.5, "set_recall@5": 0.5, "set_recall@20": 0.6, "hit@5": 0.5, "hit@20": 0.7}
RES = {"date": "2026-01-01", "commit": "abc", "backend": "local", "n_questions": 5, "n_unverified": 5,
       "retrieval": {"hybrid_rerank": {"overall": AGG, "by_category": {"crosswalk": AGG, "crosswalk_reverse": AGG}},
                     "hybrid_rerank_crosswalk": {"overall": AGG, "by_category": {"crosswalk": {**AGG, "mrr": 1.0}, "crosswalk_reverse": AGG}}},
       "guardrail": {"none": {"n": 5, "n_unverified": 5, "precision": 0.0, "recall": 0.0, "f1": 0.0, "false_refusal_rate": 0.0}}}


def test_compose_has_unverified_caveat_and_before_after():
    md = compose(RES, RES, "final.json", "base.json")
    assert "5 of them human-unverified" in md and "baseline vs after" in md and "| crosswalk | 2 | mrr | 0.500 | 1.000 |" in md


def test_update_replaces_only_marked_block(tmp_path):
    p = tmp_path / "README.md"
    p.write_text(f"before\n{START}\nold\n{END}\nafter\n")
    update(p, "NEW")
    assert p.read_text() == f"before\n{START}\nNEW\n{END}\nafter\n"
