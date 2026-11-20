import pytest

from eval import metrics as M


def test_norm_maps_nonbreaking_hyphen_and_nfkc():
    assert "agency-defined" in M.norm("The value is agency‑defined.")
    assert M.norm("Ｌevel 1") == "level 1"        # fullwidth + nbsp -> NFKC
    assert M.norm("must‑have") == "must-have"


def test_retrieval_metrics():
    r = M.retrieval_metrics(["X", "B", "A"] + ["Z"] * 17, ["A", "B"])
    assert r["mrr"] == 0.5 and r["set_recall@5"] == 1.0 and r["hit@5"] == 1.0
    miss = M.retrieval_metrics(["X"] * 20, ["A"])
    assert miss["mrr"] == 0.0 and miss["set_recall@20"] == 0.0
    late = M.retrieval_metrics(["X"] * 10 + ["A"], ["A"])
    assert late["hit@5"] == 0.0 and late["hit@20"] == 1.0 and late["mrr"] == pytest.approx(1 / 11)


def test_prf_positive_is_should_abstain():
    m = M.prf([True, True, False, False], [True, False, True, False])
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert M.prf([False], [False])["f1"] == 0.0


def test_is_refusal_handles_unicode():
    assert M.is_refusal("The provided IM8 controls do not answer this question.")
    assert not M.is_refusal("Use MFA [AC-2].")


def test_applicability_check():
    a = "AS-5 is Level 1 (should-have) for low-risk systems and Level 0 (must-have) for medium-risk systems [AS-5]."
    assert M.check_applicability(a, "medium-risk", 0)
    assert M.check_applicability(a, "low-risk", 1)
    assert not M.check_applicability(a, "low-risk", 0)
    assert not M.check_applicability("It is a must-have and a good-to-have.", "low-risk", 0)


def test_no_level_check():
    assert M.check_no_level("No profile level is assigned to BR-4 [BR-4].", None)
    assert not M.check_no_level("BR-4 is Level 1 for low-risk [BR-4].", None)
    assert not M.check_no_level("The provided IM8 controls do not answer this question.", None)
    assert M.check_no_level("LM-20 has a level for low-risk only; no level is defined for medium-risk systems [LM-20].", "medium-risk")
    assert not M.check_no_level("LM-20 is Level 2 for medium-risk systems [LM-20].", "medium-risk")


def test_agency_param_check_rejects_invented_numbers_and_accepts_nbhyphen():
    assert M.check_agency_param("The retention period is agency‑defined [LM-8].")
    assert not M.check_agency_param("Retain logs for 90 days [LM-8]; the value is agency-defined.")
    assert not M.check_agency_param("Logs must be kept [LM-8].")


def test_score_answer_out_of_scope_and_false_refusal():
    oos = {"category": "out_of_scope", "answerable": False, "gold_answer": {}, "gold_control_ids": []}
    assert M.score_answer(oos, "x", True, [], [])["correct"]
    assert not M.score_answer(oos, "answer", False, [], [])["correct"]
    q = {"category": "exact_id", "answerable": True, "gold_answer": {}, "gold_control_ids": ["AS-5"]}
    s = M.score_answer(q, "", True, [], [])
    assert s["false_refusal"] and not s["correct"]
    ok = M.score_answer(q, "AS-5 sets password rules [AS-5].", False, ["AS-5"], [])
    assert ok["correct"] and ok["citations_valid"] and ok["gold_cited_recall"] == 1.0
    bad = M.score_answer(q, "See [CK-1] and AS-5", False, ["CK-1"], ["CK-1"])
    assert not bad["citations_valid"]


def test_score_answer_crosswalk_requires_all_gold():
    q = {"category": "crosswalk", "answerable": True, "gold_answer": {}, "gold_control_ids": ["AS-1", "AS-2"]}
    assert M.score_answer(q, "AS-1 and AS-2", False, [], [])["correct"]
    assert not M.score_answer(q, "AS-1 only", False, [], [])["correct"]


def test_aggregate_reports_unverified():
    a = M.aggregate([{"x": 1.0, "verified": False}, {"x": 0.0, "verified": True}], ["x"])
    assert a == {"n": 2, "n_unverified": 1, "x": 0.5}
