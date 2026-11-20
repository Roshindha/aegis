import copy

from eval.generate_questions import generate, reference_index
from eval.validate import summary, validate


def _chunk(cid, levels, refs=(), group="as", title="Application Security", text=None):
    return {"control_id": cid, "levels": levels, "group_id": group, "group_title": title,
            "references": [{"raw": r} for r in refs],
            "text": text or f"{cid}: T\nStatement: do [agency-defined: country]"}


def fake_chunks():
    cs = []
    for i in range(1, 15):
        cs.append(_chunk(f"AS-{i}", {"low-risk": i % 3, "medium-risk": (i + 1) % 3},
                         refs=[f"MVSP 2.{i % 4}: x", f"NIST SP 800-53 AC-{i % 5}: y"]))
    cs += [_chunk("LM-20", {"low-risk": 2}, group="lm", title="Logging"),
           _chunk("BR-4", {}, group="br", title="Backup"),
           _chunk("BR-5", {}, group="br", title="Backup")]
    return cs


def test_generation_is_deterministic():
    assert generate(fake_chunks()) == generate(fake_chunks())


def test_generated_questions_validate():
    cs = fake_chunks()
    assert validate(generate(cs), cs) == []


def test_one_class_control_gets_missing_class_no_level_question():
    qs = generate(fake_chunks())
    miss = [q for q in qs if q["category"] == "no_level" and q["gold_answer"].get("risk_class") == "medium-risk"]
    assert [q["gold_control_ids"] for q in miss] == [["LM-20"]]
    assert miss[0]["gold_answer"]["level"] is None


def test_level_less_controls_get_null_level():
    qs = generate(fake_chunks())
    nl = {q["gold_control_ids"][0] for q in qs if q["category"] == "no_level" and "risk_class" not in q["gold_answer"]}
    assert nl == {"BR-4", "BR-5"}


def test_crosswalk_gold_is_reference_index_entry():
    cs = fake_chunks()
    idx = reference_index(cs)
    for q in generate(cs):
        if q["category"] == "crosswalk":
            key = (q["gold_answer"]["framework"], q["gold_answer"]["ref_id"])
            assert set(q["gold_control_ids"]) == idx[key]


def _first(qs, cat):
    return copy.deepcopy(next(q for q in qs if q["category"] == cat))


def test_validator_rejects_unknown_gold_id():
    cs = fake_chunks()
    q = _first(generate(cs), "exact_id")
    q["gold_control_ids"] = ["ZZ-99"]
    assert any("unknown control ID" in e for e in validate([q], cs))


def test_validator_rejects_wrong_level():
    cs = fake_chunks()
    q = _first(generate(cs), "applicability")
    q["gold_answer"]["level"] = (q["gold_answer"]["level"] + 1) % 3
    assert any("does not match chunk levels" in e for e in validate([q], cs))


def test_validator_rejects_wrong_group_and_crosswalk_gold():
    cs = fake_chunks()
    qs = generate(cs)
    g = _first(qs, "group_list"); g["gold_control_ids"] = g["gold_control_ids"][:-1] or ["AS-1"]
    c = _first(qs, "crosswalk"); c["gold_control_ids"] = ["AS-1", "AS-2", "AS-3", "AS-4", "AS-5"]
    errs = validate([g, c], cs)
    assert any("group_list" in e for e in errs) and any("crosswalk" in e for e in errs)


def test_validator_rejects_duplicate_qid_and_bad_unanswerable():
    cs = fake_chunks()
    q = _first(generate(cs), "exact_id")
    errs = validate([q, copy.deepcopy(q)], cs)
    assert any("duplicate qid" in e for e in errs)
    bad = {**q, "qid": "x", "answerable": False}
    assert any("unanswerable" in e for e in validate([bad], cs))


def test_validator_rejects_unknown_id_in_question_text_unless_probe():
    cs = fake_chunks()
    q = {"qid": "q1", "source": "curated", "category": "paraphrase", "question": "What does AS-99 say?",
         "gold_control_ids": ["AS-1"], "gold_answer": {}, "answerable": True, "verified": False}
    assert any("AS-99" in e for e in validate([q], cs))


def test_agency_param_label_must_exist_in_control_text():
    cs = fake_chunks()
    q = {"qid": "q1", "source": "curated", "category": "agency_param", "question": "which country?",
         "gold_control_ids": ["AS-1"], "gold_answer": {"agency_defined": True, "param_labels": ["country"]},
         "answerable": True, "verified": False}
    assert validate([q], cs) == []
    q["gold_answer"]["param_labels"] = ["nonexistent"]
    assert validate([q], cs)


def test_summary_reports_unverified_count():
    cs = fake_chunks()
    assert "Unverified: " in summary(generate(cs))


def test_committed_benchmark_is_valid():
    import pytest
    from pathlib import Path
    from aegis.corpus import load_chunks
    from eval.validate import load_questions
    p = Path("eval/questions.jsonl")
    if not p.exists():
        pytest.skip("benchmark not built")
    assert validate(load_questions(p), load_chunks()) == []
