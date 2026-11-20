import aegis.crosswalk as cw
import aegis.guardrail as gr
import aegis.pipeline as pl
from aegis.generate import Answer
from aegis.retrieve import Hit

# These tests read the committed corpus snapshot (corpus/chunks.jsonl): no network, DB or models.


def test_forward_lookup_returns_every_control_with_that_reference():
    a = cw.analyse("Which IM8 Reform controls reference MVSP 2.5?")
    assert a.forward == [("MVSP", "2.5")] and len(a.controls) == 3
    assert a.controls == sorted(a.controls)


def test_ref_ids_are_not_prefix_matched():
    nist = cw.analyse("Which controls reference NIST SP 800-53 AC-2?")
    assert nist.forward == [("NIST SP 800-53", "AC-2")]            # not also AC-2(3)
    assert cw.analyse("Which controls map to NIST SP 800-53 AC-2(3)?").forward == [("NIST SP 800-53", "AC-2(3)")]
    assert cw.analyse("Which controls reference MVSP 2.55?").forward == []


def test_framework_name_is_required_for_bare_ids():
    assert not cw.analyse("What does AC-2 say about accounts?").active
    assert not cw.analyse("Which controls reference 2.5?").active


def test_reverse_lookup_for_known_im8_id_only():
    a = cw.analyse("Which NIST, MVSP, CIS, SLSA or CCM references does AS-12 map to?")
    assert a.reverse == ["AS-12"] and not a.forward
    assert not cw.analyse("What does AS-12 require?").active
    assert not cw.analyse("Which references does ZZ-99 map to?").active


def test_ordinary_questions_are_untouched():
    for q in ("How often should backups be tested?", "Do admins need MFA?", "What does AS-5 require?"):
        assert not cw.analyse(q).active


def test_merge_puts_crosswalk_first_with_reference_line_and_dedupes():
    a = cw.analyse("Which IM8 Reform controls reference MVSP 2.5?")
    base = [Hit("x", "ZZ-1", "t", "g", "c", 1.0), Hit("y", a.controls[0], "t", "g", "plain", 1.0)]
    out = cw.merge(a, base, limit=20)
    assert [h.control_id for h in out[:3]] == a.controls and all(h.via == "crosswalk" for h in out[:3])
    assert "External references:" in out[0].content and "MVSP 2.5" in out[0].content
    assert [h.control_id for h in out].count(a.controls[0]) == 1 and out[-1].control_id == "ZZ-1"


def test_merge_is_identity_when_inactive():
    base = [Hit("x", "AS-1", "t", "g", "c", 1.0)]
    assert cw.merge(cw.Analysis(), base) == base


def test_context_size_grows_for_crosswalk_but_is_capped():
    plain = [Hit(str(i), f"AS-{i}", "t", "g", "c", 0.0) for i in range(8)]
    assert cw.context_size(plain, 5) == 5
    many = [Hit(str(i), f"AS-{i}", "t", "g", "c", 0.0, via="crosswalk") for i in range(8)]
    assert cw.context_size(many, 5) == 8
    assert cw.context_size(many * 3, 5) == cw.MAX_CONTEXT


def test_guardrail_bypasses_threshold_for_crosswalk_questions():
    assert gr.structural_bypass("Which IM8 Reform controls reference MVSP 2.5?") == "crosswalk"
    assert gr.structural_bypass("Which NIST, MVSP, CIS, SLSA or CCM references does AS-12 map to?") == "crosswalk"


def test_pipeline_emits_crosswalk_stage_and_context_has_references(monkeypatch):
    monkeypatch.setattr(pl, "semantic_search", lambda q, k: [Hit("s", "AS-1", "t", "g", "AS-1: x", 0.1)])
    monkeypatch.setattr(pl, "bm25_search", lambda q, k: [Hit("b", "AS-1", "t", "g", "AS-1: x", 0.1)])
    monkeypatch.setattr(pl, "rerank", lambda q, hits, k=5: hits[:k])
    seen = {}

    def answerer(q, hits):
        seen["hits"] = hits
        return Answer("ok", [], [], hits, "m")
    names = [e["stage"] for e in pl.run_stages("Which IM8 Reform controls reference MVSP 2.5?", answerer=answerer)]
    assert names[:2] == ["prefilter", "crosswalk"] and "answer" in names
    assert len(seen["hits"]) == 3 + 1 or len(seen["hits"]) >= 3
    assert "MVSP 2.5" in seen["hits"][0].content
