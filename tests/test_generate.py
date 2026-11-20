from types import SimpleNamespace

import aegis.generate as g
from aegis.retrieve import Hit


def _hit(cid):
    return Hit(cid, cid, "t", "g", f"{cid}: Title\nStatement: do the thing", 1.0)


def test_extract_citations_flags_ids_not_in_context():
    hits = [_hit("AS-5"), _hit("AS-6")]
    cited, invalid = g.extract_citations("Use long passwords [AS-5]. Rotate keys [CK-1]. Again [AS-5].", hits)
    assert cited == ["AS-5", "CK-1"]          # deduplicated, order kept
    assert invalid == ["CK-1"]                # CK-1 was never retrieved -> hallucinated citation


def test_build_context_contains_every_control():
    ctx = g.build_context([_hit("AS-5"), _hit("BR-2")])
    assert "AS-5: Title" in ctx and "BR-2: Title" in ctx


class FakeClient:
    def __init__(self, content):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._content, self.kwargs = content, None

    def _create(self, **kwargs):
        self.kwargs = kwargs
        msg = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_answer_question_parses_response(monkeypatch):
    fake = FakeClient("Passwords need a minimum length [AS-5].")
    monkeypatch.setattr(g, "_client", lambda: fake)
    ans = g.answer_question("password length?", [_hit("AS-5")])
    assert ans.cited_ids == ["AS-5"] and ans.invalid_citations == []
    assert fake.kwargs["temperature"] == 0
    assert "password length?" in fake.kwargs["messages"][1]["content"]


def test_empty_model_reply_does_not_crash(monkeypatch):
    monkeypatch.setattr(g, "_client", lambda: FakeClient(None))   # reasoning ate the token budget
    assert g.answer_question("q", [_hit("AS-5")]).text == ""
