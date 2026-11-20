import numpy as np

import aegis.rerank as rr
from aegis.retrieve import Hit


class FakeCrossEncoder:
    """Pretends to be the model: scores a chunk by how many query words it contains."""

    def predict(self, pairs, **kwargs):
        return np.array([sum(w in doc.lower() for w in q.lower().split()) for q, doc in pairs], dtype=float)


def _hit(cid, text, bm=None):
    return Hit(cid, cid, "t", "g", text, 0.016, semantic_rank=1, bm25_rank=bm)


def test_rerank_reorders_by_new_score(monkeypatch):
    monkeypatch.setattr(rr, "get_reranker", lambda: FakeCrossEncoder())
    hits = [_hit("NOISE", "separate hosting of systems"), _hit("BR-2", "recovery testing of backups")]
    out = rr.rerank("backups testing", hits, k=2)
    assert [h.control_id for h in out] == ["BR-2", "NOISE"]
    assert out[0].score > out[1].score          # score is now the reranker's, not RRF's


def test_rerank_keeps_rank_provenance_and_does_not_mutate_input(monkeypatch):
    monkeypatch.setattr(rr, "get_reranker", lambda: FakeCrossEncoder())
    hits = [_hit("A", "backups", bm=3)]
    out = rr.rerank("backups", hits, k=1)
    assert out[0].bm25_rank == 3
    assert hits[0].score == 0.016               # original untouched (dataclasses.replace copies)


def test_rerank_empty():
    assert rr.rerank("anything", [], k=5) == []
