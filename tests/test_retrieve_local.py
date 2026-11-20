import numpy as np

import aegis.retrieve as rv

CHUNKS = [
    {"chunk_id": "c-br-2", "control_id": "BR-2", "title": "Recovery Testing", "group_title": "Backup", "text": "BR-2: recovery testing of backups"},
    {"chunk_id": "c-as-5", "control_id": "AS-5", "title": "Passwords", "group_title": "App", "text": "AS-5: password length rules"},
]
VECS = {"BR-2": np.array([1.0, 0.0]), "AS-5": np.array([0.0, 1.0])}


def test_local_semantic_search_ranks_by_cosine(monkeypatch):
    rv._local_index.cache_clear()
    monkeypatch.setattr(rv, "backend", lambda: "local")
    monkeypatch.setattr(rv, "load_chunks", lambda: CHUNKS)
    monkeypatch.setattr(rv, "embed_documents", lambda texts, show_progress=True: np.array([VECS[t.split(":")[0]] for t in texts]))
    monkeypatch.setattr(rv, "embed_query", lambda q: np.array([0.9, 0.1]))
    hits = rv.semantic_search("anything", k=2)
    assert [h.control_id for h in hits] == ["BR-2", "AS-5"]
    assert hits[0].semantic_rank == 1 and hits[0].score > hits[1].score
    rv._local_index.cache_clear()


def test_local_bm25_search(monkeypatch):
    rv._load_corpus.cache_clear()
    monkeypatch.setattr(rv, "backend", lambda: "local")
    filler = [{"chunk_id": f"c-x-{i}", "control_id": f"XX-{i}", "title": "t", "group_title": "g",
               "text": f"XX-{i}: unrelated words number {i} about hosting"} for i in range(6)]
    monkeypatch.setattr(rv, "load_chunks", lambda: CHUNKS + filler)
    hits = rv.bm25_search("AS-5 password", k=2)
    assert hits[0].control_id == "AS-5" and hits[0].bm25_rank == 1
    rv._load_corpus.cache_clear()


def test_rrf_does_not_mutate_input_hits():
    a = rv.Hit("A", "A", "t", "g", "c", 0.5, semantic_rank=1)
    rv.rrf_fuse([[a]], k=1)
    assert a.score == 0.5
