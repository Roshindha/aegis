import numpy as np
import pytest

from aegis.corpus import DEFAULT_EMBEDDINGS, chunk_digest, load_embeddings
from aegis.retrieve import _local_rows


def test_committed_embeddings_match_committed_corpus():
    rows = _local_rows()
    vecs = load_embeddings(rows)
    assert vecs is not None, "corpus/embeddings.npz is stale: run python -m scripts.build_embeddings"
    assert vecs.shape == (len(rows), 384)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)


def test_stale_embeddings_are_rejected(tmp_path):
    rows = [("a", "A", "t", "g", "text a"), ("b", "B", "t", "g", "text b")]
    p = tmp_path / "e.npz"
    np.savez(p, ids=np.array(["a", "b"]), vectors=np.eye(2, dtype="float32"), digest=np.array(chunk_digest(rows)))
    assert load_embeddings(rows, p) is not None
    changed = [rows[0], ("b", "B", "t", "g", "different text")]
    assert load_embeddings(changed, p) is None
    assert load_embeddings(rows, tmp_path / "missing.npz") is None


def test_summarise_reports_equivalence():
    from scripts.equivalence import summarise
    r = summarise([0.9999, 1.0], [1e-5, 2e-5], [True, False], [True, True], 2)
    assert r["top5_identical"] == 1 and r["top1_identical"] == 2 and r["rerank_abs_diff_max"] == 2e-5
