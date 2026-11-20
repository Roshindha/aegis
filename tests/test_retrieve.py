from aegis.retrieve import Hit, rrf_fuse, tokenize


def test_tokenize_keeps_control_ids_whole():
    assert "as-5" in tokenize("What does AS-5 require?")
    assert "as-5" in tokenize("AS-5")          # "as" is a stopword, "as-5" must not be


def test_tokenize_drops_stopwords():
    assert "the" not in tokenize("the password policy")


def _hit(cid, sem=None, bm=None):
    return Hit(cid, cid, "t", "g", "c", 0.0, semantic_rank=sem, bm25_rank=bm)


def test_rrf_rewards_agreement():
    # B is #2 in both lists; A is #1 in only one list, C is #1 in only the other.
    semantic = [_hit("A", sem=1), _hit("B", sem=2)]
    keyword = [_hit("C", bm=1), _hit("B", bm=2)]
    fused = rrf_fuse([semantic, keyword], k=3)
    assert fused[0].chunk_id == "B"
    assert fused[0].semantic_rank == 2 and fused[0].bm25_rank == 2


def test_rrf_exact_id_hit_can_rescue_weak_semantic_rank():
    semantic = [_hit(x, sem=i) for i, x in enumerate(["X1", "X2", "X3", "AS5"], start=1)]
    keyword = [_hit("AS5", bm=1)]
    assert rrf_fuse([semantic, keyword], k=1)[0].chunk_id == "AS5"
