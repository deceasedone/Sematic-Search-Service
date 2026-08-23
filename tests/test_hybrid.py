import pytest

from app.search.hybrid import reciprocal_rank_fusion


def test_rrf_agreement_boosts_a_doc():
    bm25 = [{"doc_id": "d1"}, {"doc_id": "d2"}, {"doc_id": "d3"}]
    semantic = [{"doc_id": "d1"}, {"doc_id": "d4"}, {"doc_id": "d2"}]

    fused = reciprocal_rank_fusion(bm25, semantic, k=60)

    assert fused[0]["doc_id"] == "d1"
    expected_d1 = 1 / (60 + 1) + 1 / (60 + 1)
    assert fused[0]["rrf_score"] == pytest.approx(expected_d1)


def test_rrf_includes_docs_found_by_only_one_list():
    bm25 = [{"doc_id": "d1"}]
    semantic = [{"doc_id": "d2"}]

    fused = reciprocal_rank_fusion(bm25, semantic, k=60)
    ids = {d["doc_id"] for d in fused}
    assert ids == {"d1", "d2"}


def test_rrf_preserves_first_seen_item_fields():
    bm25 = [{"doc_id": "d1", "title": "From BM25"}]
    semantic = [{"doc_id": "d1", "title": "From semantic"}]

    fused = reciprocal_rank_fusion(bm25, semantic, k=60)
    assert fused[0]["title"] == "From BM25"


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([], [], k=60) == []