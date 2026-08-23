"""Reciprocal Rank Fusion — merges ranked result lists (e.g. BM25 + semantic)
into one ranking, without needing the two lists' scores to be comparable.
"""
from typing import Dict, List


def reciprocal_rank_fusion(
    *ranked_lists: List[Dict], k: int = 60, id_key: str = "doc_id"
) -> List[Dict]:
    """Each ranked_list is a doc-level list, best-first (already rolled up —
    see app.search.ranking.rollup_to_docs). RRF score for a doc = sum over
    lists of 1 / (k + rank), rank is 1-indexed. k=60 is the standard default
    from the original RRF paper.
    """
    scores: Dict[str, float] = {}
    items: Dict[str, Dict] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            doc_id = item[id_key]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            items.setdefault(doc_id, item)
    fused_ids = sorted(scores, key=lambda d: scores[d], reverse=True)
    return [{**items[doc_id], "rrf_score": scores[doc_id]} for doc_id in fused_ids]