"""Reciprocal Rank Fusion — merges ranked result lists (e.g. BM25 + semantic)
into one ranking, without needing the two lists' scores to be comparable.
"""
from typing import Dict, List, Optional


def reciprocal_rank_fusion(
    *ranked_lists: List[Dict],
    k: int = 60,
    weights: Optional[List[float]] = None,
    id_key: str = "doc_id",
) -> List[Dict]:
    """Each ranked_list is a doc-level list, best-first (already rolled up —
    see app.search.ranking.rollup_to_docs). RRF score for a doc = sum over
    lists of weight / (k + rank), rank is 1-indexed. k=60 is the standard
    default from the original RRF paper. weights defaults to equal (1.0 each)
    — pass e.g. [0.5, 1.0] to trust the second list more than the first, when
    eval numbers show one retriever is reliably weaker on this corpus.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    scores: Dict[str, float] = {}
    items: Dict[str, Dict] = {}
    for weight, ranked in zip(weights, ranked_lists):
        for rank, item in enumerate(ranked, start=1):
            doc_id = item[id_key]
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
            items.setdefault(doc_id, item)
    fused_ids = sorted(scores, key=lambda d: scores[d], reverse=True)
    return [{**items[doc_id], "rrf_score": scores[doc_id]} for doc_id in fused_ids]
