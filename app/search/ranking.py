"""Collapse chunk-level search hits down to unique parent documents — the
index is chunk-grained, but qrels and the end user care about documents."""
from typing import Dict, List


def rollup_to_docs(chunk_results: List[Dict]) -> List[Dict]:
    seen = set()
    docs = []
    for r in chunk_results:
        if r["doc_id"] in seen:
            continue
        seen.add(r["doc_id"])
        docs.append(r)
    return docs
