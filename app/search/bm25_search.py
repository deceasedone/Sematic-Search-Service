"""In-memory BM25 keyword index — the baseline this project measures itself
against. Fine at FiQA's ~58k-doc scale; revisit if the corpus grows much larger.
"""
import re
from typing import Dict, List

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        corpus_tokens = [tokenize(f"{c.get('title') or ''} {c['text']}") for c in chunks]
        self._bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None

    def search(self, query: str, k: int = 10) -> List[Dict]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in ranked]
