"""Semantic search over chunk embeddings using pgvector cosine distance."""
from typing import Dict, List, Optional

from app.db import get_conn
from app.search.embeddings import EmbeddingProvider

_QUERY_SQL = """
SELECT id, doc_id, title, text, 1 - (embedding <=> %s::vector) AS score
FROM chunks
WHERE embedding IS NOT NULL
ORDER BY embedding <=> %s::vector
LIMIT %s
"""


class SemanticIndex:
    def __init__(self, provider: EmbeddingProvider, provider_name: str = "local"):
        self.provider = provider
        self.provider_name = provider_name

    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Embeds `query` (cached by text+provider) and searches. Opens its
        own connection — fine for one-off calls (e.g. a live API request).
        For many queries in a loop, use search_by_vector() with a connection
        you reuse — see scripts/run_phase2_eval.py, which deliberately
        bypasses this cache to measure real, uncached latency.
        """
        query_vec = self._embed_cached(query)
        with get_conn() as conn:
            return self.search_by_vector(query_vec, k, conn=conn)

    def _embed_cached(self, query: str) -> List[float]:
        from app.cache import get_cached_embedding, set_cached_embedding  # deferred: avoid Redis dependency for eval/tests that don't need it

        cached = get_cached_embedding(self.provider_name, query)
        if cached is not None:
            return cached
        vector = self.provider.embed([query], is_query=True)[0]
        set_cached_embedding(self.provider_name, query, vector)
        return vector

    def search_by_vector(self, query_vec: List[float], k: int = 10, conn=None) -> List[Dict]:
        if conn is not None:
            return self._run_query(conn, query_vec, k)
        with get_conn() as owned_conn:
            return self._run_query(owned_conn, query_vec, k)

    @staticmethod
    def _run_query(conn, query_vec: List[float], k: int) -> List[Dict]:
        with conn.cursor() as cur:
            cur.execute(_QUERY_SQL, (query_vec, query_vec, k))
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]