"""Redis cache: full search results, and separately, query embeddings.

Two layers on purpose — a result-cache hit skips everything (BM25, pgvector,
the embedding model); an embedding-cache hit (on a result-cache miss, e.g.
same query text but a different k or mode) still skips the most expensive
part, re-embedding the query.
"""
import hashlib
import json
from typing import Any, Dict, List, Optional

import redis

from app.config import EMBEDDING_CACHE_TTL, REDIS_URL, SEARCH_CACHE_TTL

_client: Optional[redis.Redis] = None


def get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(REDIS_URL, decode_responses=True)
    return _client


def ping() -> bool:
    try:
        return get_client().ping()
    except redis.RedisError:
        return False


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def get_cached_search(mode: str, query: str, k: int) -> Optional[Dict[str, Any]]:
    try:
        raw = get_client().get(f"search:{mode}:{k}:{_hash(query)}")
    except redis.RedisError:
        return None
    return json.loads(raw) if raw else None


def set_cached_search(mode: str, query: str, k: int, response: Dict[str, Any]) -> None:
    try:
        get_client().setex(f"search:{mode}:{k}:{_hash(query)}", SEARCH_CACHE_TTL, json.dumps(response))
    except redis.RedisError:
        pass  # cache is a speed optimization, never a hard dependency


def get_cached_embedding(provider_name: str, text: str) -> Optional[List[float]]:
    try:
        raw = get_client().get(f"embedding:{provider_name}:{_hash(text)}")
    except redis.RedisError:
        return None
    return json.loads(raw) if raw else None


def set_cached_embedding(provider_name: str, text: str, vector: List[float]) -> None:
    try:
        get_client().setex(
            f"embedding:{provider_name}:{_hash(text)}", EMBEDDING_CACHE_TTL, json.dumps(vector)
        )
    except redis.RedisError:
        pass
