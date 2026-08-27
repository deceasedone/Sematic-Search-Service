"""FastAPI app. Three search modes: keyword (BM25), semantic (pgvector),
hybrid (RRF fusion of both). Phase 3 adds Redis result + embedding caching,
per-IP rate limiting, structured logging, and stricter request validation.
"""
import time
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, Query, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from app.cache import get_cached_search, ping as redis_ping, set_cached_search
from app.config import (
    DEFAULT_SEARCH_MODE,
    EMBEDDING_PROVIDER,
    HYBRID_BM25_WEIGHT,
    HYBRID_SEMANTIC_WEIGHT,
    RATE_LIMIT,
    REDIS_URL,
)
from app.db import fetch_all_chunks
from app.logging_config import configure_logging, logger
from app.search.bm25_search import BM25Index
from app.search.embeddings import get_embedding_provider
from app.search.hybrid import reciprocal_rank_fusion
from app.search.ranking import rollup_to_docs
from app.search.semantic_search import SemanticIndex

_bm25_index: Optional[BM25Index] = None
_semantic_index: Optional[SemanticIndex] = None

SearchMode = Literal["keyword", "semantic", "hybrid"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    global _bm25_index, _semantic_index
    chunks = fetch_all_chunks()
    _bm25_index = BM25Index(chunks)
    _semantic_index = SemanticIndex(
        get_embedding_provider(EMBEDDING_PROVIDER), provider_name=EMBEDDING_PROVIDER
    )
    logger.info(f"startup chunks_indexed={len(chunks)} embedding_provider={EMBEDDING_PROVIDER}")
    yield


limiter = Limiter(key_func=get_remote_address, storage_uri=REDIS_URL)

app = FastAPI(title="Semantic Search Service", version="0.3.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


@app.get("/health")
def health():
    redis_ok = redis_ping()
    return {
        "status": "ok" if redis_ok else "degraded",
        "chunks_indexed": len(_bm25_index.chunks) if _bm25_index else 0,
        "redis": "ok" if redis_ok else "unreachable",
    }


@app.get("/search")
@limiter.limit(RATE_LIMIT)
def search(
    request: Request,
    q: str = Query(..., min_length=1),
    k: int = Query(10, ge=1, le=100),
    mode: SearchMode = DEFAULT_SEARCH_MODE,
):
    if _bm25_index is None or _semantic_index is None:
        return {"error": "index not ready"}

    cached = get_cached_search(mode, q, k)
    if cached is not None:
        logger.info(f"search query={q!r} mode={mode} k={k} cached=True")
        return {**cached, "cached": True}

    start = time.perf_counter()
    fetch_k = max(k * 5, 50)

    if mode == "keyword":
        doc_hits = rollup_to_docs(_bm25_index.search(q, k=fetch_k))[:k]
    elif mode == "semantic":
        doc_hits = rollup_to_docs(_semantic_index.search(q, k=fetch_k))[:k]
    else:
        bm25_docs = rollup_to_docs(_bm25_index.search(q, k=fetch_k))
        semantic_docs = rollup_to_docs(_semantic_index.search(q, k=fetch_k))
        doc_hits = reciprocal_rank_fusion(
            bm25_docs, semantic_docs, weights=[HYBRID_BM25_WEIGHT, HYBRID_SEMANTIC_WEIGHT]
        )[:k]

    latency_ms = (time.perf_counter() - start) * 1000

    response = {
        "query": q,
        "mode": mode,
        "results": [
            {
                "id": d["doc_id"],
                "title": d.get("title") or "",
                "score": round(d.get("rrf_score", d.get("score", 0.0)), 4),
                "snippet": d["text"][:200],
                "source": "semantic+keyword" if mode == "hybrid" else mode,
            }
            for d in doc_hits
        ],
        "latency_ms": round(latency_ms, 1),
    }

    set_cached_search(mode, q, k, response)
    logger.info(
        f"search query={q!r} mode={mode} k={k} latency_ms={latency_ms:.1f} "
        f"cached=False results={len(response['results'])}"
    )
    return {**response, "cached": False}
