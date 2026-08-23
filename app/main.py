"""FastAPI app with keyword, semantic, and hybrid search modes."""
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Query

from app.config import EMBEDDING_PROVIDER
from app.db import fetch_all_chunks
from app.search.bm25_search import BM25Index
from app.search.embeddings import get_embedding_provider
from app.search.hybrid import reciprocal_rank_fusion
from app.search.ranking import rollup_to_docs
from app.search.semantic_search import SemanticIndex

_bm25_index: Optional[BM25Index] = None
_semantic_index: Optional[SemanticIndex] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bm25_index, _semantic_index
    chunks = fetch_all_chunks()
    _bm25_index = BM25Index(chunks)
    _semantic_index = SemanticIndex(get_embedding_provider(EMBEDDING_PROVIDER))
    print(f"BM25 index built over {len(chunks)} chunks. Semantic index ready (pgvector-backed).")
    yield


app = FastAPI(title="Semantic Search Service", version="0.2.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": len(_bm25_index.chunks) if _bm25_index else 0}


@app.get("/search")
def search(q: str = Query(..., min_length=1), k: int = 10, mode: str = "hybrid"):
    if _bm25_index is None or _semantic_index is None:
        return {"error": "index not ready"}
    if mode not in ("keyword", "semantic", "hybrid"):
        return {"error": f"mode='{mode}' not recognized — use keyword, semantic, or hybrid"}

    start = time.perf_counter()
    fetch_k = max(k * 5, 50)

    if mode == "keyword":
        doc_hits = rollup_to_docs(_bm25_index.search(q, k=fetch_k))[:k]
    elif mode == "semantic":
        doc_hits = rollup_to_docs(_semantic_index.search(q, k=fetch_k))[:k]
    else:
        bm25_docs = rollup_to_docs(_bm25_index.search(q, k=fetch_k))
        semantic_docs = rollup_to_docs(_semantic_index.search(q, k=fetch_k))
        doc_hits = reciprocal_rank_fusion(bm25_docs, semantic_docs)[:k]

    latency_ms = (time.perf_counter() - start) * 1000

    return {
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