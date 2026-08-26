import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://search_app:devpassword@localhost:5432/semantic_search"
)
# Must match the active embedding provider; changing it requires rebuilding the
# embedding column before re-embedding.
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
# Weighted RRF fusion for hybrid mode. Equal weights (1.0/1.0) let a weaker
# retriever's mediocre-but-nonzero matches out-fuse a stronger retriever's
# genuine misses — measured on FiQA: semantic (nDCG@10 0.363) is well ahead
# of BM25 (0.214), so BM25's vote counts for less by default here.
HYBRID_BM25_WEIGHT = float(os.getenv("HYBRID_BM25_WEIGHT", "0.5"))
HYBRID_SEMANTIC_WEIGHT = float(os.getenv("HYBRID_SEMANTIC_WEIGHT", "1.0"))
# Measured on FiQA: semantic alone beats hybrid on every metric (see
# results_phase2_comparison.json), so the live API defaults to it rather
# than hybrid — override per-request with ?mode=hybrid or ?mode=keyword.
DEFAULT_SEARCH_MODE = os.getenv("DEFAULT_SEARCH_MODE", "semantic")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SEARCH_CACHE_TTL = int(os.getenv("SEARCH_CACHE_TTL", "300"))       # 5 min
EMBEDDING_CACHE_TTL = int(os.getenv("EMBEDDING_CACHE_TTL", "3600"))  # 1 hour

RATE_LIMIT = os.getenv("RATE_LIMIT", "30/minute")