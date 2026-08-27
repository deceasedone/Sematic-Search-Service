import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://search_app:devpassword@localhost:5432/semantic_search"
)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
HYBRID_BM25_WEIGHT = float(os.getenv("HYBRID_BM25_WEIGHT", "0.5"))
HYBRID_SEMANTIC_WEIGHT = float(os.getenv("HYBRID_SEMANTIC_WEIGHT", "1.0"))
DEFAULT_SEARCH_MODE = os.getenv("DEFAULT_SEARCH_MODE", "semantic")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SEARCH_CACHE_TTL = int(os.getenv("SEARCH_CACHE_TTL", "300"))
EMBEDDING_CACHE_TTL = int(os.getenv("EMBEDDING_CACHE_TTL", "3600"))

RATE_LIMIT = os.getenv("RATE_LIMIT", "30/minute")
DB_POOL_MIN_SIZE = int(os.getenv("DB_POOL_MIN_SIZE", "2"))
DB_POOL_MAX_SIZE = int(os.getenv("DB_POOL_MAX_SIZE", "20"))
