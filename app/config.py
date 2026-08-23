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