"""Pluggable embedding backends behind one interface."""
from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @abstractmethod
    def embed(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """is_query: True when embedding a search query rather than a stored
        document. Providers that don't distinguish (Local) ignore it;
        providers with asymmetric retrieval modes (Gemini) use it to pick
        the right task type for better retrieval quality.
        """
        ...


class LocalEmbeddingProvider(EmbeddingProvider):
    """sentence-transformers, run on-machine — free, offline, default choice."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Google Gemini embeddings (google-genai SDK) — has a free tier, no
    local GPU/CPU cost. Requires GEMINI_API_KEY and `pip install google-genai`.

    Uses gemini-embedding-001 (GA, stable) rather than the newer preview
    multimodal models, since this project only ever embeds text.
    """

    def __init__(self, model_name: str = "gemini-embedding-001", dimension: int = 768):
        import os

        from google import genai

        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self._model_name = model_name
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        from google.genai.types import EmbedContentConfig

        response = self._client.models.embed_content(
            model=self._model_name,
            contents=texts,
            config=EmbedContentConfig(
                task_type="RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT",
                output_dimensionality=self._dim,
            ),
        )
        return [e.values for e in response.embeddings]


def get_embedding_provider(name: str = "local") -> EmbeddingProvider:
    if name == "local":
        return LocalEmbeddingProvider()
    if name == "gemini":
        from app.config import EMBEDDING_DIM

        return GeminiEmbeddingProvider(dimension=EMBEDDING_DIM)
    raise ValueError(f"Unknown embedding provider: {name!r}")