"""Embed every chunk (that doesn't already have one) and store the vector
in pgvector.

Usage:
    python -m scripts.embed_chunks
"""
from app.config import EMBEDDING_PROVIDER
from app.db import fetch_chunks_missing_embeddings, update_embeddings
from app.search.embeddings import get_embedding_provider

BATCH_SIZE = 256


def main() -> None:
    provider = get_embedding_provider(EMBEDDING_PROVIDER)
    chunks = fetch_chunks_missing_embeddings()
    print(f"Embedding {len(chunks)} chunks with '{EMBEDDING_PROVIDER}' (dim={provider.dimension})...")

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [f"{c.get('title') or ''} {c['text']}" for c in batch]
        vectors = provider.embed(texts)
        update_embeddings(zip((c["id"] for c in batch), vectors))
        print(f"  ...{min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print("Done.")


if __name__ == "__main__":
    main()