"""Chunk data/fiqa/corpus.jsonl and load it into Postgres.
Usage: python -m scripts.ingest
"""
import json
from pathlib import Path

from app.db import count_chunks, init_schema, insert_chunks
from app.search.chunking import chunk_text

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "fiqa" / "corpus.jsonl"
BATCH_SIZE = 5000


def iter_corpus():
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    init_schema()
    rows = []
    n_docs = 0
    for doc in iter_corpus():
        n_docs += 1
        doc_id = doc["_id"]
        title = doc.get("title", "")
        text = doc.get("text", "")
        for chunk in chunk_text(text):
            chunk_id = f"{doc_id}::{chunk.index}"
            rows.append((chunk_id, doc_id, chunk.index, title, chunk.text, chunk.word_count))
        if len(rows) >= BATCH_SIZE:
            insert_chunks(rows)
            rows = []
        if n_docs % 10000 == 0:
            print(f"  ...{n_docs} documents processed")
    if rows:
        insert_chunks(rows)
    print(f"Ingested {n_docs} documents -> {count_chunks()} chunks in Postgres.")


if __name__ == "__main__":
    main()
