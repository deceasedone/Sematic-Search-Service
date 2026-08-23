"""Postgres connection + schema management (psycopg3, no ORM)."""
from contextlib import contextmanager
from typing import Dict, Iterable, List, Tuple

import psycopg
from pgvector.psycopg import register_vector

from app.config import DATABASE_URL, EMBEDDING_DIM

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    chunk_index INT NOT NULL,
    title TEXT,
    text TEXT NOT NULL,
    word_count INT NOT NULL,
    embedding vector({EMBEDDING_DIM})
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks
    USING hnsw (embedding vector_cosine_ops);
"""


@contextmanager
def get_conn():
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    try:
        yield conn
    finally:
        conn.close()


def init_schema() -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(SCHEMA)
        conn.commit()


def clear_chunks() -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE chunks;")
        conn.commit()


def insert_chunks(rows: Iterable[Tuple]) -> None:
    """rows: iterable of (id, doc_id, chunk_index, title, text, word_count)"""
    with get_conn() as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (id, doc_id, chunk_index, title, text, word_count)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text
            """,
            list(rows),
        )
        conn.commit()


def fetch_all_chunks() -> List[Dict]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, doc_id, title, text FROM chunks;")
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_chunks_missing_embeddings() -> List[Dict]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, title, text FROM chunks WHERE embedding IS NULL;")
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def update_embeddings(pairs: Iterable[Tuple[str, List[float]]]) -> None:
    """pairs: iterable of (chunk_id, embedding_vector)"""
    with get_conn() as conn, conn.cursor() as cur:
        cur.executemany(
            "UPDATE chunks SET embedding = %s WHERE id = %s",
            [(vec, cid) for cid, vec in pairs],
        )
        conn.commit()


def count_chunks() -> int:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chunks;")
        return cur.fetchone()[0]
