"""Split document text into overlapping word-count chunks.

Most FiQA posts are already short (well under the chunk target), so this is
usually a no-op that returns a single chunk — but it's written generally so
the same ingestion path works unmodified on longer documents later.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    index: int
    text: str
    word_count: int


def chunk_text(text: str, target_words: int = 300, overlap_words: int = 50) -> List[Chunk]:
    words = text.split()
    if not words:
        return [Chunk(index=0, text="", word_count=0)]

    if len(words) <= target_words:
        return [Chunk(index=0, text=text.strip(), word_count=len(words))]

    step = max(1, target_words - overlap_words)
    chunks: List[Chunk] = []
    start = 0
    idx = 0
    while start < len(words):
        piece = words[start : start + target_words]
        chunks.append(Chunk(index=idx, text=" ".join(piece), word_count=len(piece)))
        if start + target_words >= len(words):
            break
        start += step
        idx += 1
    return chunks
