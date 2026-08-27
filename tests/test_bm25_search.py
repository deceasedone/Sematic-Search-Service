from app.search.bm25_search import BM25Index

CHUNKS = [
    {"id": "c1", "doc_id": "d1", "title": "", "text": "apple banana cherry"},
    {"id": "c2", "doc_id": "d2", "title": "", "text": "dog cat bird"},
    {"id": "c3", "doc_id": "d3", "title": "", "text": "car train plane"},
]


def test_search_excludes_zero_overlap_results():
    index = BM25Index(CHUNKS)
    results = index.search("zebra elephant giraffe", k=10)
    assert results == []


def test_search_returns_genuine_term_matches():
    index = BM25Index(CHUNKS)
    results = index.search("apple", k=10)
    assert len(results) == 1
    assert results[0]["doc_id"] == "d1"
    assert results[0]["score"] > 0
