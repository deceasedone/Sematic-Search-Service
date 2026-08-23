from app.search.ranking import rollup_to_docs


def test_rollup_keeps_best_occurrence_and_order():
    chunk_results = [
        {"doc_id": "d1", "id": "d1::0", "score": 5.0},
        {"doc_id": "d2", "id": "d2::0", "score": 4.0},
        {"doc_id": "d1", "id": "d1::1", "score": 3.0},
        {"doc_id": "d3", "id": "d3::0", "score": 1.0},
    ]
    docs = rollup_to_docs(chunk_results)
    assert [d["doc_id"] for d in docs] == ["d1", "d2", "d3"]
    assert docs[0]["id"] == "d1::0"


def test_rollup_empty_input():
    assert rollup_to_docs([]) == []
