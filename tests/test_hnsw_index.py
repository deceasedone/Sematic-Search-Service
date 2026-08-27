import numpy as np

from app.search.hnsw_index import HNSWIndex


def _normalized_random_vectors(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _brute_force_topk(query, vectors, k):
    dists = 1.0 - vectors @ query
    return set(np.argsort(dists)[:k].tolist())


def test_hnsw_recall_against_brute_force():
    n, dim, k = 500, 32, 10
    vectors = _normalized_random_vectors(n, dim, seed=1)
    queries = _normalized_random_vectors(20, dim, seed=2)

    index = HNSWIndex(M=16, ef_construction=200, seed=42)
    index.build([(i, vectors[i]) for i in range(n)])

    recalls = []
    for q in queries:
        gt = _brute_force_topk(q, vectors, k)
        hits = {node_id for _, node_id in index.search(q, k=k, ef=50)}
        recalls.append(len(gt & hits) / k)

    mean_recall = sum(recalls) / len(recalls)
    assert mean_recall > 0.85, f"mean recall too low: {mean_recall}"


def test_hnsw_returns_query_itself_as_nearest():
    n, dim = 200, 16
    vectors = _normalized_random_vectors(n, dim, seed=3)
    index = HNSWIndex(M=16, ef_construction=100, seed=1)
    index.build([(i, vectors[i]) for i in range(n)])

    hits = index.search(vectors[5], k=1, ef=50)
    assert hits[0][1] == 5
    assert hits[0][0] < 1e-9


def test_hnsw_empty_index_returns_empty():
    index = HNSWIndex()
    assert index.search(np.array([1.0, 0.0]), k=5) == []


def test_hnsw_single_item():
    index = HNSWIndex()
    v = np.array([1.0, 0.0])
    index.insert(0, v)
    assert index.search(v, k=5) == [(0.0, 0)]
