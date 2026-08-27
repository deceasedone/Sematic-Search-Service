"""Phase 6: benchmark the hand-built HNSW index against hnswlib and exact
brute-force search, on recall / build time / query latency, using this
project's own real embedded chunks.

Uses corpus vectors as queries (a standard approach for benchmarking ANN
*index* implementations against each other) — that's different from Phase
2's retrieval-quality eval against human relevance judgments, and answers a
different question: not "does this retrieve the right document" but "how
well does this index approximate exact nearest-neighbor search."

Usage:
    python -m scripts.benchmark_hnsw --n 5000 --queries 200
"""
import argparse
import time

import numpy as np

from app.db import fetch_chunks_with_embeddings
from app.search.hnsw_index import HNSWIndex
from eval.metrics import latency_percentiles

try:
    import hnswlib

    HAVE_HNSWLIB = True
except ImportError:
    HAVE_HNSWLIB = False


def brute_force_topk(query: np.ndarray, vectors: np.ndarray, ids: list, k: int) -> set:
    dists = 1.0 - vectors @ query
    idx = np.argsort(dists)[:k]
    return {ids[i] for i in idx}


def main(n: int, n_queries: int, k: int = 10) -> None:
    print(f"Fetching up to {n} embedded chunks from Postgres...")
    rows = fetch_chunks_with_embeddings(limit=n)
    print(f"  {len(rows)} chunks with embeddings")
    if len(rows) < 100:
        raise SystemExit("Not enough embedded chunks — run `python -m scripts.embed_chunks` first.")

    ids = [r["id"] for r in rows]
    vectors = np.array([r["embedding"].to_numpy() for r in rows], dtype=np.float64)

    rng = np.random.default_rng(42)
    query_idx = rng.choice(len(vectors), size=min(n_queries, len(vectors)), replace=False)
    queries = vectors[query_idx]

    print("\nComputing brute-force ground truth...")
    ground_truth, bf_latencies = [], []
    for q in queries:
        t0 = time.perf_counter()
        gt_ids = brute_force_topk(q, vectors, ids, k)
        bf_latencies.append((time.perf_counter() - t0) * 1000)
        ground_truth.append(gt_ids)

    results = {}

    print("\nBuilding hand-built HNSW...")
    t0 = time.perf_counter()
    my_index = HNSWIndex(M=16, ef_construction=200)
    my_index.build([(i, vectors[i]) for i in range(len(vectors))])
    my_build_time = time.perf_counter() - t0
    print(f"  built in {my_build_time:.1f}s")

    my_latencies, my_recalls = [], []
    for q, gt in zip(queries, ground_truth):
        t0 = time.perf_counter()
        hits = my_index.search(q, k=k, ef=50)
        my_latencies.append((time.perf_counter() - t0) * 1000)
        hit_ids = {ids[node_id] for _, node_id in hits}
        my_recalls.append(len(gt & hit_ids) / k)

    results["hand-built HNSW"] = {
        "build_time_s": my_build_time,
        "recall@10": float(np.mean(my_recalls)),
        "latency_ms": latency_percentiles(my_latencies),
    }

    if HAVE_HNSWLIB:
        print("\nBuilding hnswlib index...")
        dim = vectors.shape[1]
        t0 = time.perf_counter()
        lib_index = hnswlib.Index(space="cosine", dim=dim)
        lib_index.init_index(max_elements=len(vectors), M=16, ef_construction=200)
        lib_index.add_items(vectors, np.arange(len(vectors)))
        lib_index.set_ef(50)
        hnswlib_build_time = time.perf_counter() - t0
        print(f"  built in {hnswlib_build_time:.1f}s")

        lib_latencies, lib_recalls = [], []
        for q, gt in zip(queries, ground_truth):
            t0 = time.perf_counter()
            labels, _ = lib_index.knn_query(q, k=k)
            lib_latencies.append((time.perf_counter() - t0) * 1000)
            hit_ids = {ids[i] for i in labels[0]}
            lib_recalls.append(len(gt & hit_ids) / k)

        results["hnswlib"] = {
            "build_time_s": hnswlib_build_time,
            "recall@10": float(np.mean(lib_recalls)),
            "latency_ms": latency_percentiles(lib_latencies),
        }
    else:
        print("\nhnswlib not installed — skipping (pip install hnswlib)")

    results["brute force (numpy)"] = {
        "build_time_s": 0.0,
        "recall@10": 1.0,
        "latency_ms": latency_percentiles(bf_latencies),
    }

    print("\n=== Phase 6 comparison ===")
    print(f"{'Index':<20} {'build(s)':>10} {'recall@10':>10} {'p50(ms)':>10} {'p95(ms)':>10}")
    for name, r in results.items():
        print(
            f"{name:<20} {r['build_time_s']:>10.1f} {r['recall@10']:>10.4f} "
            f"{r['latency_ms'][50]:>10.2f} {r['latency_ms'][95]:>10.2f}"
        )

    print(
        "\nFor reference (not directly comparable — different metric, and "
        "includes real network+DB overhead these in-process numbers don't): "
        "pgvector's own HNSW, already live in this service, scored "
        "nDCG@10=0.363 in Phase 2's retrieval-quality eval and served "
        "semantic queries at p50~650ms / p95~3.5s under 50-user concurrent "
        "load in Phase 4."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000, help="number of chunks to benchmark over")
    parser.add_argument("--queries", type=int, default=200, help="number of query vectors to test")
    args = parser.parse_args()
    main(args.n, args.queries)
