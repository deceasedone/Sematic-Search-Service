"""Compare BM25, semantic, and hybrid search on FiQA's labeled test queries.

Usage:
    python -m scripts.run_phase2_eval
"""
import json
import time
from pathlib import Path

from app.config import EMBEDDING_PROVIDER
from app.db import fetch_all_chunks, get_conn
from app.search.bm25_search import BM25Index
from app.search.embeddings import get_embedding_provider
from app.search.hybrid import reciprocal_rank_fusion
from app.search.ranking import rollup_to_docs
from app.search.semantic_search import SemanticIndex
from eval.metrics import evaluate, latency_percentiles, load_qrels

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "fiqa"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results_phase2_comparison.json"
PROGRESS_EVERY = 50


def load_queries() -> dict:
    queries = {}
    with open(DATA_DIR / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                queries[q["_id"]] = q["text"]
    return queries


def main() -> None:
    print("Loading chunks from Postgres...")
    chunks = fetch_all_chunks()
    print(f"  {len(chunks)} chunks")
    if not chunks:
        raise SystemExit("No chunks in Postgres — run `python -m scripts.ingest` first.")

    print("Building BM25 index...")
    bm25 = BM25Index(chunks)

    provider = get_embedding_provider(EMBEDDING_PROVIDER)
    semantic = SemanticIndex(provider)

    queries = load_queries()
    qrels = load_qrels(DATA_DIR / "qrels" / "test.tsv")
    eval_queries = {qid: text for qid, text in queries.items() if qid in qrels}
    qids = list(eval_queries.keys())
    texts = [eval_queries[qid] for qid in qids]
    print(f"  {len(qids)} labeled test queries")

    print(f"Embedding all {len(texts)} queries in one batch (provider={EMBEDDING_PROVIDER})...")
    t0 = time.perf_counter()
    query_vectors = provider.embed(texts, is_query=True)
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    keyword_run, semantic_run, hybrid_run = {}, {}, {}
    keyword_lat, semantic_lat, hybrid_lat = [], [], []

    print("Scoring keyword + semantic + hybrid per query (one reused DB connection)...")
    with get_conn() as conn:
        for i, (qid, text, vec) in enumerate(zip(qids, texts, query_vectors), start=1):
            t0 = time.perf_counter()
            bm25_docs = rollup_to_docs(bm25.search(text, k=50))
            t1 = time.perf_counter()
            semantic_docs = rollup_to_docs(semantic.search_by_vector(vec, k=50, conn=conn))
            t2 = time.perf_counter()
            hybrid_docs = reciprocal_rank_fusion(bm25_docs, semantic_docs)
            t3 = time.perf_counter()

            keyword_run[qid] = [d["doc_id"] for d in bm25_docs]
            semantic_run[qid] = [d["doc_id"] for d in semantic_docs]
            hybrid_run[qid] = [d["doc_id"] for d in hybrid_docs]

            keyword_lat.append((t1 - t0) * 1000)
            semantic_lat.append((t2 - t1) * 1000)
            hybrid_lat.append((t3 - t0) * 1000)

            if i % PROGRESS_EVERY == 0 or i == len(qids):
                print(f"  ...{i}/{len(qids)} queries")

    results = {}
    for mode, run, lat in (
        ("keyword", keyword_run, keyword_lat),
        ("semantic", semantic_run, semantic_lat),
        ("hybrid", hybrid_run, hybrid_lat),
    ):
        results[mode] = {
            "metrics": evaluate(run, qrels, k=10),
            "latency_ms": latency_percentiles(lat),
        }

    print("\n=== Comparison ===")
    print(f"{'Method':<10} {'recall@10':>10} {'precision@10':>13} {'mrr':>8} {'ndcg@10':>9}")
    for mode, r in results.items():
        m = r["metrics"]
        print(
            f"{mode:<10} {m['recall@10']:>10.4f} {m['precision@10']:>13.4f} "
            f"{m['mrr']:>8.4f} {m['ndcg@10']:>9.4f}"
        )

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()