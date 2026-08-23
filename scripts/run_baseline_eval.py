"""Run the BM25 keyword baseline over FiQA's labeled test queries and score it.
Usage: python -m scripts.run_baseline_eval
"""
import json
import time
from pathlib import Path

from app.db import fetch_all_chunks
from app.search.bm25_search import BM25Index
from app.search.ranking import rollup_to_docs
from eval.metrics import evaluate, latency_percentiles, load_qrels

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "fiqa"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results_bm25_baseline.json"


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
    index = BM25Index(chunks)

    queries = load_queries()
    qrels = load_qrels(DATA_DIR / "qrels" / "test.tsv")
    eval_queries = {qid: text for qid, text in queries.items() if qid in qrels}
    print(f"  {len(eval_queries)} labeled test queries")

    run, latencies = {}, []
    for qid, text in eval_queries.items():
        start = time.perf_counter()
        chunk_hits = index.search(text, k=50)
        doc_hits = rollup_to_docs(chunk_hits)
        latencies.append((time.perf_counter() - start) * 1000)
        run[qid] = [d["doc_id"] for d in doc_hits]

    metrics = evaluate(run, qrels, k=10)
    lat = latency_percentiles(latencies)

    print("\n=== BM25 baseline ===")
    for name, val in metrics.items():
        print(f"{name}: {val:.4f}")
    print(f"latency p50: {lat[50]:.1f} ms | p95: {lat[95]:.1f} ms  (in-process search only)")

    RESULTS_PATH.write_text(
        json.dumps({"metrics": metrics, "latency_ms": lat, "n_queries": len(eval_queries)}, indent=2)
    )
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
