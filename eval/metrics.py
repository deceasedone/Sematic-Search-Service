"""Information-retrieval evaluation metrics: recall@k, precision@k, MRR, nDCG@k,
plus latency percentile helpers.

All ranking metrics are macro-averaged across queries (each query weighted
equally), which is the standard convention for BEIR / TREC-style evaluation.

Data shapes:
    Qrels = {query_id: {doc_id: relevance}}   relevance is an int; >0 = relevant
    Run   = {query_id: [doc_id, ...]}         already ranked best-first
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np

Qrels = Dict[str, Dict[str, int]]
Run = Dict[str, List[str]]


def load_qrels(path: Union[str, Path]) -> Qrels:
    """Load a BEIR-format qrels TSV: 'query-id\\tcorpus-id\\tscore' (+ header row)."""
    qrels: Qrels = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for row in reader:
            if not row:
                continue
            qid, docid, score = row[0], row[1], int(row[2])
            qrels.setdefault(qid, {})[docid] = score
    return qrels


def load_run(path: Union[str, Path]) -> Run:
    """Load a run file: JSON {query_id: [doc_id, ...]}, already ranked best-first."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_run(run: Run, path: Union[str, Path]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run, f)


def _relevant_docs(rels: Dict[str, int]) -> set:
    return {d for d, r in rels.items() if r > 0}


def recall_at_k(run: Run, qrels: Qrels, k: int = 10) -> float:
    scores = []
    for qid, rels in qrels.items():
        relevant = _relevant_docs(rels)
        if not relevant:
            continue
        retrieved = set(run.get(qid, [])[:k])
        scores.append(len(relevant & retrieved) / len(relevant))
    return sum(scores) / len(scores) if scores else 0.0


def precision_at_k(run: Run, qrels: Qrels, k: int = 10) -> float:
    scores = []
    for qid, rels in qrels.items():
        relevant = _relevant_docs(rels)
        if not relevant:
            continue
        retrieved = run.get(qid, [])[:k]
        if not retrieved:
            scores.append(0.0)
            continue
        hits = sum(1 for d in retrieved if d in relevant)
        scores.append(hits / len(retrieved))
    return sum(scores) / len(scores) if scores else 0.0


def mrr(run: Run, qrels: Qrels, k: int = 1000) -> float:
    """Mean Reciprocal Rank of the first relevant doc, within the top-k."""
    scores = []
    for qid, rels in qrels.items():
        relevant = _relevant_docs(rels)
        if not relevant:
            continue
        retrieved = run.get(qid, [])[:k]
        rr = 0.0
        for rank, docid in enumerate(retrieved, start=1):
            if docid in relevant:
                rr = 1.0 / rank
                break
        scores.append(rr)
    return sum(scores) / len(scores) if scores else 0.0


def ndcg_at_k(run: Run, qrels: Qrels, k: int = 10) -> float:
    scores = []
    for qid, rels in qrels.items():
        if not rels:
            continue
        retrieved = run.get(qid, [])[:k]
        dcg = sum(
            rels.get(docid, 0) / math.log2(rank + 1)
            for rank, docid in enumerate(retrieved, start=1)
        )
        ideal_rels = sorted(rels.values(), reverse=True)[:k]
        idcg = sum(
            rel / math.log2(rank + 1) for rank, rel in enumerate(ideal_rels, start=1)
        )
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def latency_percentiles(
    latencies_ms: Sequence[float], percentiles: Sequence[int] = (50, 95)
) -> Dict[int, float]:
    if not latencies_ms:
        return {p: 0.0 for p in percentiles}
    arr = np.asarray(latencies_ms, dtype=float)
    return {p: float(np.percentile(arr, p)) for p in percentiles}


def evaluate(run: Run, qrels: Qrels, k: int = 10) -> Dict[str, float]:
    """Convenience: compute the standard metric set at once, for a results table."""
    return {
        f"recall@{k}": recall_at_k(run, qrels, k),
        f"precision@{k}": precision_at_k(run, qrels, k),
        "mrr": mrr(run, qrels),
        f"ndcg@{k}": ndcg_at_k(run, qrels, k),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python -m eval.metrics <run.json> <qrels.tsv>")
        raise SystemExit(1)
    run_data = load_run(sys.argv[1])
    qrels_data = load_qrels(sys.argv[2])
    for name, val in evaluate(run_data, qrels_data, k=10).items():
        print(f"{name}: {val:.4f}")
