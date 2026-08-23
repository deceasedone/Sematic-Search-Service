# Semantic Search-as-a-Service

Finds documents by meaning, not keywords. Built in measured phases — every
phase ends in a number (recall@k, MRR, p95 latency), not just working code.

## Locked-in decisions

| Choice | Pick | Why |
|---|---|---|
| Embedding | Local `sentence-transformers` (`all-MiniLM-L6-v2` to start), behind a pluggable `EmbeddingProvider` interface so an API model (OpenAI/Cohere/Voyage) can be swapped in later without touching calling code | Free/offline to iterate on; mirrors the provider-agnostic factory pattern already used in Argus Lite |
| Vector store | pgvector | One system (Postgres) for documents + vectors, simplest mental model. Qdrant considered later as a Phase 5 tradeoff write-up, not a rebuild |
| Corpus + eval set | **FiQA-2018** (BEIR) — real StackExchange Investment/Personal-Finance posts, 2009–2017. 57,638 documents, 648 labeled test queries, ~1,700 relevance judgments | A real Stack Exchange dump *and* pre-labeled (BEIR qrels), so Phase 0's "build a gold eval set" is mostly done for free. 10–15x larger than the toy BEIR sets (SciFact/NFCorpus). Live index can later be padded with more unlabeled Investment-tag posts to push scale further for the Phase 4/5 demo, without touching the labeled eval subset |

## Repo layout (Phase 0)

```
semantic-search-service/
├── eval/
│   └── metrics.py       # recall@k, precision@k, MRR, nDCG@k, latency percentiles
├── scripts/
│   └── download_fiqa.py # pulls + unzips FiQA-2018 into data/fiqa/
├── tests/
│   └── test_metrics.py  # unit tests against hand-computed values
├── data/                # gitignored — created by the download script
└── requirements.txt
```

## Run it

```bash
pip install -r requirements.txt
python -m pytest tests/ -v        # verify the metrics library
python scripts/download_fiqa.py   # pulls data/fiqa/{corpus,queries}.jsonl + qrels/
```

> Run the download script on your own machine, not inside a restricted
> sandbox — it needs to reach `public.ukp.informatik.tu-darmstadt.de`, which
> isn't a package registry.

Once downloaded, `eval/metrics.load_qrels("data/fiqa/qrels/test.tsv")` gives
you the `Qrels` structure any future search mode's results get scored
against.

## Phase checklist

- [x] **Phase 0** — corpus chosen (FiQA-2018), eval set = its labeled test
      split (648 queries), `metrics.py` written and unit-tested (8/8 passing)
- [ ] Phase 1 — ingest FiQA corpus into Postgres, BM25 keyword baseline, first
      recall@10 / MRR numbers
- [ ] Phase 2 — sentence-transformers embeddings + pgvector, hybrid RRF
      (BM25 + semantic), comparison table
- [ ] Phase 3 — Redis cache, rate limiting, `/health`, docker-compose
- [ ] Phase 4 — Locust load test, SLA
- [ ] Phase 5 — README write-up (architecture diagram + numbers) + live demo
- [ ] Phase 6 — (optional) hand-built mini-HNSW vs pgvector vs hnswlib

## Later

Plan is to eventually connect this service to Argus Lite as a retrieval
backend — that's a late-stage integration, not part of the phases above.
