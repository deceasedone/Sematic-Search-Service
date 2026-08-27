"""Locust load test for GET /search.

IMPORTANT — read before running: Phase 3 added a per-IP rate limit
(RATE_LIMIT, default 30/minute). Locust runs from one machine, so every
simulated user shares the same source IP and therefore the same rate-limit
bucket — without raising it, you'll see a wall of 429s that measures the
rate limiter, not the service. Start the app with a much higher limit for
this test:

    (bash)       RATE_LIMIT=100000/minute uvicorn app.main:app
    (PowerShell) $env:RATE_LIMIT="100000/minute"; uvicorn app.main:app

This isn't cheating the result — a real load test simulating many distinct
end users would need distinct client identities (API keys) to exercise
per-user rate limits meaningfully. A single-origin tool like Locust can't do
that without spoofing source IPs, so this isolates throughput/latency from
the rate-limiting feature on purpose. Note that fact in the SLA writeup.

Usage:
    locust -f locustfile.py --host http://localhost:8000 \
        --users 50 --spawn-rate 5 --run-time 2m --headless \
        --csv results/loadtest --html results/loadtest.html
"""
import json
import random
from pathlib import Path

from locust import HttpUser, between, task

QUERIES_PATH = Path(__file__).resolve().parent / "data" / "fiqa" / "queries.jsonl"


def load_queries():
    queries = []
    with open(QUERIES_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line)["text"])
    return queries


QUERY_POOL = load_queries()
# A small repeated slice alongside the full pool means some requests are
# cache hits (repeat queries) and some are cache misses (novel queries) —
# closer to a real mixed workload than an all-hit or all-miss test.
HOT_QUERIES = QUERY_POOL[:20]


class SearchUser(HttpUser):
    wait_time = between(0.2, 1.5)

    @task(3)
    def search_hot_query(self):
        """Repeats a small pool of queries — mostly cache hits after warmup."""
        q = random.choice(HOT_QUERIES)
        self.client.get("/search", params={"q": q, "k": 10, "mode": "semantic"}, name="/search [hot]")

    @task(1)
    def search_novel_query(self):
        """Draws from the full pool — mostly cache misses."""
        q = random.choice(QUERY_POOL)
        self.client.get("/search", params={"q": q, "k": 10, "mode": "semantic"}, name="/search [novel]")

    @task(1)
    def search_hybrid(self):
        q = random.choice(QUERY_POOL)
        self.client.get("/search", params={"q": q, "k": 10, "mode": "hybrid"}, name="/search [hybrid]")
