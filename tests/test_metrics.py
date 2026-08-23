import math

import pytest

from eval.metrics import (
    latency_percentiles,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

# Two queries, hand-computable by inspection:
#   q1 relevant = {d1, d2}; ranked run = [d3, d1, d5, d2]  (d3, d5 irrelevant/unknown)
#   q2 relevant = {d4};     ranked run = [d6, d4]
QRELS = {
    "q1": {"d1": 1, "d2": 1, "d3": 0},
    "q2": {"d4": 1},
}
RUN = {
    "q1": ["d3", "d1", "d5", "d2"],
    "q2": ["d6", "d4"],
}


def test_recall_at_10_finds_everything():
    # Both queries' full relevant sets appear somewhere in their (short) run.
    assert recall_at_k(RUN, QRELS, k=10) == pytest.approx(1.0)


def test_recall_at_1_finds_nothing():
    # Top-1 for both queries is an irrelevant doc.
    assert recall_at_k(RUN, QRELS, k=1) == pytest.approx(0.0)


def test_precision_at_10():
    # q1: 2 relevant hits / 4 retrieved = 0.5
    # q2: 1 relevant hit / 2 retrieved = 0.5
    assert precision_at_k(RUN, QRELS, k=10) == pytest.approx(0.5)


def test_mrr_first_relevant_rank():
    # q1: first relevant doc (d1) is at rank 2 -> 1/2
    # q2: first relevant doc (d4) is at rank 2 -> 1/2
    assert mrr(RUN, QRELS) == pytest.approx(0.5)


def test_ndcg_at_10_matches_manual_calc():
    # q1 DCG = 1/log2(3) [d1@rank2] + 1/log2(5) [d2@rank4]
    #    IDCG (ideal order [1,1,0]) = 1/log2(2) + 1/log2(3)
    q1_dcg = 1 / math.log2(3) + 1 / math.log2(5)
    q1_idcg = 1 / math.log2(2) + 1 / math.log2(3)
    q1_ndcg = q1_dcg / q1_idcg

    # q2 DCG = 1/log2(3) [d4@rank2]; IDCG (ideal [1]) = 1/log2(2)
    q2_dcg = 1 / math.log2(3)
    q2_idcg = 1 / math.log2(2)
    q2_ndcg = q2_dcg / q2_idcg

    expected = (q1_ndcg + q2_ndcg) / 2
    assert ndcg_at_k(RUN, QRELS, k=10) == pytest.approx(expected)


def test_empty_run_scores_zero():
    empty_run = {"q1": [], "q2": []}
    assert recall_at_k(empty_run, QRELS) == pytest.approx(0.0)
    assert mrr(empty_run, QRELS) == pytest.approx(0.0)
    assert ndcg_at_k(empty_run, QRELS) == pytest.approx(0.0)


def test_latency_percentiles():
    latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    result = latency_percentiles(latencies, percentiles=(50, 95))
    assert result[50] == pytest.approx(55.0)  # numpy linear interpolation
    assert 95 <= result[95] <= 100


def test_latency_percentiles_empty():
    assert latency_percentiles([], percentiles=(50, 95)) == {50: 0.0, 95: 0.0}
