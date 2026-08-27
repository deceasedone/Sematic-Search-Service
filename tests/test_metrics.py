import math

import pytest

from eval.metrics import (
    latency_percentiles,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

QRELS = {
    "q1": {"d1": 1, "d2": 1, "d3": 0},
    "q2": {"d4": 1},
}
RUN = {
    "q1": ["d3", "d1", "d5", "d2"],
    "q2": ["d6", "d4"],
}


def test_recall_at_10_finds_everything():
    assert recall_at_k(RUN, QRELS, k=10) == pytest.approx(1.0)


def test_recall_at_1_finds_nothing():
    assert recall_at_k(RUN, QRELS, k=1) == pytest.approx(0.0)


def test_precision_at_10():
    assert precision_at_k(RUN, QRELS, k=10) == pytest.approx(0.5)


def test_mrr_first_relevant_rank():
    assert mrr(RUN, QRELS) == pytest.approx(0.5)


def test_ndcg_at_10_matches_manual_calc():
    q1_dcg = 1 / math.log2(3) + 1 / math.log2(5)
    q1_idcg = 1 / math.log2(2) + 1 / math.log2(3)
    q1_ndcg = q1_dcg / q1_idcg

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
    assert result[50] == pytest.approx(55.0)
    assert 95 <= result[95] <= 100


def test_latency_percentiles_empty():
    assert latency_percentiles([], percentiles=(50, 95)) == {50: 0.0, 95: 0.0}
