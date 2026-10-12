import pytest

from knowledgeops.evaluation.retrieval import (
    RetrievalEvaluationResult,
    hit_rate_at_k,
    mean_reciprocal_rank,
)


def test_hit_rate_at_one() -> None:
    results = [
        RetrievalEvaluationResult(
            expected_source="a.txt",
            retrieved_sources=[
                "a.txt",
                "b.txt",
            ],
        ),
        RetrievalEvaluationResult(
            expected_source="b.txt",
            retrieved_sources=[
                "a.txt",
                "b.txt",
            ],
        ),
    ]

    assert hit_rate_at_k(results, 1) == 0.5


def test_hit_rate_at_two() -> None:
    results = [
        RetrievalEvaluationResult(
            expected_source="a.txt",
            retrieved_sources=[
                "a.txt",
                "b.txt",
            ],
        ),
        RetrievalEvaluationResult(
            expected_source="b.txt",
            retrieved_sources=[
                "a.txt",
                "b.txt",
            ],
        ),
    ]

    assert hit_rate_at_k(results, 2) == 1.0


def test_mean_reciprocal_rank() -> None:
    results = [
        RetrievalEvaluationResult(
            expected_source="a.txt",
            retrieved_sources=[
                "a.txt",
                "b.txt",
            ],
        ),
        RetrievalEvaluationResult(
            expected_source="b.txt",
            retrieved_sources=[
                "a.txt",
                "b.txt",
            ],
        ),
    ]

    assert mean_reciprocal_rank(
        results
    ) == pytest.approx(0.75)