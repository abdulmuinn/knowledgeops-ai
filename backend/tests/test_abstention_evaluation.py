import pytest

from knowledgeops.evaluation.abstention import (
    evaluate_threshold,
    find_best_threshold,
)


def test_threshold_with_perfect_separation() -> None:
    result = evaluate_threshold(
        answerable_scores=[
            0.70,
            0.80,
        ],
        unanswerable_scores=[
            0.10,
            0.20,
        ],
        threshold=0.50,
    )

    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
    assert result.accuracy == 1.0


def test_threshold_with_overlap() -> None:
    result = evaluate_threshold(
        answerable_scores=[
            0.40,
            0.80,
        ],
        unanswerable_scores=[
            0.30,
            0.50,
        ],
        threshold=0.45,
    )

    assert result.precision == pytest.approx(
        0.5
    )
    assert result.recall == pytest.approx(
        0.5
    )
    assert result.f1 == pytest.approx(
        0.5
    )
    assert result.accuracy == pytest.approx(
        0.5
    )


def test_find_best_threshold() -> None:
    result = find_best_threshold(
        answerable_scores=[
            0.60,
            0.80,
        ],
        unanswerable_scores=[
            0.10,
            0.20,
        ],
    )

    assert 0.20 < result.threshold < 0.60
    assert result.f1 == 1.0