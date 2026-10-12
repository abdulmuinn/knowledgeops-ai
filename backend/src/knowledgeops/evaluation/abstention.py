from dataclasses import dataclass


@dataclass(frozen=True)
class AbstentionMetrics:
    threshold: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    accuracy: float


def evaluate_threshold(
    answerable_scores: list[float],
    unanswerable_scores: list[float],
    threshold: float,
) -> AbstentionMetrics:
    true_positive = sum(
        score >= threshold
        for score in answerable_scores
    )

    false_negative = (
        len(answerable_scores)
        - true_positive
    )

    false_positive = sum(
        score >= threshold
        for score in unanswerable_scores
    )

    true_negative = (
        len(unanswerable_scores)
        - false_positive
    )

    precision_denominator = (
        true_positive + false_positive
    )

    precision = (
        true_positive / precision_denominator
        if precision_denominator
        else 0.0
    )

    recall_denominator = (
        true_positive + false_negative
    )

    recall = (
        true_positive / recall_denominator
        if recall_denominator
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    total = (
        true_positive
        + false_positive
        + true_negative
        + false_negative
    )

    accuracy = (
        (true_positive + true_negative)
        / total
        if total
        else 0.0
    )

    return AbstentionMetrics(
        threshold=threshold,
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
    )


def find_best_threshold(
    answerable_scores: list[float],
    unanswerable_scores: list[float],
) -> AbstentionMetrics:
    all_scores = sorted(
        set(
            answerable_scores
            + unanswerable_scores
        )
    )

    if not all_scores:
        raise ValueError(
            "scores cannot be empty"
        )

    candidates: list[float] = []

    for left, right in zip(
        all_scores,
        all_scores[1:],
    ):
        candidates.append(
            (left + right) / 2
        )

    candidates.append(
        all_scores[0] - 0.000001
    )

    candidates.append(
        all_scores[-1] + 0.000001
    )

    evaluations = [
        evaluate_threshold(
            answerable_scores,
            unanswerable_scores,
            threshold,
        )
        for threshold in candidates
    ]

    return max(
        evaluations,
        key=lambda result: (
            result.f1,
            result.accuracy,
            result.recall,
        ),
    )