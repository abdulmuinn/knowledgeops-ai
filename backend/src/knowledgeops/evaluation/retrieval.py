from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True)
class RetrievalEvaluationResult:
    expected_source: str
    retrieved_sources: list[str]


def hit_rate_at_k(
    results: Sequence[RetrievalEvaluationResult],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not results:
        return 0.0

    hits = sum(
        result.expected_source
        in result.retrieved_sources[:k]
        for result in results
    )

    return hits / len(results)


def mean_reciprocal_rank(
    results: Sequence[RetrievalEvaluationResult],
) -> float:
    if not results:
        return 0.0

    reciprocal_ranks: list[float] = []

    for result in results:
        rank = 0

        for index, source in enumerate(
            result.retrieved_sources,
            start=1,
        ):
            if source == result.expected_source:
                rank = index
                break

        reciprocal_ranks.append(
            0.0 if rank == 0 else 1.0 / rank
        )

    return sum(reciprocal_ranks) / len(
        reciprocal_ranks
    )