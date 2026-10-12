from collections.abc import Sequence
from typing import Protocol

from knowledgeops.reranking.models import (
    RerankedResult,
)
from knowledgeops.retrieval.models import (
    SearchResult,
)


class PairScorer(Protocol):
    def score(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[float]:
        ...


def rerank_results(
    query: str,
    results: Sequence[SearchResult],
    scorer: PairScorer,
    top_n: int | None = None,
) -> list[RerankedResult]:
    query = query.strip()

    if not query:
        raise ValueError(
            "query cannot be empty"
        )

    if top_n is not None and top_n <= 0:
        raise ValueError(
            "top_n must be greater than 0"
        )

    if not results:
        return []

    documents = [
        result.content
        for result in results
    ]

    scores = scorer.score(
        query=query,
        documents=documents,
    )

    if len(scores) != len(results):
        raise RuntimeError(
            "Reranker score count does not "
            "match result count"
        )

    reranked = [
        RerankedResult(
            result=result,
            rerank_score=float(score),
        )
        for result, score in zip(
            results,
            scores,
        )
    ]

    reranked.sort(
        key=lambda item: item.rerank_score,
        reverse=True,
    )

    if top_n is not None:
        reranked = reranked[:top_n]

    return reranked