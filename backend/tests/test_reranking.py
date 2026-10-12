import uuid

import pytest

from knowledgeops.reranking.service import (
    rerank_results,
)
from knowledgeops.retrieval.models import (
    SearchResult,
)


class FakeScorer:
    def score(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        return [
            0.2,
            0.9,
            0.4,
        ]


class WrongCountScorer:
    def score(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        return [0.5]


def make_result(
    content: str,
    index: int,
) -> SearchResult:
    return SearchResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename=f"doc-{index}.txt",
        content=content,
        page_number=None,
        chunk_index=index,
        score=0.5,
    )


def test_reranker_reorders_results() -> None:
    results = [
        make_result("Document A", 0),
        make_result("Document B", 1),
        make_result("Document C", 2),
    ]

    reranked = rerank_results(
        query="test query",
        results=results,
        scorer=FakeScorer(),
    )

    assert reranked[0].result.content == (
        "Document B"
    )

    assert reranked[1].result.content == (
        "Document C"
    )

    assert reranked[2].result.content == (
        "Document A"
    )


def test_reranker_top_n() -> None:
    results = [
        make_result("Document A", 0),
        make_result("Document B", 1),
        make_result("Document C", 2),
    ]

    reranked = rerank_results(
        query="test query",
        results=results,
        scorer=FakeScorer(),
        top_n=2,
    )

    assert len(reranked) == 2


def test_empty_results_return_empty_list() -> None:
    reranked = rerank_results(
        query="test query",
        results=[],
        scorer=FakeScorer(),
    )

    assert reranked == []


def test_empty_query_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="empty",
    ):
        rerank_results(
            query="   ",
            results=[
                make_result(
                    "Document A",
                    0,
                )
            ],
            scorer=FakeScorer(),
        )


def test_invalid_top_n_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="top_n",
    ):
        rerank_results(
            query="test query",
            results=[
                make_result(
                    "Document A",
                    0,
                )
            ],
            scorer=FakeScorer(),
            top_n=0,
        )


def test_score_count_must_match_results() -> None:
    results = [
        make_result("Document A", 0),
        make_result("Document B", 1),
    ]

    with pytest.raises(
        RuntimeError,
        match="score count",
    ):
        rerank_results(
            query="test query",
            results=results,
            scorer=WrongCountScorer(),
        )