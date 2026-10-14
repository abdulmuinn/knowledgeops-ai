import uuid

import pytest

from knowledgeops.rag.context import (
    build_context,
)
from knowledgeops.reranking.models import (
    RerankedResult,
)
from knowledgeops.retrieval.models import (
    SearchResult,
)


def make_reranked_result(
    filename: str,
    content: str,
    chunk_index: int,
    dense_score: float,
    rerank_score: float,
    page_number: int | None = None,
) -> RerankedResult:
    search_result = SearchResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename=filename,
        content=content,
        page_number=page_number,
        chunk_index=chunk_index,
        score=dense_score,
    )

    return RerankedResult(
        result=search_result,
        rerank_score=rerank_score,
    )


def test_context_preserves_reranked_order() -> None:
    results = [
        make_reranked_result(
            filename="first.txt",
            content="First evidence.",
            chunk_index=0,
            dense_score=0.40,
            rerank_score=5.0,
        ),
        make_reranked_result(
            filename="second.txt",
            content="Second evidence.",
            chunk_index=0,
            dense_score=0.60,
            rerank_score=2.0,
        ),
    ]

    context = build_context(results)

    assert context.sources[0].filename == (
        "first.txt"
    )

    assert context.sources[1].filename == (
        "second.txt"
    )


def test_context_assigns_stable_citations() -> None:
    results = [
        make_reranked_result(
            filename="first.txt",
            content="First evidence.",
            chunk_index=0,
            dense_score=0.50,
            rerank_score=3.0,
        ),
        make_reranked_result(
            filename="second.txt",
            content="Second evidence.",
            chunk_index=1,
            dense_score=0.40,
            rerank_score=2.0,
        ),
    ]

    context = build_context(results)

    assert context.sources[0].citation_id == (
        "S1"
    )

    assert context.sources[1].citation_id == (
        "S2"
    )

    assert "[S1]" in context.text
    assert "[S2]" in context.text


def test_context_preserves_source_metadata() -> None:
    result = make_reranked_result(
        filename="policy.txt",
        content="Security approval is required.",
        chunk_index=4,
        dense_score=0.52,
        rerank_score=6.4,
        page_number=3,
    )

    context = build_context([result])

    source = context.sources[0]

    assert source.filename == "policy.txt"
    assert source.page_number == 3
    assert source.chunk_index == 4
    assert source.dense_score == 0.52
    assert source.rerank_score == 6.4


def test_context_respects_max_sources() -> None:
    results = [
        make_reranked_result(
            filename=f"doc-{index}.txt",
            content=f"Evidence {index}",
            chunk_index=index,
            dense_score=0.5,
            rerank_score=float(
                10 - index
            ),
        )
        for index in range(5)
    ]

    context = build_context(
        results,
        max_sources=2,
    )

    assert len(context.sources) == 2
    assert "[S3]" not in context.text


def test_invalid_max_sources_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="max_sources",
    ):
        build_context(
            [],
            max_sources=0,
        )