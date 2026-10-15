import uuid

import pytest

from knowledgeops.rag.pipeline import (
    answer_question,
)
from knowledgeops.retrieval.models import (
    SearchResult,
)


class FakeRetriever:
    def __init__(
        self,
        results: list[SearchResult],
    ) -> None:
        self.results = results

        self.query_received: str | None = None
        self.top_k_received: int | None = None
        self.min_score_received: float | None = None

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
        document_ids=None,
    ) -> list[SearchResult]:
        self.query_received = query
        self.top_k_received = top_k
        self.min_score_received = min_score

        return self.results


class FakeReranker:
    def __init__(
        self,
        scores: list[float],
    ) -> None:
        self.scores = scores
        self.call_count = 0

    def score(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        self.call_count += 1
        return self.scores


class FakeLLM:
    def __init__(
        self,
        response: str,
    ) -> None:
        self.response = response
        self.call_count = 0

    def generate(
        self,
        system: str,
        user: str,
    ) -> str:
        self.call_count += 1
        return self.response


def make_result(
    filename: str,
    content: str,
    score: float,
    index: int,
) -> SearchResult:
    return SearchResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename=filename,
        content=content,
        page_number=1,
        chunk_index=index,
        score=score,
    )


def test_pipeline_returns_grounded_answer() -> None:
    retriever = FakeRetriever(
        [
            make_result(
                filename="security.txt",
                content=(
                    "Production access requires "
                    "Security Team approval."
                ),
                score=0.60,
                index=0,
            )
        ]
    )

    reranker = FakeReranker(
        [5.0]
    )

    llm = FakeLLM(
        "Security Team approval is required [S1]."
    )

    response = answer_question(
        question=(
            "Who approves production access?"
        ),
        retriever=retriever,
        reranker=reranker,
        llm=llm,
    )

    assert response.status == "answered"
    assert response.answer is not None

    assert response.sources[0].filename == (
        "security.txt"
    )


def test_pipeline_abstains_when_retrieval_empty() -> None:
    retriever = FakeRetriever([])

    reranker = FakeReranker([])
    llm = FakeLLM(
        "Should not be called"
    )

    response = answer_question(
        question="Unknown benefit question",
        retriever=retriever,
        reranker=reranker,
        llm=llm,
    )

    assert response.status == "abstained"
    assert reranker.call_count == 0
    assert llm.call_count == 0


def test_pipeline_passes_retrieval_configuration() -> None:
    retriever = FakeRetriever([])

    response = answer_question(
        question="Policy question",
        retriever=retriever,
        reranker=FakeReranker([]),
        llm=FakeLLM("unused"),
        top_k=5,
        min_dense_score=0.42,
    )

    assert response.status == "abstained"

    assert retriever.top_k_received == 5

    assert retriever.min_score_received == (
        0.42
    )


def test_pipeline_applies_reranking() -> None:
    retriever = FakeRetriever(
        [
            make_result(
                filename="wrong.txt",
                content="Less relevant evidence.",
                score=0.70,
                index=0,
            ),
            make_result(
                filename="correct.txt",
                content=(
                    "Security Team approves "
                    "production access."
                ),
                score=0.60,
                index=1,
            ),
        ]
    )

    reranker = FakeReranker(
        [
            1.0,
            8.0,
        ]
    )

    llm = FakeLLM(
        "Security Team approves access [S1]."
    )

    response = answer_question(
        question=(
            "Who approves production access?"
        ),
        retriever=retriever,
        reranker=reranker,
        llm=llm,
    )

    assert response.status == "answered"

    assert response.sources[0].filename == (
        "correct.txt"
    )


def test_empty_question_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="question",
    ):
        answer_question(
            question="   ",
            retriever=FakeRetriever([]),
            reranker=FakeReranker([]),
            llm=FakeLLM("unused"),
        )


def test_invalid_pipeline_configuration_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="top_k",
    ):
        answer_question(
            question="Question",
            retriever=FakeRetriever([]),
            reranker=FakeReranker([]),
            llm=FakeLLM("unused"),
            top_k=0,
        )