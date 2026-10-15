import uuid
from collections.abc import Sequence
from typing import Protocol

from knowledgeops.llm.base import (
    LLMClient,
)
from knowledgeops.rag.models import (
    RAGResponse,
)
from knowledgeops.rag.service import (
    generate_grounded_answer,
)
from knowledgeops.reranking.service import (
    PairScorer,
    rerank_results,
)
from knowledgeops.retrieval.models import (
    SearchResult,
)


class Retriever(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
        document_ids: Sequence[
            uuid.UUID
        ] | None = None,
    ) -> list[SearchResult]:
        ...


def answer_question(
    question: str,
    retriever: Retriever,
    reranker: PairScorer,
    llm: LLMClient,
    top_k: int = 3,
    rerank_top_n: int = 3,
    max_sources: int = 3,
    min_dense_score: float | None = None,
    document_ids: Sequence[
        uuid.UUID
    ] | None = None,
) -> RAGResponse:
    question = question.strip()

    if not question:
        raise ValueError(
            "question cannot be empty"
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0"
        )

    if rerank_top_n <= 0:
        raise ValueError(
            "rerank_top_n must be greater than 0"
        )

    if max_sources <= 0:
        raise ValueError(
            "max_sources must be greater than 0"
        )

    retrieved = retriever.search(
        query=question,
        top_k=top_k,
        min_score=min_dense_score,
        document_ids=document_ids,
    )

    if not retrieved:
        return generate_grounded_answer(
            question=question,
            results=[],
            llm=llm,
            max_sources=max_sources,
        )

    reranked = rerank_results(
        query=question,
        results=retrieved,
        scorer=reranker,
        top_n=rerank_top_n,
    )

    return generate_grounded_answer(
        question=question,
        results=reranked,
        llm=llm,
        max_sources=max_sources,
    )