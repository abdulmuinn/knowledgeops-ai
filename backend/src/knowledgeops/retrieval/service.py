import uuid
from collections.abc import Sequence
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from knowledgeops.db.models import Document, DocumentChunk
from knowledgeops.embeddings.constants import EMBEDDING_DIMENSION
from knowledgeops.retrieval.models import SearchResult


class QueryEmbedder(Protocol):
    def embed_text(
        self,
        text: str,
    ) -> list[float]:
        ...


def semantic_search(
    session: Session,
    query: str,
    embedder: QueryEmbedder,
    top_k: int = 5,
    min_score: float | None = None,
    document_ids: Sequence[uuid.UUID] | None = None,
) -> list[SearchResult]:
    query = query.strip()

    if not query:
        raise ValueError("query cannot be empty")

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0"
        )

    if document_ids is not None and not document_ids:
        return []

    query_embedding = embedder.embed_text(query)

    if len(query_embedding) != EMBEDDING_DIMENSION:
        raise RuntimeError(
            "Unexpected query embedding dimension"
        )

    distance = (
        DocumentChunk.embedding.cosine_distance(
            query_embedding
        )
    )

    statement = (
        select(
            DocumentChunk,
            Document,
            distance.label("distance"),
        )
        .join(
            Document,
            Document.id == DocumentChunk.document_id,
        )
        .where(
            DocumentChunk.embedding.is_not(None),
            Document.status == "ready",
        )
    )

    if document_ids is not None:
        statement = statement.where(
            Document.id.in_(document_ids)
        )

    statement = (
        statement
        .order_by(distance)
        .limit(top_k)
    )

    rows = session.execute(statement).all()

    results: list[SearchResult] = []

    for chunk, document, cosine_distance in rows:
        score = 1.0 - float(cosine_distance)

        if (
            min_score is not None
            and score < min_score
        ):
            continue

        results.append(
            SearchResult(
                chunk_id=chunk.id,
                document_id=document.id,
                filename=document.filename,
                content=chunk.content,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                score=score,
            )
        )

    return results