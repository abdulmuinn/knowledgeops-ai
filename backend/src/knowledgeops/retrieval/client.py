import uuid
from collections.abc import Sequence

from sqlalchemy.orm import Session

from knowledgeops.retrieval.models import (
    SearchResult,
)
from knowledgeops.retrieval.service import (
    QueryEmbedder,
    semantic_search,
)


class SemanticRetriever:
    def __init__(
        self,
        session: Session,
        embedder: QueryEmbedder,
    ) -> None:
        self.session = session
        self.embedder = embedder

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
        document_ids: Sequence[
            uuid.UUID
        ] | None = None,
    ) -> list[SearchResult]:
        return semantic_search(
            session=self.session,
            query=query,
            embedder=self.embedder,
            top_k=top_k,
            min_score=min_score,
            document_ids=document_ids,
        )