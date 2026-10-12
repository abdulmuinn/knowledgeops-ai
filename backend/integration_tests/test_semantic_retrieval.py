import uuid

from knowledgeops.db.models import (
    Document,
    DocumentChunk,
)
from knowledgeops.db.session import SessionLocal
from knowledgeops.embeddings.constants import (
    EMBEDDING_DIMENSION,
)
from knowledgeops.retrieval.service import (
    semantic_search,
)


def make_vector(position: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSION
    vector[position] = 1.0
    return vector


class FakeQueryEmbedder:
    def embed_text(
        self,
        text: str,
    ) -> list[float]:
        return make_vector(0)


def test_semantic_search_orders_by_similarity() -> None:
    document_id = uuid.uuid4()

    document = Document(
        id=document_id,
        filename="retrieval-test.txt",
        title="Retrieval Test",
        mime_type="text/plain",
        checksum=uuid.uuid4().hex * 2,
        status="ready",
    )

    document.chunks.extend(
        [
            DocumentChunk(
                chunk_index=0,
                content=(
                    "Production access requires "
                    "Security Team approval."
                ),
                page_number=1,
                embedding=make_vector(0),
            ),
            DocumentChunk(
                chunk_index=1,
                content=(
                    "Employees can submit "
                    "travel expense claims."
                ),
                page_number=2,
                embedding=make_vector(1),
            ),
        ]
    )

    with SessionLocal() as session:
        try:
            session.add(document)
            session.commit()

            results = semantic_search(
                session=session,
                query=(
                    "Who approves production access?"
                ),
                embedder=FakeQueryEmbedder(),
                top_k=2,
            )

            assert len(results) == 2

            assert results[0].content == (
                "Production access requires "
                "Security Team approval."
            )

            assert results[0].score > results[1].score
            assert results[0].score == 1.0

        finally:
            stored_document = session.get(
                Document,
                document_id,
            )

            if stored_document is not None:
                session.delete(stored_document)
                session.commit()