from pathlib import Path

import pytest
from sqlalchemy import select

from knowledgeops.db.models import Document, DocumentChunk
from knowledgeops.db.session import SessionLocal
from knowledgeops.embeddings.constants import EMBEDDING_DIMENSION
from knowledgeops.ingestion.service import (
    DuplicateDocumentError,
    ingest_document,
)


class FakeEmbedder:
    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [
            [0.01] * EMBEDDING_DIMENSION
            for _ in texts
        ]


def test_ingestion_persists_document_and_chunks(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "integration-policy.txt"

    file_path.write_text(
        (
            "KnowledgeOps AI Integration Policy\n\n"
            "Production access requires explicit "
            "Security Team approval."
        ),
        encoding="utf-8",
    )

    embedder = FakeEmbedder()

    with SessionLocal() as session:
        document = None

        try:
            document = ingest_document(
                session=session,
                file_path=file_path,
                embedder=embedder,
            )

            stored_document = session.scalar(
                select(Document).where(
                    Document.id == document.id
                )
            )

            assert stored_document is not None
            assert stored_document.status == "ready"
            assert stored_document.filename == (
                "integration-policy.txt"
            )

            stored_chunks = session.scalars(
                select(DocumentChunk).where(
                    DocumentChunk.document_id
                    == document.id
                )
            ).all()

            assert len(stored_chunks) == 1
            assert len(stored_chunks[0].embedding) == (
                EMBEDDING_DIMENSION
            )

            with pytest.raises(
                DuplicateDocumentError
            ):
                ingest_document(
                    session=session,
                    file_path=file_path,
                    embedder=embedder,
                )

        finally:
            if document is not None:
                stored_document = session.get(
                    Document,
                    document.id,
                )

                if stored_document is not None:
                    session.delete(stored_document)
                    session.commit()