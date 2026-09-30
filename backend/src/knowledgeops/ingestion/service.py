from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from knowledgeops.db.models import Document, DocumentChunk
from knowledgeops.embeddings.constants import EMBEDDING_DIMENSION
from knowledgeops.ingestion.checksum import calculate_sha256
from knowledgeops.ingestion.chunking import chunk_document
from knowledgeops.ingestion.parser import parse_document


class Embedder(Protocol):
    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        ...


class DuplicateDocumentError(ValueError):
    pass


class EmptyDocumentError(ValueError):
    pass


def ingest_document(
    session: Session,
    file_path: Path,
    embedder: Embedder,
) -> Document:
    checksum = calculate_sha256(file_path)

    existing_document = session.scalar(
        select(Document).where(
            Document.checksum == checksum
        )
    )

    if existing_document is not None:
        raise DuplicateDocumentError(
            f"Document already exists: {existing_document.id}"
        )

    parsed_document = parse_document(file_path)

    chunks = chunk_document(parsed_document)

    if not chunks:
        raise EmptyDocumentError(
            "Document contains no extractable text"
        )

    embeddings = embedder.embed_texts(
        [
            chunk.content
            for chunk in chunks
        ]
    )

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            "Embedding count does not match chunk count"
        )

    if any(
        len(embedding) != EMBEDDING_DIMENSION
        for embedding in embeddings
    ):
        raise RuntimeError(
            "Unexpected embedding dimension"
        )

    document = Document(
        filename=parsed_document.filename,
        title=file_path.stem,
        mime_type=parsed_document.mime_type,
        checksum=checksum,
        status="ready",
    )

    for chunk, embedding in zip(
        chunks,
        embeddings,
        strict=True,
    ):
        document.chunks.append(
            DocumentChunk(
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                page_number=chunk.page_number,
                section=None,
                embedding=embedding,
            )
        )

    session.add(document)

    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()

        raise DuplicateDocumentError(
            "Document already exists"
        ) from exc
    except Exception:
        session.rollback()
        raise

    session.refresh(document)

    return document