import argparse
from pathlib import Path

from knowledgeops.db.session import SessionLocal
from knowledgeops.embeddings.service import EmbeddingService
from knowledgeops.ingestion.service import (
    DuplicateDocumentError,
    ingest_document,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "file_path",
        type=Path,
    )

    args = parser.parse_args()

    if not args.file_path.exists():
        raise FileNotFoundError(
            f"File not found: {args.file_path}"
        )

    embedder = EmbeddingService()

    with SessionLocal() as session:
        try:
            document = ingest_document(
                session=session,
                file_path=args.file_path,
                embedder=embedder,
            )
        except DuplicateDocumentError as exc:
            print(f"Duplicate document: {exc}")
            return

        print("Document ingestion PASSED")
        print(f"ID: {document.id}")
        print(f"Filename: {document.filename}")
        print(f"Status: {document.status}")
        print(f"Chunks: {len(document.chunks)}")


if __name__ == "__main__":
    main()