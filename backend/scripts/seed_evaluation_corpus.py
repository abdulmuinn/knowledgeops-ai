from pathlib import Path

from knowledgeops.db.session import SessionLocal
from knowledgeops.embeddings.service import EmbeddingService
from knowledgeops.ingestion.service import (
    DuplicateDocumentError,
    ingest_document,
)


CORPUS_DIR = Path("evaluation/knowledge_base")


def main() -> None:
    files = sorted(CORPUS_DIR.glob("*.txt"))

    if not files:
        raise RuntimeError(
            f"No evaluation documents found in {CORPUS_DIR}"
        )

    embedder = EmbeddingService()

    inserted = 0
    skipped = 0

    with SessionLocal() as session:
        for file_path in files:
            try:
                document = ingest_document(
                    session=session,
                    file_path=file_path,
                    embedder=embedder,
                )

                inserted += 1

                print(
                    f"INGESTED: {document.filename} "
                    f"({len(document.chunks)} chunks)"
                )

            except DuplicateDocumentError:
                skipped += 1
                print(
                    f"SKIPPED: {file_path.name} "
                    "(already exists)"
                )

    print()
    print(f"Inserted: {inserted}")
    print(f"Skipped: {skipped}")
    print(f"Total corpus files: {len(files)}")


if __name__ == "__main__":
    main()