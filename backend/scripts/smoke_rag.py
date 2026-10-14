import os
from pathlib import Path

from sqlalchemy import select

from knowledgeops.db.models import Document
from knowledgeops.db.session import SessionLocal
from knowledgeops.embeddings.service import EmbeddingService
from knowledgeops.ingestion.checksum import calculate_sha256
from knowledgeops.llm.gemini_client import GeminiClient
from knowledgeops.rag.pipeline import answer_question
from knowledgeops.reranking.cross_encoder import (
    CrossEncoderReranker,
)
from knowledgeops.retrieval.client import (
    SemanticRetriever,
)


CORPUS_DIR = Path(
    "evaluation/hard_knowledge_base"
)


QUESTIONS = [
    (
        "ANSWERABLE",
        (
            "Who must approve standard employee "
            "access to production systems?"
        ),
    ),
    (
        "UNANSWERABLE",
        (
            "Does the company reimburse monthly "
            "office parking fees?"
        ),
    ),
]


def load_document_ids(
    session,
) -> list:
    document_ids = []

    for file_path in sorted(
        CORPUS_DIR.glob("*.txt")
    ):
        checksum = calculate_sha256(
            file_path
        )

        document = session.scalar(
            select(Document).where(
                Document.checksum == checksum
            )
        )

        if document is None:
            raise RuntimeError(
                "Evaluation document not ingested: "
                f"{file_path.name}"
            )

        document_ids.append(
            document.id
        )

    return document_ids


def main() -> None:
    model = os.getenv(
        "GEMINI_MODEL",
        "gemini-3.8-flash",
    )

    embedder = EmbeddingService()
    reranker = CrossEncoderReranker()

    llm = GeminiClient(
        model=model,
        temperature=0.1,
    )

    with SessionLocal() as session:
        retriever = SemanticRetriever(
            session=session,
            embedder=embedder,
        )

        document_ids = load_document_ids(
            session
        )

        print(
            f"Gemini model: {model}"
        )

        print(
            f"Knowledge documents: "
            f"{len(document_ids)}"
        )

        for label, question in QUESTIONS:
            print()
            print("=" * 60)
            print(label)
            print("=" * 60)
            print(
                f"Question: {question}"
            )

            response = answer_question(
                question=question,
                retriever=retriever,
                reranker=reranker,
                llm=llm,
                top_k=3,
                rerank_top_n=3,
                max_sources=3,

                # Deliberately None for this smoke test.
                # We want Gemini to decide whether
                # retrieved evidence is sufficient.
                min_dense_score=None,

                document_ids=document_ids,
            )

            print(
                f"Status: {response.status}"
            )

            if response.answer is not None:
                print(
                    f"Answer: {response.answer}"
                )
            else:
                print(
                    "Answer: <abstained>"
                )

            if response.sources:
                print("Sources:")

                for source in response.sources:
                    print(
                        f"  [{source.citation_id}] "
                        f"{source.filename} "
                        f"chunk={source.chunk_index} "
                        f"dense={source.dense_score:.4f} "
                        f"rerank={source.rerank_score:.4f}"
                    )
            else:
                print(
                    "Sources: none"
                )


if __name__ == "__main__":
    main()