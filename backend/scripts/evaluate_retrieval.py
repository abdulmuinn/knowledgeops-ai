import json
from pathlib import Path

from sqlalchemy import select

from knowledgeops.db.models import Document
from knowledgeops.db.session import SessionLocal
from knowledgeops.embeddings.service import EmbeddingService
from knowledgeops.evaluation.retrieval import (
    RetrievalEvaluationResult,
    hit_rate_at_k,
    mean_reciprocal_rank,
)
from knowledgeops.ingestion.checksum import calculate_sha256
from knowledgeops.retrieval.service import semantic_search


CASES_PATH = Path("evaluation/retrieval_cases.json")
CORPUS_DIR = Path("evaluation/knowledge_base")


def load_cases() -> list[dict[str, str]]:
    with CASES_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main() -> None:
    cases = load_cases()

    embedder = EmbeddingService()

    evaluation_results: list[
        RetrievalEvaluationResult
    ] = []

    with SessionLocal() as session:
        document_ids = []

        for file_path in sorted(
            CORPUS_DIR.glob("*.txt")
        ):
            checksum = calculate_sha256(file_path)

            document = session.scalar(
                select(Document).where(
                    Document.checksum == checksum
                )
            )

            if document is None:
                raise RuntimeError(
                    f"Evaluation document not ingested: "
                    f"{file_path.name}"
                )

            document_ids.append(document.id)

        for index, case in enumerate(
            cases,
            start=1,
        ):
            query = case["query"]
            expected_source = case[
                "expected_source"
            ]

            results = semantic_search(
                session=session,
                query=query,
                embedder=embedder,
                top_k=3,
                document_ids=document_ids,
            )

            retrieved_sources = [
                result.filename
                for result in results
            ]

            evaluation_results.append(
                RetrievalEvaluationResult(
                    expected_source=expected_source,
                    retrieved_sources=retrieved_sources,
                )
            )

            top_source = (
                retrieved_sources[0]
                if retrieved_sources
                else "NO RESULT"
            )

            status = (
                "PASS"
                if top_source == expected_source
                else "MISS"
            )

            print()
            print(
                f"[{index:02d}] {status}"
            )
            print(f"Query: {query}")
            print(
                f"Expected: {expected_source}"
            )

            for rank, result in enumerate(
                results,
                start=1,
            ):
                print(
                    f"  #{rank} "
                    f"{result.filename} "
                    f"score={result.score:.4f}"
                )

    hit_1 = hit_rate_at_k(
        evaluation_results,
        1,
    )

    hit_3 = hit_rate_at_k(
        evaluation_results,
        3,
    )

    mrr = mean_reciprocal_rank(
        evaluation_results
    )

    print()
    print("=" * 50)
    print("RETRIEVAL BASELINE")
    print("=" * 50)
    print(
        f"Queries : {len(evaluation_results)}"
    )
    print(f"Hit@1   : {hit_1:.3f}")
    print(f"Hit@3   : {hit_3:.3f}")
    print(f"MRR     : {mrr:.3f}")


if __name__ == "__main__":
    main()