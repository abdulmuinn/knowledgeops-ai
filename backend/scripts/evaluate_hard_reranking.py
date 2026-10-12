import json
from pathlib import Path

from sqlalchemy import select

from knowledgeops.db.models import Document
from knowledgeops.db.session import SessionLocal
from knowledgeops.embeddings.service import (
    EmbeddingService,
)
from knowledgeops.evaluation.retrieval import (
    RetrievalEvaluationResult,
    hit_rate_at_k,
    mean_reciprocal_rank,
)
from knowledgeops.ingestion.checksum import (
    calculate_sha256,
)
from knowledgeops.reranking.cross_encoder import (
    CrossEncoderReranker,
)
from knowledgeops.reranking.service import (
    rerank_results,
)
from knowledgeops.retrieval.service import (
    semantic_search,
)


CASES_PATH = Path(
    "evaluation/hard_retrieval_cases.json"
)

CORPUS_DIR = Path(
    "evaluation/hard_knowledge_base"
)


def load_cases() -> list[dict[str, object]]:
    with CASES_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main() -> None:
    cases = load_cases()

    embedder = EmbeddingService()
    reranker = CrossEncoderReranker()

    baseline_results: list[
        RetrievalEvaluationResult
    ] = []

    reranked_results: list[
        RetrievalEvaluationResult
    ] = []

    with SessionLocal() as session:
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
                    "Hard evaluation document "
                    "not ingested: "
                    f"{file_path.name}"
                )

            document_ids.append(
                document.id
            )

        for index, case in enumerate(
            cases,
            start=1,
        ):
            if not bool(case["answerable"]):
                continue

            query = str(case["query"])
            expected_source = str(
                case["expected_source"]
            )

            retrieved = semantic_search(
                session=session,
                query=query,
                embedder=embedder,
                top_k=3,
                document_ids=document_ids,
            )

            reranked = rerank_results(
                query=query,
                results=retrieved,
                scorer=reranker,
            )

            baseline_sources = [
                result.filename
                for result in retrieved
            ]

            reranked_sources = [
                item.result.filename
                for item in reranked
            ]

            baseline_results.append(
                RetrievalEvaluationResult(
                    expected_source=expected_source,
                    retrieved_sources=baseline_sources,
                )
            )

            reranked_results.append(
                RetrievalEvaluationResult(
                    expected_source=expected_source,
                    retrieved_sources=reranked_sources,
                )
            )

            baseline_top = (
                baseline_sources[0]
                if baseline_sources
                else "NO RESULT"
            )

            reranked_top = (
                reranked_sources[0]
                if reranked_sources
                else "NO RESULT"
            )

            baseline_pass = (
                baseline_top == expected_source
            )

            reranked_pass = (
                reranked_top == expected_source
            )

            if (
                not baseline_pass
                or not reranked_pass
            ):
                print()
                print(
                    f"[{index:02d}] "
                    f"{'FIXED' if not baseline_pass and reranked_pass else 'CHECK'}"
                )

                print(f"Query: {query}")
                print(
                    f"Expected: {expected_source}"
                )

                print()
                print("BASELINE")

                for rank, result in enumerate(
                    retrieved,
                    start=1,
                ):
                    print(
                        f"  #{rank} "
                        f"{result.filename} "
                        f"dense={result.score:.4f}"
                    )

                print()
                print("RERANKED")

                for rank, item in enumerate(
                    reranked,
                    start=1,
                ):
                    print(
                        f"  #{rank} "
                        f"{item.result.filename} "
                        f"rerank={item.rerank_score:.4f} "
                        f"dense={item.result.score:.4f}"
                    )

    baseline_hit_1 = hit_rate_at_k(
        baseline_results,
        1,
    )

    baseline_hit_3 = hit_rate_at_k(
        baseline_results,
        3,
    )

    baseline_mrr = mean_reciprocal_rank(
        baseline_results
    )

    reranked_hit_1 = hit_rate_at_k(
        reranked_results,
        1,
    )

    reranked_hit_3 = hit_rate_at_k(
        reranked_results,
        3,
    )

    reranked_mrr = mean_reciprocal_rank(
        reranked_results
    )

    print()
    print("=" * 60)
    print("RERANKING EVALUATION")
    print("=" * 60)

    print()
    print("DENSE RETRIEVAL")
    print(
        f"Hit@1 : {baseline_hit_1:.3f}"
    )
    print(
        f"Hit@3 : {baseline_hit_3:.3f}"
    )
    print(
        f"MRR   : {baseline_mrr:.3f}"
    )

    print()
    print("CROSS-ENCODER RERANKING")
    print(
        f"Hit@1 : {reranked_hit_1:.3f}"
    )
    print(
        f"Hit@3 : {reranked_hit_3:.3f}"
    )
    print(
        f"MRR   : {reranked_mrr:.3f}"
    )

    print()
    print("IMPROVEMENT")
    print(
        f"Hit@1 delta : "
        f"{reranked_hit_1 - baseline_hit_1:+.3f}"
    )
    print(
        f"MRR delta   : "
        f"{reranked_mrr - baseline_mrr:+.3f}"
    )


if __name__ == "__main__":
    main()