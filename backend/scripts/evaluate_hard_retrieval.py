import json
from pathlib import Path
from statistics import mean

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
from knowledgeops.retrieval.service import (
    semantic_search,
)
from knowledgeops.evaluation.abstention import (
    find_best_threshold,
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


def summarize_scores(
    scores: list[float],
) -> tuple[float, float, float]:
    if not scores:
        return 0.0, 0.0, 0.0

    return (
        mean(scores),
        min(scores),
        max(scores),
    )


def main() -> None:
    cases = load_cases()

    embedder = EmbeddingService()

    answerable_results: list[
        RetrievalEvaluationResult
    ] = []

    answerable_top_scores: list[float] = []
    unanswerable_top_scores: list[float] = []

    answerable_count = 0
    unanswerable_count = 0

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
                    Document.checksum
                    == checksum
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
            query = str(case["query"])

            answerable = bool(
                case["answerable"]
            )

            expected_source = (
                case.get("expected_source")
            )

            results = semantic_search(
                session=session,
                query=query,
                embedder=embedder,
                top_k=3,
                document_ids=document_ids,
            )

            top_score = (
                results[0].score
                if results
                else 0.0
            )

            print()
            print(
                f"[{index:02d}] "
                f"{'ANSWERABLE' if answerable else 'UNANSWERABLE'}"
            )
            print(f"Query: {query}")

            if answerable:
                answerable_count += 1

                expected_source_str = str(
                    expected_source
                )

                retrieved_sources = [
                    result.filename
                    for result in results
                ]

                answerable_results.append(
                    RetrievalEvaluationResult(
                        expected_source=(
                            expected_source_str
                        ),
                        retrieved_sources=(
                            retrieved_sources
                        ),
                    )
                )

                answerable_top_scores.append(
                    top_score
                )

                top_source = (
                    retrieved_sources[0]
                    if retrieved_sources
                    else "NO RESULT"
                )

                status = (
                    "PASS"
                    if top_source
                    == expected_source_str
                    else "MISS"
                )

                print(
                    f"Status: {status}"
                )
                print(
                    "Expected: "
                    f"{expected_source_str}"
                )

            else:
                unanswerable_count += 1

                unanswerable_top_scores.append(
                    top_score
                )

                print(
                    "Expected: NO ANSWER"
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
        answerable_results,
        1,
    )

    hit_3 = hit_rate_at_k(
        answerable_results,
        3,
    )

    mrr = mean_reciprocal_rank(
        answerable_results
    )

    (
        answerable_mean,
        answerable_min,
        answerable_max,
    ) = summarize_scores(
        answerable_top_scores
    )

    (
        unanswerable_mean,
        unanswerable_min,
        unanswerable_max,
    ) = summarize_scores(
        unanswerable_top_scores
    )

    print()
    print("=" * 60)
    print("HARD RETRIEVAL EVALUATION")
    print("=" * 60)

    print()
    print("ANSWERABLE")
    print(
        f"Queries          : "
        f"{answerable_count}"
    )
    print(
        f"Hit@1            : "
        f"{hit_1:.3f}"
    )
    print(
        f"Hit@3            : "
        f"{hit_3:.3f}"
    )
    print(
        f"MRR              : "
        f"{mrr:.3f}"
    )
    print(
        f"Top-1 score mean : "
        f"{answerable_mean:.4f}"
    )
    print(
        f"Top-1 score min  : "
        f"{answerable_min:.4f}"
    )
    print(
        f"Top-1 score max  : "
        f"{answerable_max:.4f}"
    )

    print()
    print("UNANSWERABLE")
    print(
        f"Queries          : "
        f"{unanswerable_count}"
    )
    print(
        f"Top-1 score mean : "
        f"{unanswerable_mean:.4f}"
    )
    print(
        f"Top-1 score min  : "
        f"{unanswerable_min:.4f}"
    )
    print(
        f"Top-1 score max  : "
        f"{unanswerable_max:.4f}"
    )

    print()
    print("SCORE SEPARATION")
    print(
        "Lowest answerable: "
        f"{answerable_min:.4f}"
    )
    print(
        "Highest unanswerable: "
        f"{unanswerable_max:.4f}"
    )

    gap = (
        answerable_min
        - unanswerable_max
    )

    print(
        f"Separation gap   : {gap:.4f}"
    )
    best_threshold = find_best_threshold(
        answerable_scores=answerable_top_scores,
        unanswerable_scores=unanswerable_top_scores,
    )
    
    print()
    print("ABSTENTION THRESHOLD")
    print(
        f"Best threshold   : "
        f"{best_threshold.threshold:.4f}"
    )
    print(
        f"Precision        : "
        f"{best_threshold.precision:.3f}"
    )
    print(
        f"Recall           : "
        f"{best_threshold.recall:.3f}"
    )
    print(
        f"F1               : "
        f"{best_threshold.f1:.3f}"
    )
    print(
        f"Accuracy         : "
        f"{best_threshold.accuracy:.3f}"
    )
    print(
        f"True positive    : "
        f"{best_threshold.true_positive}"
    )
    print(
        f"False positive   : "
        f"{best_threshold.false_positive}"
    )
    print(
        f"True negative    : "
        f"{best_threshold.true_negative}"
    )
    print(
        f"False negative   : "
        f"{best_threshold.false_negative}"
    )


if __name__ == "__main__":
    main()