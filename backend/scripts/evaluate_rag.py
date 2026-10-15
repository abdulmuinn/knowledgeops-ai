import json
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import cast, Literal

from sqlalchemy import select

from knowledgeops.db.models import Document
from knowledgeops.db.session import SessionLocal
from knowledgeops.embeddings.service import EmbeddingService
from knowledgeops.evaluation.rag import (
    RAGEvaluationCase,
    evaluate_rag_responses,
)
from knowledgeops.ingestion.checksum import calculate_sha256
from knowledgeops.llm.gemini_client import GeminiClient
from knowledgeops.rag.models import ContextSource, RAGResponse
from knowledgeops.rag.pipeline import answer_question
from knowledgeops.reranking.cross_encoder import CrossEncoderReranker
from knowledgeops.retrieval.client import SemanticRetriever


CASES_PATH = Path("evaluation/hard_retrieval_cases.json")
CORPUS_DIR = Path("evaluation/hard_knowledge_base")
RESULTS_PATH = Path("evaluation/rag_results.jsonl")
SUMMARY_PATH = Path("evaluation/rag_summary.json")

TOP_K = 3
RERANK_TOP_N = 3
MAX_SOURCES = 3
MIN_DENSE_SCORE = None

DELAY_SECONDS = float(
    os.getenv("RAG_EVAL_DELAY_SECONDS", "2.0")
)

MAX_ATTEMPTS = int(
    os.getenv("RAG_EVAL_MAX_ATTEMPTS", "4")
)


def load_cases() -> list[dict[str, object]]:
    with CASES_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_document_ids(session) -> list[uuid.UUID]:
    document_ids: list[uuid.UUID] = []

    for file_path in sorted(CORPUS_DIR.glob("*.txt")):
        checksum = calculate_sha256(file_path)

        document = session.scalar(
            select(Document).where(
                Document.checksum == checksum
            )
        )

        if document is None:
            raise RuntimeError(
                "Hard evaluation document not ingested: "
                + file_path.name
            )

        document_ids.append(document.id)

    return document_ids


def source_to_dict(
    source: ContextSource,
) -> dict[str, object]:
    return {
        "citation_id": source.citation_id,
        "chunk_id": str(source.chunk_id),
        "document_id": str(source.document_id),
        "filename": source.filename,
        "page_number": source.page_number,
        "chunk_index": source.chunk_index,
        "content": source.content,
        "dense_score": source.dense_score,
        "rerank_score": source.rerank_score,
    }


def source_from_dict(
    data: dict[str, object],
) -> ContextSource:
    return ContextSource(
        citation_id=str(data["citation_id"]),
        chunk_id=uuid.UUID(str(data["chunk_id"])),
        document_id=uuid.UUID(str(data["document_id"])),
        filename=str(data["filename"]),
        page_number=(
            int(data["page_number"])
            if data["page_number"] is not None
            else None
        ),
        chunk_index=int(data["chunk_index"]),
        content=str(data["content"]),
        dense_score=float(data["dense_score"]),
        rerank_score=float(data["rerank_score"]),
    )


def response_to_dict(
    response: RAGResponse,
) -> dict[str, object]:
    return {
        "status": response.status,
        "answer": response.answer,
        "raw_output": response.raw_output,
        "sources": [
            source_to_dict(source)
            for source in response.sources
        ],
    }


def response_from_dict(
    data: dict[str, object],
) -> RAGResponse:
    sources_data = data.get("sources", [])

    if not isinstance(sources_data, list):
        raise ValueError(
            "checkpoint sources must be a list"
        )

    status_value = str(data["status"])

    if status_value not in {
        "answered",
        "abstained",
    }:
        raise ValueError(
            "invalid checkpoint response status"
        )

    status = cast(
        Literal["answered", "abstained"],
        status_value,
    )

    return RAGResponse(
        status=status,
        answer=(
            str(data["answer"])
            if data["answer"] is not None
            else None
        ),
        sources=tuple(
            source_from_dict(source)
            for source in sources_data
            if isinstance(source, dict)
        ),
        raw_output=str(data["raw_output"]),
    )


def load_checkpoint(
    model: str,
) -> dict[int, dict[str, object]]:
    if not RESULTS_PATH.exists():
        return {}

    completed: dict[int, dict[str, object]] = {}

    with RESULTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)
            record_model = str(record["model"])

            if record_model != model:
                raise RuntimeError(
                    "Existing checkpoint uses model "
                    + record_model
                    + ", but current model is "
                    + model
                    + ". Rename or delete "
                    + "evaluation/rag_results.jsonl "
                    + "before changing models."
                )

            index = int(record["case_index"])
            completed[index] = record

    return completed


def save_checkpoint(
    record: dict[str, object],
) -> None:
    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_PATH.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
        )
        file.write("\n")


def is_retryable_error(
    exc: Exception,
) -> bool:
    message = (
        type(exc).__name__
        + ": "
        + str(exc)
    ).upper()

    retryable_markers = (
        "429",
        "503",
        "RESOURCE_EXHAUSTED",
        "UNAVAILABLE",
    )

    return any(
        marker in message
        for marker in retryable_markers
    )


def run_with_retry(
    *,
    question: str,
    retriever: SemanticRetriever,
    reranker: CrossEncoderReranker,
    llm: GeminiClient,
    document_ids: list[uuid.UUID],
) -> RAGResponse:
    for attempt in range(
        1,
        MAX_ATTEMPTS + 1,
    ):
        try:
            return answer_question(
                question=question,
                retriever=retriever,
                reranker=reranker,
                llm=llm,
                top_k=TOP_K,
                rerank_top_n=RERANK_TOP_N,
                max_sources=MAX_SOURCES,
                min_dense_score=MIN_DENSE_SCORE,
                document_ids=document_ids,
            )

        except Exception as exc:
            if (
                not is_retryable_error(exc)
                or attempt == MAX_ATTEMPTS
            ):
                raise

            wait_seconds = 5 * (
                2 ** (attempt - 1)
            )

            print(
                "Temporary provider error. "
                + "Retry "
                + str(attempt)
                + "/"
                + str(MAX_ATTEMPTS)
                + " in "
                + str(wait_seconds)
                + "s..."
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        "retry loop exited unexpectedly"
    )


def build_evaluation_case(
    case: dict[str, object],
) -> RAGEvaluationCase:
    answerable = bool(case["answerable"])

    expected_source_raw = case[
        "expected_source"
    ]

    expected_source = (
        str(expected_source_raw)
        if expected_source_raw is not None
        else None
    )

    return RAGEvaluationCase(
        question=str(case["query"]),
        expected_status=(
            "answered"
            if answerable
            else "abstained"
        ),
        expected_source=expected_source,
    )


def print_case_result(
    *,
    index: int,
    total: int,
    case: dict[str, object],
    response: RAGResponse,
    resumed: bool,
) -> None:
    answerable = bool(case["answerable"])

    expected_status = (
        "answered"
        if answerable
        else "abstained"
    )

    expected_source = (
        str(case["expected_source"])
        if case["expected_source"] is not None
        else None
    )

    actual_sources = [
        source.filename
        for source in response.sources
    ]

    actual_sources_text = (
        ", ".join(actual_sources)
        if actual_sources
        else "NONE"
    )

    expected_source_text = (
        expected_source
        if expected_source is not None
        else "NONE"
    )

    decision_ok = (
        response.status == expected_status
    )

    source_ok = (
        expected_source is None
        or expected_source in actual_sources
    )

    passed = decision_ok and source_ok

    label = (
        "ANSWERABLE"
        if answerable
        else "UNANSWERABLE"
    )

    resume_label = (
        " [RESUMED]"
        if resumed
        else ""
    )

    result_text = (
        "PASS"
        if passed
        else "FAIL"
    )

    print()
    print("=" * 70)
    print(
        "["
        + f"{index:02d}"
        + "/"
        + f"{total:02d}"
        + "] "
        + label
        + resume_label
    )
    print("=" * 70)

    print(
        "Question : "
        + str(case["query"])
    )
    print(
        "Expected : "
        + expected_status
    )
    print(
        "Actual   : "
        + response.status
    )
    print(
        "Expected source: "
        + expected_source_text
    )
    print(
        "Actual sources  : "
        + actual_sources_text
    )
    print(
        "Result   : "
        + result_text
    )

    if response.answer is not None:
        print(
            "Answer   : "
            + response.answer
        )
    else:
        print(
            "Answer   : <abstained>"
        )


def main() -> None:
    cases = load_cases()

    model = os.getenv(
        "GEMINI_MODEL",
        "gemini-3.5-flash-lite",
    )

    print(
        "Gemini model : "
        + model
    )
    print(
        "Cases        : "
        + str(len(cases))
    )
    print(
        "Checkpoint   : "
        + str(RESULTS_PATH)
    )

    embedder = EmbeddingService()
    reranker = CrossEncoderReranker()

    llm = GeminiClient(
        model=model,
        temperature=0.1,
    )

    completed = load_checkpoint(
        model=model
    )

    responses: list[RAGResponse] = []

    evaluation_cases: list[
        RAGEvaluationCase
    ] = []

    with SessionLocal() as session:
        document_ids = load_document_ids(
            session
        )

        retriever = SemanticRetriever(
            session=session,
            embedder=embedder,
        )

        print(
            "Knowledge documents: "
            + str(len(document_ids))
        )

        for index, case in enumerate(
            cases,
            start=1,
        ):
            question = str(case["query"])

            evaluation_cases.append(
                build_evaluation_case(case)
            )

            if index in completed:
                record = completed[index]

                if (
                    str(record["query"])
                    != question
                ):
                    raise RuntimeError(
                        "Checkpoint query does not "
                        + "match case "
                        + str(index)
                        + "."
                    )

                response_data = record[
                    "response"
                ]

                if not isinstance(
                    response_data,
                    dict,
                ):
                    raise RuntimeError(
                        "Invalid checkpoint "
                        + "response format."
                    )

                response = (
                    response_from_dict(
                        response_data
                    )
                )

                resumed = True

            else:
                response = run_with_retry(
                    question=question,
                    retriever=retriever,
                    reranker=reranker,
                    llm=llm,
                    document_ids=document_ids,
                )

                record = {
                    "case_index": index,
                    "model": model,
                    "query": question,
                    "answerable": bool(
                        case["answerable"]
                    ),
                    "expected_source": case[
                        "expected_source"
                    ],
                    "response": (
                        response_to_dict(
                            response
                        )
                    ),
                }

                save_checkpoint(record)
                resumed = False

            responses.append(response)

            print_case_result(
                index=index,
                total=len(cases),
                case=case,
                response=response,
                resumed=resumed,
            )

            if not resumed:
                time.sleep(
                    DELAY_SECONDS
                )

    result = evaluate_rag_responses(
        evaluation_cases,
        responses,
    )

    print()
    print("=" * 70)
    print(
        "KNOWLEDGEOPS AI - "
        + "RAG EVALUATION"
    )
    print("=" * 70)

    print(
        "Total cases          : "
        + str(result.total)
    )
    print(
        "Correct decisions    : "
        + str(result.correct_decisions)
    )
    print(
        "Decision accuracy    : "
        + f"{result.decision_accuracy:.3f}"
    )

    print()

    print(
        "Answerable cases     : "
        + str(result.answerable_total)
    )
    print(
        "Answerable correct   : "
        + str(result.answerable_correct)
    )
    print(
        "Answerable accuracy  : "
        + f"{result.answerable_accuracy:.3f}"
    )

    print()

    print(
        "Unanswerable cases   : "
        + str(result.unanswerable_total)
    )
    print(
        "Correct abstentions  : "
        + str(result.abstention_correct)
    )
    print(
        "Abstention accuracy  : "
        + f"{result.abstention_accuracy:.3f}"
    )

    print()

    print(
        "Citation presence    : "
        + f"{result.citation_presence_rate:.3f}"
    )
    print(
        "Source accuracy      : "
        + f"{result.source_accuracy:.3f}"
    )

    summary = {
        "model": model,
        "top_k": TOP_K,
        "rerank_top_n": RERANK_TOP_N,
        "max_sources": MAX_SOURCES,
        "min_dense_score": (
            MIN_DENSE_SCORE
        ),
        **asdict(result),
    }

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "Summary saved to: "
        + str(SUMMARY_PATH)
    )
    print(
        "Case results saved to: "
        + str(RESULTS_PATH)
    )


if __name__ == "__main__":
    main()
