import json
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Literal, cast

from knowledgeops.evaluation.answer_quality import (
    AnswerQualityJudgement,
    judge_answer_quality,
    summarize_quality_judgements,
)
from knowledgeops.llm.gemini_client import GeminiClient
from knowledgeops.rag.models import ContextSource, RAGResponse


RAG_RESULTS_PATH = Path("evaluation/rag_results.jsonl")
QUALITY_RESULTS_PATH = Path(
    "evaluation/answer_quality_results.jsonl"
)
QUALITY_SUMMARY_PATH = Path(
    "evaluation/answer_quality_summary.json"
)

DELAY_SECONDS = float(
    os.getenv("ANSWER_QUALITY_DELAY_SECONDS", "2.0")
)

MAX_ATTEMPTS = int(
    os.getenv("ANSWER_QUALITY_MAX_ATTEMPTS", "4")
)


def load_rag_records() -> list[dict[str, object]]:
    if not RAG_RESULTS_PATH.exists():
        raise RuntimeError(
            "RAG results not found: "
            + str(RAG_RESULTS_PATH)
        )

    records: list[dict[str, object]] = []

    with RAG_RESULTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            if not isinstance(record, dict):
                raise ValueError(
                    "Each RAG result must be a JSON object."
                )

            records.append(record)

    records.sort(
        key=lambda item: int(
            item["case_index"]
        )
    )

    return records


def source_from_dict(
    data: dict[str, object],
) -> ContextSource:
    return ContextSource(
        citation_id=str(data["citation_id"]),
        chunk_id=uuid.UUID(
            str(data["chunk_id"])
        ),
        document_id=uuid.UUID(
            str(data["document_id"])
        ),
        filename=str(data["filename"]),
        page_number=(
            int(data["page_number"])
            if data["page_number"] is not None
            else None
        ),
        chunk_index=int(
            data["chunk_index"]
        ),
        content=str(data["content"]),
        dense_score=float(
            data["dense_score"]
        ),
        rerank_score=float(
            data["rerank_score"]
        ),
    )


def response_from_dict(
    data: dict[str, object],
) -> RAGResponse:
    sources_data = data.get("sources", [])

    if not isinstance(
        sources_data,
        list,
    ):
        raise ValueError(
            "response sources must be a list"
        )

    status_value = str(data["status"])

    if status_value not in {
        "answered",
        "abstained",
    }:
        raise ValueError(
            "invalid RAG response status"
        )

    status = cast(
        Literal[
            "answered",
            "abstained",
        ],
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
            if isinstance(
                source,
                dict,
            )
        ),
        raw_output=str(
            data["raw_output"]
        ),
    )


def judgement_to_dict(
    judgement: AnswerQualityJudgement,
) -> dict[str, object]:
    return {
        "correctness": (
            judgement.correctness
        ),
        "relevance": (
            judgement.relevance
        ),
        "groundedness": (
            judgement.groundedness
        ),
        "citation_support": (
            judgement.citation_support
        ),
        "rationale": (
            judgement.rationale
        ),
    }


def judgement_from_dict(
    data: dict[str, object],
) -> AnswerQualityJudgement:
    return AnswerQualityJudgement(
        correctness=int(
            data["correctness"]
        ),
        relevance=int(
            data["relevance"]
        ),
        groundedness=int(
            data["groundedness"]
        ),
        citation_support=int(
            data["citation_support"]
        ),
        rationale=str(
            data["rationale"]
        ),
    )


def load_checkpoint(
    judge_model: str,
) -> dict[int, dict[str, object]]:
    if not QUALITY_RESULTS_PATH.exists():
        return {}

    completed: dict[
        int,
        dict[str, object],
    ] = {}

    with QUALITY_RESULTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            if not isinstance(
                record,
                dict,
            ):
                raise ValueError(
                    "Quality checkpoint must contain JSON objects."
                )

            record_model = str(
                record["judge_model"]
            )

            if record_model != judge_model:
                raise RuntimeError(
                    "Existing quality checkpoint uses judge model "
                    + record_model
                    + ", but current judge model is "
                    + judge_model
                    + ". Rename or delete "
                    + str(QUALITY_RESULTS_PATH)
                    + " before changing judge models."
                )

            case_index = int(
                record["case_index"]
            )

            completed[
                case_index
            ] = record

    return completed


def save_checkpoint(
    record: dict[str, object],
) -> None:
    QUALITY_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with QUALITY_RESULTS_PATH.open(
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

    markers = (
        "429",
        "503",
        "RESOURCE_EXHAUSTED",
        "UNAVAILABLE",
    )

    return any(
        marker in message
        for marker in markers
    )


def judge_with_retry(
    *,
    question: str,
    response: RAGResponse,
    llm: GeminiClient,
) -> AnswerQualityJudgement:
    for attempt in range(
        1,
        MAX_ATTEMPTS + 1,
    ):
        try:
            return judge_answer_quality(
                question=question,
                response=response,
                llm=llm,
            )

        except Exception as exc:
            if (
                not is_retryable_error(exc)
                or attempt == MAX_ATTEMPTS
            ):
                raise

            wait_seconds = (
                5
                * (
                    2
                    ** (
                        attempt - 1
                    )
                )
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

            time.sleep(
                wait_seconds
            )

    raise RuntimeError(
        "retry loop exited unexpectedly"
    )


def print_judgement(
    *,
    case_index: int,
    question: str,
    judgement: AnswerQualityJudgement,
    resumed: bool,
) -> None:
    resume_text = (
        " [RESUMED]"
        if resumed
        else ""
    )

    print()
    print("=" * 70)

    print(
        "["
        + f"{case_index:02d}"
        + "] ANSWER QUALITY"
        + resume_text
    )

    print("=" * 70)
    print(
        "Question         : "
        + question
    )
    print(
        "Correctness      : "
        + str(judgement.correctness)
        + "/2"
    )
    print(
        "Relevance        : "
        + str(judgement.relevance)
        + "/2"
    )
    print(
        "Groundedness     : "
        + str(judgement.groundedness)
        + "/2"
    )
    print(
        "Citation support : "
        + str(judgement.citation_support)
        + "/2"
    )
    print(
        "Rationale        : "
        + judgement.rationale
    )


def main() -> None:
    records = load_rag_records()

    generator_models = {
        str(record["model"])
        for record in records
    }

    if len(generator_models) != 1:
        raise RuntimeError(
            "RAG results contain multiple generator models."
        )

    generator_model = next(
        iter(generator_models)
    )

    judge_model = os.getenv(
        "GEMINI_JUDGE_MODEL",
        os.getenv(
            "GEMINI_MODEL",
            "gemini-3.5-flash-lite",
        ),
    )

    answerable_records = [
        record
        for record in records
        if bool(record["answerable"])
    ]

    if not answerable_records:
        raise RuntimeError(
            "No answerable RAG records found."
        )

    print(
        "Generator model : "
        + generator_model
    )
    print(
        "Judge model     : "
        + judge_model
    )
    print(
        "Answerable cases: "
        + str(len(answerable_records))
    )
    print(
        "Checkpoint      : "
        + str(QUALITY_RESULTS_PATH)
    )

    llm = GeminiClient(
        model=judge_model,
        temperature=0.0,
    )

    completed = load_checkpoint(
        judge_model=judge_model
    )

    judgements: list[
        AnswerQualityJudgement
    ] = []

    for record in answerable_records:
        case_index = int(
            record["case_index"]
        )

        question = str(
            record["query"]
        )

        response_data = record[
            "response"
        ]

        if not isinstance(
            response_data,
            dict,
        ):
            raise RuntimeError(
                "Invalid response format for case "
                + str(case_index)
                + "."
            )

        response = response_from_dict(
            response_data
        )

        if response.status != "answered":
            raise RuntimeError(
                "Answerable case "
                + str(case_index)
                + " was not answered. "
                + "Run deterministic RAG evaluation first."
            )

        if case_index in completed:
            checkpoint = completed[
                case_index
            ]

            if (
                str(checkpoint["query"])
                != question
            ):
                raise RuntimeError(
                    "Quality checkpoint query does not match case "
                    + str(case_index)
                    + "."
                )

            judgement_data = checkpoint[
                "judgement"
            ]

            if not isinstance(
                judgement_data,
                dict,
            ):
                raise RuntimeError(
                    "Invalid quality checkpoint format."
                )

            judgement = (
                judgement_from_dict(
                    judgement_data
                )
            )

            resumed = True

        else:
            judgement = judge_with_retry(
                question=question,
                response=response,
                llm=llm,
            )

            checkpoint = {
                "case_index": (
                    case_index
                ),
                "generator_model": (
                    generator_model
                ),
                "judge_model": (
                    judge_model
                ),
                "query": question,
                "answer": (
                    response.answer
                ),
                "sources": [
                    source.filename
                    for source
                    in response.sources
                ],
                "judgement": (
                    judgement_to_dict(
                        judgement
                    )
                ),
            }

            save_checkpoint(
                checkpoint
            )

            resumed = False

        judgements.append(
            judgement
        )

        print_judgement(
            case_index=case_index,
            question=question,
            judgement=judgement,
            resumed=resumed,
        )

        if not resumed:
            time.sleep(
                DELAY_SECONDS
            )

    summary = (
        summarize_quality_judgements(
            judgements
        )
    )

    summary_data = {
        "generator_model": (
            generator_model
        ),
        "judge_model": (
            judge_model
        ),
        "rubric_scale": "0-2",
        "note": (
            "LLM-as-a-judge evaluation. "
            + "Scores are not objective ground-truth metrics."
        ),
        **asdict(summary),
    }

    QUALITY_SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with QUALITY_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary_data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print(
        "KNOWLEDGEOPS AI - "
        + "ANSWER QUALITY EVALUATION"
    )
    print("=" * 70)

    print(
        "Cases evaluated       : "
        + str(summary.total)
    )
    print(
        "Correctness mean      : "
        + f"{summary.correctness_mean:.3f}"
        + " / 2"
    )
    print(
        "Relevance mean        : "
        + f"{summary.relevance_mean:.3f}"
        + " / 2"
    )
    print(
        "Groundedness mean     : "
        + f"{summary.groundedness_mean:.3f}"
        + " / 2"
    )
    print(
        "Citation support mean : "
        + f"{summary.citation_support_mean:.3f}"
        + " / 2"
    )
    print(
        "Overall mean          : "
        + f"{summary.overall_mean:.3f}"
        + " / 2"
    )
    print(
        "Perfect cases         : "
        + str(summary.perfect_cases)
        + "/"
        + str(summary.total)
    )
    print(
        "Perfect case rate     : "
        + f"{summary.perfect_case_rate:.3f}"
    )

    print()
    print(
        "Summary saved to: "
        + str(QUALITY_SUMMARY_PATH)
    )
    print(
        "Case results saved to: "
        + str(QUALITY_RESULTS_PATH)
    )


if __name__ == "__main__":
    main()
