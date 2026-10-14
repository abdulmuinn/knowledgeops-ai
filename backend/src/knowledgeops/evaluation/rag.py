from dataclasses import dataclass
from typing import Literal

from knowledgeops.rag.models import (
    RAGResponse,
)


ExpectedStatus = Literal[
    "answered",
    "abstained",
]


@dataclass(frozen=True)
class RAGEvaluationCase:
    question: str
    expected_status: ExpectedStatus
    expected_source: str | None = None


@dataclass(frozen=True)
class RAGEvaluationResult:
    total: int
    correct_decisions: int
    decision_accuracy: float

    answerable_total: int
    answerable_correct: int
    answerable_accuracy: float

    unanswerable_total: int
    abstention_correct: int
    abstention_accuracy: float

    citation_required: int
    citation_present: int
    citation_presence_rate: float

    source_required: int
    source_correct: int
    source_accuracy: float

def evaluate_rag_responses(
    cases: list[RAGEvaluationCase],
    responses: list[RAGResponse],
) -> RAGEvaluationResult:
    if len(cases) != len(responses):
        raise ValueError(
            "cases and responses must have "
            "the same length"
        )

    if not cases:
        raise ValueError(
            "evaluation cases cannot be empty"
        )

    correct_decisions = 0

    answerable_total = 0
    answerable_correct = 0

    unanswerable_total = 0
    abstention_correct = 0

    citation_required = 0
    citation_present = 0

    source_required = 0
    source_correct = 0

    for case, response in zip(
        cases,
        responses,
        strict=True,
    ):
        if (
            response.status
            == case.expected_status
        ):
            correct_decisions += 1

        if (
            case.expected_status
            == "answered"
        ):
            answerable_total += 1
            citation_required += 1

            if response.status == "answered":
                answerable_correct += 1

            if response.sources:
                citation_present += 1

            if case.expected_source is not None:
                source_required += 1

                filenames = {
                    source.filename
                    for source
                    in response.sources
                }

                if (
                    case.expected_source
                    in filenames
                ):
                    source_correct += 1

        else:
            unanswerable_total += 1

            if response.status == "abstained":
                abstention_correct += 1

    total = len(cases)

    return RAGEvaluationResult(
        total=total,
        correct_decisions=correct_decisions,
        decision_accuracy=(
            correct_decisions / total
        ),
        answerable_total=answerable_total,
        answerable_correct=answerable_correct,
        answerable_accuracy=(
            answerable_correct
            / answerable_total
            if answerable_total
            else 0.0
        ),
        unanswerable_total=unanswerable_total,
        abstention_correct=abstention_correct,
        abstention_accuracy=(
            abstention_correct
            / unanswerable_total
            if unanswerable_total
            else 0.0
        ),
        citation_required=citation_required,
        citation_present=citation_present,
        citation_presence_rate=(
            citation_present
            / citation_required
            if citation_required
            else 0.0
        ),
        source_required=source_required,
        source_correct=source_correct,
        source_accuracy=(
            source_correct
            / source_required
            if source_required
            else 0.0
        ),
    )