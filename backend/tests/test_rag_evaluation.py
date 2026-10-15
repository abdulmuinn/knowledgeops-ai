import uuid

import pytest

from knowledgeops.evaluation.rag import (
    RAGEvaluationCase,
    evaluate_rag_responses,
)
from knowledgeops.rag.models import (
    ContextSource,
    RAGResponse,
)


def make_source(
    filename: str,
) -> ContextSource:
    return ContextSource(
        citation_id="S1",
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename=filename,
        page_number=1,
        chunk_index=0,
        content="Evidence",
        dense_score=0.6,
        rerank_score=5.0,
    )


def test_perfect_rag_evaluation() -> None:
    cases = [
        RAGEvaluationCase(
            question="Who approves access?",
            expected_status="answered",
            expected_source="security.txt",
        ),
        RAGEvaluationCase(
            question="Parking reimbursement?",
            expected_status="abstained",
        ),
    ]

    responses = [
        RAGResponse(
            status="answered",
            answer=(
                "Security approves access [S1]."
            ),
            sources=(
                make_source(
                    "security.txt"
                ),
            ),
            raw_output=(
                "Security approves access [S1]."
            ),
        ),
        RAGResponse(
            status="abstained",
            answer=None,
            sources=(),
            raw_output=(
                "INSUFFICIENT_EVIDENCE"
            ),
        ),
    ]

    result = evaluate_rag_responses(
        cases,
        responses,
    )

    assert result.total == 2
    assert result.decision_accuracy == 1.0
    assert result.answerable_accuracy == 1.0
    assert result.abstention_accuracy == 1.0
    assert result.citation_presence_rate == 1.0
    assert result.source_accuracy == 1.0


def test_wrong_abstention_is_counted() -> None:
    cases = [
        RAGEvaluationCase(
            question="Who approves access?",
            expected_status="answered",
            expected_source="security.txt",
        ),
    ]

    responses = [
        RAGResponse(
            status="abstained",
            answer=None,
            sources=(),
            raw_output=(
                "INSUFFICIENT_EVIDENCE"
            ),
        ),
    ]

    result = evaluate_rag_responses(
        cases,
        responses,
    )

    assert result.decision_accuracy == 0.0
    assert result.answerable_accuracy == 0.0
    assert result.citation_presence_rate == 0.0
    assert result.source_accuracy == 0.0


def test_wrong_source_is_counted() -> None:
    cases = [
        RAGEvaluationCase(
            question="Who approves access?",
            expected_status="answered",
            expected_source="security.txt",
        ),
    ]

    responses = [
        RAGResponse(
            status="answered",
            answer="Manager approves [S1].",
            sources=(
                make_source(
                    "leave-policy.txt"
                ),
            ),
            raw_output=(
                "Manager approves [S1]."
            ),
        ),
    ]

    result = evaluate_rag_responses(
        cases,
        responses,
    )

    assert result.decision_accuracy == 1.0
    assert result.citation_presence_rate == 1.0
    assert result.source_accuracy == 0.0


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="same length",
    ):
        evaluate_rag_responses(
            cases=[
                RAGEvaluationCase(
                    question="Question",
                    expected_status="abstained",
                )
            ],
            responses=[],
        )


def test_empty_evaluation_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        evaluate_rag_responses(
            cases=[],
            responses=[],
        )