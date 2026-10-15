import uuid

import pytest

from knowledgeops.evaluation.answer_quality import (
    build_quality_judge_prompt,
    judge_answer_quality,
    parse_quality_judgement,
    summarize_quality_judgements,
    AnswerQualityJudgement,
)
from knowledgeops.rag.models import (
    ContextSource,
    RAGResponse,
)


class FakeLLM:
    def __init__(
        self,
        response: str,
    ) -> None:
        self.response = response
        self.call_count = 0

    def generate(
        self,
        system: str,
        user: str,
    ) -> str:
        self.call_count += 1
        return self.response


def make_source() -> ContextSource:
    return ContextSource(
        citation_id="S1",
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename="security-policy.txt",
        page_number=1,
        chunk_index=0,
        content=(
            "Production access requires "
            "Security Team approval."
        ),
        dense_score=0.60,
        rerank_score=6.0,
    )


def make_response() -> RAGResponse:
    answer = (
        "Security Team approval "
        "is required [S1]."
    )

    return RAGResponse(
        status="answered",
        answer=answer,
        sources=(
            make_source(),
        ),
        raw_output=answer,
    )


def test_quality_prompt_contains_answer_and_evidence() -> None:
    source = make_source()

    prompt = build_quality_judge_prompt(
        question=(
            "Who approves production access?"
        ),
        answer=(
            "Security Team approves it [S1]."
        ),
        sources=(
            source,
        ),
    )

    assert (
        "Who approves production access?"
        in prompt.user
    )

    assert (
        "Security Team approves it [S1]."
        in prompt.user
    )

    assert (
        "Production access requires "
        "Security Team approval."
        in prompt.user
    )


def test_parse_quality_judgement() -> None:
    raw = """
    {
      "correctness": 2,
      "relevance": 2,
      "groundedness": 2,
      "citation_support": 2,
      "rationale": "Fully supported."
    }
    """

    result = parse_quality_judgement(
        raw
    )

    assert result.correctness == 2
    assert result.relevance == 2
    assert result.groundedness == 2
    assert result.citation_support == 2


def test_parse_fenced_json() -> None:
    raw = """
    ```json
    {
      "correctness": 2,
      "relevance": 1,
      "groundedness": 2,
      "citation_support": 2,
      "rationale": "Mostly relevant."
    }
    ```
    """

    result = parse_quality_judgement(
        raw
    )

    assert result.relevance == 1


def test_invalid_score_is_rejected() -> None:
    raw = """
    {
      "correctness": 3,
      "relevance": 2,
      "groundedness": 2,
      "citation_support": 2,
      "rationale": "Invalid score."
    }
    """

    with pytest.raises(
        ValueError,
        match="correctness",
    ):
        parse_quality_judgement(
            raw
        )


def test_judge_answer_quality_uses_llm() -> None:
    llm = FakeLLM(
        """
        {
          "correctness": 2,
          "relevance": 2,
          "groundedness": 2,
          "citation_support": 2,
          "rationale": "Fully supported."
        }
        """
    )

    result = judge_answer_quality(
        question=(
            "Who approves production access?"
        ),
        response=make_response(),
        llm=llm,
    )

    assert llm.call_count == 1
    assert result.correctness == 2


def test_quality_summary() -> None:
    judgements = [
        AnswerQualityJudgement(
            correctness=2,
            relevance=2,
            groundedness=2,
            citation_support=2,
            rationale="Perfect.",
        ),
        AnswerQualityJudgement(
            correctness=2,
            relevance=1,
            groundedness=2,
            citation_support=2,
            rationale="Some extra detail.",
        ),
    ]

    summary = (
        summarize_quality_judgements(
            judgements
        )
    )

    assert summary.total == 2

    assert (
        summary.correctness_mean
        == 2.0
    )

    assert (
        summary.relevance_mean
        == 1.5
    )

    assert (
        summary.overall_mean
        == 1.875
    )

    assert summary.perfect_cases == 1

    assert (
        summary.perfect_case_rate
        == 0.5
    )