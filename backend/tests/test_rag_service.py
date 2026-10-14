import uuid

import pytest

from knowledgeops.rag.prompt import (
    ABSTENTION_TOKEN,
)
from knowledgeops.rag.service import (
    generate_grounded_answer,
)
from knowledgeops.reranking.models import (
    RerankedResult,
)
from knowledgeops.retrieval.models import (
    SearchResult,
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


def make_result(
    filename: str = "policy.txt",
    content: str = (
        "Production access requires "
        "Security Team approval."
    ),
) -> RerankedResult:
    result = SearchResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename=filename,
        content=content,
        page_number=1,
        chunk_index=0,
        score=0.55,
    )

    return RerankedResult(
        result=result,
        rerank_score=6.0,
    )


def test_grounded_answer_with_valid_citation() -> None:
    llm = FakeLLM(
        "Security Team approval is required [S1]."
    )

    response = generate_grounded_answer(
        question=(
            "Who approves production access?"
        ),
        results=[make_result()],
        llm=llm,
    )

    assert response.status == "answered"

    assert response.answer == (
        "Security Team approval is required [S1]."
    )

    assert len(response.sources) == 1

    assert response.sources[0].citation_id == (
        "S1"
    )


def test_model_can_abstain() -> None:
    llm = FakeLLM(
        ABSTENTION_TOKEN
    )

    response = generate_grounded_answer(
        question="What is the dental benefit?",
        results=[make_result()],
        llm=llm,
    )

    assert response.status == "abstained"
    assert response.answer is None
    assert response.sources == ()


def test_no_evidence_abstains_without_llm() -> None:
    llm = FakeLLM(
        "This should never be returned."
    )

    response = generate_grounded_answer(
        question="Unknown question",
        results=[],
        llm=llm,
    )

    assert response.status == "abstained"
    assert response.answer is None
    assert llm.call_count == 0


def test_answer_without_citation_is_rejected() -> None:
    llm = FakeLLM(
        "Security Team approval is required."
    )

    with pytest.raises(
        RuntimeError,
        match="does not contain citations",
    ):
        generate_grounded_answer(
            question=(
                "Who approves production access?"
            ),
            results=[make_result()],
            llm=llm,
        )


def test_invalid_citation_is_rejected() -> None:
    llm = FakeLLM(
        "Security Team approval is required [S99]."
    )

    with pytest.raises(
        RuntimeError,
        match="invalid citation",
    ):
        generate_grounded_answer(
            question=(
                "Who approves production access?"
            ),
            results=[make_result()],
            llm=llm,
        )


def test_duplicate_citations_are_deduplicated() -> None:
    llm = FakeLLM(
        "Approval is required [S1]. "
        "Security Team approves it [S1]."
    )

    response = generate_grounded_answer(
        question=(
            "Who approves production access?"
        ),
        results=[make_result()],
        llm=llm,
    )

    assert response.status == "answered"
    assert len(response.sources) == 1