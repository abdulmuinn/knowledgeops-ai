import uuid

import pytest

from knowledgeops.rag.models import (
    BuiltContext,
    ContextSource,
)
from knowledgeops.rag.prompt import (
    ABSTENTION_TOKEN,
    SYSTEM_PROMPT,
    build_rag_prompt,
)


def make_context() -> BuiltContext:
    source = ContextSource(
        citation_id="S1",
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename="security-policy.txt",
        page_number=2,
        chunk_index=0,
        content=(
            "Production access requires "
            "Security Team approval."
        ),
        dense_score=0.55,
        rerank_score=6.2,
    )

    return BuiltContext(
        text=(
            "[S1]\n"
            "Source: security-policy.txt\n"
            "Page: 2\n"
            "Chunk: 0\n"
            "Content:\n"
            "Production access requires "
            "Security Team approval."
        ),
        sources=(source,),
    )


def test_prompt_contains_question_and_evidence() -> None:
    context = make_context()

    prompt = build_rag_prompt(
        question=(
            "Who approves production access?"
        ),
        context=context,
    )

    assert (
        "Who approves production access?"
        in prompt.user
    )

    assert (
        "Production access requires "
        "Security Team approval."
        in prompt.user
    )


def test_prompt_contains_allowed_citations() -> None:
    context = make_context()

    prompt = build_rag_prompt(
        question="Who approves access?",
        context=context,
    )

    assert "ALLOWED CITATIONS" in prompt.user
    assert "[S1]" in prompt.user


def test_prompt_requires_grounded_abstention() -> None:
    context = make_context()

    prompt = build_rag_prompt(
        question="Unknown policy question",
        context=context,
    )

    assert ABSTENTION_TOKEN in prompt.system
    assert ABSTENTION_TOKEN in prompt.user

    assert (
        "only the evidence"
        in SYSTEM_PROMPT.lower()
    )


def test_empty_question_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="question",
    ):
        build_rag_prompt(
            question="   ",
            context=make_context(),
        )


def test_empty_context_is_rejected() -> None:
    empty_context = BuiltContext(
        text="",
        sources=(),
    )

    with pytest.raises(
        ValueError,
        match="context",
    ):
        build_rag_prompt(
            question="What is the policy?",
            context=empty_context,
        )