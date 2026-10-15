import pytest

from knowledgeops.llm.service import (
    generate_text,
)
from knowledgeops.rag.models import (
    RAGPrompt,
)


class FakeLLM:
    def __init__(
        self,
        response: str,
    ) -> None:
        self.response = response
        self.system_received: str | None = None
        self.user_received: str | None = None

    def generate(
        self,
        system: str,
        user: str,
    ) -> str:
        self.system_received = system
        self.user_received = user

        return self.response


def test_generate_text_returns_model_response() -> None:
    client = FakeLLM(
        "Security Team approval is required [S1]."
    )

    prompt = RAGPrompt(
        system="Use only evidence.",
        user="Who approves access?",
    )

    response = generate_text(
        client=client,
        prompt=prompt,
    )

    assert response == (
        "Security Team approval is required [S1]."
    )


def test_generate_text_passes_prompts_to_client() -> None:
    client = FakeLLM(
        "Answer [S1]."
    )

    prompt = RAGPrompt(
        system="System instructions",
        user="User question",
    )

    generate_text(
        client=client,
        prompt=prompt,
    )

    assert client.system_received == (
        "System instructions"
    )

    assert client.user_received == (
        "User question"
    )


def test_empty_model_response_is_rejected() -> None:
    client = FakeLLM("   ")

    prompt = RAGPrompt(
        system="System instructions",
        user="User question",
    )

    with pytest.raises(
        RuntimeError,
        match="empty response",
    ):
        generate_text(
            client=client,
            prompt=prompt,
        )


def test_empty_system_prompt_is_rejected() -> None:
    client = FakeLLM("Answer")

    prompt = RAGPrompt(
        system="   ",
        user="Question",
    )

    with pytest.raises(
        ValueError,
        match="system",
    ):
        generate_text(
            client=client,
            prompt=prompt,
        )


def test_empty_user_prompt_is_rejected() -> None:
    client = FakeLLM("Answer")

    prompt = RAGPrompt(
        system="System",
        user="   ",
    )

    with pytest.raises(
        ValueError,
        match="user",
    ):
        generate_text(
            client=client,
            prompt=prompt,
        )