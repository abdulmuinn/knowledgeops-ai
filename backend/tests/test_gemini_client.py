from typing import Any

from knowledgeops.llm.gemini_client import (
    GeminiClient,
)


class FakeResponse:
    text = (
        "Security Team approval is required [S1]."
    )


class FakeModels:
    def __init__(self) -> None:
        self.model_received: str | None = None
        self.contents_received: str | None = None
        self.config_received: Any | None = None

    def generate_content(
        self,
        model: str,
        contents: str,
        config: Any,
    ) -> FakeResponse:
        self.model_received = model
        self.contents_received = contents
        self.config_received = config

        return FakeResponse()


class FakeGemini:
    def __init__(self) -> None:
        self.models = FakeModels()


def test_gemini_client_generates_text() -> None:
    fake_client = FakeGemini()

    client = GeminiClient(
        model="test-model",
        client=fake_client,
    )

    output = client.generate(
        system="Use only evidence.",
        user="Who approves access?",
    )

    assert output == (
        "Security Team approval is required [S1]."
    )


def test_gemini_client_passes_request_fields() -> None:
    fake_client = FakeGemini()

    client = GeminiClient(
        model="test-model",
        client=fake_client,
        temperature=0.1,
    )

    client.generate(
        system="System instructions",
        user="User question",
    )

    assert (
        fake_client.models.model_received
        == "test-model"
    )

    assert (
        fake_client.models.contents_received
        == "User question"
    )

    assert (
        fake_client.models.config_received
        is not None
    )


def test_gemini_client_handles_empty_text() -> None:
    class EmptyResponse:
        text = None

    class EmptyModels:
        def generate_content(
            self,
            model: str,
            contents: str,
            config: Any,
        ) -> EmptyResponse:
            return EmptyResponse()

    class EmptyGemini:
        models = EmptyModels()

    client = GeminiClient(
        client=EmptyGemini(),
    )

    output = client.generate(
        system="System",
        user="Question",
    )

    assert output == ""