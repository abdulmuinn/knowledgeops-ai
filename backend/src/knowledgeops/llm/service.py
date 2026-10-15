from knowledgeops.llm.base import (
    LLMClient,
)
from knowledgeops.rag.models import (
    RAGPrompt,
)


def generate_text(
    client: LLMClient,
    prompt: RAGPrompt,
) -> str:
    if not prompt.system.strip():
        raise ValueError(
            "system prompt cannot be empty"
        )

    if not prompt.user.strip():
        raise ValueError(
            "user prompt cannot be empty"
        )

    response = client.generate(
        system=prompt.system,
        user=prompt.user,
    )

    response = response.strip()

    if not response:
        raise RuntimeError(
            "LLM returned an empty response"
        )

    return response