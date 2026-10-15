from collections.abc import Sequence

from knowledgeops.llm.base import (
    LLMClient,
)
from knowledgeops.llm.service import (
    generate_text,
)
from knowledgeops.rag.citations import (
    resolve_citations,
)
from knowledgeops.rag.context import (
    build_context,
)
from knowledgeops.rag.models import (
    RAGResponse,
)
from knowledgeops.rag.prompt import (
    ABSTENTION_TOKEN,
    build_rag_prompt,
)
from knowledgeops.reranking.models import (
    RerankedResult,
)


def generate_grounded_answer(
    question: str,
    results: Sequence[RerankedResult],
    llm: LLMClient,
    max_sources: int = 3,
) -> RAGResponse:
    question = question.strip()

    if not question:
        raise ValueError(
            "question cannot be empty"
        )

    if not results:
        return RAGResponse(
            status="abstained",
            answer=None,
            sources=(),
            raw_output=ABSTENTION_TOKEN,
        )

    context = build_context(
        results=results,
        max_sources=max_sources,
    )

    if not context.sources:
        return RAGResponse(
            status="abstained",
            answer=None,
            sources=(),
            raw_output=ABSTENTION_TOKEN,
        )

    prompt = build_rag_prompt(
        question=question,
        context=context,
    )

    output = generate_text(
        client=llm,
        prompt=prompt,
    )

    if output == ABSTENTION_TOKEN:
        return RAGResponse(
            status="abstained",
            answer=None,
            sources=(),
            raw_output=output,
        )

    cited_sources = resolve_citations(
        text=output,
        context=context,
    )

    return RAGResponse(
        status="answered",
        answer=output,
        sources=cited_sources,
        raw_output=output,
    )