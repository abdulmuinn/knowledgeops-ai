from collections.abc import Sequence

from knowledgeops.rag.models import (
    BuiltContext,
    ContextSource,
)
from knowledgeops.reranking.models import (
    RerankedResult,
)


def build_context(
    results: Sequence[RerankedResult],
    max_sources: int = 3,
) -> BuiltContext:
    if max_sources <= 0:
        raise ValueError(
            "max_sources must be greater than 0"
        )

    sources: list[ContextSource] = []
    blocks: list[str] = []

    for item in results[:max_sources]:
        result = item.result
        content = result.content.strip()

        if not content:
            continue

        citation_id = (
            f"S{len(sources) + 1}"
        )

        source = ContextSource(
            citation_id=citation_id,
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            filename=result.filename,
            page_number=result.page_number,
            chunk_index=result.chunk_index,
            content=content,
            dense_score=result.score,
            rerank_score=item.rerank_score,
        )

        sources.append(source)

        page = (
            str(result.page_number)
            if result.page_number is not None
            else "N/A"
        )

        blocks.append(
            "\n".join(
                [
                    f"[{citation_id}]",
                    f"Source: {result.filename}",
                    f"Page: {page}",
                    (
                        "Chunk: "
                        f"{result.chunk_index}"
                    ),
                    "Content:",
                    content,
                ]
            )
        )

    return BuiltContext(
        text="\n\n".join(blocks),
        sources=tuple(sources),
    )