import re

from knowledgeops.rag.models import (
    BuiltContext,
    ContextSource,
)


CITATION_PATTERN = re.compile(
    r"\[(S\d+)\]"
)


def extract_citation_ids(
    text: str,
) -> tuple[str, ...]:
    matches = CITATION_PATTERN.findall(text)

    return tuple(
        dict.fromkeys(matches)
    )


def resolve_citations(
    text: str,
    context: BuiltContext,
) -> tuple[ContextSource, ...]:
    citation_ids = extract_citation_ids(
        text
    )

    if not citation_ids:
        raise RuntimeError(
            "Answer does not contain citations"
        )

    source_by_id = {
        source.citation_id: source
        for source in context.sources
    }

    invalid_ids = [
        citation_id
        for citation_id in citation_ids
        if citation_id not in source_by_id
    ]

    if invalid_ids:
        raise RuntimeError(
            "Answer contains invalid citation IDs: "
            + ", ".join(invalid_ids)
        )

    return tuple(
        source_by_id[citation_id]
        for citation_id in citation_ids
    )