from dataclasses import dataclass

from knowledgeops.ingestion.models import ParsedDocument


DEFAULT_CHUNK_SIZE_WORDS = 160
DEFAULT_CHUNK_OVERLAP_WORDS = 30


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    content: str
    page_number: int | None


def chunk_document(
    document: ParsedDocument,
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> list[TextChunk]:
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be greater than 0")

    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")

    if overlap_words >= chunk_size_words:
        raise ValueError(
            "overlap_words must be smaller than chunk_size_words"
        )

    chunks: list[TextChunk] = []
    chunk_index = 0

    for page in document.pages:
        words = page.text.split()

        if not words:
            continue

        start = 0

        while start < len(words):
            end = min(
                start + chunk_size_words,
                len(words),
            )

            content = " ".join(words[start:end])

            chunks.append(
                TextChunk(
                    chunk_index=chunk_index,
                    content=content,
                    page_number=page.page_number,
                )
            )

            chunk_index += 1

            if end == len(words):
                break

            start = end - overlap_words

    return chunks