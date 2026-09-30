import pytest

from knowledgeops.ingestion.chunking import chunk_document
from knowledgeops.ingestion.models import (
    ParsedDocument,
    ParsedPage,
)


def test_short_document_creates_single_chunk() -> None:
    document = ParsedDocument(
        filename="policy.txt",
        mime_type="text/plain",
        pages=[
            ParsedPage(
                page_number=None,
                text="Production access requires approval.",
            )
        ],
    )

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].content == (
        "Production access requires approval."
    )


def test_long_document_creates_multiple_chunks() -> None:
    text = " ".join(
        f"word{i}"
        for i in range(300)
    )

    document = ParsedDocument(
        filename="policy.txt",
        mime_type="text/plain",
        pages=[
            ParsedPage(
                page_number=1,
                text=text,
            )
        ],
    )

    chunks = chunk_document(
        document,
        chunk_size_words=100,
        overlap_words=20,
    )

    assert len(chunks) == 4


def test_chunk_overlap_is_preserved() -> None:
    words = [
        f"word{i}"
        for i in range(200)
    ]

    document = ParsedDocument(
        filename="policy.txt",
        mime_type="text/plain",
        pages=[
            ParsedPage(
                page_number=1,
                text=" ".join(words),
            )
        ],
    )

    chunks = chunk_document(
        document,
        chunk_size_words=100,
        overlap_words=20,
    )

    first_chunk_words = chunks[0].content.split()
    second_chunk_words = chunks[1].content.split()

    assert first_chunk_words[-20:] == second_chunk_words[:20]


def test_page_number_is_preserved() -> None:
    document = ParsedDocument(
        filename="policy.pdf",
        mime_type="application/pdf",
        pages=[
            ParsedPage(
                page_number=7,
                text="Security approval is required.",
            )
        ],
    )

    chunks = chunk_document(document)

    assert chunks[0].page_number == 7


def test_empty_page_is_ignored() -> None:
    document = ParsedDocument(
        filename="policy.pdf",
        mime_type="application/pdf",
        pages=[
            ParsedPage(
                page_number=1,
                text="",
            ),
            ParsedPage(
                page_number=2,
                text="Valid content.",
            ),
        ],
    )

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].page_number == 2


def test_invalid_chunk_configuration() -> None:
    document = ParsedDocument(
        filename="test.txt",
        mime_type="text/plain",
        pages=[],
    )

    with pytest.raises(ValueError):
        chunk_document(
            document,
            chunk_size_words=100,
            overlap_words=100,
        )