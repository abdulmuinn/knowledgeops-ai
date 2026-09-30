import numpy as np
import pytest

from knowledgeops.embeddings.constants import (
    EMBEDDING_DIMENSION,
)
from knowledgeops.embeddings.service import (
    EmbeddingService,
)


class FakeEmbeddingModel:
    def encode(
        self,
        texts: list[str],
        **kwargs: object,
    ) -> np.ndarray:
        return np.ones(
            (
                len(texts),
                EMBEDDING_DIMENSION,
            ),
            dtype=np.float32,
        )


def test_embed_text_returns_expected_dimension() -> None:
    service = EmbeddingService(
        model=FakeEmbeddingModel(),
    )

    embedding = service.embed_text(
        "Production access requires approval."
    )

    assert len(embedding) == EMBEDDING_DIMENSION


def test_embed_multiple_texts() -> None:
    service = EmbeddingService(
        model=FakeEmbeddingModel(),
    )

    embeddings = service.embed_texts(
        [
            "Security policy",
            "Expense policy",
        ]
    )

    assert len(embeddings) == 2

    assert all(
        len(embedding) == EMBEDDING_DIMENSION
        for embedding in embeddings
    )


def test_empty_collection_returns_empty_list() -> None:
    service = EmbeddingService(
        model=FakeEmbeddingModel(),
    )

    assert service.embed_texts([]) == []


def test_blank_text_is_rejected() -> None:
    service = EmbeddingService(
        model=FakeEmbeddingModel(),
    )

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        service.embed_text("   ")