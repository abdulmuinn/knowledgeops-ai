import pytest

from knowledgeops.retrieval.service import semantic_search


class FakeEmbedder:
    def embed_text(
        self,
        text: str,
    ) -> list[float]:
        return [1.0] * 384


class WrongDimensionEmbedder:
    def embed_text(
        self,
        text: str,
    ) -> list[float]:
        return [1.0] * 10


def test_empty_query_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="empty",
    ):
        semantic_search(
            session=None,  # type: ignore[arg-type]
            query="   ",
            embedder=FakeEmbedder(),
        )


def test_invalid_top_k_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="top_k",
    ):
        semantic_search(
            session=None,  # type: ignore[arg-type]
            query="security policy",
            embedder=FakeEmbedder(),
            top_k=0,
        )


def test_invalid_embedding_dimension_is_rejected() -> None:
    with pytest.raises(
        RuntimeError,
        match="dimension",
    ):
        semantic_search(
            session=None,  # type: ignore[arg-type]
            query="security policy",
            embedder=WrongDimensionEmbedder(),
        )
        
def test_empty_document_filter_returns_no_results() -> None:
    results = semantic_search(
        session=None,  # type: ignore[arg-type]
        query="security policy",
        embedder=FakeEmbedder(),
        document_ids=[],
    )

    assert results == []