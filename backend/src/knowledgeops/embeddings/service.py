from collections.abc import Sequence
from typing import Any, Protocol

from knowledgeops.embeddings.constants import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
)


class EmbeddingModel(Protocol):
    def encode(
        self,
        sentences: Sequence[str],
        **kwargs: Any,
    ) -> Any:
        ...


class EmbeddingService:
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        model: EmbeddingModel | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = model

    @property
    def model(self) -> EmbeddingModel:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device="cpu",
            )

        return self._model

    def embed_text(self, text: str) -> list[float]:
        embeddings = self.embed_texts([text])
        return embeddings[0]

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        cleaned_texts = [
            text.strip()
            for text in texts
        ]

        if any(not text for text in cleaned_texts):
            raise ValueError(
                "texts cannot contain empty values"
            )

        embeddings = self.model.encode(
            cleaned_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        if embeddings.ndim != 2:
            raise RuntimeError(
                "embedding model returned an invalid shape"
            )

        if embeddings.shape[1] != EMBEDDING_DIMENSION:
            raise RuntimeError(
                "unexpected embedding dimension: "
                f"{embeddings.shape[1]}"
            )

        return embeddings.tolist()