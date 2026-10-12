from collections.abc import Sequence
from typing import Any


DEFAULT_RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
    ) -> None:
        self.model_name = model_name
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import (
                CrossEncoder,
            )

            self._model = CrossEncoder(
                self.model_name
            )

        return self._model

    def score(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[float]:
        if not documents:
            return []

        pairs = [
            [query, document]
            for document in documents
        ]

        scores = self.model.predict(
            pairs
        )

        return [
            float(score)
            for score in scores
        ]