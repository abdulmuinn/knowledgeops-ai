import math

from knowledgeops.embeddings.constants import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
)
from knowledgeops.embeddings.service import EmbeddingService


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    dot_product = sum(
        a * b
        for a, b in zip(
            vector_a,
            vector_b,
            strict=True,
        )
    )

    magnitude_a = math.sqrt(
        sum(value * value for value in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(value * value for value in vector_b)
    )

    return dot_product / (
        magnitude_a * magnitude_b
    )


def main() -> None:
    service = EmbeddingService()

    texts = [
        (
            "Production database access requires "
            "Security Team approval."
        ),
        (
            "Approval is required before accessing "
            "the production database."
        ),
        (
            "Employees can submit travel expense claims."
        ),
    ]

    embeddings = service.embed_texts(texts)

    similar_score = cosine_similarity(
        embeddings[0],
        embeddings[1],
    )

    unrelated_score = cosine_similarity(
        embeddings[0],
        embeddings[2],
    )

    print(f"Model: {EMBEDDING_MODEL_NAME}")
    print(f"Dimension: {len(embeddings[0])}")
    print(f"Expected dimension: {EMBEDDING_DIMENSION}")
    print()
    print(
        f"Similar sentences: {similar_score:.4f}"
    )
    print(
        f"Different topics: {unrelated_score:.4f}"
    )

    assert len(embeddings[0]) == EMBEDDING_DIMENSION
    assert similar_score > unrelated_score

    print()
    print("Embedding smoke test PASSED")


if __name__ == "__main__":
    main()