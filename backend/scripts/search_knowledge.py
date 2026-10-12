import argparse

from knowledgeops.db.session import SessionLocal
from knowledgeops.embeddings.service import EmbeddingService
from knowledgeops.retrieval.service import semantic_search


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "query",
        type=str,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    embedder = EmbeddingService()

    with SessionLocal() as session:
        results = semantic_search(
            session=session,
            query=args.query,
            embedder=embedder,
            top_k=args.top_k,
        )

    if not results:
        print("No results found")
        return

    for index, result in enumerate(
        results,
        start=1,
    ):
        print()
        print(f"Result #{index}")
        print(f"Score: {result.score:.4f}")
        print(f"Source: {result.filename}")
        print(f"Page: {result.page_number}")
        print(f"Chunk: {result.chunk_index}")
        print(f"Content: {result.content}")


if __name__ == "__main__":
    main()