import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    content: str
    page_number: int | None
    chunk_index: int
    score: float