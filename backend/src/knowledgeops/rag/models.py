import uuid
from dataclasses import dataclass

from typing import Literal


@dataclass(frozen=True)
class ContextSource:
    citation_id: str
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_number: int | None
    chunk_index: int
    content: str
    dense_score: float
    rerank_score: float


@dataclass(frozen=True)
class BuiltContext:
    text: str
    sources: tuple[ContextSource, ...]

@dataclass(frozen=True)
class RAGPrompt:
    system: str
    user: str

@dataclass(frozen=True)
class RAGResponse:
    status: Literal[
        "answered",
        "abstained",
    ]
    answer: str | None
    sources: tuple[ContextSource, ...]
    raw_output: str