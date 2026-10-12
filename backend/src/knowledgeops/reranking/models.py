from dataclasses import dataclass

from knowledgeops.retrieval.models import SearchResult


@dataclass(frozen=True)
class RerankedResult:
    result: SearchResult
    rerank_score: float