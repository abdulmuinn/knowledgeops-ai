from functools import lru_cache

from knowledgeops.agent.approval_service import (
    HumanApprovalService,
)
from knowledgeops.agent.approval_store import (
    ApprovalStore,
)
from knowledgeops.agent.orchestrator import (
    AgentOrchestrator,
)
from knowledgeops.agent.planner import (
    AgentPlanner,
)
from knowledgeops.agent.policy import (
    ApprovalPolicy,
)
from knowledgeops.llm.gemini_client import (
    GeminiClient,
)
from knowledgeops.rag.models import (
    RAGResponse,
)
from knowledgeops.tools.access_request import (
    CreateAccessRequestTool,
)
from knowledgeops.tools.executor import (
    ToolExecutor,
)
from knowledgeops.tools.registry import (
    ToolRegistry,
)


@lru_cache
def _get_embedding_service() -> object:
    from knowledgeops.embeddings.service import (
        EmbeddingService,
    )

    return EmbeddingService()


@lru_cache
def _get_reranker() -> object:
    from knowledgeops.reranking.cross_encoder import (
        CrossEncoderReranker,
    )

    return CrossEncoderReranker()


def _answer_with_rag(
    question: str,
) -> RAGResponse:
    from knowledgeops.db.session import (
        SessionLocal,
    )
    from knowledgeops.rag.pipeline import (
        answer_question,
    )
    from knowledgeops.retrieval.client import (
        SemanticRetriever,
    )

    embedder = _get_embedding_service()
    reranker = _get_reranker()

    with SessionLocal() as session:
        retriever = SemanticRetriever(
            session=session,
            embedder=embedder,  # type: ignore[arg-type]
        )

        return answer_question(
            question=question,
            retriever=retriever,
            reranker=reranker,  # type: ignore[arg-type]
            llm=GeminiClient(),
            top_k=3,
            rerank_top_n=3,
            max_sources=3,
        )


@lru_cache
def get_agent_orchestrator() -> AgentOrchestrator:
    registry = ToolRegistry(
        [
            CreateAccessRequestTool(),
        ]
    )

    policy = ApprovalPolicy()

    approval_service = HumanApprovalService(
        store=ApprovalStore(),
        policy=policy,
    )

    executor = ToolExecutor(
        registry=registry,
        approval_service=approval_service,
    )

    planner = AgentPlanner(
        llm=GeminiClient(),
        registry=registry,
    )

    return AgentOrchestrator(
        planner=planner,
        executor=executor,
        rag_answerer=_answer_with_rag,
    )