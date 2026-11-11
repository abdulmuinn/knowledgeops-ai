from fastapi.testclient import TestClient

from knowledgeops.agent.orchestrator import (
    AgentRunResult,
    AgentRunStatus,
)
from knowledgeops.api.routes.agent import (
    get_agent_orchestrator,
)
from knowledgeops.main import app
from knowledgeops.rag.models import (
    RAGResponse,
)


client = TestClient(app)


class FakeOrchestrator:
    def handle(
        self,
        user_request: str,
    ) -> AgentRunResult:
        return AgentRunResult(
            status=(
                AgentRunStatus.RAG_COMPLETED
            ),
            rag_response=RAGResponse(
                status="answered",
                answer=(
                    "Emergency access expires "
                    "after four hours [S1]."
                ),
                sources=(),
                raw_output=(
                    "Emergency access expires "
                    "after four hours [S1]."
                ),
            ),
        )


def test_agent_run_returns_rag_response() -> None:
    app.dependency_overrides[
        get_agent_orchestrator
    ] = lambda: FakeOrchestrator()

    try:
        response = client.post(
            "/agent/run",
            json={
                "message": (
                    "How long does emergency "
                    "access last?"
                )
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert (
            data["status"]
            == "rag_completed"
        )

        assert data["answer"] == (
            "Emergency access expires "
            "after four hours [S1]."
        )

        assert data["sources"] == []

    finally:
        app.dependency_overrides.clear()


def test_agent_run_rejects_empty_message() -> None:
    app.dependency_overrides[
        get_agent_orchestrator
    ] = lambda: FakeOrchestrator()

    try:
        response = client.post(
            "/agent/run",
            json={
                "message": "",
            },
        )

        assert response.status_code == 422

    finally:
        app.dependency_overrides.clear()