import uuid

from knowledgeops.agent.approval_store import (
    ApprovalNotFoundError,
)
from knowledgeops.agent.models import (
    ToolCall,
    ToolExecutionStatus,
    ToolResult,
)
from knowledgeops.agent.policy import (
    ApprovalNotApprovedError,
)

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

def test_agent_resume_returns_tool_result() -> None:
    approval_id = uuid.uuid4()

    app.dependency_overrides[
        get_agent_orchestrator
    ] = lambda: FakeResumeOrchestrator()

    try:
        response = client.post(
            f"/agent/resume/{approval_id}"
        )

        assert response.status_code == 200

        data = response.json()

        assert (
            data["status"]
            == "tool_completed"
        )

        assert (
            data["tool_name"]
            == "create_access_request"
        )

        assert (
            data["tool_result"]["status"]
            == "succeeded"
        )

        assert (
            data["tool_result"]["output"]
            ["request_id"]
            == "AR-TEST1234"
        )

    finally:
        app.dependency_overrides.clear()


def test_agent_resume_unknown_approval_returns_404() -> None:
    approval_id = uuid.uuid4()

    app.dependency_overrides[
        get_agent_orchestrator
    ] = lambda: FakeResumeOrchestrator(
        mode="not_found"
    )

    try:
        response = client.post(
            f"/agent/resume/{approval_id}"
        )

        assert response.status_code == 404

        assert response.json() == {
            "detail": "approval not found"
        }

    finally:
        app.dependency_overrides.clear()


def test_agent_resume_pending_approval_returns_409() -> None:
    approval_id = uuid.uuid4()

    app.dependency_overrides[
        get_agent_orchestrator
    ] = lambda: FakeResumeOrchestrator(
        mode="not_approved"
    )

    try:
        response = client.post(
            f"/agent/resume/{approval_id}"
        )

        assert response.status_code == 409

        assert response.json() == {
            "detail": (
                "approval is not approved"
            )
        }

    finally:
        app.dependency_overrides.clear()

class FakeResumeOrchestrator:
    def __init__(
        self,
        *,
        mode: str = "success",
    ) -> None:
        self.mode = mode

    def resume(
        self,
        approval_id: uuid.UUID,
    ) -> AgentRunResult:
        if self.mode == "not_found":
            raise ApprovalNotFoundError(
                f"approval not found: {approval_id}"
            )

        if self.mode == "not_approved":
            raise ApprovalNotApprovedError(
                "approval request is not approved"
            )

        tool_call = ToolCall(
            tool_name="create_access_request",
            arguments={
                "system": "production",
                "role": "read-only",
            },
        )

        return AgentRunResult(
            status=(
                AgentRunStatus.TOOL_COMPLETED
            ),
            tool_call=tool_call,
            tool_result=ToolResult(
                tool_call_id=tool_call.call_id,
                status=(
                    ToolExecutionStatus.SUCCEEDED
                ),
                output={
                    "request_id": "AR-TEST1234",
                    "status": "submitted",
                },
            ),
        )