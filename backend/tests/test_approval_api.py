import uuid

from fastapi.testclient import TestClient

from knowledgeops.agent.approval_service import (
    HumanApprovalService,
)
from knowledgeops.agent.approval_store import (
    ApprovalStore,
)
from knowledgeops.agent.models import (
    ApprovalRequest,
    ToolCall,
)
from knowledgeops.agent.policy import (
    ApprovalPolicy,
)
from knowledgeops.api.routes.approvals import (
    get_approval_service,
)
from knowledgeops.main import app


client = TestClient(app)


def make_service() -> HumanApprovalService:
    return HumanApprovalService(
        store=ApprovalStore(),
        policy=ApprovalPolicy(),
    )


def make_request() -> ApprovalRequest:
    return ApprovalRequest(
        tool_call=ToolCall(
            tool_name="create_access_request",
            arguments={
                "system": "production",
                "role": "read-only",
            },
        ),
        reason="Human approval required.",
    )


def test_list_pending_approvals() -> None:
    service = make_service()
    request = make_request()

    service.submit(request)

    app.dependency_overrides[
        get_approval_service
    ] = lambda: service

    try:
        response = client.get(
            "/approvals"
        )

        assert response.status_code == 200

        data = response.json()

        assert len(data) == 1

        assert (
            data[0]["approval_id"]
            == str(request.approval_id)
        )

        assert data[0]["status"] == "pending"

    finally:
        app.dependency_overrides.clear()


def test_approve_request() -> None:
    service = make_service()
    request = make_request()

    service.submit(request)

    app.dependency_overrides[
        get_approval_service
    ] = lambda: service

    try:
        response = client.post(
            f"/approvals/"
            f"{request.approval_id}/approve"
        )

        assert response.status_code == 200

        assert (
            response.json()["status"]
            == "approved"
        )

    finally:
        app.dependency_overrides.clear()


def test_reject_request() -> None:
    service = make_service()
    request = make_request()

    service.submit(request)

    app.dependency_overrides[
        get_approval_service
    ] = lambda: service

    try:
        response = client.post(
            f"/approvals/"
            f"{request.approval_id}/reject"
        )

        assert response.status_code == 200

        assert (
            response.json()["status"]
            == "rejected"
        )

    finally:
        app.dependency_overrides.clear()


def test_unknown_approval_returns_404() -> None:
    service = make_service()

    app.dependency_overrides[
        get_approval_service
    ] = lambda: service

    try:
        response = client.post(
            f"/approvals/"
            f"{uuid.uuid4()}/approve"
        )

        assert response.status_code == 404

        assert response.json() == {
            "detail": "approval not found"
        }

    finally:
        app.dependency_overrides.clear()


def test_approval_cannot_be_decided_twice() -> None:
    service = make_service()
    request = make_request()

    service.submit(request)

    service.approve(
        request.approval_id
    )

    app.dependency_overrides[
        get_approval_service
    ] = lambda: service

    try:
        response = client.post(
            f"/approvals/"
            f"{request.approval_id}/reject"
        )

        assert response.status_code == 409

    finally:
        app.dependency_overrides.clear()