import uuid

import pytest

from knowledgeops.agent.approval_service import (
    HumanApprovalService,
)
from knowledgeops.agent.approval_store import (
    ApprovalNotFoundError,
    ApprovalStore,
)
from knowledgeops.agent.models import (
    ApprovalRequest,
    ApprovalStatus,
    ToolCall,
)
from knowledgeops.agent.policy import (
    ApprovalPolicy,
)


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


def test_submit_stores_pending_request() -> None:
    service = make_service()
    request = make_request()

    submitted = service.submit(
        request
    )

    assert submitted == request
    assert (
        service.get(
            request.approval_id
        )
        == request
    )


def test_non_pending_request_cannot_be_submitted() -> None:
    service = make_service()

    request = make_request()

    already_approved = ApprovalRequest(
        tool_call=request.tool_call,
        reason=request.reason,
        status=ApprovalStatus.APPROVED,
        approval_id=request.approval_id,
    )

    with pytest.raises(
        ValueError,
        match="only pending",
    ):
        service.submit(
            already_approved
        )


def test_approve_updates_stored_request() -> None:
    service = make_service()
    request = make_request()

    service.submit(
        request
    )

    approved = service.approve(
        request.approval_id
    )

    assert (
        approved.status
        == ApprovalStatus.APPROVED
    )
    assert (
        approved.approval_id
        == request.approval_id
    )
    assert (
        service.get(
            request.approval_id
        )
        == approved
    )


def test_reject_updates_stored_request() -> None:
    service = make_service()
    request = make_request()

    service.submit(
        request
    )

    rejected = service.reject(
        request.approval_id
    )

    assert (
        rejected.status
        == ApprovalStatus.REJECTED
    )
    assert (
        service.get(
            request.approval_id
        )
        == rejected
    )


def test_request_cannot_be_decided_twice() -> None:
    service = make_service()
    request = make_request()

    service.submit(
        request
    )

    service.approve(
        request.approval_id
    )

    with pytest.raises(
        ValueError,
        match="only pending",
    ):
        service.reject(
            request.approval_id
        )


def test_unknown_approval_id_is_rejected() -> None:
    service = make_service()

    with pytest.raises(
        ApprovalNotFoundError,
        match="approval not found",
    ):
        service.approve(
            uuid.uuid4()
        )