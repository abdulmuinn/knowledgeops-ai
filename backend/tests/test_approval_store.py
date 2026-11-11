import uuid

import pytest

from knowledgeops.agent.approval_store import (
    ApprovalNotFoundError,
    ApprovalStore,
)
from knowledgeops.agent.models import (
    ApprovalRequest,
    ApprovalStatus,
    ToolCall,
)


def make_request() -> ApprovalRequest:
    return ApprovalRequest(
        tool_call=ToolCall(
            tool_name="create_access_request",
            arguments={
                "system": "production",
            },
        ),
        reason="Human approval required.",
    )


def test_save_and_get_approval() -> None:
    store = ApprovalStore()
    request = make_request()

    store.save(request)

    loaded = store.get(
        request.approval_id
    )

    assert loaded == request
    assert len(store) == 1


def test_unknown_approval_is_rejected() -> None:
    store = ApprovalStore()

    unknown_id = uuid.uuid4()

    with pytest.raises(
        ApprovalNotFoundError,
        match="approval not found",
    ):
        store.get(
            unknown_id
        )


def test_pending_returns_only_pending_requests() -> None:
    store = ApprovalStore()

    pending_request = make_request()

    approved_request = ApprovalRequest(
        tool_call=ToolCall(
            tool_name="create_access_request",
            arguments={
                "system": "staging",
            },
        ),
        reason="Human approval required.",
        status=ApprovalStatus.APPROVED,
    )

    store.save(
        pending_request
    )
    store.save(
        approved_request
    )

    pending = store.pending()

    assert pending == (
        pending_request,
    )


def test_save_replaces_same_approval_id() -> None:
    store = ApprovalStore()

    original = make_request()

    updated = ApprovalRequest(
        tool_call=original.tool_call,
        reason=original.reason,
        status=ApprovalStatus.APPROVED,
        approval_id=original.approval_id,
    )

    store.save(
        original
    )
    store.save(
        updated
    )

    loaded = store.get(
        original.approval_id
    )

    assert (
        loaded.status
        == ApprovalStatus.APPROVED
    )
    assert len(store) == 1