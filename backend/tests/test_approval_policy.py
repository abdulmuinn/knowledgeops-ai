import pytest

from knowledgeops.agent.models import (
    ApprovalStatus,
    ToolCall,
    ToolRiskLevel,
)
from knowledgeops.agent.policy import (
    ApprovalBindingError,
    ApprovalNotApprovedError,
    ApprovalPolicy,
    ApprovalRequiredError,
)
from knowledgeops.tools.base import (
    ToolDefinition,
)


def make_definition(
    name: str,
    risk_level: ToolRiskLevel,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="Test tool.",
        risk_level=risk_level,
        input_schema={
            "type": "object",
            "properties": {},
        },
    )


def test_read_only_tool_does_not_require_approval() -> None:
    policy = ApprovalPolicy()

    definition = make_definition(
        "lookup_policy",
        ToolRiskLevel.READ_ONLY,
    )

    assert (
        policy.requires_approval(
            definition
        )
        is False
    )


def test_state_changing_tool_requires_approval() -> None:
    policy = ApprovalPolicy()

    definition = make_definition(
        "create_access_request",
        ToolRiskLevel.STATE_CHANGING,
    )

    assert (
        policy.requires_approval(
            definition
        )
        is True
    )


def test_read_only_tool_creates_no_approval_request() -> None:
    policy = ApprovalPolicy()

    tool_call = ToolCall(
        tool_name="lookup_policy",
        arguments={
            "query": "remote work",
        },
    )

    definition = make_definition(
        "lookup_policy",
        ToolRiskLevel.READ_ONLY,
    )

    request = policy.create_request(
        tool_call,
        definition,
    )

    assert request is None


def test_state_changing_tool_creates_pending_request() -> None:
    policy = ApprovalPolicy()

    tool_call = ToolCall(
        tool_name="create_access_request",
        arguments={
            "role": "read-only",
        },
    )

    definition = make_definition(
        "create_access_request",
        ToolRiskLevel.STATE_CHANGING,
    )

    request = policy.create_request(
        tool_call,
        definition,
    )

    assert request is not None
    assert (
        request.status
        == ApprovalStatus.PENDING
    )
    assert (
        request.tool_call
        == tool_call
    )


def test_state_changing_tool_without_approval_is_blocked() -> None:
    policy = ApprovalPolicy()

    tool_call = ToolCall(
        tool_name="create_access_request",
        arguments={},
    )

    definition = make_definition(
        "create_access_request",
        ToolRiskLevel.STATE_CHANGING,
    )

    with pytest.raises(
        ApprovalRequiredError,
        match="approval",
    ):
        policy.ensure_execution_allowed(
            tool_call,
            definition,
        )


def test_pending_approval_cannot_execute() -> None:
    policy = ApprovalPolicy()

    tool_call = ToolCall(
        tool_name="create_access_request",
        arguments={},
    )

    definition = make_definition(
        "create_access_request",
        ToolRiskLevel.STATE_CHANGING,
    )

    request = policy.create_request(
        tool_call,
        definition,
    )

    assert request is not None

    with pytest.raises(
        ApprovalNotApprovedError,
        match="not approved",
    ):
        policy.ensure_execution_allowed(
            tool_call,
            definition,
            request,
        )


def test_approved_call_can_execute() -> None:
    policy = ApprovalPolicy()

    tool_call = ToolCall(
        tool_name="create_access_request",
        arguments={
            "role": "read-only",
        },
    )

    definition = make_definition(
        "create_access_request",
        ToolRiskLevel.STATE_CHANGING,
    )

    request = policy.create_request(
        tool_call,
        definition,
    )

    assert request is not None

    approved = policy.approve(
        request
    )

    policy.ensure_execution_allowed(
        tool_call,
        definition,
        approved,
    )

    assert (
        approved.status
        == ApprovalStatus.APPROVED
    )


def test_rejected_call_cannot_execute() -> None:
    policy = ApprovalPolicy()

    tool_call = ToolCall(
        tool_name="create_access_request",
        arguments={},
    )

    definition = make_definition(
        "create_access_request",
        ToolRiskLevel.STATE_CHANGING,
    )

    request = policy.create_request(
        tool_call,
        definition,
    )

    assert request is not None

    rejected = policy.reject(
        request
    )

    with pytest.raises(
        ApprovalNotApprovedError,
        match="not approved",
    ):
        policy.ensure_execution_allowed(
            tool_call,
            definition,
            rejected,
        )


def test_approval_cannot_be_reused_for_different_call() -> None:
    policy = ApprovalPolicy()

    first_call = ToolCall(
        tool_name="create_access_request",
        arguments={
            "role": "read-only",
        },
    )

    second_call = ToolCall(
        tool_name="create_access_request",
        arguments={
            "role": "admin",
        },
    )

    definition = make_definition(
        "create_access_request",
        ToolRiskLevel.STATE_CHANGING,
    )

    request = policy.create_request(
        first_call,
        definition,
    )

    assert request is not None

    approved = policy.approve(
        request
    )

    with pytest.raises(
        ApprovalBindingError,
        match="does not match",
    ):
        policy.ensure_execution_allowed(
            second_call,
            definition,
            approved,
        )


def test_tool_call_must_match_definition() -> None:
    policy = ApprovalPolicy()

    tool_call = ToolCall(
        tool_name="create_access_request",
        arguments={},
    )

    definition = make_definition(
        "lookup_policy",
        ToolRiskLevel.READ_ONLY,
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        policy.ensure_execution_allowed(
            tool_call,
            definition,
        )