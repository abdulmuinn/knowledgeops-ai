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
    ToolExecutionStatus,
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
from knowledgeops.tools.executor import (
    ToolExecutor,
)
from knowledgeops.tools.registry import (
    ToolRegistry,
)


class FakeTool:
    def __init__(
        self,
        *,
        name: str,
        risk_level: ToolRiskLevel,
        fail_execution: bool = False,
    ) -> None:
        self.execute_calls = 0
        self.fail_execution = fail_execution

        self._definition = ToolDefinition(
            name=name,
            description="Fake test tool.",
            risk_level=risk_level,
            input_schema={
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                    },
                },
                "required": [
                    "value",
                ],
            },
        )

    @property
    def definition(
        self,
    ) -> ToolDefinition:
        return self._definition

    def validate_arguments(
        self,
        arguments: dict[str, object],
    ) -> None:
        value = arguments.get("value")

        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise ValueError(
                "value must be a non-empty string"
            )

    def execute(
        self,
        arguments: dict[str, object],
    ) -> object:
        self.execute_calls += 1

        if self.fail_execution:
            raise RuntimeError(
                "simulated execution failure"
            )

        return {
            "received": arguments["value"],
        }


def make_executor(
    tool: FakeTool,
) -> tuple[
    ToolExecutor,
    HumanApprovalService,
]:
    policy = ApprovalPolicy()

    approval_service = HumanApprovalService(
        store=ApprovalStore(),
        policy=policy,
    )

    executor = ToolExecutor(
        registry=ToolRegistry(
            [
                tool,
            ]
        ),
        approval_service=approval_service,
    )

    return (
        executor,
        approval_service,
    )


def test_read_only_tool_executes_without_approval() -> None:
    tool = FakeTool(
        name="lookup_policy",
        risk_level=ToolRiskLevel.READ_ONLY,
    )

    executor, _ = make_executor(
        tool
    )

    tool_call = ToolCall(
        tool_name="lookup_policy",
        arguments={
            "value": "remote work",
        },
    )

    result = executor.execute(
        tool_call
    )

    assert (
        result.status
        == ToolExecutionStatus.SUCCEEDED
    )
    assert result.output == {
        "received": "remote work",
    }
    assert tool.execute_calls == 1


def test_prepare_read_only_returns_no_approval() -> None:
    tool = FakeTool(
        name="lookup_policy",
        risk_level=ToolRiskLevel.READ_ONLY,
    )

    executor, _ = make_executor(
        tool
    )

    tool_call = ToolCall(
        tool_name="lookup_policy",
        arguments={
            "value": "security",
        },
    )

    approval = executor.prepare(
        tool_call
    )

    assert approval is None
    assert tool.execute_calls == 0


def test_prepare_state_changing_stores_pending_approval() -> None:
    tool = FakeTool(
        name="create_access_request",
        risk_level=(
            ToolRiskLevel.STATE_CHANGING
        ),
    )

    executor, service = make_executor(
        tool
    )

    tool_call = ToolCall(
        tool_name="create_access_request",
        arguments={
            "value": "production",
        },
    )

    approval = executor.prepare(
        tool_call
    )

    assert approval is not None
    assert (
        approval.status
        == ApprovalStatus.PENDING
    )
    assert (
        service.get(
            approval.approval_id
        )
        == approval
    )
    assert tool.execute_calls == 0


def test_state_changing_tool_without_approval_is_blocked() -> None:
    tool = FakeTool(
        name="create_access_request",
        risk_level=(
            ToolRiskLevel.STATE_CHANGING
        ),
    )

    executor, _ = make_executor(
        tool
    )

    tool_call = ToolCall(
        tool_name="create_access_request",
        arguments={
            "value": "production",
        },
    )

    with pytest.raises(
        ApprovalRequiredError
    ):
        executor.execute(
            tool_call
        )

    assert tool.execute_calls == 0


def test_manual_approval_object_is_not_trusted() -> None:
    tool = FakeTool(
        name="create_access_request",
        risk_level=(
            ToolRiskLevel.STATE_CHANGING
        ),
    )

    executor, _ = make_executor(
        tool
    )

    tool_call = ToolCall(
        tool_name="create_access_request",
        arguments={
            "value": "production",
        },
    )

    forged = ApprovalRequest(
        tool_call=tool_call,
        reason="Forged approval.",
        status=ApprovalStatus.APPROVED,
    )

    with pytest.raises(
        TypeError,
        match="approval_id must be a UUID",
    ):
        executor.execute(
            tool_call,
            forged,  # type: ignore[arg-type]
        )

    assert tool.execute_calls == 0


def test_unknown_approval_id_is_rejected() -> None:
    tool = FakeTool(
        name="create_access_request",
        risk_level=(
            ToolRiskLevel.STATE_CHANGING
        ),
    )

    executor, _ = make_executor(
        tool
    )

    tool_call = ToolCall(
        tool_name="create_access_request",
        arguments={
            "value": "production",
        },
    )

    with pytest.raises(
        ApprovalNotFoundError,
        match="approval not found",
    ):
        executor.execute(
            tool_call,
            uuid.uuid4(),
        )

    assert tool.execute_calls == 0


def test_pending_approval_id_does_not_execute() -> None:
    tool = FakeTool(
        name="create_access_request",
        risk_level=(
            ToolRiskLevel.STATE_CHANGING
        ),
    )

    executor, _ = make_executor(
        tool
    )

    tool_call = ToolCall(
        tool_name="create_access_request",
        arguments={
            "value": "production",
        },
    )

    approval = executor.prepare(
        tool_call
    )

    assert approval is not None

    with pytest.raises(
        ApprovalNotApprovedError,
        match="not approved",
    ):
        executor.execute(
            tool_call,
            approval.approval_id,
        )

    assert tool.execute_calls == 0


def test_rejected_approval_id_does_not_execute() -> None:
    tool = FakeTool(
        name="create_access_request",
        risk_level=(
            ToolRiskLevel.STATE_CHANGING
        ),
    )

    executor, service = make_executor(
        tool
    )

    tool_call = ToolCall(
        tool_name="create_access_request",
        arguments={
            "value": "production",
        },
    )

    approval = executor.prepare(
        tool_call
    )

    assert approval is not None

    service.reject(
        approval.approval_id
    )

    with pytest.raises(
        ApprovalNotApprovedError,
        match="not approved",
    ):
        executor.execute(
            tool_call,
            approval.approval_id,
        )

    assert tool.execute_calls == 0


def test_approval_cannot_be_used_for_different_call() -> None:
    tool = FakeTool(
        name="create_access_request",
        risk_level=(
            ToolRiskLevel.STATE_CHANGING
        ),
    )

    executor, service = make_executor(
        tool
    )

    first_call = ToolCall(
        tool_name="create_access_request",
        arguments={
            "value": "read-only",
        },
    )

    second_call = ToolCall(
        tool_name="create_access_request",
        arguments={
            "value": "admin",
        },
    )

    approval = executor.prepare(
        first_call
    )

    assert approval is not None

    service.approve(
        approval.approval_id
    )

    with pytest.raises(
        ApprovalBindingError,
        match="does not match",
    ):
        executor.execute(
            second_call,
            approval.approval_id,
        )

    assert tool.execute_calls == 0


def test_approved_matching_call_executes() -> None:
    tool = FakeTool(
        name="create_access_request",
        risk_level=(
            ToolRiskLevel.STATE_CHANGING
        ),
    )

    executor, service = make_executor(
        tool
    )

    tool_call = ToolCall(
        tool_name="create_access_request",
        arguments={
            "value": "production",
        },
    )

    approval = executor.prepare(
        tool_call
    )

    assert approval is not None

    service.approve(
        approval.approval_id
    )

    result = executor.execute(
        tool_call,
        approval.approval_id,
    )

    assert (
        result.status
        == ToolExecutionStatus.SUCCEEDED
    )
    assert result.output == {
        "received": "production",
    }
    assert tool.execute_calls == 1


def test_invalid_arguments_are_blocked_before_execution() -> None:
    tool = FakeTool(
        name="lookup_policy",
        risk_level=ToolRiskLevel.READ_ONLY,
    )

    executor, _ = make_executor(
        tool
    )

    tool_call = ToolCall(
        tool_name="lookup_policy",
        arguments={
            "value": "",
        },
    )

    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        executor.execute(
            tool_call
        )

    assert tool.execute_calls == 0


def test_execution_error_becomes_failed_result() -> None:
    tool = FakeTool(
        name="lookup_policy",
        risk_level=ToolRiskLevel.READ_ONLY,
        fail_execution=True,
    )

    executor, _ = make_executor(
        tool
    )

    tool_call = ToolCall(
        tool_name="lookup_policy",
        arguments={
            "value": "security",
        },
    )

    result = executor.execute(
        tool_call
    )

    assert (
        result.status
        == ToolExecutionStatus.FAILED
    )
    assert (
        result.error
        == "simulated execution failure"
    )
    assert result.output is None
    assert tool.execute_calls == 1