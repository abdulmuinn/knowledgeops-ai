import pytest

from knowledgeops.agent.models import (
    ApprovalStatus,
    ToolCall,
    ToolExecutionStatus,
    ToolRiskLevel,
)
from knowledgeops.agent.policy import (
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
) -> ToolExecutor:
    registry = ToolRegistry(
        [
            tool,
        ]
    )

    return ToolExecutor(
        registry=registry,
        approval_policy=ApprovalPolicy(),
    )


def test_read_only_tool_executes_without_approval() -> None:
    tool = FakeTool(
        name="lookup_policy",
        risk_level=ToolRiskLevel.READ_ONLY,
    )

    executor = make_executor(tool)

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

    executor = make_executor(tool)

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


def test_prepare_state_changing_returns_pending_approval() -> None:
    tool = FakeTool(
        name="create_access_request",
        risk_level=(
            ToolRiskLevel.STATE_CHANGING
        ),
    )

    executor = make_executor(tool)

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
    assert tool.execute_calls == 0


def test_state_changing_tool_never_executes_without_approval() -> None:
    tool = FakeTool(
        name="create_access_request",
        risk_level=(
            ToolRiskLevel.STATE_CHANGING
        ),
    )

    executor = make_executor(tool)

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


def test_pending_approval_does_not_execute_tool() -> None:
    tool = FakeTool(
        name="create_access_request",
        risk_level=(
            ToolRiskLevel.STATE_CHANGING
        ),
    )

    executor = make_executor(tool)

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
        ApprovalNotApprovedError
    ):
        executor.execute(
            tool_call,
            approval,
        )

    assert tool.execute_calls == 0


def test_approved_state_changing_tool_executes() -> None:
    tool = FakeTool(
        name="create_access_request",
        risk_level=(
            ToolRiskLevel.STATE_CHANGING
        ),
    )

    executor = make_executor(tool)

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

    approved = (
        executor.approval_policy.approve(
            approval
        )
    )

    result = executor.execute(
        tool_call,
        approved,
    )

    assert (
        result.status
        == ToolExecutionStatus.SUCCEEDED
    )
    assert tool.execute_calls == 1


def test_invalid_arguments_are_blocked_before_execution() -> None:
    tool = FakeTool(
        name="lookup_policy",
        risk_level=ToolRiskLevel.READ_ONLY,
    )

    executor = make_executor(tool)

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

    executor = make_executor(tool)

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