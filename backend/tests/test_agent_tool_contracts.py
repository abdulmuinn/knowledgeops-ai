import uuid

import pytest

from knowledgeops.agent.models import (
    ApprovalRequest,
    ApprovalStatus,
    ToolCall,
    ToolExecutionStatus,
    ToolResult,
    ToolRiskLevel,
)
from knowledgeops.tools.base import (
    ToolDefinition,
)
from knowledgeops.tools.registry import (
    ToolRegistry,
)


class FakeReadOnlyTool:
    @property
    def definition(
        self,
    ) -> ToolDefinition:
        return ToolDefinition(
            name="lookup_policy",
            description="Look up an internal policy.",
            risk_level=ToolRiskLevel.READ_ONLY,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                    },
                },
                "required": [
                    "query",
                ],
            },
        )

    def validate_arguments(
        self,
        arguments: dict[str, object],
    ) -> None:
        query = arguments.get("query")

        if (
            not isinstance(query, str)
            or not query.strip()
        ):
            raise ValueError(
                "query must be a non-empty string"
            )

    def execute(
        self,
        arguments: dict[str, object],
    ) -> object:
        self.validate_arguments(arguments)

        return {
            "policy": "example",
        }


class FakeStateChangingTool:
    @property
    def definition(
        self,
    ) -> ToolDefinition:
        return ToolDefinition(
            name="create_access_request",
            description=(
                "Create an access request."
            ),
            risk_level=(
                ToolRiskLevel.STATE_CHANGING
            ),
            input_schema={
                "type": "object",
                "properties": {},
            },
        )

    def validate_arguments(
        self,
        arguments: dict[str, object],
    ) -> None:
        return None

    def execute(
        self,
        arguments: dict[str, object],
    ) -> object:
        return {
            "request_id": "REQ-001",
        }


def test_tool_call_generates_unique_ids() -> None:
    first = ToolCall(
        tool_name="lookup_policy",
        arguments={},
    )
    second = ToolCall(
        tool_name="lookup_policy",
        arguments={},
    )

    assert isinstance(
        first.call_id,
        uuid.UUID,
    )
    assert first.call_id != second.call_id


def test_empty_tool_call_name_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="tool_name",
    ):
        ToolCall(
            tool_name="   ",
            arguments={},
        )


def test_approval_request_defaults_to_pending() -> None:
    tool_call = ToolCall(
        tool_name="create_access_request",
        arguments={
            "system": "production",
        },
    )

    approval = ApprovalRequest(
        tool_call=tool_call,
        reason="Changes external state.",
    )

    assert (
        approval.status
        == ApprovalStatus.PENDING
    )
    assert approval.tool_call is tool_call


def test_empty_approval_reason_is_rejected() -> None:
    tool_call = ToolCall(
        tool_name="create_access_request",
        arguments={},
    )

    with pytest.raises(
        ValueError,
        match="reason",
    ):
        ApprovalRequest(
            tool_call=tool_call,
            reason="   ",
        )


def test_successful_result_cannot_have_error() -> None:
    with pytest.raises(
        ValueError,
        match="cannot contain an error",
    ):
        ToolResult(
            tool_call_id=uuid.uuid4(),
            status=(
                ToolExecutionStatus.SUCCEEDED
            ),
            output={
                "ok": True,
            },
            error="unexpected failure",
        )


def test_failed_result_requires_error() -> None:
    with pytest.raises(
        ValueError,
        match="must contain an error",
    ):
        ToolResult(
            tool_call_id=uuid.uuid4(),
            status=ToolExecutionStatus.FAILED,
        )


def test_valid_tool_definition() -> None:
    definition = ToolDefinition(
        name="lookup_policy",
        description="Look up a policy.",
        risk_level=ToolRiskLevel.READ_ONLY,
        input_schema={
            "type": "object",
            "properties": {},
        },
    )

    assert definition.name == "lookup_policy"
    assert (
        definition.risk_level
        == ToolRiskLevel.READ_ONLY
    )


def test_tool_definition_requires_object_schema() -> None:
    with pytest.raises(
        ValueError,
        match="input_schema",
    ):
        ToolDefinition(
            name="lookup_policy",
            description="Look up a policy.",
            risk_level=(
                ToolRiskLevel.READ_ONLY
            ),
            input_schema={
                "type": "string",
            },
        )


def test_registry_registers_and_gets_tool() -> None:
    tool = FakeReadOnlyTool()
    registry = ToolRegistry()

    registry.register(tool)

    assert len(registry) == 1
    assert registry.contains(
        "lookup_policy"
    )
    assert (
        registry.get("lookup_policy")
        is tool
    )


def test_registry_rejects_duplicate_tool() -> None:
    registry = ToolRegistry(
        [
            FakeReadOnlyTool(),
        ]
    )

    with pytest.raises(
        ValueError,
        match="already registered",
    ):
        registry.register(
            FakeReadOnlyTool()
        )


def test_registry_rejects_unknown_tool() -> None:
    registry = ToolRegistry()

    with pytest.raises(
        KeyError,
        match="not registered",
    ):
        registry.get(
            "delete_entire_database"
        )


def test_registry_definitions_are_sorted() -> None:
    registry = ToolRegistry(
        [
            FakeReadOnlyTool(),
            FakeStateChangingTool(),
        ]
    )

    definitions = (
        registry.definitions()
    )

    assert [
        definition.name
        for definition in definitions
    ] == [
        "create_access_request",
        "lookup_policy",
    ]