import pytest

from knowledgeops.agent.models import (
    ToolRiskLevel,
)
from knowledgeops.tools.access_request import (
    CreateAccessRequestTool,
)


def test_access_request_tool_definition() -> None:
    tool = CreateAccessRequestTool()

    definition = tool.definition

    assert (
        definition.name
        == "create_access_request"
    )

    assert (
        definition.risk_level
        == ToolRiskLevel.STATE_CHANGING
    )

    assert (
        definition.input_schema["type"]
        == "object"
    )


def test_access_request_tool_executes() -> None:
    tool = CreateAccessRequestTool()

    result = tool.execute(
        {
            "system": "production",
            "role": "read-only",
        }
    )

    assert isinstance(
        result,
        dict,
    )

    assert result["system"] == "production"
    assert result["role"] == "read-only"
    assert result["status"] == "submitted"

    request_id = result["request_id"]

    assert isinstance(
        request_id,
        str,
    )

    assert request_id.startswith(
        "AR-"
    )


def test_access_request_rejects_empty_system() -> None:
    tool = CreateAccessRequestTool()

    with pytest.raises(
        ValueError,
        match="system",
    ):
        tool.execute(
            {
                "system": "",
                "role": "read-only",
            }
        )


def test_access_request_rejects_empty_role() -> None:
    tool = CreateAccessRequestTool()

    with pytest.raises(
        ValueError,
        match="role",
    ):
        tool.execute(
            {
                "system": "production",
                "role": "",
            }
        )