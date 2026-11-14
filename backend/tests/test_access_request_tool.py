import pytest

from knowledgeops.agent.models import (
    ToolRiskLevel,
)
from knowledgeops.tools.access_request import (
    CreateAccessRequestTool,
)


class FakeN8NClient:
    def __init__(self) -> None:
        self.calls: list[
            dict[str, str]
        ] = []

    def create_access_request(
        self,
        *,
        system: str,
        role: str,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "system": system,
                "role": role,
            }
        )

        return {
            "request_id": "AR-N8N1234",
            "system": system,
            "role": role,
            "status": "submitted",
            "created_at": (
                "2026-11-14T04:00:00Z"
            ),
        }


def make_tool() -> tuple[
    CreateAccessRequestTool,
    FakeN8NClient,
]:
    client = FakeN8NClient()

    tool = CreateAccessRequestTool(
        client=client,  # type: ignore[arg-type]
    )

    return tool, client


def test_access_request_tool_definition() -> None:
    tool, _ = make_tool()

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


def test_access_request_tool_delegates_to_client() -> None:
    tool, client = make_tool()

    result = tool.execute(
        {
            "system": "production",
            "role": "read-only",
        }
    )

    assert client.calls == [
        {
            "system": "production",
            "role": "read-only",
        }
    ]

    assert isinstance(
        result,
        dict,
    )

    assert (
        result["request_id"]
        == "AR-N8N1234"
    )

    assert result["status"] == "submitted"


def test_access_request_rejects_empty_system() -> None:
    tool, client = make_tool()

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

    assert client.calls == []


def test_access_request_rejects_empty_role() -> None:
    tool, client = make_tool()

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

    assert client.calls == []