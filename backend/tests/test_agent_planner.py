import pytest

from knowledgeops.agent.decision import (
    AgentDecisionType,
)
from knowledgeops.agent.planner import (
    AgentPlanner,
    build_planner_prompt,
    parse_agent_decision,
)
from knowledgeops.agent.models import (
    ToolRiskLevel,
)
from knowledgeops.tools.base import (
    ToolDefinition,
)
from knowledgeops.tools.registry import (
    ToolRegistry,
)


class FakeLLM:
    def __init__(
        self,
        response: str,
    ) -> None:
        self.response = response
        self.calls = 0
        self.last_system: str | None = None
        self.last_user: str | None = None

    def generate(
        self,
        system: str,
        user: str,
    ) -> str:
        self.calls += 1
        self.last_system = system
        self.last_user = user

        return self.response


class FakeAccessRequestTool:
    @property
    def definition(
        self,
    ) -> ToolDefinition:
        return ToolDefinition(
            name="create_access_request",
            description=(
                "Create an internal access request."
            ),
            risk_level=(
                ToolRiskLevel.STATE_CHANGING
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "system": {
                        "type": "string",
                    },
                    "role": {
                        "type": "string",
                    },
                },
                "required": [
                    "system",
                    "role",
                ],
            },
        )

    def validate_arguments(
        self,
        arguments: dict[str, object],
    ) -> None:
        system = arguments.get(
            "system"
        )
        role = arguments.get(
            "role"
        )

        if (
            not isinstance(system, str)
            or not system.strip()
        ):
            raise ValueError(
                "system must be a non-empty string"
            )

        if (
            not isinstance(role, str)
            or not role.strip()
        ):
            raise ValueError(
                "role must be a non-empty string"
            )

    def execute(
        self,
        arguments: dict[str, object],
    ) -> object:
        raise AssertionError(
            "planner must never execute tools"
        )


def make_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            FakeAccessRequestTool(),
        ]
    )


def test_build_prompt_contains_registered_tool() -> None:
    prompt = build_planner_prompt(
        "Create production access for me.",
        make_registry(),
    )

    assert "create_access_request" in prompt
    assert "state_changing" in prompt
    assert (
        "Create production access for me."
        in prompt
    )


def test_empty_user_request_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        build_planner_prompt(
            "   ",
            make_registry(),
        )


def test_parse_rag_decision() -> None:
    decision = parse_agent_decision(
        '{"decision": "rag"}',
        make_registry(),
    )

    assert (
        decision.decision_type
        == AgentDecisionType.RAG
    )
    assert decision.tool_call is None


def test_parse_valid_tool_decision() -> None:
    decision = parse_agent_decision(
        """
        {
            "decision": "tool",
            "tool_name": "create_access_request",
            "arguments": {
                "system": "production",
                "role": "read-only"
            }
        }
        """,
        make_registry(),
    )

    assert (
        decision.decision_type
        == AgentDecisionType.TOOL
    )
    assert decision.tool_call is not None
    assert (
        decision.tool_call.tool_name
        == "create_access_request"
    )
    assert decision.tool_call.arguments == {
        "system": "production",
        "role": "read-only",
    }


def test_hallucinated_tool_is_rejected() -> None:
    with pytest.raises(
        KeyError,
        match="not registered",
    ):
        parse_agent_decision(
            """
            {
                "decision": "tool",
                "tool_name": "delete_everything",
                "arguments": {}
            }
            """,
            make_registry(),
        )


def test_invalid_tool_arguments_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="role",
    ):
        parse_agent_decision(
            """
            {
                "decision": "tool",
                "tool_name": "create_access_request",
                "arguments": {
                    "system": "production",
                    "role": ""
                }
            }
            """,
            make_registry(),
        )


def test_invalid_json_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="invalid JSON",
    ):
        parse_agent_decision(
            '{"decision": "tool"',
            make_registry(),
        )


def test_rag_decision_cannot_smuggle_tool_fields() -> None:
    with pytest.raises(
        ValueError,
        match="cannot contain tool fields",
    ):
        parse_agent_decision(
            """
            {
                "decision": "rag",
                "tool_name": "create_access_request",
                "arguments": {
                    "system": "production",
                    "role": "admin"
                }
            }
            """,
            make_registry(),
        )


def test_planner_uses_llm_and_returns_decision() -> None:
    llm = FakeLLM(
        """
        {
            "decision": "tool",
            "tool_name": "create_access_request",
            "arguments": {
                "system": "production",
                "role": "read-only"
            }
        }
        """
    )

    planner = AgentPlanner(
        llm=llm,
        registry=make_registry(),
    )

    decision = planner.plan(
        "Please create read-only production access."
    )

    assert llm.calls == 1
    assert (
        decision.decision_type
        == AgentDecisionType.TOOL
    )
    assert decision.tool_call is not None


def test_empty_llm_output_is_rejected() -> None:
    planner = AgentPlanner(
        llm=FakeLLM("   "),
        registry=make_registry(),
    )

    with pytest.raises(
        ValueError,
        match="empty output",
    ):
        planner.plan(
            "Create production access."
        )