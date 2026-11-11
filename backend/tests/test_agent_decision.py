import pytest

from knowledgeops.agent.decision import (
    AgentDecision,
    AgentDecisionType,
)
from knowledgeops.agent.models import (
    ToolCall,
)


def test_rag_decision_is_valid() -> None:
    decision = AgentDecision(
        decision_type=AgentDecisionType.RAG,
    )

    assert (
        decision.decision_type
        == AgentDecisionType.RAG
    )
    assert decision.tool_call is None


def test_tool_decision_requires_tool_call() -> None:
    with pytest.raises(
        ValueError,
        match="requires",
    ):
        AgentDecision(
            decision_type=AgentDecisionType.TOOL,
        )


def test_rag_decision_rejects_tool_call() -> None:
    tool_call = ToolCall(
        tool_name="lookup_policy",
        arguments={},
    )

    with pytest.raises(
        ValueError,
        match="cannot contain",
    ):
        AgentDecision(
            decision_type=AgentDecisionType.RAG,
            tool_call=tool_call,
        )


def test_tool_decision_contains_proposed_call() -> None:
    tool_call = ToolCall(
        tool_name="create_access_request",
        arguments={
            "system": "production",
            "role": "read-only",
        },
    )

    decision = AgentDecision(
        decision_type=AgentDecisionType.TOOL,
        tool_call=tool_call,
    )

    assert (
        decision.decision_type
        == AgentDecisionType.TOOL
    )
    assert decision.tool_call == tool_call