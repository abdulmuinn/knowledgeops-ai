from dataclasses import dataclass
from enum import StrEnum

from knowledgeops.agent.models import (
    ToolCall,
)


class AgentDecisionType(StrEnum):
    RAG = "rag"
    TOOL = "tool"


@dataclass(frozen=True)
class AgentDecision:
    decision_type: AgentDecisionType
    tool_call: ToolCall | None = None

    def __post_init__(self) -> None:
        if (
            self.decision_type
            == AgentDecisionType.RAG
            and self.tool_call is not None
        ):
            raise ValueError(
                "RAG decision cannot contain "
                "a tool call"
            )

        if (
            self.decision_type
            == AgentDecisionType.TOOL
            and self.tool_call is None
        ):
            raise ValueError(
                "tool decision requires "
                "a tool call"
            )