import json

from knowledgeops.agent.decision import (
    AgentDecision,
    AgentDecisionType,
)
from knowledgeops.agent.models import ToolCall
from knowledgeops.llm.base import LLMClient
from knowledgeops.tools.registry import ToolRegistry


PLANNER_SYSTEM_PROMPT = """
You are the action planner for KnowledgeOps AI.

Choose exactly one route:

1. "rag"
   Use this for informational questions that should be answered
   from the internal knowledge base.

2. "tool"
   Use this only when the user is explicitly asking the system
   to perform an action supported by one of the available tools.

Rules:
- Use only tools listed in AVAILABLE TOOLS.
- Never invent a tool.
- Never claim that a tool has already executed.
- Never claim that human approval has been granted.
- Tool execution and approval are handled by other components.
- Return JSON only.
- Do not include markdown fences.

Valid outputs:

{"decision": "rag"}

or

{
  "decision": "tool",
  "tool_name": "registered_tool_name",
  "arguments": {}
}
""".strip()


def build_planner_prompt(
    user_request: str,
    registry: ToolRegistry,
) -> str:
    user_request = user_request.strip()

    if not user_request:
        raise ValueError(
            "user request cannot be empty"
        )

    tools = [
        {
            "name": definition.name,
            "description": definition.description,
            "risk_level": (
                definition.risk_level.value
            ),
            "input_schema": (
                definition.input_schema
            ),
        }
        for definition
        in registry.definitions()
    ]

    return (
        "AVAILABLE TOOLS:\n"
        f"{json.dumps(tools, indent=2)}\n\n"
        "USER REQUEST:\n"
        f"{user_request}"
    )


def parse_agent_decision(
    raw_output: str,
    registry: ToolRegistry,
) -> AgentDecision:
    raw_output = raw_output.strip()

    start = raw_output.find("{")
    
    if start == -1:
        raise ValueError(
            "planner output must contain JSON"
        )
    
    end = raw_output.rfind("}")
    
    if end == -1 or end < start:
        raise ValueError(
            "planner output is invalid JSON"
        )
    
    try:
        data = json.loads(
            raw_output[start : end + 1]
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            "planner output is invalid JSON"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            "planner output must be an object"
        )

    decision = data.get(
        "decision"
    )

    if decision == AgentDecisionType.RAG:
        if (
            "tool_name" in data
            or "arguments" in data
        ):
            raise ValueError(
                "RAG decision cannot contain "
                "tool fields"
            )

        return AgentDecision(
            decision_type=(
                AgentDecisionType.RAG
            )
        )

    if decision == AgentDecisionType.TOOL:
        tool_name = data.get(
            "tool_name"
        )
        arguments = data.get(
            "arguments"
        )

        if (
            not isinstance(tool_name, str)
            or not tool_name.strip()
        ):
            raise ValueError(
                "tool decision requires "
                "a valid tool_name"
            )

        if not isinstance(
            arguments,
            dict,
        ):
            raise ValueError(
                "tool decision requires "
                "an arguments object"
            )

        tool = registry.get(
            tool_name
        )

        tool.validate_arguments(
            arguments
        )

        return AgentDecision(
            decision_type=(
                AgentDecisionType.TOOL
            ),
            tool_call=ToolCall(
                tool_name=tool_name,
                arguments=arguments,
            ),
        )

    raise ValueError(
        "planner decision must be "
        "'rag' or 'tool'"
    )


class AgentPlanner:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
    ) -> None:
        self.llm = llm
        self.registry = registry

    def plan(
        self,
        user_request: str,
    ) -> AgentDecision:
        user_prompt = build_planner_prompt(
            user_request,
            self.registry,
        )

        raw_output = self.llm.generate(
            system=PLANNER_SYSTEM_PROMPT,
            user=user_prompt,
        )

        if not raw_output.strip():
            raise ValueError(
                "planner returned empty output"
            )

        return parse_agent_decision(
            raw_output,
            self.registry,
        )