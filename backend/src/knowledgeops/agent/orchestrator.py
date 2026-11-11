import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from knowledgeops.agent.models import (
    ApprovalRequest,
    ToolCall,
    ToolResult,
)
from knowledgeops.agent.planner import AgentPlanner
from knowledgeops.rag.models import RAGResponse
from knowledgeops.tools.executor import ToolExecutor


class AgentRunStatus(StrEnum):
    RAG_COMPLETED = "rag_completed"
    TOOL_COMPLETED = "tool_completed"
    APPROVAL_REQUIRED = "approval_required"


@dataclass(frozen=True)
class AgentRunResult:
    status: AgentRunStatus
    rag_response: RAGResponse | None = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    approval_request: ApprovalRequest | None = None


class AgentOrchestrator:
    def __init__(
        self,
        planner: AgentPlanner,
        executor: ToolExecutor,
        rag_answerer: Callable[
            [str],
            RAGResponse,
        ],
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.rag_answerer = rag_answerer

    def handle(
        self,
        user_request: str,
    ) -> AgentRunResult:
        decision = self.planner.plan(
            user_request
        )

        if decision.tool_call is None:
            response = self.rag_answerer(
                user_request
            )

            return AgentRunResult(
                status=AgentRunStatus.RAG_COMPLETED,
                rag_response=response,
            )

        tool_call = decision.tool_call

        approval = self.executor.prepare(
            tool_call
        )

        if approval is not None:
            return AgentRunResult(
                status=(
                    AgentRunStatus.APPROVAL_REQUIRED
                ),
                tool_call=tool_call,
                approval_request=approval,
            )

        result = self.executor.execute(
            tool_call
        )

        return AgentRunResult(
            status=(
                AgentRunStatus.TOOL_COMPLETED
            ),
            tool_call=tool_call,
            tool_result=result,
        )

    def resume(
        self,
        approval_id: uuid.UUID,
    ) -> AgentRunResult:
        approval = (
            self.executor.approval_service.get(
                approval_id
            )
        )

        tool_call = approval.tool_call

        result = self.executor.execute(
            tool_call,
            approval_id,
        )

        return AgentRunResult(
            status=(
                AgentRunStatus.TOOL_COMPLETED
            ),
            tool_call=tool_call,
            tool_result=result,
        )