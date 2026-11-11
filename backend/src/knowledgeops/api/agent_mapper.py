from knowledgeops.agent.orchestrator import (
    AgentRunResult,
    AgentRunStatus,
)
from knowledgeops.api.agent_models import (
    AgentRunResponse,
    ApprovalResponse,
    SourceResponse,
    ToolResultResponse,
)


def map_agent_run_result(
    result: AgentRunResult,
) -> AgentRunResponse:
    if (
        result.status
        == AgentRunStatus.RAG_COMPLETED
    ):
        if result.rag_response is None:
            raise ValueError(
                "RAG result requires rag_response"
            )

        return AgentRunResponse(
            status="rag_completed",
            answer=result.rag_response.answer,
            sources=[
                SourceResponse(
                    citation_id=source.citation_id,
                    filename=source.filename,
                    page_number=source.page_number,
                    chunk_index=source.chunk_index,
                )
                for source
                in result.rag_response.sources
            ],
        )

    if (
        result.status
        == AgentRunStatus.APPROVAL_REQUIRED
    ):
        if (
            result.tool_call is None
            or result.approval_request is None
        ):
            raise ValueError(
                "approval-required result "
                "requires tool call and approval"
            )

        approval = result.approval_request

        return AgentRunResponse(
            status="approval_required",
            tool_name=result.tool_call.tool_name,
            tool_arguments=(
                result.tool_call.arguments
            ),
            approval=ApprovalResponse(
                approval_id=approval.approval_id,
                status=approval.status.value,
                tool_name=(
                    approval.tool_call.tool_name
                ),
                arguments=(
                    approval.tool_call.arguments
                ),
                reason=approval.reason,
            ),
        )

    if (
        result.status
        == AgentRunStatus.TOOL_COMPLETED
    ):
        if (
            result.tool_call is None
            or result.tool_result is None
        ):
            raise ValueError(
                "completed tool result requires "
                "tool call and tool result"
            )

        tool_result = result.tool_result

        return AgentRunResponse(
            status="tool_completed",
            tool_name=result.tool_call.tool_name,
            tool_arguments=(
                result.tool_call.arguments
            ),
            tool_result=ToolResultResponse(
                status=tool_result.status.value,
                output=tool_result.output,
                error=tool_result.error,
            ),
        )

    raise ValueError(
        f"unsupported agent run status: "
        f"{result.status}"
    )