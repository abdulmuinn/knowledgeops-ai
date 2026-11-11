from knowledgeops.agent.models import (
    ApprovalRequest,
    ToolCall,
    ToolExecutionStatus,
    ToolResult,
)
from knowledgeops.agent.policy import (
    ApprovalPolicy,
)
from knowledgeops.tools.registry import (
    ToolRegistry,
)


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        approval_policy: ApprovalPolicy,
    ) -> None:
        self.registry = registry
        self.approval_policy = approval_policy

    def prepare(
        self,
        tool_call: ToolCall,
    ) -> ApprovalRequest | None:
        tool = self.registry.get(
            tool_call.tool_name
        )

        tool.validate_arguments(
            tool_call.arguments
        )

        return self.approval_policy.create_request(
            tool_call,
            tool.definition,
        )

    def execute(
        self,
        tool_call: ToolCall,
        approval: ApprovalRequest | None = None,
    ) -> ToolResult:
        tool = self.registry.get(
            tool_call.tool_name
        )

        tool.validate_arguments(
            tool_call.arguments
        )

        self.approval_policy.ensure_execution_allowed(
            tool_call,
            tool.definition,
            approval,
        )

        try:
            output = tool.execute(
                tool_call.arguments
            )

        except Exception as exc:
            error = str(exc).strip()

            if not error:
                error = exc.__class__.__name__

            return ToolResult(
                tool_call_id=tool_call.call_id,
                status=ToolExecutionStatus.FAILED,
                error=error,
            )

        return ToolResult(
            tool_call_id=tool_call.call_id,
            status=ToolExecutionStatus.SUCCEEDED,
            output=output,
        )