import uuid

from knowledgeops.agent.approval_service import (
    HumanApprovalService,
)
from knowledgeops.agent.models import (
    ApprovalRequest,
    ToolCall,
    ToolExecutionStatus,
    ToolResult,
)
from knowledgeops.tools.registry import (
    ToolRegistry,
)


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        approval_service: HumanApprovalService,
    ) -> None:
        self.registry = registry
        self.approval_service = approval_service
        self.approval_policy = (
            approval_service.policy
        )

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

        request = (
            self.approval_policy.create_request(
                tool_call,
                tool.definition,
            )
        )

        if request is not None:
            self.approval_service.submit(
                request
            )

        return request

    def execute(
        self,
        tool_call: ToolCall,
        approval_id: uuid.UUID | None = None,
    ) -> ToolResult:
        if (
            approval_id is not None
            and not isinstance(
                approval_id,
                uuid.UUID,
            )
        ):
            raise TypeError(
                "approval_id must be a UUID"
            )
        tool = self.registry.get(
            tool_call.tool_name
        )

        tool.validate_arguments(
            tool_call.arguments
        )

        approval = None

        if (
            self.approval_policy.requires_approval(
                tool.definition
            )
            and approval_id is not None
        ):
            approval = (
                self.approval_service.get(
                    approval_id
                )
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
                status=(
                    ToolExecutionStatus.FAILED
                ),
                error=error,
            )

        return ToolResult(
            tool_call_id=tool_call.call_id,
            status=(
                ToolExecutionStatus.SUCCEEDED
            ),
            output=output,
        )