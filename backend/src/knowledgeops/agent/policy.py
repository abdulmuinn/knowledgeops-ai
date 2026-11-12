from dataclasses import replace

from knowledgeops.agent.models import (
    ApprovalRequest,
    ApprovalStatus,
    ToolCall,
    ToolRiskLevel,
)
from knowledgeops.tools.base import (
    ToolDefinition,
)


class ApprovalRequiredError(
    PermissionError
):
    pass


class ApprovalBindingError(
    PermissionError
):
    pass


class ApprovalNotApprovedError(
    PermissionError
):
    pass


class ApprovalPolicy:
    def requires_approval(
        self,
        definition: ToolDefinition,
    ) -> bool:
        return (
            definition.risk_level
            == ToolRiskLevel.STATE_CHANGING
        )

    def create_request(
        self,
        tool_call: ToolCall,
        definition: ToolDefinition,
    ) -> ApprovalRequest | None:
        self._validate_tool_binding(
            tool_call,
            definition,
        )

        if not self.requires_approval(
            definition
        ):
            return None

        return ApprovalRequest(
            tool_call=tool_call,
            reason=(
                f"Tool '{definition.name}' "
                "changes state and requires "
                "human approval."
            ),
        )

    def approve(
        self,
        request: ApprovalRequest,
    ) -> ApprovalRequest:
        self._validate_pending(
            request
        )

        return replace(
            request,
            status=ApprovalStatus.APPROVED,
        )

    def reject(
        self,
        request: ApprovalRequest,
    ) -> ApprovalRequest:
        self._validate_pending(
            request
        )

        return replace(
            request,
            status=ApprovalStatus.REJECTED,
        )

    def ensure_execution_allowed(
        self,
        tool_call: ToolCall,
        definition: ToolDefinition,
        approval: ApprovalRequest | None = None,
    ) -> None:
        self._validate_tool_binding(
            tool_call,
            definition,
        )

        if not self.requires_approval(
            definition
        ):
            return

        if approval is None:
            raise ApprovalRequiredError(
                "human approval is required "
                "before executing this tool"
            )

        if approval.tool_call != tool_call:
            raise ApprovalBindingError(
                "approval does not match "
                "the requested tool call"
            )

        if (
            approval.status
            != ApprovalStatus.APPROVED
        ):
            raise ApprovalNotApprovedError(
                "approval request is not approved"
            )

    @staticmethod
    def _validate_tool_binding(
        tool_call: ToolCall,
        definition: ToolDefinition,
    ) -> None:
        if (
            tool_call.tool_name
            != definition.name
        ):
            raise ValueError(
                "tool call does not match "
                "tool definition"
            )

    @staticmethod
    def _validate_pending(
        request: ApprovalRequest,
    ) -> None:
        if (
            request.status
            != ApprovalStatus.PENDING
        ):
            raise ValueError(
                "only pending approval requests "
                "can be decided"
            )