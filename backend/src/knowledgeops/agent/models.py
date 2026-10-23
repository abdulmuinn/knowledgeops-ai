import uuid
from dataclasses import dataclass, field
from enum import StrEnum


class ToolRiskLevel(StrEnum):
    READ_ONLY = "read_only"
    STATE_CHANGING = "state_changing"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ToolExecutionStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class ToolCall:
    tool_name: str
    arguments: dict[str, object]
    call_id: uuid.UUID = field(
        default_factory=uuid.uuid4
    )

    def __post_init__(self) -> None:
        if not self.tool_name.strip():
            raise ValueError(
                "tool_name cannot be empty"
            )


@dataclass(frozen=True)
class ApprovalRequest:
    tool_call: ToolCall
    reason: str
    status: ApprovalStatus = (
        ApprovalStatus.PENDING
    )
    approval_id: uuid.UUID = field(
        default_factory=uuid.uuid4
    )

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError(
                "approval reason cannot be empty"
            )


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: uuid.UUID
    status: ToolExecutionStatus
    output: object | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if (
            self.status
            == ToolExecutionStatus.SUCCEEDED
            and self.error is not None
        ):
            raise ValueError(
                "successful tool result "
                "cannot contain an error"
            )

        if (
            self.status
            == ToolExecutionStatus.FAILED
            and (
                self.error is None
                or not self.error.strip()
            )
        ):
            raise ValueError(
                "failed tool result "
                "must contain an error"
            )