import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentRunRequest(BaseModel):
    message: str = Field(
        min_length=1,
    )


class SourceResponse(BaseModel):
    citation_id: str
    filename: str
    page_number: int | None = None
    chunk_index: int


class ApprovalResponse(BaseModel):
    approval_id: uuid.UUID
    status: str
    tool_name: str
    arguments: dict[str, Any]
    reason: str


class ToolResultResponse(BaseModel):
    status: str
    output: Any | None = None
    error: str | None = None


class AgentRunResponse(BaseModel):
    status: Literal[
        "rag_completed",
        "tool_completed",
        "approval_required",
    ]

    answer: str | None = None

    sources: list[
        SourceResponse
    ] = Field(
        default_factory=list
    )

    tool_name: str | None = None

    tool_arguments: dict[
        str,
        Any,
    ] | None = None

    approval: (
        ApprovalResponse | None
    ) = None

    tool_result: (
        ToolResultResponse | None
    ) = None