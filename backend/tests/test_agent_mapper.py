import uuid

import pytest

from knowledgeops.agent.models import (
    ApprovalRequest,
    ToolCall,
    ToolExecutionStatus,
    ToolResult,
)
from knowledgeops.agent.orchestrator import (
    AgentRunResult,
    AgentRunStatus,
)
from knowledgeops.api.agent_mapper import (
    map_agent_run_result,
)
from knowledgeops.rag.models import (
    ContextSource,
    RAGResponse,
)


def make_source() -> ContextSource:
    return ContextSource(
        citation_id="S1",
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename="access_control_policy.txt",
        page_number=None,
        chunk_index=0,
        content=(
            "Emergency access expires "
            "after four hours."
        ),
        dense_score=0.62,
        rerank_score=8.4,
    )


def test_maps_rag_completed_result() -> None:
    rag_response = RAGResponse(
        status="answered",
        answer=(
            "Emergency access expires "
            "after four hours [S1]."
        ),
        sources=(
            make_source(),
        ),
        raw_output=(
            "Emergency access expires "
            "after four hours [S1]."
        ),
    )

    result = AgentRunResult(
        status=AgentRunStatus.RAG_COMPLETED,
        rag_response=rag_response,
    )

    response = map_agent_run_result(
        result
    )

    assert response.status == "rag_completed"

    assert response.answer == (
        "Emergency access expires "
        "after four hours [S1]."
    )

    assert len(response.sources) == 1

    source = response.sources[0]

    assert source.citation_id == "S1"
    assert (
        source.filename
        == "access_control_policy.txt"
    )
    assert source.chunk_index == 0

    assert response.approval is None
    assert response.tool_result is None


def test_maps_approval_required_result() -> None:
    tool_call = ToolCall(
        tool_name="create_access_request",
        arguments={
            "system": "production",
            "role": "read-only",
        },
    )

    approval = ApprovalRequest(
        tool_call=tool_call,
        reason=(
            "Tool changes state and requires "
            "human approval."
        ),
    )

    result = AgentRunResult(
        status=(
            AgentRunStatus.APPROVAL_REQUIRED
        ),
        tool_call=tool_call,
        approval_request=approval,
    )

    response = map_agent_run_result(
        result
    )

    assert (
        response.status
        == "approval_required"
    )

    assert (
        response.tool_name
        == "create_access_request"
    )

    assert response.tool_arguments == {
        "system": "production",
        "role": "read-only",
    }

    assert response.approval is not None

    assert (
        response.approval.approval_id
        == approval.approval_id
    )

    assert (
        response.approval.status
        == "pending"
    )

    assert (
        response.approval.tool_name
        == "create_access_request"
    )

    assert response.tool_result is None


def test_maps_tool_completed_result() -> None:
    tool_call = ToolCall(
        tool_name="lookup_policy",
        arguments={
            "query": "remote work",
        },
    )

    tool_result = ToolResult(
        tool_call_id=tool_call.call_id,
        status=(
            ToolExecutionStatus.SUCCEEDED
        ),
        output={
            "policy": "remote_work_policy.txt",
        },
    )

    result = AgentRunResult(
        status=AgentRunStatus.TOOL_COMPLETED,
        tool_call=tool_call,
        tool_result=tool_result,
    )

    response = map_agent_run_result(
        result
    )

    assert (
        response.status
        == "tool_completed"
    )

    assert (
        response.tool_name
        == "lookup_policy"
    )

    assert response.tool_result is not None

    assert (
        response.tool_result.status
        == "succeeded"
    )

    assert response.tool_result.output == {
        "policy": "remote_work_policy.txt",
    }

    assert (
        response.tool_result.error
        is None
    )


def test_rag_completed_requires_rag_response() -> None:
    result = AgentRunResult(
        status=AgentRunStatus.RAG_COMPLETED,
    )

    with pytest.raises(
        ValueError,
        match="requires rag_response",
    ):
        map_agent_run_result(
            result
        )


def test_approval_required_needs_tool_and_approval() -> None:
    result = AgentRunResult(
        status=(
            AgentRunStatus.APPROVAL_REQUIRED
        ),
    )

    with pytest.raises(
        ValueError,
        match="requires tool call and approval",
    ):
        map_agent_run_result(
            result
        )