import pytest

from knowledgeops.agent.approval_service import (
    HumanApprovalService,
)
from knowledgeops.agent.approval_store import (
    ApprovalStore,
)
from knowledgeops.agent.decision import (
    AgentDecision,
    AgentDecisionType,
)
from knowledgeops.agent.models import (
    ApprovalStatus,
    ToolCall,
    ToolExecutionStatus,
    ToolRiskLevel,
)
from knowledgeops.agent.orchestrator import (
    AgentOrchestrator,
    AgentRunStatus,
)
from knowledgeops.agent.policy import (
    ApprovalNotApprovedError,
    ApprovalPolicy,
)
from knowledgeops.rag.models import (
    RAGResponse,
)
from knowledgeops.tools.base import (
    ToolDefinition,
)
from knowledgeops.tools.executor import (
    ToolExecutor,
)
from knowledgeops.tools.registry import (
    ToolRegistry,
)


class FakePlanner:
    def __init__(
        self,
        decision: AgentDecision,
    ) -> None:
        self.decision = decision

    def plan(
        self,
        user_request: str,
    ) -> AgentDecision:
        return self.decision


class FakeTool:
    def __init__(
        self,
        name: str,
        risk_level: ToolRiskLevel,
    ) -> None:
        self.execute_calls = 0

        self._definition = ToolDefinition(
            name=name,
            description="Fake tool.",
            risk_level=risk_level,
            input_schema={
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                    },
                },
                "required": [
                    "value",
                ],
            },
        )

    @property
    def definition(
        self,
    ) -> ToolDefinition:
        return self._definition

    def validate_arguments(
        self,
        arguments: dict[str, object],
    ) -> None:
        value = arguments.get("value")

        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise ValueError(
                "value must be a non-empty string"
            )

    def execute(
        self,
        arguments: dict[str, object],
    ) -> object:
        self.execute_calls += 1

        return {
            "received": arguments["value"],
        }


def make_executor(
    tool: FakeTool,
) -> tuple[
    ToolExecutor,
    HumanApprovalService,
]:
    policy = ApprovalPolicy()

    service = HumanApprovalService(
        store=ApprovalStore(),
        policy=policy,
    )

    executor = ToolExecutor(
        registry=ToolRegistry(
            [
                tool,
            ]
        ),
        approval_service=service,
    )

    return executor, service


def test_rag_route_calls_rag_answerer() -> None:
    tool = FakeTool(
        "lookup_policy",
        ToolRiskLevel.READ_ONLY,
    )

    executor, _ = make_executor(
        tool
    )

    planner = FakePlanner(
        AgentDecision(
            decision_type=(
                AgentDecisionType.RAG
            )
        )
    )

    calls: list[str] = []

    def rag_answerer(
        question: str,
    ) -> RAGResponse:
        calls.append(question)

        return RAGResponse(
            status="answered",
            answer="Four hours [S1].",
            sources=(),
            raw_output="Four hours [S1].",
        )

    orchestrator = AgentOrchestrator(
        planner=planner,  # type: ignore[arg-type]
        executor=executor,
        rag_answerer=rag_answerer,
    )

    result = orchestrator.handle(
        "How long does emergency access last?"
    )

    assert (
        result.status
        == AgentRunStatus.RAG_COMPLETED
    )
    assert result.rag_response is not None
    assert calls == [
        "How long does emergency access last?"
    ]
    assert tool.execute_calls == 0


def test_read_only_tool_executes_immediately() -> None:
    tool = FakeTool(
        "lookup_policy",
        ToolRiskLevel.READ_ONLY,
    )

    executor, _ = make_executor(
        tool
    )

    tool_call = ToolCall(
        tool_name="lookup_policy",
        arguments={
            "value": "remote work",
        },
    )

    planner = FakePlanner(
        AgentDecision(
            decision_type=(
                AgentDecisionType.TOOL
            ),
            tool_call=tool_call,
        )
    )

    orchestrator = AgentOrchestrator(
        planner=planner,  # type: ignore[arg-type]
        executor=executor,
        rag_answerer=lambda _: RAGResponse(
            status="abstained",
            answer=None,
            sources=(),
            raw_output="INSUFFICIENT_EVIDENCE",
        ),
    )

    result = orchestrator.handle(
        "Look up remote work policy."
    )

    assert (
        result.status
        == AgentRunStatus.TOOL_COMPLETED
    )
    assert result.tool_result is not None
    assert (
        result.tool_result.status
        == ToolExecutionStatus.SUCCEEDED
    )
    assert tool.execute_calls == 1


def test_state_changing_tool_requires_approval() -> None:
    tool = FakeTool(
        "create_access_request",
        ToolRiskLevel.STATE_CHANGING,
    )

    executor, _ = make_executor(
        tool
    )

    tool_call = ToolCall(
        tool_name="create_access_request",
        arguments={
            "value": "production",
        },
    )

    planner = FakePlanner(
        AgentDecision(
            decision_type=(
                AgentDecisionType.TOOL
            ),
            tool_call=tool_call,
        )
    )

    orchestrator = AgentOrchestrator(
        planner=planner,  # type: ignore[arg-type]
        executor=executor,
        rag_answerer=lambda _: RAGResponse(
            status="abstained",
            answer=None,
            sources=(),
            raw_output="INSUFFICIENT_EVIDENCE",
        ),
    )

    result = orchestrator.handle(
        "Create production access."
    )

    assert (
        result.status
        == AgentRunStatus.APPROVAL_REQUIRED
    )
    assert result.approval_request is not None
    assert (
        result.approval_request.status
        == ApprovalStatus.PENDING
    )
    assert tool.execute_calls == 0


def test_approved_request_can_be_resumed() -> None:
    tool = FakeTool(
        "create_access_request",
        ToolRiskLevel.STATE_CHANGING,
    )

    executor, service = make_executor(
        tool
    )

    tool_call = ToolCall(
        tool_name="create_access_request",
        arguments={
            "value": "production",
        },
    )

    planner = FakePlanner(
        AgentDecision(
            decision_type=(
                AgentDecisionType.TOOL
            ),
            tool_call=tool_call,
        )
    )

    orchestrator = AgentOrchestrator(
        planner=planner,  # type: ignore[arg-type]
        executor=executor,
        rag_answerer=lambda _: RAGResponse(
            status="abstained",
            answer=None,
            sources=(),
            raw_output="INSUFFICIENT_EVIDENCE",
        ),
    )

    pending = orchestrator.handle(
        "Create production access."
    )

    assert pending.approval_request is not None

    approval_id = (
        pending.approval_request.approval_id
    )

    service.approve(
        approval_id
    )

    result = orchestrator.resume(
        approval_id
    )

    assert (
        result.status
        == AgentRunStatus.TOOL_COMPLETED
    )
    assert result.tool_result is not None
    assert (
        result.tool_result.status
        == ToolExecutionStatus.SUCCEEDED
    )
    assert tool.execute_calls == 1


def test_pending_request_cannot_be_resumed() -> None:
    tool = FakeTool(
        "create_access_request",
        ToolRiskLevel.STATE_CHANGING,
    )

    executor, _ = make_executor(
        tool
    )

    tool_call = ToolCall(
        tool_name="create_access_request",
        arguments={
            "value": "production",
        },
    )

    planner = FakePlanner(
        AgentDecision(
            decision_type=(
                AgentDecisionType.TOOL
            ),
            tool_call=tool_call,
        )
    )

    orchestrator = AgentOrchestrator(
        planner=planner,  # type: ignore[arg-type]
        executor=executor,
        rag_answerer=lambda _: RAGResponse(
            status="abstained",
            answer=None,
            sources=(),
            raw_output="INSUFFICIENT_EVIDENCE",
        ),
    )

    pending = orchestrator.handle(
        "Create production access."
    )

    assert pending.approval_request is not None

    with pytest.raises(
        ApprovalNotApprovedError
    ):
        orchestrator.resume(
            pending.approval_request.approval_id
        )

    assert tool.execute_calls == 0


def test_rejected_request_cannot_be_resumed() -> None:
    tool = FakeTool(
        "create_access_request",
        ToolRiskLevel.STATE_CHANGING,
    )

    executor, service = make_executor(
        tool
    )

    tool_call = ToolCall(
        tool_name="create_access_request",
        arguments={
            "value": "production",
        },
    )

    planner = FakePlanner(
        AgentDecision(
            decision_type=(
                AgentDecisionType.TOOL
            ),
            tool_call=tool_call,
        )
    )

    orchestrator = AgentOrchestrator(
        planner=planner,  # type: ignore[arg-type]
        executor=executor,
        rag_answerer=lambda _: RAGResponse(
            status="abstained",
            answer=None,
            sources=(),
            raw_output="INSUFFICIENT_EVIDENCE",
        ),
    )

    pending = orchestrator.handle(
        "Create production access."
    )

    assert pending.approval_request is not None

    approval_id = (
        pending.approval_request.approval_id
    )

    service.reject(
        approval_id
    )

    with pytest.raises(
        ApprovalNotApprovedError
    ):
        orchestrator.resume(
            approval_id
        )

    assert tool.execute_calls == 0