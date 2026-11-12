import uuid

from knowledgeops.agent.approval_store import (
    ApprovalNotFoundError,
)
from knowledgeops.agent.policy import (
    ApprovalNotApprovedError,
)

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from knowledgeops.agent.orchestrator import (
    AgentOrchestrator,
)
from knowledgeops.api.agent_mapper import (
    map_agent_run_result,
)
from knowledgeops.api.agent_models import (
    AgentRunRequest,
    AgentRunResponse,
)


router = APIRouter(
    prefix="/agent",
    tags=["agent"],
)


def get_agent_orchestrator() -> AgentOrchestrator:
    from knowledgeops.agent.runtime import (
        get_agent_orchestrator as get_runtime_orchestrator,
    )

    return get_runtime_orchestrator()


@router.post(
    "/run",
    response_model=AgentRunResponse,
)
def run_agent(
    request: AgentRunRequest,
    orchestrator: AgentOrchestrator = Depends(
        get_agent_orchestrator
    ),
) -> AgentRunResponse:
    result = orchestrator.handle(
        request.message
    )

    return map_agent_run_result(
        result
    )

@router.post(
    "/resume/{approval_id}",
    response_model=AgentRunResponse,
)
def resume_agent(
    approval_id: uuid.UUID,
    orchestrator: AgentOrchestrator = Depends(
        get_agent_orchestrator
    ),
) -> AgentRunResponse:
    try:
        result = orchestrator.resume(
            approval_id
        )

    except ApprovalNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="approval not found",
        ) from exc

    except ApprovalNotApprovedError as exc:
        raise HTTPException(
            status_code=409,
            detail="approval is not approved",
        ) from exc

    return map_agent_run_result(
        result
    )