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
    raise HTTPException(
        status_code=503,
        detail="agent service is not configured",
    )


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