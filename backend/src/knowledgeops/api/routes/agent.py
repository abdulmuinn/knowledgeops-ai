from fastapi import (
    APIRouter,
    Depends,
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