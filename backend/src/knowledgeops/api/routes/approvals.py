import uuid

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from knowledgeops.agent.approval_service import (
    HumanApprovalService,
)
from knowledgeops.agent.approval_store import (
    ApprovalNotFoundError,
)
from knowledgeops.api.agent_mapper import (
    map_approval_request,
)
from knowledgeops.api.agent_models import (
    ApprovalResponse,
)


router = APIRouter(
    prefix="/approvals",
    tags=["approvals"],
)


def get_approval_service(
) -> HumanApprovalService:
    from knowledgeops.agent.runtime import (
        get_human_approval_service,
    )

    return get_human_approval_service()


@router.get(
    "",
    response_model=list[ApprovalResponse],
)
def list_pending_approvals(
    service: HumanApprovalService = Depends(
        get_approval_service
    ),
) -> list[ApprovalResponse]:
    return [
        map_approval_request(request)
        for request in service.pending()
    ]


@router.post(
    "/{approval_id}/approve",
    response_model=ApprovalResponse,
)
def approve_request(
    approval_id: uuid.UUID,
    service: HumanApprovalService = Depends(
        get_approval_service
    ),
) -> ApprovalResponse:
    try:
        request = service.approve(
            approval_id
        )

    except ApprovalNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="approval not found",
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    return map_approval_request(
        request
    )


@router.post(
    "/{approval_id}/reject",
    response_model=ApprovalResponse,
)
def reject_request(
    approval_id: uuid.UUID,
    service: HumanApprovalService = Depends(
        get_approval_service
    ),
) -> ApprovalResponse:
    try:
        request = service.reject(
            approval_id
        )

    except ApprovalNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="approval not found",
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    return map_approval_request(
        request
    )