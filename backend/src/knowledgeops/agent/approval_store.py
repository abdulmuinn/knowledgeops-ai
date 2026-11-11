import uuid

from knowledgeops.agent.models import (
    ApprovalRequest,
    ApprovalStatus,
)


class ApprovalNotFoundError(
    KeyError
):
    pass


class ApprovalStore:
    def __init__(self) -> None:
        self._requests: dict[
            uuid.UUID,
            ApprovalRequest,
        ] = {}

    def save(
        self,
        request: ApprovalRequest,
    ) -> None:
        self._requests[
            request.approval_id
        ] = request

    def get(
        self,
        approval_id: uuid.UUID,
    ) -> ApprovalRequest:
        try:
            return self._requests[
                approval_id
            ]

        except KeyError:
            raise ApprovalNotFoundError(
                f"approval not found: "
                f"{approval_id}"
            ) from None

    def pending(
        self,
    ) -> tuple[
        ApprovalRequest,
        ...,
    ]:
        return tuple(
            request
            for request
            in self._requests.values()
            if (
                request.status
                == ApprovalStatus.PENDING
            )
        )

    def __len__(
        self,
    ) -> int:
        return len(
            self._requests
        )