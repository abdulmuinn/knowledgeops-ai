import uuid

from knowledgeops.agent.approval_store import (
    ApprovalStore,
)
from knowledgeops.agent.models import (
    ApprovalRequest,
    ApprovalStatus,
)
from knowledgeops.agent.policy import (
    ApprovalPolicy,
)


class HumanApprovalService:
    def __init__(
        self,
        store: ApprovalStore,
        policy: ApprovalPolicy,
    ) -> None:
        self.store = store
        self.policy = policy

    def submit(
        self,
        request: ApprovalRequest,
    ) -> ApprovalRequest:
        if (
            request.status
            != ApprovalStatus.PENDING
        ):
            raise ValueError(
                "only pending approval requests "
                "can be submitted"
            )

        self.store.save(request)

        return request

    def get(
        self,
        approval_id: uuid.UUID,
    ) -> ApprovalRequest:
        return self.store.get(
            approval_id
        )

    def pending(
        self,
    ) -> tuple[
        ApprovalRequest,
        ...,
    ]:
        return self.store.pending()

    def approve(
        self,
        approval_id: uuid.UUID,
    ) -> ApprovalRequest:
        request = self.store.get(
            approval_id
        )

        approved = self.policy.approve(
            request
        )

        self.store.save(
            approved
        )

        return approved

    def reject(
        self,
        approval_id: uuid.UUID,
    ) -> ApprovalRequest:
        request = self.store.get(
            approval_id
        )

        rejected = self.policy.reject(
            request
        )

        self.store.save(
            rejected
        )

        return rejected