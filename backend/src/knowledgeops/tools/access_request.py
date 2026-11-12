import uuid

from knowledgeops.agent.models import (
    ToolRiskLevel,
)
from knowledgeops.tools.base import (
    ToolDefinition,
)


class CreateAccessRequestTool:
    @property
    def definition(
        self,
    ) -> ToolDefinition:
        return ToolDefinition(
            name="create_access_request",
            description=(
                "Create an internal system access "
                "request for a user."
            ),
            risk_level=(
                ToolRiskLevel.STATE_CHANGING
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "system": {
                        "type": "string",
                    },
                    "role": {
                        "type": "string",
                    },
                },
                "required": [
                    "system",
                    "role",
                ],
            },
        )

    def validate_arguments(
        self,
        arguments: dict[str, object],
    ) -> None:
        system = arguments.get(
            "system"
        )
        role = arguments.get(
            "role"
        )

        if (
            not isinstance(system, str)
            or not system.strip()
        ):
            raise ValueError(
                "system must be a non-empty string"
            )

        if (
            not isinstance(role, str)
            or not role.strip()
        ):
            raise ValueError(
                "role must be a non-empty string"
            )

    def execute(
        self,
        arguments: dict[str, object],
    ) -> object:
        self.validate_arguments(
            arguments
        )

        return {
            "request_id": (
                f"AR-{uuid.uuid4().hex[:8].upper()}"
            ),
            "system": arguments["system"],
            "role": arguments["role"],
            "status": "submitted",
        }