from dataclasses import dataclass
from typing import Protocol

from knowledgeops.agent.models import (
    ToolRiskLevel,
)


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    risk_level: ToolRiskLevel
    input_schema: dict[str, object]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError(
                "tool name cannot be empty"
            )

        if not self.description.strip():
            raise ValueError(
                "tool description cannot be empty"
            )

        schema_type = self.input_schema.get(
            "type"
        )

        if schema_type != "object":
            raise ValueError(
                "tool input_schema must "
                "describe an object"
            )


class Tool(Protocol):
    @property
    def definition(
        self,
    ) -> ToolDefinition:
        ...

    def validate_arguments(
        self,
        arguments: dict[str, object],
    ) -> None:
        ...

    def execute(
        self,
        arguments: dict[str, object],
    ) -> object:
        ...