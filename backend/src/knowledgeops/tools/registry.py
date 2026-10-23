from collections.abc import Iterable

from knowledgeops.tools.base import (
    Tool,
    ToolDefinition,
)


class ToolRegistry:
    def __init__(
        self,
        tools: Iterable[Tool] | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}

        if tools is not None:
            for tool in tools:
                self.register(tool)

    def register(
        self,
        tool: Tool,
    ) -> None:
        name = tool.definition.name

        if name in self._tools:
            raise ValueError(
                f"tool already registered: {name}"
            )

        self._tools[name] = tool

    def get(
        self,
        name: str,
    ) -> Tool:
        name = name.strip()

        if not name:
            raise ValueError(
                "tool name cannot be empty"
            )

        try:
            return self._tools[name]

        except KeyError:
            raise KeyError(
                f"tool not registered: {name}"
            ) from None

    def contains(
        self,
        name: str,
    ) -> bool:
        return name in self._tools

    def definitions(
        self,
    ) -> tuple[ToolDefinition, ...]:
        return tuple(
            self._tools[name].definition
            for name in sorted(
                self._tools
            )
        )

    def __len__(self) -> int:
        return len(
            self._tools
        )