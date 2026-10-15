from typing import Protocol


class LLMClient(Protocol):
    def generate(
        self,
        system: str,
        user: str,
    ) -> str:
        ...