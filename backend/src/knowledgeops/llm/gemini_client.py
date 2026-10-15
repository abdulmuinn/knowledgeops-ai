from typing import Any


DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


class GeminiClient:
    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        client: Any | None = None,
        temperature: float = 0.1,
    ) -> None:
        self.model = model
        self._client = client
        self.temperature = temperature

    @property
    def client(self) -> Any:
        if self._client is None:
            from google import genai

            self._client = genai.Client()

        return self._client

    def generate(
        self,
        system: str,
        user: str,
    ) -> str:
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.temperature,
            ),
        )

        return response.text or ""