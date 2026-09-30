from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedPage:
    page_number: int | None
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    filename: str
    mime_type: str
    pages: list[ParsedPage]

    @property
    def text(self) -> str:
        return "\n\n".join(
            page.text
            for page in self.pages
            if page.text.strip()
        )