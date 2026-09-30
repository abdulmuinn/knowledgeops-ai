from pathlib import Path

from pypdf import PdfReader

from knowledgeops.ingestion.models import (
    ParsedDocument,
    ParsedPage,
)


SUPPORTED_EXTENSIONS = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
}


class UnsupportedDocumentTypeError(ValueError):
    pass


def parse_document(file_path: Path) -> ParsedDocument:
    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentTypeError(
            f"Unsupported document type: {suffix}"
        )

    if suffix == ".pdf":
        return _parse_pdf(file_path)

    return _parse_text_document(
        file_path=file_path,
        mime_type=SUPPORTED_EXTENSIONS[suffix],
    )


def _parse_pdf(file_path: Path) -> ParsedDocument:
    reader = PdfReader(file_path)

    pages: list[ParsedPage] = []

    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        pages.append(
            ParsedPage(
                page_number=index,
                text=_normalize_text(text),
            )
        )

    return ParsedDocument(
        filename=file_path.name,
        mime_type="application/pdf",
        pages=pages,
    )


def _parse_text_document(
    file_path: Path,
    mime_type: str,
) -> ParsedDocument:
    text = file_path.read_text(
        encoding="utf-8",
    )

    return ParsedDocument(
        filename=file_path.name,
        mime_type=mime_type,
        pages=[
            ParsedPage(
                page_number=None,
                text=_normalize_text(text),
            )
        ],
    )


def _normalize_text(text: str) -> str:
    lines = (
        line.strip()
        for line in text.splitlines()
    )

    return "\n".join(
        line
        for line in lines
        if line
    )