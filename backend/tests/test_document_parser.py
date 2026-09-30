from pathlib import Path

import pytest

from knowledgeops.ingestion.checksum import calculate_sha256
from knowledgeops.ingestion.parser import (
    UnsupportedDocumentTypeError,
    parse_document,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_parse_txt_document() -> None:
    file_path = FIXTURES_DIR / "sample.txt"

    document = parse_document(file_path)

    assert document.filename == "sample.txt"
    assert document.mime_type == "text/plain"
    assert len(document.pages) == 1

    assert "Production database access" in document.text


def test_checksum_is_deterministic() -> None:
    file_path = FIXTURES_DIR / "sample.txt"

    checksum_1 = calculate_sha256(file_path)
    checksum_2 = calculate_sha256(file_path)

    assert checksum_1 == checksum_2
    assert len(checksum_1) == 64


def test_unsupported_document_type() -> None:
    file_path = FIXTURES_DIR / "sample.csv"

    with pytest.raises(UnsupportedDocumentTypeError):
        parse_document(file_path)