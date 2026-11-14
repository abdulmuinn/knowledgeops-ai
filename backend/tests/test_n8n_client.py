import io
import json
import urllib.error
import urllib.request

import pytest

from knowledgeops.integrations.n8n import (
    N8NAccessRequestClient,
    N8NAccessRequestError,
)


class FakeResponse:
    def __init__(
        self,
        body: str,
    ) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> bool:
        return False

    def read(self) -> bytes:
        return self.body.encode(
            "utf-8"
        )


def test_blank_webhook_url_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="webhook_url",
    ):
        N8NAccessRequestClient(
            "   "
        )


def test_create_access_request_succeeds(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(
        request,
        timeout,
    ):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(
            request.data.decode("utf-8")
        )

        return FakeResponse(
            json.dumps(
                {
                    "request_id": "AR-TEST1234",
                    "system": "production",
                    "role": "read-only",
                    "status": "submitted",
                    "created_at": (
                        "2026-11-14T04:00:00Z"
                    ),
                }
            )
        )

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        fake_urlopen,
    )

    client = N8NAccessRequestClient(
        "http://localhost:5678/"
        "webhook/knowledgeops/access-request"
    )

    result = client.create_access_request(
        system="production",
        role="read-only",
    )

    assert captured["payload"] == {
        "system": "production",
        "role": "read-only",
    }

    assert result["request_id"] == (
        "AR-TEST1234"
    )
    assert result["status"] == "submitted"


def test_http_error_is_wrapped(
    monkeypatch,
) -> None:
    def fake_urlopen(
        request,
        timeout,
    ):
        raise urllib.error.HTTPError(
            url=request.full_url,
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=io.BytesIO(
                b'{"error":"workflow failed"}'
            ),
        )

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        fake_urlopen,
    )

    client = N8NAccessRequestClient(
        "http://localhost:5678/webhook/test"
    )

    with pytest.raises(
        N8NAccessRequestError,
        match="HTTP 500",
    ):
        client.create_access_request(
            system="production",
            role="read-only",
        )


def test_connection_error_is_wrapped(
    monkeypatch,
) -> None:
    def fake_urlopen(
        request,
        timeout,
    ):
        raise urllib.error.URLError(
            "connection refused"
        )

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        fake_urlopen,
    )

    client = N8NAccessRequestClient(
        "http://localhost:5678/webhook/test"
    )

    with pytest.raises(
        N8NAccessRequestError,
        match="could not reach n8n",
    ):
        client.create_access_request(
            system="production",
            role="read-only",
        )


def test_invalid_json_is_rejected(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: FakeResponse(
            "not-json"
        ),
    )

    client = N8NAccessRequestClient(
        "http://localhost:5678/webhook/test"
    )

    with pytest.raises(
        N8NAccessRequestError,
        match="invalid JSON",
    ):
        client.create_access_request(
            system="production",
            role="read-only",
        )


def test_missing_request_id_is_rejected(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: FakeResponse(
            json.dumps(
                {
                    "system": "production",
                    "role": "read-only",
                    "status": "submitted",
                }
            )
        ),
    )

    client = N8NAccessRequestClient(
        "http://localhost:5678/webhook/test"
    )

    with pytest.raises(
        N8NAccessRequestError,
        match="request_id",
    ):
        client.create_access_request(
            system="production",
            role="read-only",
        )


def test_invalid_status_is_rejected(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: FakeResponse(
            json.dumps(
                {
                    "request_id": "AR-TEST1234",
                    "system": "production",
                    "role": "read-only",
                    "status": "failed",
                }
            )
        ),
    )

    client = N8NAccessRequestClient(
        "http://localhost:5678/webhook/test"
    )

    with pytest.raises(
        N8NAccessRequestError,
        match="invalid status",
    ):
        client.create_access_request(
            system="production",
            role="read-only",
        )


def test_mismatched_system_is_rejected(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: FakeResponse(
            json.dumps(
                {
                    "request_id": "AR-TEST1234",
                    "system": "staging",
                    "role": "read-only",
                    "status": "submitted",
                }
            )
        ),
    )

    client = N8NAccessRequestClient(
        "http://localhost:5678/webhook/test"
    )

    with pytest.raises(
        N8NAccessRequestError,
        match="system does not match",
    ):
        client.create_access_request(
            system="production",
            role="read-only",
        )