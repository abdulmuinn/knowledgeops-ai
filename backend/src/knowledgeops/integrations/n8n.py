import json
import urllib.error
import urllib.request
from typing import Any


class N8NAccessRequestError(RuntimeError):
    pass


class N8NAccessRequestClient:
    def __init__(
        self,
        webhook_url: str,
        timeout_seconds: float = 10.0,
    ) -> None:
        if not webhook_url.strip():
            raise ValueError(
                "webhook_url must not be empty"
            )

        self.webhook_url = webhook_url
        self.timeout_seconds = timeout_seconds

    def create_access_request(
        self,
        *,
        system: str,
        role: str,
    ) -> dict[str, Any]:
        payload = {
            "system": system,
            "role": role,
        }

        request = urllib.request.Request(
            self.webhook_url,
            data=json.dumps(
                payload
            ).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                raw_body = response.read().decode(
                    "utf-8"
                )

        except urllib.error.HTTPError as exc:
            body = exc.read().decode(
                "utf-8",
                errors="replace",
            )

            raise N8NAccessRequestError(
                f"n8n returned HTTP {exc.code}: "
                f"{body}"
            ) from exc

        except urllib.error.URLError as exc:
            raise N8NAccessRequestError(
                f"could not reach n8n: "
                f"{exc.reason}"
            ) from exc

        try:
            result = json.loads(
                raw_body
            )

        except json.JSONDecodeError as exc:
            raise N8NAccessRequestError(
                "n8n returned invalid JSON"
            ) from exc

        if not isinstance(
            result,
            dict,
        ):
            raise N8NAccessRequestError(
                "n8n response must be a JSON object"
            )

        request_id = result.get(
            "request_id"
        )

        if (
            not isinstance(request_id, str)
            or not request_id.strip()
        ):
            raise N8NAccessRequestError(
                "n8n response is missing request_id"
            )

        if result.get("status") != "submitted":
            raise N8NAccessRequestError(
                "n8n response has invalid status"
            )

        if result.get("system") != system:
            raise N8NAccessRequestError(
                "n8n response system does not match request"
            )

        if result.get("role") != role:
            raise N8NAccessRequestError(
                "n8n response role does not match request"
            )

        return result