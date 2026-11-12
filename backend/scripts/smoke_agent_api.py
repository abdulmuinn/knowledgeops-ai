import json
import os
import urllib.error
import urllib.request


BASE_URL = os.getenv(
    "KNOWLEDGEOPS_API_URL",
    "http://127.0.0.1:8000",
)


def request_json(
    method: str,
    path: str,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    data = None

    if payload is not None:
        data = json.dumps(
            payload
        ).encode("utf-8")

    request = urllib.request.Request(
        url=f"{BASE_URL}{path}",
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=60,
        ) as response:
            body = response.read().decode(
                "utf-8"
            )

    except urllib.error.HTTPError as exc:
        body = exc.read().decode(
            "utf-8"
        )

        raise RuntimeError(
            f"{method} {path} failed "
            f"with HTTP {exc.code}: {body}"
        ) from exc

    parsed = json.loads(body)

    if not isinstance(
        parsed,
        dict,
    ):
        raise RuntimeError(
            f"{method} {path} returned "
            "a non-object JSON response"
        )

    return parsed


def main() -> None:
    print(
        f"KnowledgeOps API: {BASE_URL}"
    )

    print()
    print("=" * 60)
    print("1. CREATE ACTION")
    print("=" * 60)

    run_response = request_json(
        "POST",
        "/agent/run",
        {
            "message": (
                "Create a read-only production "
                "access request for me."
            )
        },
    )

    print(
        json.dumps(
            run_response,
            indent=2,
        )
    )

    if (
        run_response.get("status")
        != "approval_required"
    ):
        raise RuntimeError(
            "expected approval_required"
        )

    approval = run_response.get(
        "approval"
    )

    if not isinstance(
        approval,
        dict,
    ):
        raise RuntimeError(
            "approval object missing"
        )

    approval_id = approval.get(
        "approval_id"
    )

    if not isinstance(
        approval_id,
        str,
    ):
        raise RuntimeError(
            "approval_id missing"
        )

    print()
    print("=" * 60)
    print("2. APPROVE")
    print("=" * 60)

    approve_response = request_json(
        "POST",
        (
            f"/approvals/"
            f"{approval_id}/approve"
        ),
    )

    print(
        json.dumps(
            approve_response,
            indent=2,
        )
    )

    if (
        approve_response.get("status")
        != "approved"
    ):
        raise RuntimeError(
            "approval was not approved"
        )

    print()
    print("=" * 60)
    print("3. RESUME")
    print("=" * 60)

    resume_response = request_json(
        "POST",
        (
            f"/agent/resume/"
            f"{approval_id}"
        ),
    )

    print(
        json.dumps(
            resume_response,
            indent=2,
        )
    )

    if (
        resume_response.get("status")
        != "tool_completed"
    ):
        raise RuntimeError(
            "expected tool_completed"
        )

    tool_result = resume_response.get(
        "tool_result"
    )

    if not isinstance(
        tool_result,
        dict,
    ):
        raise RuntimeError(
            "tool_result missing"
        )

    if (
        tool_result.get("status")
        != "succeeded"
    ):
        raise RuntimeError(
            "tool did not succeed"
        )

    output = tool_result.get(
        "output"
    )

    if not isinstance(
        output,
        dict,
    ):
        raise RuntimeError(
            "tool output missing"
        )

    if (
        output.get("status")
        != "submitted"
    ):
        raise RuntimeError(
            "access request was not submitted"
        )

    print()
    print("=" * 60)
    print("SUCCESS")
    print("=" * 60)
    print(
        "Human approval HTTP flow "
        "completed successfully."
    )


if __name__ == "__main__":
    main()