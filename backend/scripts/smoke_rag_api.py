import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


BASE_URL = "http://127.0.0.1:8000"
LOG_PATH = Path("/tmp/knowledgeops-rag-api.log")


def prepare_environment() -> dict[str, str]:
    env = os.environ.copy()

    database_url = env.get(
        "DATABASE_URL"
    )

    if database_url:
        database_url = database_url.replace(
            "@localhost:",
            "@host.docker.internal:",
        )

        database_url = database_url.replace(
            "@127.0.0.1:",
            "@host.docker.internal:",
        )

        env["DATABASE_URL"] = database_url

    return env


def wait_for_api(
    timeout_seconds: int = 30,
) -> None:
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"{BASE_URL}/health",
                timeout=2,
            ) as response:
                if response.status == 200:
                    return

        except Exception:
            time.sleep(1)

    raise RuntimeError(
        "API did not become healthy"
    )


def post_json(
    path: str,
    payload: dict[str, object],
) -> dict[str, object]:
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
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
            timeout=180,
        ) as response:
            body = response.read().decode(
                "utf-8"
            )

    except urllib.error.HTTPError as exc:
        body = exc.read().decode(
            "utf-8"
        )

        raise RuntimeError(
            f"HTTP {exc.code}: {body}"
        ) from exc

    parsed = json.loads(body)

    if not isinstance(
        parsed,
        dict,
    ):
        raise RuntimeError(
            "API returned non-object JSON"
        )

    return parsed


def main() -> None:
    env = prepare_environment()

    with LOG_PATH.open(
        "w",
        encoding="utf-8",
    ) as log_file:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "knowledgeops.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

        try:
            wait_for_api()

            print("=" * 60)
            print("RAG HTTP SMOKE")
            print("=" * 60)

            response = post_json(
                "/agent/run",
                {
                    "message": (
                        "Who must approve standard "
                        "employee access to "
                        "production systems?"
                    )
                },
            )

            print(
                json.dumps(
                    response,
                    indent=2,
                )
            )

            if (
                response.get("status")
                != "rag_completed"
            ):
                raise RuntimeError(
                    "expected rag_completed"
                )

            answer = response.get(
                "answer"
            )

            if (
                not isinstance(answer, str)
                or not answer.strip()
            ):
                raise RuntimeError(
                    "RAG answer is missing"
                )

            sources = response.get(
                "sources"
            )

            if (
                not isinstance(sources, list)
                or not sources
            ):
                raise RuntimeError(
                    "RAG sources are missing"
                )

            print()
            print("=" * 60)
            print("SUCCESS")
            print("=" * 60)
            print(
                "Real RAG HTTP flow completed "
                "successfully."
            )

        finally:
            process.terminate()

            try:
                process.wait(
                    timeout=10
                )

            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    try:
        main()

    except Exception:
        print()
        print("==== API LOG ====")

        if LOG_PATH.exists():
            print(
                LOG_PATH.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            )

        raise