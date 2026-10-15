import os

from google import genai


MODELS = [
    os.getenv(
        "GEMINI_MODEL",
        "gemini-3.5-flash-lite",
    ),
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
]


def main() -> None:
    client = genai.Client()

    models = list(
        dict.fromkeys(MODELS)
    )

    for model in models:
        print()
        print("=" * 50)
        print(f"Testing: {model}")
        print("=" * 50)

        try:
            response = (
                client.models.generate_content(
                    model=model,
                    contents=(
                        "Reply with exactly: OK"
                    ),
                )
            )

            print(
                f"SUCCESS: {model}"
            )

            print(
                f"Response: {response.text}"
            )

            return

        except Exception as exc:
            print(
                f"FAILED: {model}"
            )

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    raise RuntimeError(
        "All Gemini models failed"
    )


if __name__ == "__main__":
    main()