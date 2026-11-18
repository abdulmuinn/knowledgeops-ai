FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app

RUN groupadd --system appgroup \
    && useradd \
        --system \
        --gid appgroup \
        --create-home \
        appuser

COPY pyproject.toml ./pyproject.toml
COPY alembic.ini ./alembic.ini
COPY migrations ./migrations
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch \
    && python -m pip install \
        --no-cache-dir \
        .

RUN mkdir -p /home/appuser/.cache/huggingface \
    && chown -R appuser:appgroup \
        /home/appuser \
        /app

USER appuser

EXPOSE 8000

HEALTHCHECK \
    --interval=10s \
    --timeout=3s \
    --start-period=30s \
    --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

CMD ["python", "-m", "uvicorn", "knowledgeops.main:app", "--host", "0.0.0.0", "--port", "8000"]