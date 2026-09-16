from fastapi import FastAPI

app = FastAPI(
    title="KnowledgeOps AI API",
    description="Backend API for KnowledgeOps AI",
    version="0.1.0",
)


@app.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "knowledgeops-api",
        "version": "0.1.0",
    }