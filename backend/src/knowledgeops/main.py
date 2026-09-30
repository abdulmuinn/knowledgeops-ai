from fastapi import FastAPI

from knowledgeops.api.routes.health import router as health_router


app = FastAPI(
    title="KnowledgeOps AI API",
    description="Backend API for KnowledgeOps AI",
    version="0.1.0",
)

app.include_router(health_router)