from fastapi import FastAPI

from knowledgeops.api.routes.agent import (
    router as agent_router,
)
from knowledgeops.api.routes.health import (
    router as health_router,
)

from knowledgeops.api.routes.approvals import (
    router as approvals_router,
)

app = FastAPI(
    title="KnowledgeOps AI API",
    description="Backend API for KnowledgeOps AI",
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(agent_router)
app.include_router(approvals_router)