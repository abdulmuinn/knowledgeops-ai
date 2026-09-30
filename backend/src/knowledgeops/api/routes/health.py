from fastapi import APIRouter, HTTPException
from sqlalchemy.exc import SQLAlchemyError

from knowledgeops.db.session import check_database_connection


router = APIRouter(tags=["System"])


@router.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "knowledgeops-api",
        "version": "0.1.0",
    }


@router.get("/health/db")
def database_health_check() -> dict[str, str]:
    try:
        check_database_connection()
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="database unavailable",
        ) from exc

    return {
        "status": "healthy",
        "database": "postgresql",
    }