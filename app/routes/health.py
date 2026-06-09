from __future__ import annotations

from fastapi import APIRouter, Query

from app.models.schemas import HealthResponse
from app.services.database import database_service
from app.services.vectorstore import vector_store
from app.config import settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    company_id: str | None = Query(None, description="Optional company ID for per-company stats"),
) -> HealthResponse:
    db_ok = database_service._pool is not None

    if company_id:
        stats = vector_store.get_company_stats(company_id)
        return HealthResponse(
            status="ok" if db_ok else "degraded",
            company_id=company_id,
            index_loaded=True,
            total_chunks=stats["total_chunks"],
            total_documents=stats["total_documents"],
            index_bit_width=settings.turbovec_bit_width,
        )

    # Global stats across all loaded stores
    total_chunks = 0
    total_documents = 0
    for cid in list(vector_store._stores.keys()):
        stats = vector_store.get_company_stats(cid)
        total_chunks += stats["total_chunks"]
        total_documents += stats["total_documents"]

    return HealthResponse(
        status="ok" if db_ok else "degraded",
        company_id=None,
        index_loaded=True,
        total_chunks=total_chunks,
        total_documents=total_documents,
        index_bit_width=settings.turbovec_bit_width,
    )
