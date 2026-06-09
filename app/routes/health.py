from __future__ import annotations

from fastapi import APIRouter

from app.models.schemas import HealthResponse
from app.services.vectorstore import vector_store

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        index_loaded=True,
        total_chunks=vector_store.total_chunks,
        total_documents=vector_store.total_documents,
        index_bit_width=4,
    )
