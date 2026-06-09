from __future__ import annotations

from pydantic import BaseModel, Field


# --- Ingest ---


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunks_indexed: int
    message: str


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    chunks: int


# --- Query ---


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user question")
    company_id: str = Field(..., min_length=1, description="Tenant company ID")
    building_id: str | None = Field(default=None, description="Optional building filter")
    sites_id: str | None = Field(default=None, description="Optional site filter")
    top_k: int = Field(
        default=5, ge=1, le=50, description="Number of relevant chunks to retrieve"
    )


class SourceChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    document_id: str
    filename: str
    building_id: str | None = None
    sites_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    question: str


# --- Delete ---


class DeleteResponse(BaseModel):
    document_id: str
    chunks_removed: int
    message: str


# --- Health ---


class HealthResponse(BaseModel):
    status: str
    company_id: str | None = None
    index_loaded: bool
    total_chunks: int
    total_documents: int
    index_bit_width: int
