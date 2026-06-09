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
    top_k: int = Field(
        default=5, ge=1, le=50, description="Number of relevant chunks to retrieve"
    )


class SourceChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    document_id: str
    filename: str


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
    index_loaded: bool
    total_chunks: int
    total_documents: int
    index_bit_width: int
