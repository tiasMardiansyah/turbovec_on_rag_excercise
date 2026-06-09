from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, UploadFile
from openai import APIError

from app.models.schemas import (
    DeleteResponse,
    DocumentInfo,
    IngestResponse,
)
from app.services import chunking, document
from app.services.vectorstore import vector_store

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse)
async def ingest_document(file: UploadFile) -> IngestResponse:
    filename = file.filename or "unknown"
    if not _is_supported(filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Accepted: {document.SUPPORTED_EXTENSIONS}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        docs = document.load_document(content, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not docs or not any(d.page_content.strip() for d in docs):
        raise HTTPException(
            status_code=422, detail="No text could be extracted from the file"
        )

    chunks = chunking.split_documents(docs)
    if not chunks:
        raise HTTPException(
            status_code=422, detail="Document produced no processable chunks"
        )

    doc_id = uuid.uuid4().hex[:12]

    try:
        n_chunks = vector_store.add_documents(doc_id, filename, chunks)
    except APIError as e:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI embedding failed: {e.message}. "
            "Check your OPENAI_API_KEY in .env.",
        )

    return IngestResponse(
        document_id=doc_id,
        filename=filename,
        chunks_indexed=n_chunks,
        message=f"Successfully indexed {n_chunks} chunks",
    )


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents() -> list[DocumentInfo]:
    docs = vector_store.list_documents()
    return [DocumentInfo(**d) for d in docs]


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str) -> DeleteResponse:
    removed = vector_store.remove_document(document_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return DeleteResponse(
        document_id=document_id,
        chunks_removed=removed,
        message=f"Removed {removed} chunks",
    )


def _is_supported(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return f".{ext}" in document.SUPPORTED_EXTENSIONS
