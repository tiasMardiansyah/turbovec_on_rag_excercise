from __future__ import annotations

import uuid

from fastapi import APIRouter, Form, HTTPException, Query, UploadFile
from openai import APIError

from app.models.schemas import (
    DeleteResponse,
    DocumentInfo,
    IngestResponse,
)
from app.services import chunking, document
from app.services.database import ChunkRow, database_service
from app.services.vectorstore import vector_store

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile,
    company_id: str = Form(..., min_length=1, description="Tenant company ID"),
    building_id: str | None = Form(None, description="Optional building ID"),
    sites_id: str | None = Form(None, description="Optional site ID"),
) -> IngestResponse:
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

    # Add to turbovec
    try:
        chunk_ids = vector_store.add_documents(
            company_id=company_id,
            doc_id=doc_id,
            filename=filename,
            documents=chunks,
            building_id=building_id,
            sites_id=sites_id,
        )
    except APIError as e:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI embedding failed: {e.message}. "
            "Check your OPENAI_API_KEY in .env.",
        )

    # Persist to PostgreSQL
    try:
        await database_service.ensure_company(company_id, company_id)
        await database_service.insert_document(
            document_id=doc_id,
            company_id=company_id,
            filename=filename,
            chunk_count=len(chunk_ids),
        )
        rows = [
            ChunkRow(
                chunk_id=cid,
                document_id=doc_id,
                company_id=company_id,
                chunk_index=i,
                text=chunks[i].page_content,
                building_id=building_id,
                sites_id=sites_id,
                metadata=chunks[i].metadata,
            )
            for i, cid in enumerate(chunk_ids)
        ]
        await database_service.batch_insert_chunks(rows)
    except Exception as e:
        # Rollback turbovec on DB failure
        vector_store.remove_document(company_id, doc_id)
        raise HTTPException(
            status_code=502,
            detail=f"Database write failed: {e}",
        )

    return IngestResponse(
        document_id=doc_id,
        filename=filename,
        chunks_indexed=len(chunk_ids),
        message=f"Successfully indexed {len(chunk_ids)} chunks",
    )


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents(
    company_id: str = Query(..., min_length=1, description="Tenant company ID"),
) -> list[DocumentInfo]:
    docs = vector_store.list_documents(company_id)
    return [DocumentInfo(**d) for d in docs]


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(
    document_id: str,
    company_id: str = Query(..., min_length=1, description="Tenant company ID"),
) -> DeleteResponse:
    removed = vector_store.remove_document(company_id, document_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete from DB (cascades to chunks)
    await database_service.delete_document(document_id)

    return DeleteResponse(
        document_id=document_id,
        chunks_removed=removed,
        message=f"Removed {removed} chunks",
    )


def _is_supported(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return f".{ext}" in document.SUPPORTED_EXTENSIONS
