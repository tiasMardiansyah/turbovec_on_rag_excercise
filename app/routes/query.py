from __future__ import annotations

from fastapi import APIRouter, HTTPException
from openai import APIError

from app.models.schemas import QueryRequest, QueryResponse, SourceChunk
from app.services import llm
from app.services.database import database_service
from app.services.vectorstore import vector_store

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    stats = vector_store.get_company_stats(request.company_id)
    if stats["total_chunks"] == 0:
        return QueryResponse(
            answer="No documents have been indexed for this company. Please upload a document first.",
            sources=[],
            question=request.question,
        )

    # Hierarchical search via turbovec
    try:
        results = vector_store.search(
            company_id=request.company_id,
            query=request.question,
            top_k=request.top_k,
            building_id=request.building_id,
            sites_id=request.sites_id,
        )
    except APIError as e:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI embedding failed: {e.message}. "
            "Check your OPENAI_API_KEY in .env.",
        )

    if not results:
        return QueryResponse(
            answer="No relevant information found for your question.",
            sources=[],
            question=request.question,
        )

    # Extract chunk IDs from turbovec results
    chunk_ids = [
        doc.metadata.get("chunk_id", "") for doc, _score in results
    ]

    # Enrich from PostgreSQL
    db_chunks = await database_service.fetch_chunks(request.company_id, chunk_ids)
    db_by_id = {c.chunk_id: c for c in db_chunks}

    # Fetch document metadata for filenames
    doc_ids = list({c.document_id for c in db_chunks}) if db_chunks else []
    doc_filenames: dict[str, str] = {}
    for did in doc_ids:
        doc = await database_service.get_document(did)
        if doc:
            doc_filenames[did] = doc.filename

    # Build enriched context for LLM
    context_texts: list[str] = []
    enriched_sources: list[SourceChunk] = []

    for i, (doc, score) in enumerate(results):
        cid = doc.metadata.get("chunk_id", "")
        db_chunk = db_by_id.get(cid)

        text = db_chunk.text if db_chunk else doc.page_content
        context_texts.append(text)

        enriched_sources.append(
            SourceChunk(
                chunk_id=cid or str(i),
                text=text,
                score=float(score),
                document_id=db_chunk.document_id if db_chunk else doc.metadata.get("document_id", ""),
                filename=doc_filenames.get(db_chunk.document_id, "") if db_chunk else doc.metadata.get("filename", ""),
                building_id=db_chunk.building_id if db_chunk else doc.metadata.get("building_id"),
                sites_id=db_chunk.sites_id if db_chunk else doc.metadata.get("sites_id"),
            )
        )

    # Generate answer
    try:
        answer = llm.generate_answer(request.question, context_texts)
    except APIError as e:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI generation failed: {e.message}. "
            "Check your OPENAI_API_KEY in .env.",
        )

    return QueryResponse(
        answer=answer,
        sources=enriched_sources,
        question=request.question,
    )
