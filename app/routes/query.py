from __future__ import annotations

from fastapi import APIRouter, HTTPException
from openai import APIError

from app.models.schemas import QueryRequest, QueryResponse, SourceChunk
from app.services import llm
from app.services.vectorstore import vector_store

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    if vector_store.total_chunks == 0:
        return QueryResponse(
            answer="No documents have been indexed yet. Please upload a document first.",
            sources=[],
            question=request.question,
        )

    try:
        results = vector_store.search(request.question, top_k=request.top_k)
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

    # results is List[Tuple[Document, float]]
    context_texts = [doc.page_content for doc, _score in results]

    try:
        answer = llm.generate_answer(request.question, context_texts)
    except APIError as e:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI generation failed: {e.message}. "
            "Check your OPENAI_API_KEY in .env.",
        )

    sources = [
        SourceChunk(
            chunk_id=str(doc.metadata.get("chunk_id", i)),
            text=doc.page_content,
            score=float(score),
            document_id=doc.metadata.get("document_id", ""),
            filename=doc.metadata.get("filename", ""),
        )
        for i, (doc, score) in enumerate(results)
    ]

    return QueryResponse(
        answer=answer,
        sources=sources,
        question=request.question,
    )
