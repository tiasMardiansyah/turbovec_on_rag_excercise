from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes import health, ingest, query


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly load the vector store (triggers index load from disk)
    from app.services.vectorstore import vector_store

    print(
        f"Index ready: {vector_store.total_chunks} chunks, "
        f"{vector_store.total_documents} documents"
    )
    yield


app = FastAPI(
    title="RAG API — LangChain + turbovec + OpenAI",
    description=(
        "A Retrieval-Augmented Generation template using "
        "FastAPI, LangChain orchestration, turbovec for vector search, "
        "and OpenAI for embeddings + LLM."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(health.router, prefix="/api/v1")
app.include_router(ingest.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
