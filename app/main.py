from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes import health, ingest, query


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize PostgreSQL connection pool
    from app.services.database import database_service

    await database_service.init_pool()
    print("PostgreSQL connection pool initialized")

    # Initialize vector store (start background flush thread)
    from app.services.vectorstore import vector_store

    vector_store.start_flush_thread()
    print(
        f"Vector store ready: {len(vector_store._stores)} company stores loaded, "
        f"flush interval={settings.store_flush_interval}s"
    )

    yield

    # Shutdown: flush dirty stores and close DB pool
    vector_store.stop_flush_thread()
    await database_service.close_pool()
    print("Shutdown complete: stores flushed, DB pool closed")


app = FastAPI(
    title="RAG API — LangChain + turbovec + OpenAI",
    description=(
        "A Retrieval-Augmented Generation template using "
        "FastAPI, LangChain orchestration, turbovec for vector search, "
        "and OpenAI for embeddings + LLM. Supports multi-company isolation "
        "with hierarchical search (sitesId > buildingId > companyId)."
    ),
    version="3.0.0",
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
