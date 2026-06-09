"""Async PostgreSQL service for multi-company RAG metadata storage."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import asyncpg

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ChunkRow:
    """A single chunk row from the database."""

    chunk_id: str
    document_id: str
    company_id: str
    chunk_index: int
    text: str
    building_id: str | None
    sites_id: str | None
    metadata: dict[str, Any]


@dataclass
class DocumentRow:
    """A single document row from the database."""

    document_id: str
    company_id: str
    filename: str
    file_url: str | None
    chunk_count: int


class DatabaseService:
    """Manages asyncpg connection pool for RAG metadata."""

    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Database pool not initialized. Call init_pool() first.")
        return self._pool

    async def init_pool(self) -> None:
        """Create the connection pool from settings."""
        self._pool = await asyncpg.create_pool(
            settings.postgres_dsn,
            min_size=2,
            max_size=10,
        )
        logger.info("PostgreSQL connection pool created")

    async def close_pool(self) -> None:
        """Gracefully close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL connection pool closed")

    async def ensure_company(self, company_id: str, name: str) -> None:
        """Insert company if it doesn't exist."""
        await self.pool.execute(
            """
            INSERT INTO companies (company_id, name)
            VALUES ($1, $2)
            ON CONFLICT (company_id) DO NOTHING
            """,
            company_id,
            name,
        )

    async def insert_document(
        self,
        document_id: str,
        company_id: str,
        filename: str,
        file_url: str | None = None,
        chunk_count: int = 0,
    ) -> None:
        """Insert a document record."""
        await self.pool.execute(
            """
            INSERT INTO documents (document_id, company_id, filename, file_url, chunk_count)
            VALUES ($1, $2, $3, $4, $5)
            """,
            document_id,
            company_id,
            filename,
            file_url,
            chunk_count,
        )

    async def update_document_chunk_count(self, document_id: str, chunk_count: int) -> None:
        """Update the chunk_count on a document."""
        await self.pool.execute(
            "UPDATE documents SET chunk_count = $2 WHERE document_id = $1",
            document_id,
            chunk_count,
        )

    async def batch_insert_chunks(self, rows: list[ChunkRow]) -> None:
        """Bulk insert chunks using COPY for maximum throughput."""
        if not rows:
            return

        records = [
            (
                r.chunk_id,
                r.document_id,
                r.company_id,
                r.chunk_index,
                r.text,
                r.building_id,
                r.sites_id,
                json.dumps(r.metadata),
            )
            for r in rows
        ]

        async with self.pool.acquire() as conn:
            await conn.copy_records_to_table(
                "chunks",
                records=records,
                columns=[
                    "chunk_id",
                    "document_id",
                    "company_id",
                    "chunk_index",
                    "text",
                    "building_id",
                    "sites_id",
                    "metadata",
                ],
            )

    async def fetch_chunks(
        self, company_id: str, chunk_ids: list[str]
    ) -> list[ChunkRow]:
        """Fetch chunks by company and chunk IDs for search result enrichment."""
        if not chunk_ids:
            return []

        rows = await self.pool.fetch(
            """
            SELECT chunk_id, document_id, company_id, chunk_index,
                   text, building_id, sites_id, metadata
            FROM chunks
            WHERE company_id = $1 AND chunk_id = ANY($2::varchar[])
            """,
            company_id,
            chunk_ids,
        )
        return [_row_to_chunk(r) for r in rows]

    async def fetch_all_chunk_ids(self, company_id: str) -> list[str]:
        """Get all chunk IDs for a company (used for allowlist rebuild)."""
        rows = await self.pool.fetch(
            "SELECT chunk_id FROM chunks WHERE company_id = $1",
            company_id,
        )
        return [r["chunk_id"] for r in rows]

    async def delete_document(self, document_id: str) -> int:
        """Delete a document (cascades to chunks). Returns deleted chunk count."""
        chunk_count = await self.pool.fetchval(
            "SELECT COUNT(*) FROM chunks WHERE document_id = $1",
            document_id,
        )
        await self.pool.execute(
            "DELETE FROM documents WHERE document_id = $1",
            document_id,
        )
        return chunk_count or 0

    async def get_document(self, document_id: str) -> DocumentRow | None:
        """Fetch a single document by ID."""
        row = await self.pool.fetchrow(
            """
            SELECT document_id, company_id, filename, file_url, chunk_count
            FROM documents
            WHERE document_id = $1
            """,
            document_id,
        )
        if row is None:
            return None
        return DocumentRow(
            document_id=row["document_id"],
            company_id=row["company_id"],
            filename=row["filename"],
            file_url=row["file_url"],
            chunk_count=row["chunk_count"],
        )

    async def list_documents(self, company_id: str) -> list[DocumentRow]:
        """List all documents for a company."""
        rows = await self.pool.fetch(
            """
            SELECT document_id, company_id, filename, file_url, chunk_count
            FROM documents
            WHERE company_id = $1
            ORDER BY created_at DESC
            """,
            company_id,
        )
        return [
            DocumentRow(
                document_id=r["document_id"],
                company_id=r["company_id"],
                filename=r["filename"],
                file_url=r["file_url"],
                chunk_count=r["chunk_count"],
            )
            for r in rows
        ]

    async def get_company_stats(self, company_id: str) -> dict[str, int]:
        """Get document and chunk counts for a company."""
        row = await self.pool.fetchrow(
            """
            SELECT
                COUNT(DISTINCT d.document_id) AS total_documents,
                COALESCE(SUM(d.chunk_count), 0) AS total_chunks
            FROM documents d
            WHERE d.company_id = $1
            """,
            company_id,
        )
        return {
            "total_documents": row["total_documents"] if row else 0,
            "total_chunks": row["total_chunks"] if row else 0,
        }


def _row_to_chunk(r: asyncpg.Record) -> ChunkRow:
    """Convert an asyncpg Record to a ChunkRow."""
    meta = r["metadata"]
    if isinstance(meta, str):
        meta = json.loads(meta)
    return ChunkRow(
        chunk_id=r["chunk_id"],
        document_id=r["document_id"],
        company_id=r["company_id"],
        chunk_index=r["chunk_index"],
        text=r["text"],
        building_id=r["building_id"],
        sites_id=r["sites_id"],
        metadata=meta if isinstance(meta, dict) else {},
    )


database_service = DatabaseService()
