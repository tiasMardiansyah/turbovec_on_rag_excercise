-- Multi-Company RAG Schema
-- PostgreSQL migration for turbovec-backed RAG with hierarchical metadata

BEGIN;

-- ─── Companies ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS companies (
    company_id      VARCHAR(64) PRIMARY KEY,
    name            VARCHAR(255) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ─── Documents ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS documents (
    document_id     VARCHAR(128) PRIMARY KEY,
    company_id      VARCHAR(64) NOT NULL REFERENCES companies(company_id),
    filename        VARCHAR(512) NOT NULL,
    file_url        TEXT,
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_company ON documents(company_id);

-- ─── Chunks ───────────────────────────────────────────────────────────────────
-- chunk_id format: "{doc_id}__{chunk_index}" for deterministic, readable IDs

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        VARCHAR(200) PRIMARY KEY,
    document_id     VARCHAR(128) NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    company_id      VARCHAR(64) NOT NULL REFERENCES companies(company_id),
    chunk_index     INTEGER NOT NULL,
    text            TEXT NOT NULL,
    building_id     VARCHAR(64),
    sites_id        VARCHAR(64),
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Hierarchical search indexes (company > building > sites)
CREATE INDEX IF NOT EXISTS idx_chunks_company ON chunks(company_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_building ON chunks(company_id, building_id);
CREATE INDEX IF NOT EXISTS idx_chunks_sites ON chunks(company_id, building_id, sites_id);
CREATE INDEX IF NOT EXISTS idx_chunks_metadata ON chunks USING GIN(metadata);

COMMIT;
