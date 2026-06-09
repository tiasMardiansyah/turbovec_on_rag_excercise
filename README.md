# RAG API — turbovec + PostgreSQL + OpenAI

Multi-tenant Retrieval-Augmented Generation service built with FastAPI, [turbovec](https://pypi.org/project/turbovec/) for quantized vector search, PostgreSQL for metadata storage, and OpenAI for embeddings + LLM.

Supports **multi-company isolation** with **hierarchical search** (sitesId > buildingId > companyId) — designed as a drop-in replacement for ChromaDB-based RAG pipelines.

## Architecture

```
┌─ FastAPI ────────────────────────────────────────────────────┐
│                                                              │
│  POST /api/v1/ingest                                         │
│    file + company_id + building_id? + sites_id?              │
│    → chunk → embed → turbovec add + PostgreSQL insert        │
│                                                              │
│  POST /api/v1/query                                          │
│    question + company_id + building_id? + sites_id? + top_k  │
│    → hierarchical turbovec search → DB enrichment → LLM      │
│                                                              │
│  GET  /api/v1/health  (?company_id)                          │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  VectorStoreService                                    │  │
│  │  Per-company .tvim files (lazy-loaded, LRU eviction)   │  │
│  │  Pre-built allowlist caches for hierarchical search    │  │
│  │  Background flush thread for .tvim persistence         │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  DatabaseService (asyncpg)                             │  │
│  │  PostgreSQL: companies / documents / chunks            │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Why turbovec?

Turbovec quantizes embeddings to 2-4 bits per dimension (~8x compression) and searches with SIMD kernels — no network hop to a vector database server. For multi-tenant RAG with hierarchical filtering, this eliminates the latency of repeated ChromaDB/external DB round trips.

## Prerequisites

- Python 3.10+
- PostgreSQL 14+
- OpenAI API key

## Setup

### 1. PostgreSQL

Create a database and run the migration:

```bash
createdb rag
psql -d rag -f migrations/001_initial.sql
```

This creates three tables:

```
companies  → company_id (PK), name, created_at
documents  → document_id (PK), company_id (FK), filename, file_url, chunk_count
chunks     → chunk_id (PK, "{doc_id}__{i}"), document_id (FK CASCADE),
              company_id (FK), chunk_index, text, building_id, sites_id, metadata (JSONB)
```

With indexes optimized for hierarchical lookups: `(company_id)`, `(company_id, building_id)`, `(company_id, building_id, sites_id)`, `GIN(metadata)`.

### 2. Environment

```bash
cp .env.example .env
```

Edit `.env` with your values:

```env
OPENAI_API_KEY=sk-your-key
POSTGRES_DSN=postgresql://user:pass@localhost:5432/rag
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run

```bash
python -m app.main
```

Server starts at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

## API Reference

### Ingest a document

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@report.pdf" \
  -F "company_id=acme-corp" \
  -F "building_id=hq-east" \
  -F "sites_id=floor-3"
```

`building_id` and `sites_id` are optional. The document is chunked, embedded, indexed in turbovec, and stored in PostgreSQL.

### Query (hierarchical search)

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the emergency evacuation procedure?",
    "company_id": "acme-corp",
    "building_id": "hq-east",
    "sites_id": "floor-3",
    "top_k": 8
  }'
```

The hierarchical search priority is:

1. **sites-specific**: `companyId + buildingId + sitesId` — if ≥ k-2 results found, return them
2. **building-specific**: `companyId + buildingId` — combine with sites results, deduplicated
3. **company-wide**: `companyId` only — supplement if combined results < 2, excluding chunks already scoped to a building

If `building_id` and `sites_id` are omitted, searches all company documents.

### List documents

```bash
curl "http://localhost:8000/api/v1/ingest/documents?company_id=acme-corp"
```

### Delete a document

```bash
curl -X DELETE "http://localhost:8000/api/v1/ingest/documents/{document_id}?company_id=acme-corp"
```

Removes from both turbovec and PostgreSQL (chunks cascade-delete via FK).

### Health check

```bash
# Global
curl http://localhost:8000/api/v1/health

# Per-company
curl "http://localhost:8000/api/v1/health?company_id=acme-corp"
```

## Configuration

All settings via `.env` or environment variables:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | `sk-placeholder` | OpenAI API key |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `OPENAI_LLM_MODEL` | `gpt-4o-mini` | Chat completion model |
| `TURBOVEC_BIT_WIDTH` | `4` | Quantization width (2 or 4 bits) |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `DEFAULT_TOP_K` | `5` | Default number of results |
| `POSTGRES_DSN` | `postgresql://user:pass@localhost:5432/rag` | PostgreSQL connection string |
| `STORE_FLUSH_INTERVAL` | `30` | Seconds between .tvim flushes to disk |
| `MAX_STORES_IN_MEMORY` | `100` | Max company stores in memory (LRU eviction) |
| `INDEX_DIR` | `./data/index` | Directory for per-company .tvim files |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |

## How It Works

### Data Flow — Ingest

```
PDF/DOCX upload
    → extract text (PyPDFLoader / Docx2txtLoader)
    → split into chunks (RecursiveCharacterTextSplitter)
    → embed chunks (OpenAI text-embedding-3-small)
    → add to per-company turbovec index (in-memory, quantized)
    → mark .tvim as dirty (background flush every 30s)
    → insert metadata to PostgreSQL (companies, documents, chunks)
```

### Data Flow — Query

```
User question + company_id + building_id? + sites_id?
    → embed question (OpenAI)
    → hierarchical search in turbovec:
        1. resolve pre-built allowlist (O(1) dict lookup, not O(N) scan)
        2. SIMD search restricted to allowlist handles
        3. fallback through hierarchy if results are sparse
    → return chunk IDs + scores
    → enrich from PostgreSQL (text, filename, metadata)
    → build context for LLM
    → generate answer (gpt-4o-mini)
```

### Per-Company Isolation

Each company gets its own `.tvim` index file:

```
data/index/
  acme-corp/
    store/
      index.tvim          ← quantized vectors
      docstore.json       ← text/metadata side-car
  other-corp/
    store/
      index.tvim
      docstore.json
```

Stores are lazy-loaded on first access and evicted via LRU when `MAX_STORES_IN_MEMORY` is exceeded. Each company has its own threading lock — no cross-company contention.

### Pre-Built Allowlist Caches

Every `CompanyStore` maintains cached handle sets for O(1) filter resolution:

```python
company_handles:  set[int]                        # all chunks for company
building_handles: dict[str, set[int]]             # building_id → handles
sites_handles:    dict[str, set[int]]             # "{building}__{sites}" → handles
```

Rebuilt after every insert/delete. This avoids the O(N) metadata scan that turbovec's default filter would require per query.

### Persistence Strategy

- **In-memory index** is always current (queries use it directly)
- **Dirty flag** set on every mutation
- **Background thread** flushes dirty `.tvim` files to disk every `STORE_FLUSH_INTERVAL` seconds
- **Atomic writes** via temp file + `os.replace()`
- **PostgreSQL** is the durable metadata store — used for enrichment and crash recovery

## Project Structure

```
├── app/
│   ├── main.py                  # FastAPI app, lifespan (DB pool + flush thread)
│   ├── config.py                # Settings (pydantic, .env)
│   ├── models/
│   │   └── schemas.py           # Request/response models
│   ├── routes/
│   │   ├── ingest.py            # Upload, list, delete documents
│   │   ├── query.py             # Hierarchical search + LLM answer
│   │   └── health.py            # Health check (global or per-company)
│   └── services/
│       ├── chunking.py          # RecursiveCharacterTextSplitter wrapper
│       ├── database.py          # asyncpg pool, batch inserts, chunk lookups
│       ├── document.py          # PDF/DOCX loader
│       ├── llm.py               # RAG answer generation (ChatOpenAI)
│       └── vectorstore.py       # Multi-company turbovec manager
├── migrations/
│   └── 001_initial.sql          # PostgreSQL schema
├── client_example.py            # Example client script
├── requirements.txt
└── .env.example
```

## Migration from ChromaDB

This project replaces a ChromaDB-based implementation. Key differences:

| Aspect | ChromaDB (before) | turbovec + PostgreSQL (after) |
|---|---|---|
| Vector storage | External ChromaDB server | In-process quantized index |
| Metadata storage | ChromaDB collections | PostgreSQL tables |
| Search latency | 3 network round trips per query | In-memory SIMD, no network hop |
| Filtering | HNSW native metadata filter | Pre-built allowlist caches (O(1)) |
| Company isolation | Collection-level filtering | Separate .tvim files per company |
| Hierarchical fallback | 3 sequential search calls | Allowlist-based, single pass per scope |
| Persistence | ChromaDB server manages | .tvim files + PostgreSQL |
| Crash recovery | ChromaDB server handles | PostgreSQL + re-embed from DB |

## License

MIT
