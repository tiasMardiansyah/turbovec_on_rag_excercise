---
source: Context7 API + GitHub raw README
library: turbovec
package: turbovec
topic: LangChain integration
fetched: 2026-06-09T12:00:00Z
official_docs: https://github.com/ryancodrai/turbovec
---

# turbovec — LangChain Integration

## Overview

`turbovec.langchain.TurboQuantVectorStore` is a [LangChain `VectorStore`](https://python.langchain.com/docs/integrations/vectorstores/) backed by an `IdMapIndex`. It implements the same public surface as `langchain_core.vectorstores.in_memory.InMemoryVectorStore` and can be used as a **drop-in replacement** wherever the in-memory store is used.

A 10 million document corpus takes 31 GB of RAM as float32. turbovec fits it in 4 GB — and searches it faster than FAISS.

## Install

```bash
pip install turbovec[langchain]
```

This installs `turbovec` plus its LangChain dependencies.

## Basic Usage

```python
from langchain_huggingface import HuggingFaceEmbeddings
from turbovec.langchain import TurboQuantVectorStore

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

store = TurboQuantVectorStore.from_texts(
    texts=["Document 1...", "Document 2...", "Document 3..."],
    embedding=embeddings,
    bit_width=4,
)

retriever = store.as_retriever(search_kwargs={"k": 5})
```

> The dimensionality of the underlying quantized index is inferred from the embedding model on the first `add_*` call — no need to specify it up front.

## Construction Patterns

```python
# No-arg: lazy. dim is inferred from the first add.
store = TurboQuantVectorStore(embeddings)

# from_texts: same lazy behaviour, plus immediate ingest.
store = TurboQuantVectorStore.from_texts(texts, embeddings, bit_width=4)

# Pre-built index: bring your own IdMapIndex (e.g. one loaded from disk).
from turbovec import IdMapIndex
store = TurboQuantVectorStore(embeddings, index=IdMapIndex(1536, 4))
```

`bit_width` is `2` or `4` and is fixed once the index is created.

## Adding with Explicit IDs

```python
store.add_texts(
    texts=["a", "b", "c"],
    ids=["doc-a", "doc-b", "doc-c"],
    metadatas=[{"source": "x"}, {"source": "y"}, {"source": "z"}],
)

# add_documents honours per-Document.id, falling back to a UUID per
# document if .id is missing — partial ids are not dropped wholesale.
from langchain_core.documents import Document

store.add_documents([
    Document(id="explicit", page_content="..."),
    Document(page_content="..."),  # gets a UUID
])
```

If an id is already present, `add_texts` **upserts** — the existing entry is removed and the new one added with the same id.

Async equivalents (`aadd_texts`, `aadd_documents`) use the embedding model's `aembed_documents`.

## Search

```python
# By string query (uses the embedding function)
docs = store.similarity_search("what is turbovec?", k=5)

# With scores
docs_and_scores = store.similarity_search_with_score("...", k=5)

# By raw vector
import numpy as np
qvec = np.random.randn(768).astype(np.float32)
qvec /= np.linalg.norm(qvec)
docs = store.similarity_search_by_vector(qvec.tolist(), k=5)
```

**Scores**: Raw inner products. Because vectors are L2-normalized on insert, inner product equals cosine similarity — higher is better, range `[-1, 1]`.

`similarity_search_with_relevance_scores` and `as_retriever(search_type="similarity_score_threshold")` work: the raw cosine is mapped to `[0, 1]` via `(sim + 1) / 2`.

Async equivalents: `asimilarity_search`, `asimilarity_search_with_score`, `asimilarity_search_by_vector`, `aget_by_ids`.

## Filters

`similarity_search`, `similarity_search_with_score`, and `similarity_search_by_vector` all accept a `filter` keyword:

```python
# Dict — AND of exact equality on Document.metadata.
docs = store.similarity_search(
    "query", k=5, filter={"source": "manual", "version": 2},
)

# Callable — predicate over the Document.
docs = store.similarity_search(
    "query", k=5, filter=lambda doc: doc.metadata.get("score", 0) > 0.8,
)
```

Filters are resolved to an id allowlist **before** scoring; the kernel only ever inserts allowed documents into the per-query heap.

## Document Retrieval by ID

```python
docs = store.get_by_ids(["doc-a", "doc-c"])
# Missing ids are silently skipped.
```

## Delete

```python
store.delete(["doc-a", "doc-b"])  # missing ids silently skipped, returns None
```

Delete is O(1) per id. `delete(None)` is a no-op (matches `InMemoryVectorStore` contract).

## Save / Load

```python
store.dump("./my-store")
# ... later ...
store = TurboQuantVectorStore.load("./my-store", embedding=embeddings)
```

Writes two files under the given folder path:
- `index.tvim` — the `IdMapIndex` payload
- `docstore.json` — JSON-encoded document text, metadata, and id maps

Document metadata must be JSON-serializable.

## Full API Surface (Summary)

| Method | Description |
|--------|-------------|
| `TurboQuantVectorStore(embedding, index=None)` | Constructor |
| `.from_texts(texts, embedding, bit_width=4, **kwargs)` | Create + ingest texts |
| `.from_documents(docs, embedding, bit_width=4, **kwargs)` | Create + ingest Documents |
| `.add_texts(texts, metadatas=None, ids=None, **kwargs)` | Add texts with optional ids/metadata |
| `.add_documents(documents, **kwargs)` | Add Document objects |
| `.similarity_search(query, k=4, filter=None, **kwargs)` | Search by query string |
| `.similarity_search_with_score(query, k=4, filter=None)` | Search with cosine scores |
| `.similarity_search_by_vector(embedding, k=4, filter=None)` | Search by raw vector |
| `.similarity_search_with_relevance_scores(query, k=4)` | Scores normalized to [0, 1] |
| `.get_by_ids(ids)` | Retrieve docs by ID |
| `.delete(ids)` | Delete docs by ID |
| `.dump(path)` | Persist to disk |
| `.load(path, embedding=)` | Load from disk (class method) |
| `.as_retriever(search_type=, search_kwargs=)` | Get LangChain Retriever |

## Known Limitations

- **Max-marginal-relevance search is NOT supported.** Raises `NotImplementedError`.
- **Embeddings are not retained.** Original embedding vectors are not recoverable after quantization.
- **JSON-serializable metadata only.** Non-JSON values fail at save time.

## Migration: InMemoryVectorStore → turbovec

```python
# Before (InMemoryVectorStore)
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
store = InMemoryVectorStore(embeddings)
store = InMemoryVectorStore.from_texts(texts, embeddings)

# After (turbovec drop-in replacement)
from turbovec.langchain import TurboQuantVectorStore
store = TurboQuantVectorStore(embeddings)
store = TurboQuantVectorStore.from_texts(texts, embeddings, bit_width=4)
```

Same public surface, same persistence semantics, same retriever and pipeline wiring — swap the import and add `bit_width`.
