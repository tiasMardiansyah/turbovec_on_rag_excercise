"""Multi-company VectorStore service with per-company turbovec indices.

Each company gets its own .tvim file, lazy-loaded on first access,
with pre-built allowlist caches for hierarchical search:
    sitesId > buildingId > companyId
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from turbovec.langchain import TurboQuantVectorStore

from app.config import settings

logger = logging.getLogger(__name__)


class CompanyStore:
    """Holds a per-company TurboQuantVectorStore plus allowlist caches."""

    def __init__(self, store: TurboQuantVectorStore) -> None:
        self.store = store
        self.lock = threading.Lock()
        self.dirty = False

        # Pre-built allowlist caches: scope → set of u64 handles
        self.company_handles: set[int] = set()
        self.building_handles: dict[str, set[int]] = {}
        self.sites_handles: dict[str, set[int]] = {}  # key: "{building}__{sites}"

    def rebuild_allowlists(self) -> None:
        """Rebuild all allowlist caches from the store's docs dict.

        O(N) but only called after insert/delete, not per query.
        """
        self.company_handles.clear()
        self.building_handles.clear()
        self.sites_handles.clear()

        for sid, (text, meta) in self.store._docs.items():
            handle = self.store._str_to_u64.get(sid)
            if handle is None:
                continue

            self.company_handles.add(handle)

            building = meta.get("building_id", "")
            sites = meta.get("sites_id", "")

            if building:
                self.building_handles.setdefault(building, set()).add(handle)
            if building and sites:
                key = f"{building}__{sites}"
                self.sites_handles.setdefault(key, set()).add(handle)

    def get_allowlist(
        self, building_id: str | None, sites_id: str | None
    ) -> set[int] | None:
        """Return the most specific allowlist for the given hierarchy.

        Returns None to mean "no filter" (use all company handles).
        Returns empty set to mean "no matches".
        """
        if sites_id and building_id:
            key = f"{building_id}__{sites_id}"
            sites_set = self.sites_handles.get(key, set())
            if sites_set:
                return sites_set
            # Fall back to building
            building_set = self.building_handles.get(building_id, set())
            if building_set:
                return building_set
            # Fall back to company
            return self.company_handles if self.company_handles else None

        if building_id:
            building_set = self.building_handles.get(building_id, set())
            if building_set:
                return building_set
            return self.company_handles if self.company_handles else None

        return None  # no filter — search everything

    @property
    def total_chunks(self) -> int:
        return len(self.store._docs)

    @property
    def total_documents(self) -> int:
        doc_ids: set[str] = set()
        for _sid, (_text, meta) in self.store._docs.items():
            did = meta.get("document_id", "")
            if did:
                doc_ids.add(did)
        return len(doc_ids)


class VectorStoreService:
    """Manages per-company TurboQuantVectorStore instances with LRU eviction."""

    def __init__(self) -> None:
        self._embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )
        self._stores: OrderedDict[str, CompanyStore] = OrderedDict()
        self._global_lock = threading.Lock()
        self._dirty_companies: set[str] = set()
        self._dirty_lock = threading.Lock()
        self._flush_thread: threading.Thread | None = None

    def start_flush_thread(self) -> None:
        """Start the background flush thread. Call once at app startup."""
        self._flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="tvim-flush"
        )
        self._flush_thread.start()
        logger.info("Turbovec flush thread started (interval=%ds)", settings.store_flush_interval)

    def stop_flush_thread(self) -> None:
        """Signal the flush thread to stop and do a final flush."""
        self._running = False
        self._flush_all_dirty()
        logger.info("Turbovec flush thread stopped")

    def _flush_loop(self) -> None:
        self._running = True
        while self._running:
            time.sleep(settings.store_flush_interval)
            self._flush_all_dirty()

    def _flush_all_dirty(self) -> None:
        """Flush all dirty stores to disk."""
        with self._dirty_lock:
            dirty = set(self._dirty_companies)
            self._dirty_companies.clear()

        for company_id in dirty:
            try:
                cs = self._stores.get(company_id)
                if cs and cs.dirty:
                    self._atomic_dump(company_id, cs)
                    cs.dirty = False
            except Exception:
                logger.exception("Failed to flush store for company %s", company_id)

    def _atomic_dump(self, company_id: str, cs: CompanyStore) -> None:
        """Write .tvim to temp file, then atomically rename."""
        store_path = settings.index_dir / company_id / "store"
        store_path.mkdir(parents=True, exist_ok=True)

        if cs.total_chunks == 0:
            return

        tmp_path = store_path / "index.tvim.tmp"
        cs.store.dump(str(store_path / "tmp_dump"))

        # turbovec.dump() writes to a directory, so use it directly
        # then rename the resulting directory
        final_path = store_path
        tmp_dir = store_path.parent / "store_tmp"

        try:
            cs.store.dump(str(tmp_dir))
            # Replace each file atomically
            for f in tmp_dir.iterdir():
                target = final_path / f.name
                tmp_target = final_path / f"{f.name}.tmp"
                f.replace(tmp_target)
                os.replace(str(tmp_target), str(target))
        finally:
            # Clean up temp dir
            if tmp_dir.exists():
                for f in tmp_dir.iterdir():
                    f.unlink(missing_ok=True)
                tmp_dir.rmdir()

    # --- Store lifecycle ---

    def _get_store(self, company_id: str) -> CompanyStore:
        """Get or load a CompanyStore. Evicts LRU if over limit."""
        with self._global_lock:
            if company_id in self._stores:
                self._stores.move_to_end(company_id)
                return self._stores[company_id]

            # Evict if needed
            while len(self._stores) >= settings.max_stores_in_memory:
                evicted_id, evicted = self._stores.popitem(last=False)
                if evicted.dirty:
                    self._atomic_dump(evicted_id, evicted)
                logger.info("Evicted store for company %s from memory", evicted_id)

            store = self._load_or_create(company_id)
            self._stores[company_id] = store
            return store

    def _load_or_create(self, company_id: str) -> CompanyStore:
        """Load from disk or create a fresh CompanyStore."""
        store_path = settings.index_dir / company_id / "store"
        if (store_path / "index.tvim").exists():
            try:
                store = TurboQuantVectorStore.load(
                    str(store_path), embedding=self._embeddings
                )
                logger.info("Loaded existing store for company %s", company_id)
            except Exception:
                logger.warning(
                    "Failed to load store for company %s, creating fresh",
                    company_id,
                    exc_info=True,
                )
                store = TurboQuantVectorStore(
                    self._embeddings, bit_width=settings.turbovec_bit_width
                )
        else:
            store = TurboQuantVectorStore(
                self._embeddings, bit_width=settings.turbovec_bit_width
            )
            logger.info("Created new store for company %s", company_id)

        cs = CompanyStore(store)
        cs.rebuild_allowlists()
        return cs

    # --- Public API ---

    def add_documents(
        self,
        company_id: str,
        doc_id: str,
        filename: str,
        documents: list[Document],
        building_id: str | None = None,
        sites_id: str | None = None,
    ) -> list[str]:
        """Add documents to the company's vector store. Returns chunk IDs."""
        # Generate deterministic chunk IDs
        chunk_ids = [f"{doc_id}__{i}" for i in range(len(documents))]

        # Tag metadata
        for i, doc in enumerate(documents):
            doc.metadata["document_id"] = doc_id
            doc.metadata["filename"] = filename
            doc.metadata["chunk_id"] = chunk_ids[i]
            if building_id:
                doc.metadata["building_id"] = building_id
            if sites_id:
                doc.metadata["sites_id"] = sites_id

        cs = self._get_store(company_id)

        with cs.lock:
            cs.store.add_documents(documents, ids=chunk_ids)
            cs.rebuild_allowlists()
            cs.dirty = True

        self._mark_dirty(company_id)
        return chunk_ids

    def search(
        self,
        company_id: str,
        query: str,
        top_k: int,
        building_id: str | None = None,
        sites_id: str | None = None,
    ) -> list[tuple[Document, float]]:
        """Hierarchical search: sitesId > buildingId > companyId.

        Returns (Document, score) pairs with chunk_id in metadata.
        """
        cs = self._get_store(company_id)

        if not sites_id and not building_id:
            return self._search_scope(cs, query, top_k)

        if sites_id and building_id:
            return self._search_sites_hierarchy(cs, query, top_k, building_id, sites_id)

        if building_id:
            return self._search_building_hierarchy(cs, query, top_k, building_id)

        return []

    def _search_scope(
        self,
        cs: CompanyStore,
        query: str,
        top_k: int,
        handles: set[int] | None = None,
    ) -> list[tuple[Document, float]]:
        """Core search with optional allowlist. Returns (Document, score) pairs."""
        with cs.lock:
            if handles is not None:
                if not handles:
                    return []
                allowlist = np.array(list(handles), dtype=np.uint64)
                qvec = np.asarray(
                    cs.store._embedding.embed_query(query), dtype=np.float32
                )
                if qvec.ndim == 1:
                    qvec = qvec[None, :]
                if not qvec.flags["C_CONTIGUOUS"]:
                    qvec = np.ascontiguousarray(qvec)
                k = min(top_k, len(cs.store._index))
                if k == 0:
                    return []
                scores, idx_handles = cs.store._index.search(qvec, k, allowlist=allowlist)
                results: list[tuple[Document, float]] = []
                for score, handle in zip(scores[0], idx_handles[0]):
                    sid = cs.store._u64_to_str.get(int(handle))
                    if sid is None:
                        continue
                    text, meta = cs.store._docs[sid]
                    results.append(
                        (Document(id=sid, page_content=text, metadata=dict(meta)), float(score))
                    )
                return results
            else:
                return cs.store.similarity_search_with_score(query, k=top_k)

    def _search_sites_hierarchy(
        self,
        cs: CompanyStore,
        query: str,
        top_k: int,
        building_id: str,
        sites_id: str,
    ) -> list[tuple[Document, float]]:
        """sitesId + buildingId → buildingId → companyId fallback."""
        sites_key = f"{building_id}__{sites_id}"
        sites_handles = cs.sites_handles.get(sites_key, set())

        # Try sites-specific first
        sites_results = self._search_scope(cs, query, top_k, sites_handles or None)
        if len(sites_results) >= max(top_k - 2, 1):
            return sites_results[:top_k]

        # Combine with building-specific (dedup)
        building_handles = cs.building_handles.get(building_id, set())
        building_results = self._search_scope(cs, query, top_k, building_handles or None)

        combined = list(sites_results)
        seen = {doc.metadata.get("chunk_id", "") for doc, _ in combined}
        for doc, score in building_results:
            cid = doc.metadata.get("chunk_id", "")
            if cid not in seen and len(combined) < top_k:
                combined.append((doc, score))
                seen.add(cid)

        if len(combined) >= 2:
            return combined[:top_k]

        # Supplement with company-wide (exclude results that have a building_id)
        company_results = self._search_scope(cs, query, top_k, cs.company_handles or None)
        for doc, score in company_results:
            cid = doc.metadata.get("chunk_id", "")
            if cid not in seen and not doc.metadata.get("building_id") and len(combined) < top_k:
                combined.append((doc, score))
                seen.add(cid)

        return combined[:top_k]

    def _search_building_hierarchy(
        self,
        cs: CompanyStore,
        query: str,
        top_k: int,
        building_id: str,
    ) -> list[tuple[Document, float]]:
        """buildingId → companyId fallback."""
        building_handles = cs.building_handles.get(building_id, set())
        building_results = self._search_scope(cs, query, top_k, building_handles or None)

        if len(building_results) >= 2:
            return building_results[:top_k]

        # Supplement with company-wide (exclude results that have a building_id)
        company_results = self._search_scope(cs, query, top_k, cs.company_handles or None)
        combined = list(building_results)
        seen = {doc.metadata.get("chunk_id", "") for doc, _ in combined}

        for doc, score in company_results:
            cid = doc.metadata.get("chunk_id", "")
            if cid not in seen and not doc.metadata.get("building_id") and len(combined) < top_k:
                combined.append((doc, score))
                seen.add(cid)

        return combined[:top_k]

    def remove_document(self, company_id: str, document_id: str) -> int:
        """Remove all chunks for a document. Returns count removed."""
        cs = self._get_store(company_id)

        with cs.lock:
            # Find all chunk IDs belonging to this document
            chunk_ids = [
                sid
                for sid, (text, meta) in cs.store._docs.items()
                if meta.get("document_id") == document_id
            ]
            if not chunk_ids:
                return 0

            cs.store.delete(chunk_ids)
            cs.rebuild_allowlists()
            cs.dirty = True

        self._mark_dirty(company_id)
        return len(chunk_ids)

    def list_documents(self, company_id: str) -> list[dict]:
        """Return document metadata from the in-memory store."""
        cs = self._get_store(company_id)
        docs: dict[str, dict] = {}
        for sid, (text, meta) in cs.store._docs.items():
            did = meta.get("document_id", "")
            if not did:
                continue
            if did not in docs:
                docs[did] = {
                    "document_id": did,
                    "filename": meta.get("filename", ""),
                    "chunks": 0,
                }
            docs[did]["chunks"] += 1
        return list(docs.values())

    def get_company_stats(self, company_id: str) -> dict[str, int]:
        """Get total chunks and documents for a company."""
        cs = self._get_store(company_id)
        return {
            "total_chunks": cs.total_chunks,
            "total_documents": cs.total_documents,
        }

    def _mark_dirty(self, company_id: str) -> None:
        with self._dirty_lock:
            self._dirty_companies.add(company_id)


vector_store = VectorStoreService()
