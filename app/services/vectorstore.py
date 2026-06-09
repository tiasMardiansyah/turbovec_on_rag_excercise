from __future__ import annotations

import json
import threading
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from turbovec.langchain import TurboQuantVectorStore

from app.config import settings


class VectorStoreService:
    """Manages a turbovec TurboQuantVectorStore with document-level tracking."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )
        self._store_path = settings.index_dir / "store"
        self._doc_meta_path = settings.index_dir / "doc_meta.json"

        if (self._store_path / "index.tvim").exists():
            self._store = TurboQuantVectorStore.load(
                str(self._store_path), embedding=self._embeddings
            )
        else:
            self._store = TurboQuantVectorStore(
                self._embeddings, bit_width=settings.turbovec_bit_width
            )

        # document_id → {filename, chunk_ids}
        self._doc_meta: dict[str, dict] = {}
        self._load_meta()

    # --- Public API ---

    def add_documents(
        self, doc_id: str, filename: str, documents: list[Document]
    ) -> int:
        """Add LangChain Documents to the vector store. Returns chunk count."""
        for doc in documents:
            doc.metadata["document_id"] = doc_id
            doc.metadata["filename"] = filename

        with self._lock:
            chunk_ids = self._store.add_documents(documents)

        self._doc_meta[doc_id] = {
            "filename": filename,
            "chunk_ids": chunk_ids,
        }
        self._save()
        return len(chunk_ids)

    def search(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        """Search by query string (auto-embeds). Returns (Document, score) pairs."""
        with self._lock:
            return self._store.similarity_search_with_score(query, k=top_k)

    def remove_document(self, doc_id: str) -> int:
        """Remove all chunks for a document. Returns count removed."""
        if doc_id not in self._doc_meta:
            return 0
        chunk_ids = self._doc_meta[doc_id]["chunk_ids"]
        with self._lock:
            self._store.delete(chunk_ids)
        del self._doc_meta[doc_id]
        self._save()
        return len(chunk_ids)

    def list_documents(self) -> list[dict]:
        """Return list of {document_id, filename, chunks}."""
        return [
            {
                "document_id": doc_id,
                "filename": info["filename"],
                "chunks": len(info["chunk_ids"]),
            }
            for doc_id, info in self._doc_meta.items()
        ]

    @property
    def total_chunks(self) -> int:
        return sum(len(info["chunk_ids"]) for info in self._doc_meta.values())

    @property
    def total_documents(self) -> int:
        return len(self._doc_meta)

    # --- Persistence ---

    def _save(self) -> None:
        settings.index_dir.mkdir(parents=True, exist_ok=True)
        if self.total_chunks > 0:
            self._store.dump(str(self._store_path))
        with open(self._doc_meta_path, "w") as f:
            json.dump(self._doc_meta, f)

    def _load_meta(self) -> None:
        if self._doc_meta_path.exists():
            with open(self._doc_meta_path) as f:
                self._doc_meta = json.load(f)


vector_store = VectorStoreService()
