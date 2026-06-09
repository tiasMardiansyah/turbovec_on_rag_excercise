from __future__ import annotations

import tempfile
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx"}


def load_document(content: bytes, filename: str) -> list[Document]:
    """Load a PDF or DOCX file into LangChain Documents.

    Writes to a temp file (loaders need file paths), then extracts text.
    """
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {ext}. Supported: {SUPPORTED_EXTENSIONS}"
        )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = Docx2txtLoader(tmp_path)
        return loader.load()
    finally:
        Path(tmp_path).unlink(missing_ok=True)
