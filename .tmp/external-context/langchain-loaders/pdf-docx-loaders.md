---
source: Context7 API + LangChain docs
library: LangChain
package: langchain-community
topic: PDF and DOCX document loaders
fetched: 2026-06-09T12:00:00Z
official_docs: https://python.langchain.com/docs/integrations/document_loaders/
---

# LangChain Document Loaders — PDF & DOCX

All loaders return `List[Document]` where each `Document` has:
- `page_content: str` — extracted text
- `metadata: dict` — source info, page numbers, etc.

---

## PyPDFLoader (PDF)

**Package**: `langchain-community` + `pypdf`

### Install

```bash
pip install pypdf langchain-community
# or
uv add pypdf langchain-community
```

### Import

```python
from langchain_community.document_loaders import PyPDFLoader
```

### Basic Usage

```python
# Load from local file
loader = PyPDFLoader("path/to/document.pdf")
pages = loader.load()

# Load from URL
loader = PyPDFLoader("https://arxiv.org/pdf/2303.08774.pdf")
data = loader.load()
```

### Output Structure

Each page becomes a separate `Document`:

```python
for page in pages:
    print(page.page_content)       # page text
    print(page.metadata)           # {'source': 'path/to/document.pdf', 'page': 0}
```

### Lazy Loading (for large PDFs)

```python
for doc in loader.lazy_load():
    print(doc.page_content)
```

### With Custom Blob Parsing

```python
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader
from langchain_community.document_loaders import PyPDFLoader

loader = AzureBlobStorageLoader(
    "https://<storage-account>.blob.core.windows.net",
    "<container-name>",
    blob_names="<file.pdf>",
    loader_factory=PyPDFLoader,
)

for doc in loader.lazy_load():
    print(doc.page_content)  # content of each page as separate document
```

### Manual Page Extraction (low-level)

```python
import pypdf
from langchain_core.documents import Document

def load_pdf_pages(file_path: str) -> list[Document]:
    reader = pypdf.PdfReader(file_path)
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": file_path, "page": i},
        )
        for i, page in enumerate(reader.pages)
    ]

docs = load_pdf_pages("report.pdf")
print(len(docs))  # number of pages
```

### API Surface

| Method | Description |
|--------|-------------|
| `PyPDFLoader(file_path)` | Constructor — accepts local path or URL |
| `.load()` → `List[Document]` | Load all pages eagerly |
| `.lazy_load()` → `Iterator[Document]` | Stream pages one at a time |

### Metadata Fields

| Key | Type | Description |
|-----|------|-------------|
| `source` | `str` | File path or URL |
| `page` | `int` | Zero-based page number |

---

## Docx2txtLoader (DOCX)

**Package**: `langchain-community` + `docx2txt`

### Install

```bash
pip install docx2txt langchain-community
```

### Import

```python
from langchain_community.document_loaders import Docx2txtLoader
```

### Basic Usage

```python
loader = Docx2txtLoader("path/to/document.docx")
documents = loader.load()

for doc in documents:
    print(doc.page_content)
    print(doc.metadata)  # {'source': 'path/to/document.docx'}
```

### API Surface

| Method | Description |
|--------|-------------|
| `Docx2txtLoader(file_path)` | Constructor — accepts local `.docx` path |
| `.load()` → `List[Document]` | Load entire document as single Document |
| `.lazy_load()` → `Iterator[Document]` | Stream (single item for whole doc) |

> **Note**: `Docx2txtLoader` is lightweight and does not require external services. It extracts raw text but does not preserve formatting, tables structure, or images.

---

## UnstructuredWordDocumentLoader (DOCX — Advanced)

**Package**: `langchain-community` + `unstructured`

### Install

```bash
pip install unstructured langchain-community
# or for API-based processing:
pip install unstructured-client langchain-unstructured
```

### Import

```python
# Local processing
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

# API-based processing (more accurate, requires API key)
from langchain_unstructured import UnstructuredLoader
```

### Local Usage

```python
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

loader = UnstructuredWordDocumentLoader("path/to/document.docx")
data = loader.load()

print(data[0].page_content[:500])
```

### API-Based Usage (UnstructuredLoader)

```python
import os
from langchain_unstructured import UnstructuredLoader

loader = UnstructuredLoader(
    file_path="path/to/document.docx",
    api_key=os.getenv("UNSTRUCTURED_API_KEY"),
    partition_via_api=True,
)

docs = loader.load()
print(docs[0].metadata["filename"])
print(docs[0].page_content)
```

### Batch Loading Multiple Files

```python
loader = UnstructuredLoader(
    file_path=["doc1.docx", "doc2.docx"],
    api_key=os.getenv("UNSTRUCTURED_API_KEY"),
    partition_via_api=True,
)

docs = loader.load()
for doc in docs:
    print(doc.metadata["filename"], ": ", doc.page_content[:100])
```

### Comparison: Docx2txtLoader vs UnstructuredWordDocumentLoader

| Feature | Docx2txtLoader | UnstructuredWordDocumentLoader |
|---------|----------------|-------------------------------|
| **Install size** | Tiny (`docx2txt`) | Large (`unstructured` + deps) |
| **External deps** | None | Optional API service |
| **Formatting** | Raw text only | Preserves structure |
| **Tables** | ❌ | ✅ |
| **Images** | ❌ | ✅ (with OCR) |
| **Speed** | Fast | Slower (or API latency) |
| **Best for** | Quick text extraction | Rich document parsing |

---

## Common Pattern: Load → Split → Index

```python
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from turbovec.langchain import TurboQuantVectorStore

# Load
pdf_docs = PyPDFLoader("report.pdf").load()
docx_docs = Docx2txtLoader("notes.docx").load()
all_docs = pdf_docs + docx_docs

# Split
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(all_docs)

# Index
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
store = TurboQuantVectorStore.from_documents(chunks, embeddings, bit_width=4)
```
