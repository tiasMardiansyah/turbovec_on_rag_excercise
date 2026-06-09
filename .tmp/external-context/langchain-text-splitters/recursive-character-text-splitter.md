---
source: Context7 API
library: LangChain
package: langchain-text-splitters
topic: RecursiveCharacterTextSplitter
fetched: 2026-06-09T12:00:00Z
official_docs: https://python.langchain.com/docs/concepts/text_splitters/
---

# LangChain — RecursiveCharacterTextSplitter

## Overview

`RecursiveCharacterTextSplitter` is the **recommended text splitter for generic text use cases**. It recursively splits documents using common separators (newlines, spaces) until each chunk reaches an appropriate size, defined by `chunk_size` and `chunk_overlap`.

## Install

```bash
pip install langchain-text-splitters
```

## Import

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

## Initialization

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # Maximum characters per chunk
    chunk_overlap=200,     # Overlap characters between chunks
    add_start_index=True,  # Track character index in original document
    # separators=None,     # Custom separator list (default: ["\n\n", "\n", " ", ""])
    # length_function=len, # Custom length function
    # is_separator_regex=False,  # Treat separators as regex
)
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunk_size` | `int` | `4000` | Max characters per chunk |
| `chunk_overlap` | `int` | `200` | Overlap chars between adjacent chunks |
| `add_start_index` | `bool` | `False` | If `True`, stores the char offset as `metadata["start_index"]` |
| `separators` | `List[str]` | `["\n\n", "\n", " ", ""]` | Hierarchy of separators to try |
| `length_function` | `Callable` | `len` | Custom length function (e.g., token counter) |
| `is_separator_regex` | `bool` | `False` | Treat separator strings as regex patterns |
| `keep_separator` | `bool \| str` | `False` | Whether to keep the separator in chunks |
| `strip_whitespace` | `bool` | `True` | Strip whitespace from chunk edges |

## Usage Patterns

### Split Documents (from a loader)

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("report.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)

all_splits = text_splitter.split_documents(docs)
print(f"Split into {len(all_splits)} sub-documents.")
```

### Split Raw Text

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

chunks = text_splitter.split_text(
    "A very long piece of text that needs to be split..."
)
print(len(chunks))
```

### Split with Custom Token Length Function

```python
import tiktoken

def tiktoken_len(text: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-4o")
    return len(encoding.encode(text))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # now in tokens, not characters
    chunk_overlap=50,
    length_function=tiktoken_len,
)
```

### Split with Custom Separators

```python
# Prioritize Markdown headers, then paragraphs, then lines
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["## ", "### ", "\n\n", "\n", ". ", " ", ""],
)
```

## Output

Each split chunk is a `Document` with:
- `page_content`: The text chunk
- `metadata`: Inherited from parent document + optional `start_index`

```python
for split in all_splits:
    print(f"Content: {split.page_content[:80]}...")
    print(f"Source: {split.metadata['source']}")
    if 'start_index' in split.metadata:
        print(f"Start index: {split.metadata['start_index']}")
    print("---")
```

## How It Works

1. Tries the first separator (`"\n\n"`) to split text into pieces
2. If a piece exceeds `chunk_size`, moves to the next separator (`"\n"`)
3. Continues recursively through the separator hierarchy
4. Merges small adjacent pieces back together (up to `chunk_size`)
5. Ensures `chunk_overlap` characters are shared between adjacent chunks

This preserves paragraph/line integrity better than simple character-based splitting.

## Indexing Chunks into Vector Store

```python
from turbovec.langchain import TurboQuantVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

store = TurboQuantVectorStore.from_texts(
    texts=[split.page_content for split in all_splits],
    metadatas=[split.metadata for split in all_splits],
    embedding=embeddings,
    bit_width=4,
)

# Or directly with documents:
ids = store.add_documents(documents=all_splits)
```

## Recommended Settings by Use Case

| Use Case | chunk_size | chunk_overlap | Notes |
|----------|-----------|---------------|-------|
| General RAG | 1000 | 200 | Good balance |
| Precise retrieval | 500 | 50 | Smaller = more precise |
| Long context LLM | 2000 | 400 | Larger chunks |
| Code splitting | 1500 | 100 | Use code-specific separators |
| Token-based (GPT-4) | 800 tokens | 100 tokens | Use `length_function=tiktoken_len` |
