---
source: Context7 API
library: LangChain
package: langchain-openai
topic: OpenAIEmbeddings and ChatOpenAI
fetched: 2026-06-09T12:00:00Z
official_docs: https://python.langchain.com/docs/integrations/platforms/openai/
---

# langchain-openai — OpenAIEmbeddings & ChatOpenAI

## Install

```bash
pip install -U "langchain-openai"
```

---

## OpenAIEmbeddings

### Initialization

```python
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    # With the `text-embedding-3` class of models, you can specify the size
    # of the embeddings you want returned.
    # dimensions=1024  # Optional: reduce from default 3072
)
```

### Available Models

| Model | Default Dimensions | Notes |
|-------|--------------------|-------|
| `text-embedding-3-large` | 3072 | Recommended, supports `dimensions` param |
| `text-embedding-3-small` | 1536 | Smaller, faster |
| `text-embedding-ada-002` | 1536 | Legacy |

### Key Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `str` | Model name (default: `"text-embedding-3-large"`) |
| `dimensions` | `int \| None` | Output embedding dimensions (text-embedding-3 only) |
| `api_key` | `str \| None` | OpenAI API key (or set `OPENAI_API_KEY` env var) |
| `base_url` | `str \| None` | Custom API endpoint (e.g., Azure, local proxy) |
| `organization` | `str \| None` | OpenAI organization ID |
| `max_retries` | `int` | Max retries on API failure |
| `request_timeout` | `float \| None` | Timeout per request |

### Core Methods

```python
# Embed a single query string → List[float]
vector = embeddings.embed_query("What is the meaning of life?")

# Embed multiple documents → List[List[float]]
vectors = embeddings.embed_documents([
    "Document one text.",
    "Document two text.",
])

# Async variants
vector = await embeddings.aembed_query("What is the meaning of life?")
vectors = await embeddings.aembed_documents(["doc1", "doc2"])
```

### Usage with Vector Stores

```python
from turbovec.langchain import TurboQuantVectorStore

store = TurboQuantVectorStore.from_texts(
    texts=["Hello world", "Goodbye world"],
    embedding=embeddings,
    bit_width=4,
)

results = store.similarity_search("Hello", k=2)
```

---

## ChatOpenAI

### Initialization

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    # temperature=0.7,        # Creativity (0.0–2.0, default varies by model)
    # max_tokens=None,        # Max output tokens
    # timeout=None,           # Request timeout in seconds
    # max_retries=2,          # Max retries on failure
    # api_key="...",          # Or set OPENAI_API_KEY env var
    # base_url="...",         # Custom endpoint
    # organization="...",     # OpenAI org ID
    # streaming=False,        # Stream tokens as they arrive
    # stream_usage=True,      # Stream token usage stats
    # reasoning_effort="low", # For reasoning models
)
```

### Key Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `str` | Model name (e.g., `"gpt-4o"`, `"gpt-4o-mini"`, `"gpt-5-nano"`) |
| `temperature` | `float \| None` | Sampling temperature (0.0–2.0) |
| `max_tokens` | `int \| None` | Maximum output tokens |
| `timeout` | `int \| None` | Request timeout seconds |
| `max_retries` | `int` | Max retries (default: 2) |
| `api_key` | `str \| None` | Override `OPENAI_API_KEY` env var |
| `base_url` | `str \| None` | Custom API endpoint |
| `organization` | `str \| None` | OpenAI org ID |
| `streaming` | `bool` | Enable token streaming (default: True in some configs) |
| `stream_usage` | `bool` | Stream token usage metadata |
| `reasoning_effort` | `str \| None` | `"low"`, `"medium"`, `"high"` for reasoning models |

### Core Methods

```python
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Invoke (single turn)
response = llm.invoke([HumanMessage(content="What is 2+2?")])
print(response.content)

# With system message
response = llm.invoke([
    SystemMessage(content="You are a helpful math tutor."),
    HumanMessage(content="What is 2+2?"),
])

# Batch multiple prompts
responses = llm.batch([
    [HumanMessage(content="What is 2+2?")],
    [HumanMessage(content="What is 3+3?")],
])

# Stream tokens
for chunk in llm.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)

# Async variants
response = await llm.ainvoke([HumanMessage(content="Hello")])
async for chunk in llm.astream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

### Streaming Configuration

```python
# Enable streaming with usage stats
llm = ChatOpenAI(model="gpt-4o-mini", stream_usage=True)

# Disable streaming (for multi-agent systems or LangSmith)
llm = ChatOpenAI(model="gpt-4o-mini", streaming=False)

# HTTP proxy
llm = ChatOpenAI(model="gpt-4o-mini", openai_proxy="http://proxy.example.com:8080")
```

### With Tools / Function Calling

```python
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(description="First number")
    b: int = Field(description="Second number")

llm_with_tools = llm.bind_tools([
    {"type": "function", "function": {...}},
])
```

### Integration with RAG Pipeline

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from turbovec.langchain import TurboQuantVectorStore

# Setup
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

store = TurboQuantVectorStore.from_texts(
    texts=["turbovec is a Rust vector index with Python bindings"],
    embedding=embeddings,
    bit_width=4,
)

# Simple RAG
retriever = store.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("What is turbovec?")

from langchain_core.messages import HumanMessage, SystemMessage
context = "\n".join(d.page_content for d in docs)
response = llm.invoke([
    SystemMessage(content=f"Answer based on this context:\n{context}"),
    HumanMessage(content="What is turbovec?"),
])
print(response.content)
```
