from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = "sk-placeholder"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_llm_model: str = "gpt-4o-mini"

    # turbovec
    turbovec_bit_width: int = 4

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 100

    # Retrieval
    default_top_k: int = 5

    # PostgreSQL
    postgres_dsn: str = "postgresql://user:pass@localhost:5432/personal"

    # Multi-store
    store_flush_interval: int = 30
    max_stores_in_memory: int = 100

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Paths
    index_dir: Path = Path("./data/index")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
