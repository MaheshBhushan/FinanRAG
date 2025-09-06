import os
from dataclasses import dataclass


def env(key: str, default: str | None = None) -> str | None:
    v = os.getenv(key)
    return v if v is not None else default


@dataclass
class Settings:
    gemini_api_key: str | None = env("GEMINI_API_KEY")
    embedding_model: str = env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2") or "sentence-transformers/all-MiniLM-L6-v2"
    index_dir: str = env("INDEX_DIR", "indexes") or "indexes"
    data_dir: str = env("DATA_DIR", "data") or "data"


settings = Settings()