import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

from src.config import settings


def load_text_from_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in [".txt", ".md", ".csv"]:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return path.read_text(errors="ignore")
    if ext in [".html", ".htm"]:
        html = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "lxml")
        return soup.get_text(" ")
    if ext == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, source: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return [
        {"id": f"{source}#chunk-{i}", "text": c, "source": source}
        for i, c in enumerate(chunks)
    ]


def embed_chunks(chunks: List[dict], model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    return embeddings


def build_faiss_index(embeddings: np.ndarray, dim: int):
    try:
        import faiss  # type: ignore
    except Exception:
        return None
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def ingest_directory(data_dir: str | Path = None, index_dir: str | Path = None, model_name: str | None = None) -> Tuple[str, str]:
    data_dir = str(data_dir or settings.data_dir)
    index_dir = str(index_dir or settings.index_dir)
    model_name = model_name or settings.embedding_model

    Path(index_dir).mkdir(parents=True, exist_ok=True)

    docs: List[dict] = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in {".txt", ".md", ".csv", ".html", ".htm", ".pdf"}:
                text = load_text_from_file(p)
                if text.strip():
                    docs.extend(chunk_text(text, str(p)))

    if not docs:
        raise RuntimeError(f"No documents found to ingest in {data_dir}")

    embeddings = embed_chunks(docs, model_name)
    index = build_faiss_index(embeddings, embeddings.shape[1])

    import json
    (Path(index_dir) / "chunks.json").write_text(json.dumps(docs, ensure_ascii=False))
    np.save(Path(index_dir) / "embeddings.npy", embeddings.astype(np.float32))

    if index is not None:
        try:
            import faiss  # type: ignore
            faiss.write_index(index, str(Path(index_dir) / "vectors.faiss"))
        except Exception:
            pass

    return str(Path(index_dir) / "vectors.faiss"), str(Path(index_dir) / "chunks.json")