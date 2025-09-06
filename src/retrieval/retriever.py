from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import json
import numpy as np

from sentence_transformers import SentenceTransformer

from src.config import settings


@dataclass
class RetrievedChunk:
    id: str
    text: str
    source: str
    score: float


def load_index(index_dir: str | Path | None = None):
    index_dir = str(index_dir or settings.index_dir)
    chunks = json.loads(Path(index_dir, "chunks.json").read_text(encoding="utf-8"))
    emb_path = Path(index_dir) / "embeddings.npy"
    if emb_path.exists():
        embeddings = np.load(emb_path)
    else:
        embeddings = None
    try:
        import faiss  # type: ignore
        faiss_index = faiss.read_index(str(Path(index_dir) / "vectors.faiss"))
    except Exception:
        faiss_index = None
    return faiss_index, chunks, embeddings


def semantic_search(query: str, k: int = 5, model_name: str | None = None, index_dir: str | None = None) -> List[RetrievedChunk]:
    model_name = model_name or settings.embedding_model
    faiss_index, chunks, embeddings = load_index(index_dir)
    model = SentenceTransformer(model_name)
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    results: List[RetrievedChunk] = []
    if faiss_index is not None:
        import faiss  # type: ignore
        D, I = faiss_index.search(q, k)
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            c = chunks[idx]
            results.append(RetrievedChunk(id=c["id"], text=c["text"], source=c["source"], score=float(score)))
        return results

    if embeddings is None:
        raise RuntimeError("No FAISS index and no embeddings.npy found. Re-run ingestion.")

    sims = (embeddings @ q.T).ravel()
    topk_idx = np.argsort(-sims)[:k]
    for idx in topk_idx.tolist():
        c = chunks[idx]
        results.append(RetrievedChunk(id=c["id"], text=c["text"], source=c["source"], score=float(float(sims[idx]))))
    return results