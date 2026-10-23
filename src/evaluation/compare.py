from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.retrieval.retriever import RetrievedChunk, load_index


def load_fixtures(path: str | Path) -> list[dict[str, Any]]:
    fixtures = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(fixtures, list) or not fixtures:
        raise ValueError("Fixture file must contain a non-empty JSON array")
    for i, fixture in enumerate(fixtures):
        if not isinstance(fixture, dict) or not isinstance(fixture.get("question"), str) or not fixture["question"].strip():
            raise ValueError(f"Fixture {i} must have a non-empty question")
        if not isinstance(fixture.get("expected_supported"), bool):
            raise ValueError(f"Fixture {i} must set expected_supported to true or false")
        for key in ("expected_sources", "citations"):
            if key in fixture and (not isinstance(fixture[key], list) or not all(isinstance(x, str) for x in fixture[key])):
                raise ValueError(f"Fixture {i} field {key} must be an array of strings")
    return fixtures


class SharedVectorSearch:
    """Search the existing normalized embeddings without re-chunking or re-embedding."""

    def __init__(self, chunks: list[dict[str, str]], embeddings: Any, model_name: str, k: int):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.embeddings = embeddings
        self.model = SentenceTransformer(model_name)
        self.k = k

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        import numpy as np

        vector = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
        scores = self.embeddings @ vector
        indices = np.argsort(-scores)[:k]
        return [
            RetrievedChunk(
                id=self.chunks[i]["id"], text=self.chunks[i]["text"],
                source=self.chunks[i]["source"], score=float(scores[i]),
            )
            for i in indices.tolist()
        ]


def framework_retrievers(search: SharedVectorSearch) -> dict[str, Any]:
    """Expose the shared search through each framework's native retriever API."""
    try:
        from langchain_core.documents import Document
        from langchain_core.retrievers import BaseRetriever
        from pydantic import ConfigDict

        class LangChainRetriever(BaseRetriever):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            search: Any

            def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Document]:
                return [Document(page_content=c.text, metadata={"id": c.id, "source": c.source, "score": c.score}) for c in search.search(query, search.k)]

        langchain = LangChainRetriever(search=search)
    except ImportError as exc:
        raise RuntimeError("Install project dependencies to use the LangChain comparison pipeline") from exc

    try:
        from llama_index.core import QueryBundle
        from llama_index.core.retrievers import BaseRetriever
        from llama_index.core.schema import NodeWithScore, TextNode

        class LlamaIndexRetriever(BaseRetriever):
            def __init__(self, shared_search: SharedVectorSearch):
                super().__init__()
                self.shared_search = shared_search

            def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
                chunks = self.shared_search.search(query_bundle.query_str, self.shared_search.k)
                return [
                    NodeWithScore(node=TextNode(text=c.text, id_=c.id, metadata={"source": c.source}), score=c.score)
                    for c in chunks
                ]

        llamaindex = LlamaIndexRetriever(search)
    except ImportError as exc:
        raise RuntimeError("Install llama-index-core to use the LlamaIndex comparison pipeline") from exc
    return {"langchain": langchain, "llamaindex": llamaindex}


def _retrieve(retriever: Any, framework: str, question: str) -> list[dict[str, Any]]:
    if framework == "langchain":
        docs = retriever.invoke(question)
        return [{"id": d.metadata["id"], "source": d.metadata["source"], "score": d.metadata["score"], "text": d.page_content} for d in docs]
    from llama_index.core import QueryBundle

    nodes = retriever.retrieve(QueryBundle(query_str=question))
    return [{"id": n.node.node_id, "source": n.node.metadata["source"], "score": float(n.score or 0), "text": n.node.text} for n in nodes]


def evaluate_case(fixture: dict[str, Any], retrieved: list[dict[str, Any]]) -> dict[str, Any]:
    expected = set(fixture.get("expected_sources", []))
    retrieved_sources = {item["source"] for item in retrieved}
    citations = set(fixture.get("citations", []))
    answer = fixture.get("answer", "")
    citation_hits = citations & expected
    return {
        "id": fixture.get("id", fixture["question"]),
        "expected_supported": fixture["expected_supported"],
        "retrieval_supported": bool(retrieved_sources & expected),
        "retrieved_sources": sorted(retrieved_sources),
        "citation_precision": len(citation_hits) / len(citations) if citations else None,
        "citation_recall": len(citation_hits) / len(expected) if expected else None,
        "unsupported_answer": not fixture["expected_supported"] and bool(answer.strip()),
        "citation_failures": sorted(citations - retrieved_sources),
        "retrieved": retrieved,
    }


def compare(index_dir: str | Path, fixture_path: str | Path, model_name: str, k: int = 5) -> dict[str, Any]:
    if k < 1:
        raise ValueError("k must be at least 1")
    fixtures = load_fixtures(fixture_path)
    _, chunks, embeddings = load_index(index_dir)
    if embeddings is None:
        raise RuntimeError("Comparison requires embeddings.npy in the existing index")
    search = SharedVectorSearch(chunks, embeddings, model_name, k)
    retrievers = framework_retrievers(search)
    report: dict[str, Any] = {"shared_index": str(index_dir), "fixtures": str(fixture_path), "pipelines": {}}
    for name, retriever in retrievers.items():
        rows = []
        for fixture in fixtures:
            retrieved = _retrieve(retriever, name, fixture["question"])[:k]
            rows.append(evaluate_case(fixture, retrieved))
        supported_cases = [row for row in rows if row["expected_supported"]]
        scored = [row for row in rows if row["citation_precision"] is not None]
        report["pipelines"][name] = {
            "summary": {
                "retrieval_support_rate": sum(row["retrieval_supported"] for row in supported_cases) / len(supported_cases) if supported_cases else None,
                "unsupported_cases": sum(not row["expected_supported"] for row in rows),
                "mean_citation_precision": sum(row["citation_precision"] for row in scored) / len(scored) if scored else None,
                "mean_citation_recall": sum(row["citation_recall"] for row in scored if row["citation_recall"] is not None) / sum(row["citation_recall"] is not None for row in scored) if any(row["citation_recall"] is not None for row in scored) else None,
                "unsupported_answers": sum(row["unsupported_answer"] for row in rows),
                "citation_failures": sum(len(row["citation_failures"]) for row in rows),
            },
            "cases": rows,
        }
    return report
