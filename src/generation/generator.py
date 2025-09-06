from __future__ import annotations
from dataclasses import dataclass
from typing import List
import os

from src.retrieval.retriever import RetrievedChunk
from src.config import settings

try:
    import google.generativeai as genai
except Exception:
    genai = None  # type: ignore


@dataclass
class Answer:
    text: str
    sources: List[dict]


def format_prompt(query: str, contexts: List[RetrievedChunk]) -> str:
    citations = "\n\n".join([f"[source {i+1}] {c.source}\n{c.text}" for i, c in enumerate(contexts)])
    return (
        "You are a financial analyst assistant. Use the sources to answer clearly for non-experts.\n"
        "Explain financial ratios, highlight risks, and summarize tax implications when relevant.\n"
        "Cite sources as [source N] inline where used. If unsure, say so.\n\n"
        f"Question: {query}\n\nSources:\n{citations}\n\nAnswer:"
    )


def generate_answer(query: str, contexts: List[RetrievedChunk]) -> Answer:
    prompt = format_prompt(query, contexts)
    if genai and settings.gemini_api_key:
        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(prompt)
        text = resp.text.strip() if hasattr(resp, "text") and resp.text else ""
        if not text:
            text = "No response generated."
    else:
        text = (
            "LLM provider not configured. Please set GEMINI_API_KEY or plug in your model.\n"
            "Context provided includes citations."
        )
    sources = [{"id": c.id, "source": c.source, "score": c.score} for c in contexts]
    return Answer(text=text, sources=sources)