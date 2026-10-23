from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from src.retrieval.retriever import RetrievedChunk

ABSTAIN_PREFIX = "I cannot answer this from the indexed documents"
MIN_SCORE = 0.35
# Money or percentage figures such as "$4.2 billion", "12.5%", "EUR 310 million".
FIGURE_RE = re.compile(
    r"(?:[$€£]|USD|EUR|GBP)\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:billion|million|thousand|bn|m|k))?|\d[\d,]*(?:\.\d+)?\s?%",
    re.IGNORECASE,
)
STOPWORDS = {"the", "a", "an", "of", "for", "in", "on", "to", "and", "or", "what", "was", "were", "is", "did", "does", "how", "much", "its", "it", "by", "with", "company", "company's"}


def _terms(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9']+", text.lower()) if t not in STOPWORDS and len(t) > 2}


def _figures(text: str) -> set[str]:
    return {re.sub(r"[\s,]", "", m.group(0).lower()) for m in FIGURE_RE.finditer(text)}


def _as_chunk(item: RetrievedChunk | dict[str, Any]) -> RetrievedChunk:
    if isinstance(item, RetrievedChunk):
        return item
    return RetrievedChunk(id=str(item["id"]), text=str(item.get("text", "")), source=str(item["source"]), score=float(item.get("score", 0.0)))


def diagnose(question: str, retrieved: Iterable[RetrievedChunk | dict[str, Any]], min_score: float = MIN_SCORE) -> dict[str, Any]:
    """Explain what retrieval found and decide whether an answer is supported.

    A question is unsupported when no chunk clears ``min_score`` or when no relevant
    chunk shares a content term with the question. Relevant chunks from different
    sources that report disjoint figures count as a conflict.
    """
    chunks = [_as_chunk(c) for c in retrieved]
    query_terms = _terms(question)
    evidence = []
    for rank, chunk in enumerate(chunks, start=1):
        overlap = sorted(query_terms & _terms(chunk.text))
        evidence.append({
            "rank": rank,
            "id": chunk.id,
            "source": chunk.source,
            "score": round(chunk.score, 4),
            "above_threshold": chunk.score >= min_score,
            "term_overlap": overlap,
            "figures": sorted(_figures(chunk.text)),
        })
    relevant = [e for e in evidence if e["above_threshold"] and e["term_overlap"]]

    figures_by_source: dict[str, set[str]] = {}
    for e in relevant:
        if e["figures"]:
            figures_by_source.setdefault(e["source"], set()).update(e["figures"])
    conflicts = []
    sources = sorted(figures_by_source)
    for i, a in enumerate(sources):
        for b in sources[i + 1:]:
            if not figures_by_source[a] & figures_by_source[b]:
                conflicts.append({"sources": [a, b], "figures": {a: sorted(figures_by_source[a]), b: sorted(figures_by_source[b])}})

    reasons = []
    if not chunks:
        reasons.append("no_chunks_retrieved")
    elif not any(e["above_threshold"] for e in evidence):
        reasons.append("top_score_below_threshold")
    elif not relevant:
        reasons.append("no_term_overlap")
    if conflicts:
        reasons.append("conflicting_sources")

    scores = [c.score for c in chunks]
    return {
        "question": question,
        "min_score": min_score,
        "top_score": round(max(scores), 4) if scores else None,
        "score_gap": round(scores[0] - scores[1], 4) if len(scores) > 1 else None,
        "distinct_sources": sorted({c.source for c in chunks}),
        "evidence": evidence,
        "supporting_sources": sorted({e["source"] for e in relevant}),
        "conflicts": conflicts,
        "abstain": bool(reasons),
        "abstain_reasons": reasons,
    }


def abstention_message(diagnostics: dict[str, Any]) -> str:
    reasons = diagnostics["abstain_reasons"]
    if "conflicting_sources" in reasons:
        detail = "; ".join(f"{s} reports {', '.join(c['figures'][s])}" for c in diagnostics["conflicts"] for s in c["sources"])
        return f"{ABSTAIN_PREFIX}: the sources disagree ({detail})."
    return f"{ABSTAIN_PREFIX}: no retrieved source supports it ({', '.join(reasons)})."


def is_abstention(answer: str) -> bool:
    return answer.strip().startswith(ABSTAIN_PREFIX)


def evaluate_abstention(fixture_path: str | Path, min_score: float = MIN_SCORE) -> dict[str, Any]:
    """Run fixed cases with recorded retrieval results and check the abstention decision."""
    cases = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Abstention fixture must contain a non-empty JSON array")
    rows = []
    for i, case in enumerate(cases):
        if not isinstance(case.get("expected"), str) or case["expected"] not in {"answer", "abstain"}:
            raise ValueError(f"Case {i} must set expected to 'answer' or 'abstain'")
        diag = diagnose(case["question"], case.get("retrieved", []), min_score)
        expected_sources = set(case.get("expected_sources", []))
        rows.append({
            "id": case.get("id", case["question"]),
            "category": case.get("category"),
            "expected": case["expected"],
            "decision": "abstain" if diag["abstain"] else "answer",
            "correct": (case["expected"] == "abstain") == diag["abstain"],
            "evidence_links_ok": expected_sources <= set(diag["supporting_sources"]),
            "abstain_reasons": diag["abstain_reasons"],
            "diagnostics": diag,
        })
    unsupported = [r for r in rows if r["expected"] == "abstain"]
    return {
        "fixtures": str(fixture_path),
        "summary": {
            "cases": len(rows),
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "abstention_recall": sum(r["decision"] == "abstain" for r in unsupported) / len(unsupported) if unsupported else None,
            "invented_answers": sum(r["decision"] == "answer" for r in unsupported),
            "evidence_link_failures": sum(not r["evidence_links_ok"] for r in rows if r["expected"] == "answer"),
        },
        "cases": rows,
    }
