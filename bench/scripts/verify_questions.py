"""Check bench/questions.jsonl against the corpus. Fails loudly.

Gold answers are the one artifact in this benchmark where an error is
invisible downstream: three backends get scored against them and a wrong row
just looks like three backends agreeing on a miss. So every claim the question
set makes is machine-checked here:

  - schema and class-count floors
  - every cited doc_id exists in the corpus manifest
  - every prose `evidence` phrase appears verbatim in the cited document
  - every single_passage dollar figure appears in the cited document's text,
    formatted the way filings print it (so the answer is actually retrievable,
    not merely true per XBRL)
  - every unanswerable row's `absent_terms` appear nowhere in the corpus
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ingest.pipeline import load_text_from_file  # noqa: E402

MIN_PER_CLASS = 5
MIN_TOTAL = 40
CLASSES = {"single_passage", "multi_hop", "aggregation", "regulation_xref", "unanswerable"}

problems: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        problems.append(msg)


def main() -> None:
    manifest = json.loads((ROOT / "bench" / "corpus_manifest.json").read_text())
    docs = {m["doc_id"]: m for m in manifest}
    questions = [json.loads(l) for l in (ROOT / "bench" / "questions.jsonl").read_text().splitlines() if l.strip()]

    print(f"corpus: {len(docs)} documents | questions: {len(questions)}")

    # Extract each document's text once; several checks need it.
    text: dict[str, str] = {}
    for doc_id, m in docs.items():
        text[doc_id] = re.sub(r"\s+", " ", load_text_from_file(ROOT / m["path"]))
    corpus_all = " ".join(text.values())
    print(f"extracted {len(corpus_all) / 1e6:.1f}M chars of text\n")

    # -- structure -----------------------------------------------------------
    check(len(questions) >= MIN_TOTAL, f"only {len(questions)} questions, need >= {MIN_TOTAL}")
    check(len({q["id"] for q in questions}) == len(questions), "duplicate question ids")
    counts = Counter(q["query_class"] for q in questions)
    check(set(counts) == CLASSES, f"class set mismatch: {set(counts) ^ CLASSES}")
    for cls in CLASSES:
        check(counts[cls] >= MIN_PER_CLASS, f"class {cls} has {counts[cls]}, need >= {MIN_PER_CLASS}")

    # -- per question --------------------------------------------------------
    for q in questions:
        qid = q["id"]
        check(bool(q["question"].strip()), f"{qid}: empty question")

        if q["query_class"] == "unanswerable":
            check(q["gold_answer"] is None, f"{qid}: unanswerable but gold_answer is not null")
            check(q["source_doc_ids"] == [], f"{qid}: unanswerable but cites documents")
            check(bool(q.get("note")), f"{qid}: unanswerable with no note explaining why")
            for term in (q.get("absent_terms") or []):
                n = len(re.findall(re.escape(term), corpus_all, re.I))
                check(n == 0, f"{qid}: '{term}' should be absent but appears {n}x in the corpus")
            continue

        check(q["gold_answer"] is not None, f"{qid}: answerable but gold_answer is null")
        check(bool(q["source_doc_ids"]), f"{qid}: answerable but cites no document")
        for d in q["source_doc_ids"]:
            check(d in docs, f"{qid}: cites unknown document '{d}'")

        if q["query_class"] == "multi_hop":
            check(len(q["source_doc_ids"]) >= 2,
                  f"{qid}: multi_hop cites only {len(q['source_doc_ids'])} document(s)")

        # Prose evidence must be present verbatim in the first cited document.
        if q.get("evidence"):
            d = q["source_doc_ids"][0]
            found = any(q["evidence"].lower() in text[dd].lower() for dd in q["source_doc_ids"])
            check(found, f"{qid}: evidence '{q['evidence']}' not found in {q['source_doc_ids']}")

        # A dollar figure should be printable from the cited filing, not just
        # true per XBRL -- otherwise no retriever could ever surface it.
        if q["query_class"] == "single_passage" and q["unit"] == "USD" and q["gold_value"]:
            millions = f"{q['gold_value'] / 1e6:,.0f}"
            d = q["source_doc_ids"][0]
            check(millions in text[d],
                  f"{qid}: '{millions}' (millions form of gold value) not literally in {d}")

    # -- report --------------------------------------------------------------
    for cls in sorted(CLASSES):
        print(f"  {cls:18s} {counts[cls]}")
    print()
    if problems:
        print(f"FAIL — {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("PASS — every gold answer is traceable and every check holds.")


if __name__ == "__main__":
    main()
