"""Pull the XBRL facts our filings actually reported, as gold-answer ground truth.

Hand-reading figures out of 144 MB of inline-XBRL tables is where a question
set silently acquires wrong gold answers. Instead: SEC's companyfacts API
returns every tagged fact with the accession number that reported it, so a
fact can be pinned to one document in our corpus rather than merely to an
issuer and a year.

Output: bench/facts.json, keyed by our doc_ids.
"""
from __future__ import annotations

import gzip
import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

UA = "FinanRAG-bench mahesh nanim4490@gmail.com"
ROOT = Path(__file__).resolve().parents[2]

# Concepts that make good lookup / arithmetic / multi-hop questions and are
# tagged consistently by both financial and non-financial issuers.
CONCEPTS = [
    "Assets", "Liabilities", "StockholdersEquity", "NetIncomeLoss",
    "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
    "CashAndCashEquivalentsAtCarryingValue", "OperatingIncomeLoss",
    "EarningsPerShareDiluted", "EarningsPerShareBasic",
    "IncomeTaxExpenseBenefit", "ResearchAndDevelopmentExpense",
    "CommonStockSharesOutstanding", "InterestExpense",
    "LongTermDebtNoncurrent", "PaymentsForRepurchaseOfCommonStock",
]


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
    return raw


def main() -> None:
    manifest = json.loads((ROOT / "bench" / "corpus_manifest.json").read_text())

    # Our filings, keyed by the accession they were filed under (derived from
    # the archive URL) so XBRL facts can be attributed to a specific doc_id.
    by_accn = {}
    for m in manifest:
        accn = re.search(r"/data/\d+/(\d{18})/", m["url"])
        if accn:
            by_accn[accn.group(1)] = m

    facts: dict[str, dict] = defaultdict(dict)
    for cik in sorted({m["cik"] for m in manifest}):
        cf = json.loads(get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"))
        usgaap = cf.get("facts", {}).get("us-gaap", {})
        for concept in CONCEPTS:
            if concept not in usgaap:
                continue
            for unit, entries in usgaap[concept]["units"].items():
                for e in entries:
                    accn = (e.get("accn") or "").replace("-", "")
                    m = by_accn.get(accn)
                    if not m:
                        continue
                    # Only the consolidated full-period figure, not segments.
                    facts[m["doc_id"]].setdefault(concept, []).append({
                        "value": e["val"], "unit": unit,
                        "start": e.get("start"), "end": e.get("end"),
                        "fy": e.get("fy"), "fp": e.get("fp"), "form": e.get("form"),
                        "frame": e.get("frame"),
                    })
        print(f"CIK {cik}: done")

    out = ROOT / "bench" / "facts.json"
    out.write_text(json.dumps(facts, indent=2))
    print(f"\nfacts for {len(facts)} documents -> {out.relative_to(ROOT)}")
    for doc_id in sorted(facts):
        print(f"  {doc_id}: {len(facts[doc_id])} concepts")


if __name__ == "__main__":
    main()
