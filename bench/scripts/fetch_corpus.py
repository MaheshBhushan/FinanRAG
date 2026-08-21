"""Fetch a small, real SEC EDGAR corpus for the retrieval benchmark.

Selects filings via EDGAR's browse-edgar atom feed filtered by form type, not
via the submissions "recent" block. The recent block only spans ~1 year, which
silently gave some issuers one 10-K and others two -- and period-over-period
multi-hop questions need two fiscal years for every issuer.

Financial-sector issuers (JPM, GS, SCHW, BLK) carry the capital/adviser
regulation language that the regulation-cross-reference query class needs;
AAPL and MSFT are the non-financial control group.

Writes primary documents into data/ and a manifest carrying the stable
document ids that gold answers cite.
"""
from __future__ import annotations

import gzip
import json
import re
import time
import urllib.request
from pathlib import Path

# EDGAR requires a self-identifying User-Agent with a contact address.
UA = "FinanRAG-bench mahesh nanim4490@gmail.com"

ISSUERS = {
    "JPM": 19617,      # JPMorgan Chase  - Basel / Dodd-Frank capital rules
    "GS": 886982,      # Goldman Sachs   - same regime, different framing
    "SCHW": 316709,    # Charles Schwab  - broker-dealer net capital rules
    "BLK": 2012383,    # BlackRock, Inc. - post-2024 reorg entity, not 1364742
    "AAPL": 320193,    # Apple           - non-financial control
    "MSFT": 789019,    # Microsoft       - non-financial control
}

# Per issuer: how many of each form to take, newest first.
WANT = {"10-K": 2, "10-Q": 1}

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=90) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
    time.sleep(0.2)  # stay well under EDGAR's 10 req/s limit
    return raw


def list_filings(cik: int, form: str, count: int) -> list[dict]:
    """Most recent `count` filings of `form`, via the atom feed."""
    feed = get(
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
        f"&CIK={cik:010d}&type={form}&dateb=&owner=include&count={count * 3}&output=atom"
    ).decode("utf-8", "ignore")

    out = []
    for entry in re.findall(r"<entry>(.*?)</entry>", feed, re.S):
        # Only exact form matches: a "10-K" query also returns 10-K/A, NT 10-K.
        ftype = re.search(r"<filing-type>(.*?)</filing-type>", entry)
        if not ftype or ftype.group(1).strip() != form:
            continue
        href = re.search(r"<filing-href>(.*?)</filing-href>", entry)
        date = re.search(r"<filing-date>(.*?)</filing-date>", entry)
        if not (href and date):
            continue
        out.append({"index_url": href.group(1).strip(), "filed": date.group(1).strip()})
        if len(out) >= count:
            break
    return out


def primary_doc(index_url: str) -> tuple[str, str]:
    """(document url, period of report) from a filing index."""
    base = index_url.rsplit("/", 1)[0]
    idx = json.loads(get(base + "/index.json"))

    period = ""
    page = get(index_url).decode("utf-8", "ignore")
    m = re.search(r"Period of Report.*?(\d{4}-\d{2}-\d{2})", page, re.S)
    if m:
        period = m.group(1)

    # The primary document is the largest top-level .htm that isn't the index
    # or an XBRL sidecar. Size is a more reliable signal here than naming.
    best, best_size = None, -1
    for item in idx["directory"]["item"]:
        name = item["name"]
        if not name.lower().endswith((".htm", ".html")):
            continue
        if re.search(r"(index|-index|R\d+|FilingSummary)", name, re.I):
            continue
        size = int(item.get("size") or 0)
        if size > best_size:
            best, best_size = name, size
    if not best:
        raise RuntimeError(f"no primary document in {index_url}")
    return f"{base}/{best}", period


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for stale in DATA.glob("*"):
        stale.unlink()

    manifest = []
    for ticker, cik in ISSUERS.items():
        for form, count in WANT.items():
            for f in list_filings(cik, form, count):
                try:
                    url, period = primary_doc(f["index_url"])
                except Exception as e:
                    print(f"  SKIP {ticker} {form} {f['filed']}: {e}")
                    continue
                period = period or f["filed"]
                # Document id: what a gold answer cites.
                doc_id = f"{ticker}-{form.replace('-', '')}-{period}"
                out = DATA / f"{doc_id}.htm"
                out.write_bytes(get(url))
                manifest.append({
                    "doc_id": doc_id, "ticker": ticker, "cik": cik, "form": form,
                    "period": period, "filed": f["filed"],
                    "path": str(out.relative_to(ROOT)),
                    "bytes": out.stat().st_size, "url": url,
                })
                print(f"  {doc_id}  {out.stat().st_size / 1e6:.1f} MB")

    (ROOT / "bench" / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)} documents, {sum(m['bytes'] for m in manifest) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
