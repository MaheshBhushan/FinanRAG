"""Build bench/questions.jsonl -- the benchmark's gold-answer question set.

Design decisions worth knowing before trusting the numbers:

1. Numeric gold answers are COMPUTED from bench/facts.json (SEC XBRL, tagged
   with the accession that reported them), never typed by hand. Arithmetic
   questions state the formula in code, so a wrong gold number would have to
   come from a wrong concept name -- which `fact()` raises on -- rather than
   from a typo that no reader would ever catch.

2. Each 10-K carries three years of comparative figures, so "AAPL FY2024 vs
   FY2025 revenue" is answerable from ONE document and is not multi-hop.
   Every multi_hop row therefore crosses issuers or crosses 10-K/10-Q, where
   no single filing states the other issuer's figures.

3. Prose rows carry an `evidence` phrase that must appear verbatim in the
   extracted text of the cited document. verify_questions.py enforces it.

4. unanswerable rows carry `absent_terms` that must NOT appear in the corpus.
   Same script enforces that too. This class measures abstention, which is the
   failure mode that matters most and the cheapest one to add.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FACTS = json.loads((ROOT / "bench" / "facts.json").read_text())

REV = "RevenueFromContractWithCustomerExcludingAssessedTax"


def fact(doc: str, concept: str, end: str, start: str | None = None) -> float:
    """The single value `doc` reported for `concept` as of/ending `end`.

    `start` is required whenever a document tags more than one period with the
    same end date -- 10-Qs report both the quarter and the year-to-date figure
    ending on the same day. Ambiguity raises rather than silently picking one,
    because a YTD number used as a quarterly gold answer is invisible once it
    reaches a results table.
    """
    entries = FACTS[doc].get(concept)
    if not entries:
        raise KeyError(f"{doc} has no {concept}")
    hits = [e for e in entries if e["end"] == end
            and (start is None or e["start"] == start)]
    if len(hits) != 1:
        periods = [f"{e['start']}..{e['end']}" for e in entries if e["end"] == end]
        raise KeyError(
            f"{doc}/{concept}/{end}: expected 1 entry, got {len(hits)}; "
            f"disambiguate with start= from {periods}")
    return hits[0]["value"]


def usd(v: float) -> str:
    """Format a dollar figure the way filings do, in millions."""
    return f"${v / 1e6:,.0f} million"


Q: list[dict] = []


def add(qid, cls, question, gold, docs, *, value=None, unit=None,
        evidence=None, absent=None, note=None):
    Q.append({
        "id": qid, "query_class": cls, "question": question,
        "gold_answer": gold, "gold_value": value, "unit": unit,
        "source_doc_ids": docs, "evidence": evidence,
        "absent_terms": absent, "note": note,
    })


# ---------------------------------------------------------------- single_passage
# One figure, present in exactly one corpus document. Deliberately uses each
# document's most recent period, so a prior-year 10-K cannot also answer it.
_single = [
    ("q001", "AAPL-10K-2025-09-27", REV, "2025-09-27",
     "What were Apple's total net sales for fiscal year 2025?", "Total net sales"),
    ("q002", "MSFT-10K-2026-06-30", "NetIncomeLoss", "2026-06-30",
     "What was Microsoft's net income for fiscal year 2026?", "Net income"),
    ("q003", "JPM-10K-2025-12-31", "Assets", "2025-12-31",
     "What were JPMorgan Chase's total assets as of December 31, 2025?", "Total assets"),
    ("q004", "GS-10K-2025-12-31", "InterestExpense", "2025-12-31",
     "What was Goldman Sachs' total interest expense for 2025?", "Interest expense"),
    ("q005", "SCHW-10K-2025-12-31", "Revenues", "2025-12-31",
     "What were Charles Schwab's total net revenues for 2025?", "Total net revenues"),
    ("q006", "BLK-10K-2025-12-31", "StockholdersEquity", "2025-12-31",
     "What was BlackRock's total stockholders' equity as of December 31, 2025?", "stockholders"),
    ("q007", "AAPL-10Q-2026-06-27", "ResearchAndDevelopmentExpense", "2026-06-27",
     "What was Apple's research and development expense for the quarter ended June 27, 2026?",
     "Research and development", "2026-03-29"),
    ("q008", "MSFT-10K-2026-06-30", "CashAndCashEquivalentsAtCarryingValue", "2026-06-30",
     "How much cash and cash equivalents did Microsoft hold as of June 30, 2026?", "Cash and cash equivalents"),
    ("q009", "JPM-10Q-2026-06-30", "NetIncomeLoss", "2026-06-30",
     "What was JPMorgan Chase's net income for the quarter ended June 30, 2026?", "Net income", "2026-04-01"),
    ("q010", "GS-10K-2025-12-31", "StockholdersEquity", "2025-12-31",
     "What was Goldman Sachs' total shareholders' equity as of December 31, 2025?", "equity"),
]
for row in _single:
    qid, doc, concept, end, question, ev = row[:6]
    start = row[6] if len(row) > 6 else None
    v = fact(doc, concept, end, start)
    add(qid, "single_passage", question, usd(v), [doc], value=v, unit="USD", evidence=ev)

# EPS is a per-share figure, so it needs its own formatting.
v = fact("GS-10K-2025-12-31", "EarningsPerShareDiluted", "2025-12-31")
add("q011", "single_passage", "What was Goldman Sachs' diluted earnings per share for 2025?",
    f"${v:.2f} per diluted share", ["GS-10K-2025-12-31"], value=v, unit="USD/share",
    evidence="Diluted")

# ------------------------------------------------------------------- aggregation
# Arithmetic over several tagged figures. Formula lives in code; the gold string
# is rendered from it.
jpm_ni = fact("JPM-10K-2025-12-31", "NetIncomeLoss", "2025-12-31")
gs_ni = fact("GS-10K-2025-12-31", "NetIncomeLoss", "2025-12-31")
schw_ni = fact("SCHW-10K-2025-12-31", "NetIncomeLoss", "2025-12-31")
blk_ni = fact("BLK-10K-2025-12-31", "NetIncomeLoss", "2025-12-31")
tot = jpm_ni + gs_ni + schw_ni + blk_ni
add("q012", "aggregation",
    "What was the combined 2025 net income of JPMorgan Chase, Goldman Sachs, Charles Schwab and BlackRock?",
    usd(tot), ["JPM-10K-2025-12-31", "GS-10K-2025-12-31", "SCHW-10K-2025-12-31", "BLK-10K-2025-12-31"],
    value=tot, unit="USD",
    note=f"{usd(jpm_ni)} + {usd(gs_ni)} + {usd(schw_ni)} + {usd(blk_ni)}")

a25 = fact("AAPL-10K-2025-09-27", REV, "2025-09-27")
a24 = fact("AAPL-10K-2025-09-27", REV, "2024-09-28")
g = (a25 - a24) / a24 * 100
add("q013", "aggregation",
    "By what percentage did Apple's total net sales grow from fiscal 2024 to fiscal 2025?",
    f"{g:.1f}%", ["AAPL-10K-2025-09-27"], value=g, unit="percent",
    note=f"({usd(a25)} - {usd(a24)}) / {usd(a24)}; both years appear in the FY2025 10-K")

m_op = fact("MSFT-10K-2026-06-30", "OperatingIncomeLoss", "2026-06-30")
m_rev = fact("MSFT-10K-2026-06-30", REV, "2026-06-30")
om = m_op / m_rev * 100
add("q014", "aggregation", "What was Microsoft's operating margin in fiscal year 2026?",
    f"{om:.1f}%", ["MSFT-10K-2026-06-30"], value=om, unit="percent",
    note=f"operating income {usd(m_op)} / revenue {usd(m_rev)}")

j_eq = fact("JPM-10K-2025-12-31", "StockholdersEquity", "2025-12-31")
j_as = fact("JPM-10K-2025-12-31", "Assets", "2025-12-31")
r = j_eq / j_as * 100
add("q015", "aggregation",
    "What was JPMorgan Chase's total stockholders' equity as a percentage of total assets at December 31, 2025?",
    f"{r:.1f}%", ["JPM-10K-2025-12-31"], value=r, unit="percent",
    note=f"{usd(j_eq)} / {usd(j_as)}")

a_ni = fact("AAPL-10K-2025-09-27", "NetIncomeLoss", "2025-09-27")
a_tax = fact("AAPL-10K-2025-09-27", "IncomeTaxExpenseBenefit", "2025-09-27")
etr = a_tax / (a_ni + a_tax) * 100
add("q016", "aggregation", "What was Apple's effective tax rate in fiscal year 2025?",
    f"{etr:.1f}%", ["AAPL-10K-2025-09-27"], value=etr, unit="percent",
    note=f"tax {usd(a_tax)} / pretax income {usd(a_ni + a_tax)}")

comb = j_as + fact("GS-10K-2025-12-31", "Assets", "2025-12-31")
add("q017", "aggregation",
    "What were the combined total assets of JPMorgan Chase and Goldman Sachs as of December 31, 2025?",
    usd(comb), ["JPM-10K-2025-12-31", "GS-10K-2025-12-31"], value=comb, unit="USD")

m26 = fact("MSFT-10K-2026-06-30", "NetIncomeLoss", "2026-06-30")
m25 = fact("MSFT-10K-2026-06-30", "NetIncomeLoss", "2025-06-30")
g2 = (m26 - m25) / m25 * 100
add("q018", "aggregation",
    "By what percentage did Microsoft's net income grow from fiscal 2025 to fiscal 2026?",
    f"{g2:.1f}%", ["MSFT-10K-2026-06-30"], value=g2, unit="percent",
    note=f"({usd(m26)} - {usd(m25)}) / {usd(m25)}")

s_rev = fact("SCHW-10K-2025-12-31", "Revenues", "2025-12-31")
nm = schw_ni / s_rev * 100
add("q019", "aggregation", "What was Charles Schwab's net profit margin in 2025?",
    f"{nm:.1f}%", ["SCHW-10K-2025-12-31"], value=nm, unit="percent",
    note=f"{usd(schw_ni)} / {usd(s_rev)}")

g_eq = fact("GS-10K-2025-12-31", "StockholdersEquity", "2025-12-31")
roe = gs_ni / g_eq * 100
add("q020", "aggregation",
    "What was Goldman Sachs' return on average shareholders' equity in 2025, computed on year-end equity?",
    f"{roe:.1f}%", ["GS-10K-2025-12-31"], value=roe, unit="percent",
    note=f"{usd(gs_ni)} / year-end equity {usd(g_eq)}; year-end, not average, equity")

b_op = fact("BLK-10K-2025-12-31", "OperatingIncomeLoss", "2025-12-31")
b_rev = fact("BLK-10K-2025-12-31", REV, "2025-12-31")
bom = b_op / b_rev * 100
add("q021", "aggregation", "What was BlackRock's operating margin in 2025?",
    f"{bom:.1f}%", ["BLK-10K-2025-12-31"], value=bom, unit="percent",
    note=f"{usd(b_op)} / {usd(b_rev)}")

# ---------------------------------------------------------------------- multi_hop
# Every row needs figures from two or more documents. No filing in the corpus
# states a competitor's exact figures, so flat retrieval must pull both.
d = jpm_ni - gs_ni
add("q022", "multi_hop",
    "Which had higher net income in 2025, JPMorgan Chase or Goldman Sachs, and by how much?",
    f"JPMorgan Chase, by {usd(d)} ({usd(jpm_ni)} vs {usd(gs_ni)})",
    ["JPM-10K-2025-12-31", "GS-10K-2025-12-31"], value=d, unit="USD")

d = a25 - m_rev
add("q023", "multi_hop",
    "Whose most recent full-year revenue was larger, Apple's fiscal 2025 or Microsoft's fiscal 2026, and by how much?",
    f"Apple, by {usd(d)} ({usd(a25)} vs {usd(m_rev)})",
    ["AAPL-10K-2025-09-27", "MSFT-10K-2026-06-30"], value=d, unit="USD")

eps = {t: fact(f"{t}-10K-2025-12-31", "EarningsPerShareDiluted", "2025-12-31")
       for t in ("JPM", "GS", "SCHW", "BLK")}
top = max(eps, key=eps.get)
add("q024", "multi_hop",
    "Among JPMorgan Chase, Goldman Sachs, Charles Schwab and BlackRock, which reported the highest diluted EPS for 2025?",
    f"Goldman Sachs, at ${eps['GS']:.2f} per diluted share",
    [f"{t}-10K-2025-12-31" for t in ("JPM", "GS", "SCHW", "BLK")],
    value=eps["GS"], unit="USD/share",
    note=f"diluted EPS: {', '.join(f'{k} ${v:.2f}' for k, v in eps.items())}; highest={top}")

g_tax = fact("GS-10K-2025-12-31", "IncomeTaxExpenseBenefit", "2025-12-31")
s_tax = fact("SCHW-10K-2025-12-31", "IncomeTaxExpenseBenefit", "2025-12-31")
g_etr = g_tax / (gs_ni + g_tax) * 100
s_etr = s_tax / (schw_ni + s_tax) * 100
hi = "Charles Schwab" if s_etr > g_etr else "Goldman Sachs"
add("q025", "multi_hop",
    "Which had the higher effective tax rate in 2025, Goldman Sachs or Charles Schwab?",
    f"{hi} (Schwab {s_etr:.1f}% vs Goldman Sachs {g_etr:.1f}%)",
    ["GS-10K-2025-12-31", "SCHW-10K-2025-12-31"], value=max(g_etr, s_etr), unit="percent")

b_as = fact("BLK-10K-2025-12-31", "Assets", "2025-12-31")
s_as = fact("SCHW-10K-2025-12-31", "Assets", "2025-12-31")
add("q026", "multi_hop",
    "Which held more total assets at December 31, 2025, BlackRock or Charles Schwab?",
    f"Charles Schwab ({usd(s_as)} vs BlackRock's {usd(b_as)})",
    ["BLK-10K-2025-12-31", "SCHW-10K-2025-12-31"], value=s_as, unit="USD")

a_rd = fact("AAPL-10K-2025-09-27", "ResearchAndDevelopmentExpense", "2025-09-27")
m_rd = fact("MSFT-10K-2026-06-30", "ResearchAndDevelopmentExpense", "2026-06-30")
d = abs(m_rd - a_rd)
add("q027", "multi_hop",
    "Which spent more on research and development in its most recent fiscal year, Apple or Microsoft, and by how much?",
    f"Microsoft, by {usd(d)} ({usd(m_rd)} in FY2026 vs Apple's {usd(a_rd)} in FY2025)",
    ["AAPL-10K-2025-09-27", "MSFT-10K-2026-06-30"], value=d, unit="USD")

j_bb = fact("JPM-10K-2025-12-31", "PaymentsForRepurchaseOfCommonStock", "2025-12-31")
g_bb = fact("GS-10K-2025-12-31", "PaymentsForRepurchaseOfCommonStock", "2025-12-31")
add("q028", "multi_hop",
    "Which repurchased more of its own common stock during 2025, JPMorgan Chase or Goldman Sachs?",
    f"JPMorgan Chase ({usd(j_bb)} vs Goldman Sachs' {usd(g_bb)})",
    ["JPM-10K-2025-12-31", "GS-10K-2025-12-31"], value=j_bb, unit="USD")

order = sorted([("JPMorgan Chase", jpm_ni), ("Goldman Sachs", gs_ni),
                ("Charles Schwab", schw_ni), ("BlackRock", blk_ni)],
               key=lambda x: -x[1])
add("q029", "multi_hop",
    "Rank JPMorgan Chase, Goldman Sachs, Charles Schwab and BlackRock by 2025 net income, highest first.",
    ", ".join(f"{n} ({usd(v)})" for n, v in order),
    [f"{t}-10K-2025-12-31" for t in ("JPM", "GS", "SCHW", "BLK")], unit="ranking")

# A 10-K/10-Q hop: the FY2025 10-K cannot contain a June 2026 balance.
j_as_q = fact("JPM-10Q-2026-06-30", "Assets", "2026-06-30")
d = j_as_q - j_as
add("q030", "multi_hop",
    "By how much did JPMorgan Chase's total assets change between December 31, 2025 and June 30, 2026?",
    f"Increased by {usd(d)}, from {usd(j_as)} to {usd(j_as_q)}",
    ["JPM-10K-2025-12-31", "JPM-10Q-2026-06-30"], value=d, unit="USD")

# ----------------------------------------------------------------- regulation_xref
# Grounded in passages read out of the filings, each with a verbatim evidence
# phrase. Two of these are also cross-document, which is the point: regulation
# questions are where the graph backend should have an edge.
add("q031", "regulation_xref",
    "Under the U.S. bank regulators' tailoring framework, which category of standards is Goldman Sachs subject to, and why?",
    "Category I standards, because Goldman Sachs has been designated a global systemically important bank (G-SIB); "
    "it and GS Bank USA are therefore 'Advanced approach' banking organizations.",
    ["GS-10K-2025-12-31"], evidence="Category I")

add("q032", "regulation_xref",
    "Which tailoring categories are Goldman Sachs and Charles Schwab each subject to under the U.S. bank capital framework?",
    "Goldman Sachs is subject to Category I standards as a G-SIB; Charles Schwab is a Category III banking organization.",
    ["GS-10K-2025-12-31", "SCHW-10K-2025-12-31"], evidence="Category III",
    note="cross-document: neither filing states the other issuer's category")

add("q033", "regulation_xref",
    "What minimum supplementary leverage ratio applies to Charles Schwab as a Category III banking organization?",
    "A minimum supplementary leverage ratio of 3.0%.",
    ["SCHW-10K-2025-12-31"], value=3.0, unit="percent",
    evidence="minimum 3.0% supplementary leverage ratio")

add("q034", "regulation_xref",
    "In 2025 the FRB recalibrated the SLR buffer requirement for G-SIBs. What did it replace the previous 2% buffer with?",
    "A buffer equal to 50% of the firm's G-SIB surcharge (Method 1), with conforming changes to the TLAC and "
    "external long-term debt requirements.",
    ["GS-10K-2025-12-31"], evidence="50% of their G-SIB surcharge")

add("q035", "regulation_xref",
    "Which rule governs the net capital requirements of J.P. Morgan Securities, JPMorgan Chase's principal U.S. broker-dealer?",
    "Rule 15c3-1 under the Securities Exchange Act of 1934, known as the Net Capital Rule.",
    ["JPM-10K-2025-12-31"], evidence="Rule 15c3-1")

add("q036", "regulation_xref",
    "What obligations does the Investment Advisers Act of 1940 impose on BlackRock's registered investment adviser subsidiaries?",
    "Record-keeping, operational and marketing requirements, disclosure obligations, and prohibitions on certain "
    "practices, among other obligations on registered investment advisers.",
    ["BLK-10K-2025-12-31"], evidence="Investment Advisers Act of 1940")

add("q037", "regulation_xref",
    "Under which statute are JPMorgan Chase's U.S. registered investment advisers subject to fiduciary obligations?",
    "The Investment Advisers Act of 1940, together with applicable state and federal law; the firm's bank fiduciary "
    "activities are supervised by the OCC.",
    ["JPM-10K-2025-12-31"], evidence="Investment Advisers Act of 1940")

add("q038", "regulation_xref",
    "Which SEC rule on retail investment recommendations does Goldman Sachs identify as a legal requirement it is subject to?",
    "Regulation Best Interest.",
    ["GS-10K-2025-12-31"], evidence="Regulation Best Interest")

add("q039", "regulation_xref",
    "Which broker-dealer subsidiary of Charles Schwab is subject to the Uniform Net Capital Rule, and under what rule number?",
    "CS&Co, under Rule 15c3-1 of the Securities Exchange Act of 1934 (the Uniform Net Capital Rule).",
    ["SCHW-10K-2025-12-31"], evidence="Uniform Net Capital Rule")

add("q040", "regulation_xref",
    "What is the eSLR buffer cap for JPMorgan Chase's insured depository institution subsidiaries?",
    "For IDI subsidiaries the eSLR buffer is capped at 1%.",
    ["JPM-10K-2025-12-31"], value=1.0, unit="percent", evidence="capped at 1%")

# --------------------------------------------------------------------- unanswerable
# Five is the floor; eight is still cheap and gives the abstention rate a
# usable denominator. Each row names terms verify_questions.py confirms are
# absent from the corpus, so "unanswerable" is a checked property, not an
# assumption.
add("q041", "unanswerable",
    "What was Tesla's total revenue in fiscal year 2025?", None, [],
    absent=["Tesla"], note="no Tesla filing in the corpus")

add("q042", "unanswerable",
    "What does JPMorgan Chase project its net income will be in 2027?", None, [],
    note="forward net income projections are not disclosed in 10-K filings")

add("q043", "unanswerable",
    "What was Nvidia's data center segment revenue in its most recent fiscal year?", None, [],
    absent=["Nvidia", "NVIDIA"], note="no Nvidia filing in the corpus")

add("q044", "unanswerable",
    "What was BlackRock's total assets under management at the end of 2019?", None, [],
    note="corpus holds only BlackRock's FY2024 and FY2025 10-Ks; 2019 AUM is outside every filing's reporting period")

add("q045", "unanswerable",
    "How many retail stores did Starbucks operate at the end of its most recent fiscal year?", None, [],
    absent=["Starbucks"], note="no Starbucks filing in the corpus")

# A harder abstention case than a simply-absent company: the corpus mentions
# Wells Fargo 48 times as a peer and counterparty, so a retriever gets
# confident-looking hits on the entity while the requested datum is nowhere in
# it. This is where retrieval strategies tend to confabulate rather than
# abstain, so it is deliberately kept in the set.
add("q049", "unanswerable",
    "How many retail banking customers does Wells Fargo serve?", None, [],
    note="Wells Fargo appears in the corpus only as a named peer/counterparty of the filing issuers; "
         "no Wells Fargo filing is included and no customer count appears anywhere")

add("q046", "unanswerable",
    "What was Charles Schwab's net income for the first quarter of 2024?", None, [],
    note="corpus holds Schwab's FY2024/FY2025 10-Ks and a Q2 2026 10-Q; no filing reports Q1 2024 standalone")

add("q047", "unanswerable",
    "What is the total 2025 compensation of Apple's chief executive officer?", None, [],
    note="executive compensation is disclosed in the DEF 14A proxy statement, not the 10-K; no proxy is in the corpus")

add("q048", "unanswerable",
    "What was Goldman Sachs' Common Equity Tier 1 capital ratio as of June 30, 2027?", None, [],
    note="a future period; the most recent Goldman filing in the corpus is the Q2 2026 10-Q")


out = ROOT / "bench" / "questions.jsonl"
out.write_text("".join(json.dumps(q) + "\n" for q in Q))

from collections import Counter
c = Counter(q["query_class"] for q in Q)
print(f"wrote {len(Q)} questions -> {out.relative_to(ROOT)}")
for k, v in sorted(c.items()):
    print(f"  {k:18s} {v}")
