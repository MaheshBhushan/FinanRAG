# FinanRAG - Standalone Financial Analyst RAG

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)

A Retrieval-Augmented Generation (RAG) assistant that ingests financial statements, tax regulations, and investment reports, then answers questions with clear explanations, risk highlights, tax summaries, and inline source citations.

## Features
- Ingestion: PDF, HTML/HTM, CSV, TXT/MD
- Semantic retrieval: Sentence-Transformers + FAISS
- Generation: LLM with citation-aware prompt
- Outputs: Human-readable answer + JSON citations

## Prerequisites
- Python 3.10+
- An LLM API key (e.g., Gemini): set GEMINI_API_KEY
- FAISS:
  - Windows: recommended via Conda (see below)
  - macOS/Linux: pip often works; Conda is an alternative

## Setup

### Option A (Windows, recommended via Conda)
```bash
conda create -n finanrag python=3.10 -y
conda activate finanrag
conda install -c conda-forge faiss-cpu -y
pip install -U pip
pip install -r requirements.txt
# Optional: install package to enable the `finanrag` CLI entrypoint
pip install -e .
```

### Option B (macOS/Linux)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# If FAISS errors, use Conda instead:
#   conda create -n finanrag python=3.10 -y && conda activate finanrag
#   conda install -c conda-forge faiss-cpu -y
#   pip install -r requirements.txt
#   pip install -e .  # optional, for `finanrag` CLI
```

## Configuration
Set environment variables as needed:
- GEMINI_API_KEY: your LLM API key
- DATA_DIR: default data
- INDEX_DIR: default indexes
- EMBEDDING_MODEL: default sentence-transformers/all-MiniLM-L6-v2

Examples:
- Windows (PowerShell): `$env:GEMINI_API_KEY="..."`
- Windows (cmd): `set GEMINI_API_KEY=...`
- macOS/Linux: `export GEMINI_API_KEY=...`

## Add Documents
Place your sources under `data/`:
- Financial statements (PDF/TXT/HTML)
- Tax regulations (PDF/HTML)
- Investment reports (PDF/TXT/HTML)

## Build Index
Using the installed CLI (preferred):
```bash
finanrag ingest --data-dir data --index-dir indexes
```
Or without install:
```bash
python scripts/cli.py ingest --data-dir data --index-dir indexes
```

## Ask Questions
CLI:
```bash
finanrag ask "Explain current ratio, risks, and tax implications for X"
```
Or:
```bash
python scripts/cli.py ask "Explain current ratio, risks, and tax implications for X"
```
The tool returns an answer and JSON with `[source N]` citations and file paths.

## Compare Retrieval Pipelines

Create a held-out fixture file from questions reviewed against your document collection. Copy `fixtures/retrieval_questions.example.json` to `fixtures/retrieval_questions.json`, then replace its example questions, source paths, expected support labels, and optional answer/citation records with reviewed cases. Do not tune the fixtures against retrieval results. `expected_sources` contains the source paths that support a question; `citations` and `answer` let the report flag citation misses and unsupported answers in saved responses.

Build the normal index first, then run:

```bash
python scripts/cli.py compare --index-dir indexes --fixtures fixtures/retrieval_questions.json
```

The command exposes the same stored chunks and normalized embeddings through LangChain and LlamaIndex retrievers. It reports per-pipeline retrieval support, citation precision and recall, citation failures, and unsupported answers. It does not generate answers or make model API calls, so the `answer` and `citations` fixture fields must contain responses recorded separately when answer-level checks are needed.

## Retrieval Diagnostics and Abstention

`generate_answer` runs retrieval diagnostics before it calls the LLM. The diagnostics list each retrieved chunk with its rank, score, source, query-term overlap, and money or percentage figures. The assistant abstains with a message starting "I cannot answer this from the indexed documents" when no chunk was retrieved, when no chunk reaches the score threshold (default 0.35), when no relevant chunk shares a content term with the question, or when relevant chunks from different sources report disjoint figures. Pass `--diagnostics` to `ask` to print the full report.

`fixtures/abstention_cases.json` holds fixed questions with recorded retrieval results: supported answers, missing years, missing topics, an empty index, and conflicting revenue figures. Run them without an index or model:

```bash
python scripts/cli.py abstention --fixtures fixtures/abstention_cases.json
python -m unittest tests/test_abstention.py
```

The report shows decision accuracy, abstention recall, invented answers on unsupported cases, and evidence-link failures for supported cases.

## Examples
- "What does debt-to-equity indicate and what risks does a high D/E pose?"
- "Summarize Section 179 deduction limits and phase-out rules."
- "Compare gross margin vs. operating margin for Company A and explain the drivers."

## Troubleshooting
- FAISS not available: the app will fall back to NumPy cosine search if `embeddings.npy` exists. Otherwise, install FAISS (e.g., `conda install -c conda-forge faiss-cpu`) and re-run ingestion.
- Empty results: ensure files exist in `data/` and are readable.
- Large PDFs: ingestion may take time; ensure adequate memory.

## Notes
- This assistant provides educational information and does not constitute financial or tax advice.
