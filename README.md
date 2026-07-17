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
