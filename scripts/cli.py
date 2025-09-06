import json
import sys
from pathlib import Path
import typer
from rich import print

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingest.pipeline import ingest_directory
from src.retrieval.retriever import semantic_search
from src.generation.generator import generate_answer

app = typer.Typer(add_completion=False)


@app.command()
def ingest(data_dir: str = "data", index_dir: str = "indexes", model: str | None = None):
    p, c = ingest_directory(data_dir=data_dir, index_dir=index_dir, model_name=model)
    print({"index": p, "chunks": c})


@app.command()
def ask(query: str, k: int = 6, index_dir: str = "indexes", model: str | None = None):
    ctx = semantic_search(query, k=k, model_name=model, index_dir=index_dir)
    ans = generate_answer(query, ctx)
    print(json.dumps({"answer": ans.text, "sources": ans.sources}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()