import typer

from petmed_rag.ingestion.ingest import ingest_folder
from petmed_rag.retrieval.service import retrieve_context
from petmed_rag.generation.answer import generate_answer

app = typer.Typer(add_completion=False)


@app.command()
def ingest():
    """Ingest processed documents into the Chroma vector store."""
    n = ingest_folder(
        input_dir="./data/processed",
        persist_dir="./data/chroma",
        collection="cat-health",
        embedder="openai",
        openai_model="text-embedding-3-small",
    )
    typer.echo(f"Ingested {n} chunks")


@app.command()
def ask(question: str):
    """Ask the RAG system a question."""
    typer.echo(f"\nQuestion: {question}\n")

    chunks = retrieve_context(question, k=5)

    if not chunks:
        typer.echo("No relevant documents found.")
        raise typer.Exit()

    answer = generate_answer(question, chunks)

    typer.echo("Answer:\n")
    typer.echo(answer)

    typer.echo("\nSources:\n")

    seen = set()
    
    for c in chunks:
        source = (
            c.metadata.get("file_name")
            or c.metadata.get("source_path")
            or c.metadata.get("doc_id")
            or "unknown source"
        )

        if source not in seen:
            typer.echo(f"- {source}")
            seen.add(source)


def main():
    app()


if __name__ == "__main__":
    main()