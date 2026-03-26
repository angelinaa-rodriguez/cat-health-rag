import typer

from petmed_rag.config import PROCESSED_DIR, CHROMA_DIR
from petmed_rag.utils.citations import unique_sources
from petmed_rag.ingestion.ingest import ingest_processed_documents
from petmed_rag.retrieval.service import retrieve_context
from petmed_rag.generation.answer import generate_answer

app = typer.Typer(add_completion=False)


@app.command()
def ingest():
    """Ingest processed documents into the Chroma vector store."""
    n = ingest_processed_documents()
    typer.echo(f"Ingested {n} chunks")


@app.command()
def ask(question: str):
    """Ask the RAG system a question."""
    typer.echo(f"\nQuestion: {question}\n")

    chunks = retrieve_context(question, k=5)

    if not chunks:
        typer.echo("No relevant documents found.")
        raise typer.Exit()

    typer.echo("Retrieved context:\n")
    for i, c in enumerate(chunks, start=1):
        source = (
            c.metadata.get("title")
            or c.metadata.get("publisher")
            or c.metadata.get("file_name")
            or c.metadata.get("doc_id")
            or "Unknown source"
        )
        preview = c.text[:180].replace("\n", " ").strip()
        typer.echo(f"[{i}] {source}")
        typer.echo(f"    score={c.score:.3f}")
        typer.echo(f"    {preview}...\n")

    answer = generate_answer(question, chunks)

    typer.echo("Answer:\n")
    typer.echo(answer)

    typer.echo("\nSources:\n")
    for i, source in enumerate(unique_sources(chunks), start=1):
        typer.echo(f"[{i}] {source['title']}")


def main():
    app()


if __name__ == "__main__":
    main()