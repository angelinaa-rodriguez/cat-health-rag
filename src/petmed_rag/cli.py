from __future__ import annotations

import textwrap
import typer
import chromadb

from petmed_rag.embeddings import get_embedder
from petmed_rag.retrieval import ChromaRetriever
from petmed_rag.ingest import ingest_folder

app = typer.Typer()

DEFAULT_PERSIST_DIR = "data/chroma"
DEFAULT_COLLECTION = "petmed_rag"


@app.command()
def retrieve(
    query: str = typer.Argument(..., help="Question to retrieve evidence for."),
    k: int = typer.Option(5, "--k", "-k", help="Number of chunks to retrieve."),
    persist_dir: str = typer.Option(DEFAULT_PERSIST_DIR, "--persist-dir"),
    collection: str = typer.Option(DEFAULT_COLLECTION, "--collection"),
    show_chars: int = typer.Option(500, "--show-chars"),
) -> None:
    embedder = get_embedder()
    retriever = ChromaRetriever(
        persist_dir=persist_dir,
        collection_name=collection,
        embedder=embedder,
    )


    hits = retriever.retrieve(query=query, k=k)
    if not hits:
        typer.echo(
            "No results returned.\n"
            "This usually means your collection is empty OR you're pointing at the wrong "
            "--persist-dir / --collection.\n"
            "Try: python -m petmed_rag stats"
        )
    raise typer.Exit(code=0)


    typer.echo(f"\nQuery: {query}\nTop-{k} results:\n")
    for h in hits:
        url = h.metadata.get("source_url") or h.metadata.get("url") or "N/A"
        title = h.metadata.get("title") or h.metadata.get("source_title") or "N/A"
        chunk_id = h.metadata.get("chunk_id") or h.doc_id

        preview = (h.text or "").strip().replace("\n", " ")
        preview = textwrap.shorten(preview, width=show_chars, placeholder="…")

        typer.echo(f"#{h.rank}  score={h.score:.4f}")
        typer.echo(f"   title: {title}")
        typer.echo(f"   url:   {url}")
        typer.echo(f"   id:    {chunk_id}")
        typer.echo(f"   text:  {preview}\n")

   

@app.command()
def stats(
    persist_dir: str = typer.Option(DEFAULT_PERSIST_DIR, "--persist-dir"),
    collection: str = typer.Option(DEFAULT_COLLECTION, "--collection"),
) -> None:
    """
    Show Chroma collection stats (count + available collections).
    """
    client = chromadb.PersistentClient(path=persist_dir)
    cols = client.list_collections()

    typer.echo(f"persist_dir: {persist_dir}")
    typer.echo("collections:")
    for c in cols:
        typer.echo(f"  - {c.name}")

    try:
        col = client.get_collection(name=collection)
    except Exception:
        typer.echo(f"\nCollection '{collection}' not found in {persist_dir}.")
        raise typer.Exit(code=1)

    typer.echo(f"\nactive collection: {collection}")
    typer.echo(f"count: {col.count()}")

@app.command()
def peek(
    n: int = typer.Option(3, "--n"),
    persist_dir: str = typer.Option(DEFAULT_PERSIST_DIR, "--persist-dir"),
    collection: str = typer.Option(DEFAULT_COLLECTION, "--collection"),
) -> None:
    """
    Show a few stored chunks to confirm data is in Chroma.
    """
    embedder = get_embedder()
    retriever = ChromaRetriever(
        persist_dir=persist_dir,
        collection_name=collection,
        embedder=embedder,
    )

    # Access underlying collection directly:
    col = retriever.collection
    data = col.get(include=["documents", "metadatas"], limit=n)

    ids = data.get("ids", [])
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])

    if not ids:
        typer.echo("No items found in collection.")
        raise typer.Exit(code=0)

    for i in range(len(ids)):
        typer.echo(f"\n#{i+1} id={ids[i]}")
        typer.echo(f"meta={metas[i]}")
        preview = (docs[i] or "").strip().replace("\n", " ")
        typer.echo(f"text={preview[:300]}{'…' if len(preview) > 300 else ''}")

@app.command()
def retrieve(
    query: str,
    k: int = typer.Option(5, "-k"),
    persist_dir: str = typer.Option(..., "--persist-dir"),
    collection: str = typer.Option(..., "--collection"),
    embedder: str = typer.Option(
        "openai",
        "--embedder",
        help="Embedding backend to use for querying: openai|st",
    ),
    openai_model: str = typer.Option(
        "text-embedding-3-large",
        "--openai-model",
        help="OpenAI embedding model (must match ingest)",
    ),
    st_model: str = typer.Option(
        "all-MiniLM-L6-v2",
        "--st-model",
        help="Sentence-transformers model (must match ingest)",
    ),
    print_embedder: bool = typer.Option(
        False,
        "--print-embedder",
        help="Print which embedder/model is being used (and dim when available).",
    ),
):
    ...

@app.command()
def ingest(
    input_dir: str = typer.Option(..., "--input-dir", help="Folder of .txt/.md/.html files to embed"),
    persist_dir: str = typer.Option(..., "--persist-dir"),
    collection: str = typer.Option(..., "--collection"),
    embedder: str = typer.Option("openai", "--embedder", help="openai|st"),
    openai_model: str = typer.Option("text-embedding-3-large", "--openai-model"),
    st_model: str = typer.Option("all-MiniLM-L6-v2", "--st-model"),
    chunk_size: int = typer.Option(1200, "--chunk-size"),
    overlap: int = typer.Option(150, "--overlap"),
):
    n = ingest_folder(
        input_dir=input_dir,
        persist_dir=persist_dir,
        collection=collection,
        embedder=embedder,
        openai_model=openai_model,
        st_model=st_model,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    typer.echo(f"[petmed_rag] ingested {n} chunks into '{collection}'")