import typer

app = typer.Typer(add_completion=False)

@app.command("doctor")
def doctor_cmd():
    typer.echo("✅ doctor command registered and running")

@app.command("ping")
def ping():
    typer.echo("pong")

@app.command()
def ingest():
    from petmed_rag.ingestion.ingest import ingest_folder

    n = ingest_folder(
        input_dir="./data/processed",
        persist_dir="./data/chroma",
        collection="cat-health",
        embedder="openai",
        openai_model="text-embedding-3-small",
    )
    print(f"Ingested {n} chunks")

def main():
    # DEBUG: print registered commands at runtime
    typer.echo(f"Registered commands: {list(app.registered_commands)}")
    app()

if __name__ == "__main__":
    main()
