import typer

app = typer.Typer(add_completion=False)

@app.command("doctor")
def doctor_cmd():
    typer.echo("✅ doctor command registered and running")

@app.command("ping")
def ping():
    typer.echo("pong")

def main():
    # DEBUG: print registered commands at runtime
    typer.echo(f"Registered commands: {list(app.registered_commands)}")
    app()

if __name__ == "__main__":
    main()
